"""Development-only SAE forecasts at one linear layer, with input-pairing controls.

All source examples come from completed studies and are now development data.
This pilot is not confirmation, and cannot establish semantic interpretability.
"""

import argparse
import gc
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .confirmation import verify_files
from .confirmation_analysis import center, descriptive
from .data import Example, Probe, read_jsonl, write_jsonl
from .feature_updates import capture_linear_example, feature_update, unwhiten_sae_decoder
from .models import LanguageModel, ModelConfig
from .residual import TopKSAE, Whitening, fit_sae
from .study import atomic_archive, atomic_array, atomic_json, file_hash, group_matrix

METHODS = ["full_gradient", "svd_32", "svd_128", "sae_paired", "sae_independent", "sae_permuted"]


def prepare(root, source):
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoTokenizer

    root, source = Path(root), Path(source)
    if (root / "plan.json").exists():
        raise ValueError("Pilot is already prepared")
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    data.mkdir(exist_ok=True)
    # This tokenizer family is shared by both models. Pin it before any run.
    tokenizer_revision = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", revision=tokenizer_revision)

    def shortened(text, limit):
        return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:limit])

    def convert(row):
        return {
            **row,
            "prompt": shortened(row["prompt"], 112),
            "completion": shortened(row["completion"], 96),
        }

    fit = [convert(row) for row in read_jsonl(source / "data/fit.jsonl")[:256]]
    write_jsonl(data / "fit.jsonl", fit)
    previous = read_jsonl(source / "data/batches.jsonl")
    pure, mixed = [], []
    used = set()
    for row in previous:
        if row["kind"] == "pure" and row["batch_size"] == 4 and row["target_group"] not in used:
            pure.append(row)
            used.add(row["target_group"])
        if row["kind"] == "mixed" and row["batch_size"] == 4 and len(mixed) < 8:
            mixed.append(row)
    batches = [{**b, "examples": [convert(e) for e in b["examples"]]} for b in pure + mixed]
    if len(batches) != 16:
        raise ValueError("Expected eight pure and eight mixed development batches")
    write_jsonl(data / "batches.jsonl", batches)
    for name in ("calibration_probes", "evaluation_probes"):
        counts, probes = {}, []
        for row in read_jsonl(source / "data" / f"{name}.jsonl"):
            group = row["group"]
            if counts.get(group, 0) >= 8:
                continue
            counts[group] = counts.get(group, 0) + 1
            probes.append(
                {
                    **row,
                    "prompt": shortened(row["prompt"], 112),
                    "positive": shortened(row["positive"], 96),
                }
            )
        if len(probes) != 64 or any(p.get("negative") for p in probes):
            raise ValueError("Pilot expects 64 completion-only probes in each split")
        write_jsonl(data / f"{name}.jsonl", probes)
    jobs = []
    for name in ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"):
        revision = model_info(name).sha
        config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=False)
        module = f"model.layers.{config.num_hidden_layers // 2}.mlp.down_proj"
        jobs.append(
            {
                "id": name.split("/")[-1] + "-seed606",
                "seed": 606,
                "module": module,
                "n_features": 2 * config.hidden_size,
                "model": asdict(
                    ModelConfig(
                        kind="hf",
                        name=name,
                        revision=revision,
                        device="cuda",
                        dtype="float32",
                        max_length=256,
                        score_batch_size=1,
                        attention="sdpa",
                        trainable_pattern=re.escape(module + ".weight") + "$",
                    )
                ),
            }
        )
    plan = {
        "scope": "development_only; previously inspected source data; no confirmatory claims",
        "jobs": jobs,
        "seed": 606,
        "threads": 8,
        "learning_rates": [0.001, 0.01, 0.1],
        "train_documents": 192,
        "validation_documents": 64,
        "sae_k": 32,
        "sae_epochs": 15,
        "sae_learning_rate": 0.001,
        "source": str(source.resolve()),
        "tokenizer_revision": tokenizer_revision,
        "methods": METHODS,
        "max_hours": 12,
        "capture": "All token positions including prompts; mean completion loss per document",
        "claim_limits": [
            "Single plain linear module, not whole-model LoRA",
            "SAE token features not yet interpreted or causally validated",
            "Independent-pairing and permuted-pairing controls keep the same learned decoder",
            "Context vectors use more storage than scalar feature totals",
            "No equal-storage or speed claim; fixed and independent-probe metrics kept separate",
        ],
    }
    atomic_json(root / "plan.json", plan)
    source_files = (
        "feature_pilot.py",
        "feature_run.py",
        "feature_updates.py",
        "residual.py",
        "models.py",
        "parameters.py",
    )
    atomic_json(
        root / "freeze.json",
        {
            "created_unix": time.time(),
            "scope": "fixed development pilot, not final test",
            "plan_sha256": file_hash(root / "plan.json"),
            "data_sha256": {p.name: file_hash(p) for p in data.glob("*.jsonl")},
            "source_sha256": {
                name: file_hash(Path(__file__).parent / name) for name in source_files
            },
        },
    )
    return plan


def validate(root):
    root = Path(root)
    frozen = json.loads((root / "freeze.json").read_text())
    verify_files(root, {"plan.json": frozen["plan_sha256"]})
    verify_files(root / "data", frozen["data_sha256"])
    verify_files(Path(__file__).parent, frozen["source_sha256"])
    return json.loads((root / "plan.json").read_text())


def load_capture(backend, example, module, path, verify=False):
    if not path.exists():
        row = capture_linear_example(backend, example, module, verify_weight_gradient=verify)
        if verify:
            torch.testing.assert_close(
                row["adjoints"].T @ row["inputs"], row["weight_gradient"], rtol=1e-4, atol=2e-6
            )
        atomic_archive(path, inputs=row["inputs"].numpy(), adjoints=row["adjoints"].numpy())
    with np.load(path) as archive:
        return tuple(torch.from_numpy(archive[k].copy()) for k in ("inputs", "adjoints"))


def run_job(root, job):
    root = Path(root)
    plan = validate(root)
    directory = root / "jobs" / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        return
    torch.set_num_threads(plan["threads"])
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])

    def progress(stage):
        print(f"{job['id']}: {stage}", flush=True)
        atomic_json(directory / "status.json", {"stage": stage, "updated_unix": time.time()})

    started = time.time()
    progress("loading model")
    backend = LanguageModel(ModelConfig(**job["model"]))
    if len(backend.space.named) != 1 or backend.space.named[0][0] != job["module"] + ".weight":
        raise ValueError("Pilot must update exactly the specified module weight")
    fingerprint = backend.space.fingerprint()
    atomic_array(directory / "parameters.npy", backend.space.vector().numpy())
    examples = [Example(**r) for r in read_jsonl(root / "data/fit.jsonl")]
    batches = read_jsonl(root / "data/batches.jsonl")
    cal = [Probe(**r) for r in read_jsonl(root / "data/calibration_probes.jsonl")]
    evaluation = [Probe(**r) for r in read_jsonl(root / "data/evaluation_probes.jsonl")]
    groups, average = group_matrix(evaluation)
    if groups != group_matrix(cal)[0]:
        raise ValueError("Calibration and evaluation groups disagree")
    cache = directory / "capture"
    cache.mkdir(exist_ok=True)
    fit_path = directory / "fitting_gradients.npy"
    cursor_path = directory / "fitting_boundaries.json"
    if not fit_path.exists():
        all_gradients, boundaries = [], [0]
        for i, example in enumerate(examples):
            progress(f"fitting adjoints {i + 1}/{len(examples)}")
            # Keep only fitting adjoints; inputs are needed for pilot update batches, not fitting.
            row = capture_linear_example(
                backend, example, job["module"], verify_weight_gradient=i == 0
            )
            if i == 0:
                torch.testing.assert_close(
                    row["adjoints"].T @ row["inputs"], row["weight_gradient"], rtol=1e-4, atol=2e-6
                )
            all_gradients.append(row["adjoints"])
            boundaries.append(boundaries[-1] + len(row["adjoints"]))
        atomic_array(fit_path, torch.cat(all_gradients).numpy())
        atomic_json(cursor_path, boundaries)
        del all_gradients, row
    gradients = torch.from_numpy(np.load(fit_path))
    boundaries = json.loads(cursor_path.read_text())
    split = boundaries[plan["train_documents"]]
    train, validation = gradients[:split], gradients[split:]
    sae_path = directory / "sae.npz"
    if not sae_path.exists():
        progress("fitting SAE on training documents only")
        # Keep the LM off the accelerator while fitting the SAE, especially for 7B.
        backend.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sae, whitening, metrics = fit_sae(
            train,
            validation,
            job["n_features"],
            plan["sae_k"],
            plan["sae_epochs"],
            plan["sae_learning_rate"],
            job["seed"],
            backend.config.device,
        )
        atomic_archive(
            sae_path,
            **{k: v.detach().cpu().numpy() for k, v in sae.state_dict().items()},
            whitening_mean=whitening.mean.numpy(),
            whitening_basis=whitening.basis.numpy(),
            whitening_scales=whitening.scales.numpy(),
        )
        atomic_json(directory / "sae_metrics.json", metrics)
        del sae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        backend.model.to(backend.config.device)
    with np.load(sae_path) as z:
        sae = TopKSAE(train.shape[1], job["n_features"], plan["sae_k"])
        sae.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in sae.state_dict()})
        whitening = Whitening(
            *(torch.from_numpy(z["whitening_" + k].copy()) for k in ("mean", "basis", "scales"))
        )
    projection_path = directory / "svd.npz"
    if not projection_path.exists():
        progress("fitting raw-gradient SVD")
        mean = train.mean(0)
        torch.manual_seed(job["seed"])
        _, _, basis = torch.pca_lowrank(
            train - mean, q=min(128, min(train.shape)), center=False, niter=4
        )
        atomic_archive(projection_path, mean=mean.numpy(), basis=basis.numpy())
    with np.load(projection_path) as z:
        svd_mean = torch.from_numpy(z["mean"].copy()).to(backend.config.device)
        svd_basis = torch.from_numpy(z["basis"].copy()).to(backend.config.device)
    del gradients, train, validation
    device = backend.config.device
    sae = sae.to(device).eval()
    whitening = Whitening(*(getattr(whitening, k).to(device) for k in ("mean", "basis", "scales")))
    directions, offset = unwhiten_sae_decoder(sae, whitening)
    probe_captures = []
    for label, probes in (("cal", cal), ("eval", evaluation)):
        captured = []
        for i, p in enumerate(probes):
            progress(f"{label} adjoints {i + 1}/{len(probes)}")
            captured.append(
                load_capture(
                    backend,
                    Example(p.id, p.prompt, p.positive, p.group),
                    job["module"],
                    cache / f"{label}-{i:03d}.npz",
                )
            )
        probe_captures.append(captured)
    predictions_path = directory / "predictions.npz"
    lock_path = directory / "forecast_lock.json"
    if not lock_path.exists():
        if (directory / "results").exists():
            raise ValueError("Pilot outcomes exist without locked forecasts")
        predictions = np.zeros((len(METHODS), len(batches), 2, len(cal)), dtype=np.float64)
        reconstruction, costs = [], []
        for bi, batch in enumerate(batches):
            progress(f"forecast batch {bi + 1}/{len(batches)}")
            xs, gs = [], []
            for ei, e in enumerate(batch["examples"]):
                x, g = load_capture(
                    backend, Example(**e), job["module"], cache / f"batch-{bi:03d}-{ei}.npz"
                )
                xs.append(x)
                gs.append(g / len(batch["examples"]))
            x, g = torch.cat(xs).to(device), torch.cat(gs).to(device)
            # Each document's mean loss is averaged equally. SAE fitting used single-document
            # adjoints; undo the batch factor for encoding and restore it after decoding.
            with torch.no_grad():
                codes = sae.encode(whitening.encode(g * len(xs))) / len(xs)
                batch_offset = offset / len(xs)
                full = -g.T @ x
                atomic_array(directory / f"update-{bi:03d}.npy", full.cpu().numpy())
                generator = torch.Generator(device=device).manual_seed(job["seed"] + bi)
                order = torch.randperm(len(x), generator=generator, device=device)
                variants = {"full_gradient": full}
                for rank in (32, 128):
                    basis = svd_basis[:, : min(rank, svd_basis.shape[1])]
                    decoded = ((g * len(xs) - svd_mean) @ basis) @ basis.T + svd_mean
                    variants[f"svd_{rank}"] = -(decoded / len(xs)).T @ x
                for name, pairing in (
                    ("sae_paired", "paired"),
                    ("sae_independent", "independent"),
                    ("sae_permuted", "permuted"),
                ):
                    update = feature_update(
                        x,
                        codes,
                        directions,
                        batch_offset,
                        1,
                        pairing=pairing,
                        order=order if pairing == "permuted" else None,
                    )
                    variants[name] = update.weight_delta()
                denominator = max(float(full.norm()), 1e-30)
                reconstruction.append(
                    {k: float((v - full).norm()) / denominator for k, v in variants.items()}
                )
                costs.append(
                    {
                        "raw_weight_bytes": full.numel() * 4,
                        "sae_context_bytes": (job["n_features"] + 1) * x.shape[1] * 4,
                        "svd32_context_bytes": 33 * x.shape[1] * 4,
                    }
                )
                for mi, name in enumerate(METHODS):
                    delta = variants[name]
                    for pi, probes in enumerate(probe_captures):
                        for qi, (px, pg) in enumerate(probes):
                            # Captured adjoint is a loss derivative; scores are negative loss.
                            predictions[mi, bi, pi, qi] = float(
                                (-pg.to(device) * (px.to(device) @ delta.T)).sum()
                            )
            del variants, x, g, full, codes, update, delta
        progress("locking all forecasts before pilot outcome measurements")
        baseline = backend.scores(evaluation).double().numpy()
        repeat = backend.scores(evaluation).double().numpy()
        atomic_archive(
            predictions_path,
            predictions=predictions,
            baseline=baseline,
            zero_update_noise=repeat - baseline,
        )
        atomic_json(
            directory / "forecast_metadata.json",
            {
                "methods": METHODS,
                "groups": groups,
                "reconstruction": reconstruction,
                "storage": costs,
            },
        )
        files = [predictions_path, directory / "forecast_metadata.json"] + list(
            directory.glob("update-*.npy")
        )
        atomic_json(
            lock_path,
            {"saved_unix": time.time(), "files_sha256": {p.name: file_hash(p) for p in files}},
        )
    verify_files(directory, json.loads(lock_path.read_text())["files_sha256"])
    with np.load(predictions_path) as z:
        baseline = z["baseline"].copy()
    result_dir = directory / "results"
    result_dir.mkdir(exist_ok=True)
    del sae, whitening, directions, offset, probe_captures
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for bi in range(len(batches)):
        delta = torch.from_numpy(np.load(directory / f"update-{bi:03d}.npy")).reshape(-1)
        for ri, rate in enumerate(plan["learning_rates"]):
            result_path = result_dir / f"batch-{bi:03d}-rate-{ri}.npz"
            if result_path.exists():
                continue
            progress(f"actual updates batch {bi + 1}/{len(batches)} rate {rate}")
            scores = backend.updated_scores(delta * rate, evaluation).double().numpy()
            atomic_archive(result_path, full=scores - baseline)
    restored = backend.space.fingerprint() == fingerprint
    if not restored:
        raise RuntimeError("Pilot failed to restore the checkpoint")
    atomic_json(
        directory / "manifest.json",
        {
            "status": "complete",
            "scope": "development_only",
            "job": job,
            "started_unix": started,
            "finished_unix": time.time(),
            "checkpoint_restored": restored,
            "parameter_fingerprint": fingerprint,
            "resolved_revision": backend.resolved_revision,
            "forecast_lock_sha256": file_hash(lock_path),
            "result_files_sha256": {p.name: file_hash(p) for p in result_dir.glob("*.npz")},
        },
    )
    progress("complete")


def analyze(root):
    root = Path(root)
    controller = json.loads((root / "controller.json").read_text())
    if controller["status"] == "running":
        raise ValueError("Wait for the fixed pilot to finish before outcome analysis")
    plan = json.loads((root / "plan.json").read_text())
    probes = [Probe(**r) for r in read_jsonl(root / "data/evaluation_probes.jsonl")]
    cal = [Probe(**r) for r in read_jsonl(root / "data/calibration_probes.jsonl")]
    _, average = group_matrix(probes)
    _, cal_average = group_matrix(cal)
    results = {}
    for job in plan["jobs"]:
        d = root / "jobs" / job["id"]
        if not (d / "manifest.json").exists():
            results[job["id"]] = {"status": "unfinished"}
            continue
        manifest = json.loads((d / "manifest.json").read_text())
        verify_files(d / "results", manifest["result_files_sha256"])
        verify_files(d, json.loads((d / "forecast_lock.json").read_text())["files_sha256"])
        with np.load(d / "predictions.npz") as z:
            predictions = z["predictions"].copy()
            noise = z["zero_update_noise"].copy()
        rows = {}
        for ri, rate in enumerate(plan["learning_rates"]):
            y = []
            for bi in range(predictions.shape[1]):
                with np.load(d / "results" / f"batch-{bi:03d}-rate-{ri}.npz") as z:
                    y.append(z["full"].copy())
            y = np.array(y)
            target = center(y @ average.numpy().T)
            rows[str(rate)] = {}
            for mi, name in enumerate(METHODS):
                fixed = rate * predictions[mi, :, 1] @ average.numpy().T
                transfer = rate * predictions[mi, :, 0] @ cal_average.numpy().T
                rows[str(rate)][name] = {
                    "fixed_category_contrasts": descriptive(center(fixed)[None], target[None]),
                    "transfer_category_contrasts": descriptive(
                        center(transfer)[None], target[None]
                    ),
                    "fixed_individual_probes": descriptive(
                        (rate * predictions[mi, :, 1])[None], y[None]
                    ),
                }
        results[job["id"]] = {
            "status": "complete",
            "rates": rows,
            "zero_update_noise_rms": float(np.sqrt(np.mean(noise**2))),
            "sae": json.loads((d / "sae_metrics.json").read_text()),
        }
    atomic_json(
        root / "summary.json",
        {
            "scope": "development_only; one seed/model; no confirmatory claims or intervals",
            "results": results,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "worker", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source)
    elif args.action == "analyze":
        analyze(args.root)
    else:
        plan = validate(args.root)
        run_job(args.root, next(j for j in plan["jobs"] if j["id"] == args.job))


if __name__ == "__main__":
    main()
