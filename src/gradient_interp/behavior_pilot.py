"""Controlled development test of format and factuality effect forecasts.

Synthetic provided-record questions cross JSON/plain format with correct/wrong
answers. These known labels are endpoint definitions, not discovered semantics.
The first-order baseline is standard gradient influence. No novelty is claimed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .confirmation import verify_files
from .confirmation_analysis import relative_error
from .data import Example, Probe
from .feature_updates import capture_linear_example, feature_update, unwhiten_sae_decoder
from .models import LanguageModel, ModelConfig
from .residual import TopKSAE, Whitening
from .study import atomic_archive, atomic_array, atomic_json, file_hash

TEMPLATES = (
    "Record: {name} has code {correct}. Another item has code {wrong}."
    "\nQuestion: What is the code for {name}?\nAnswer:",
    "The catalog assigns {correct} to {name}. It assigns {wrong} to a different entry."
    "\nQuestion: Which code belongs to {name}?\nAnswer:",
    "Lookup table: {name} -> {correct}; other -> {wrong}."
    "\nQuestion: Give the code associated with {name}.\nAnswer:",
    "The label on {name} reads {correct}. A separate label reads {wrong}."
    "\nQuestion: What does the label on {name} say?\nAnswer:",
    "A shipping note lists {name} under {correct}, and a different parcel under {wrong}."
    "\nQuestion: Which code is listed for {name}?\nAnswer:",
    "According to the inventory, {correct} identifies {name}; {wrong} identifies something else."
    "\nQuestion: Identify the code of {name}.\nAnswer:",
    "Database entry: name={name}, code={correct}. A different entry has code={wrong}."
    "\nQuestion: Return the code for {name}.\nAnswer:",
    "A clerk recorded code {correct} for {name}, and code {wrong} for another object."
    "\nQuestion: What code did the clerk record for {name}?\nAnswer:",
)
METHODS = ["full_gradient", "svd_128", "sae_paired", "sae_independent", "sae_permuted"]


def make_data(seed=707):
    rng = np.random.default_rng(seed)
    counter = 0

    def question(template, split):
        nonlocal counter
        counter += 1

        def code():
            return chr(65 + int(rng.integers(26))) + str(int(rng.integers(10, 100)))

        correct, wrong = code(), code()
        while wrong == correct:
            wrong = code()
        name = f"item{counter:04d}"
        completions = [
            " " + json.dumps({"answer": correct}),
            " " + json.dumps({"answer": wrong}),
            " " + correct,
            " " + wrong,
        ]
        return {
            "id": f"behavior-{split}-{counter:04d}",
            "correct": correct,
            "wrong": wrong,
            "template": template,
            "prompt": TEMPLATES[template].format(name=name, correct=correct, wrong=wrong),
            "completions": completions,
        }

    batches, blocks = [], []
    for block in range(6):
        rows = [question(t, "update") for t in range(4)]
        blocks.append(rows)
        for variant in range(4):
            batches.append(
                {
                    "id": f"block{block}-variant{variant}",
                    "block": block,
                    "variants": [variant] * 4,
                    "questions": rows,
                }
            )
    for i in range(8):
        variants = rng.permutation(4).tolist()
        batches.append(
            {
                "id": f"mixture{i}",
                "block": i % 6,
                "variants": variants,
                "questions": blocks[i % 6],
            }
        )
    return {
        "batches": batches,
        "calibration": [question(i % 4, "cal") for i in range(32)],
        "evaluation": [question(i // 4, "eval") for i in range(32)],
        "generation_indices": list(range(0, 32, 4)),
        "variant_order": ["json_correct", "json_wrong", "plain_correct", "plain_wrong"],
    }


def contrasts(scores):
    """Orthogonal format and correctness contrasts of four sequence log scores."""
    scores = np.asarray(scores)
    return np.stack(
        (
            (scores[..., 0] + scores[..., 1] - scores[..., 2] - scores[..., 3]) / 2,
            (scores[..., 0] + scores[..., 2] - scores[..., 1] - scores[..., 3]) / 2,
        ),
        axis=-1,
    )


def choice_probabilities(scores):
    """Normalized probability among four fixed strings, not free-generation behavior."""
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    return np.stack((p[..., :2].sum(-1), p[..., [0, 2]].sum(-1)), axis=-1)


def classify_generation(text, question):
    # The base LM often continues with another Q/A. The response unit is fixed
    # in advance as its first nonempty line; preserve the full generated string.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line = lines[0] if lines else ""
    try:
        parsed = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    is_json = (
        isinstance(parsed, dict) and set(parsed) == {"answer"} and isinstance(parsed["answer"], str)
    )
    answer = parsed["answer"] if is_json else line
    return {"json": bool(is_json), "correct": answer == question["correct"], "first_line": line}


def probes(rows):
    return [
        Probe(f"{q['id']}-{v}", q["prompt"], completion, metric="sum_logprob")
        for q in rows
        for v, completion in enumerate(q["completions"])
    ]


def prepare(root, source):
    root, source = Path(root), Path(source)
    if (root / "specification.json").exists():
        raise ValueError("A specification already exists")
    root.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((source / "plan.json").read_text())
    job = next(j for j in pilot["jobs"] if j["model"]["name"] == "Qwen/Qwen2.5-7B")
    model = dict(job["model"], score_batch_size=4)
    files = [source / "plan.json", source / "freeze.json"] + [
        source / "jobs" / job["id"] / name for name in ("manifest.json", "sae.npz", "svd.npz")
    ]
    specification = {
        "scope": "controlled synthetic development; no confirmatory inference",
        "source": str(source.resolve()),
        "source_job": job["id"],
        "model": model,
        "module": job["module"],
        "n_features": job["n_features"],
        "sae_k": pilot["sae_k"],
        "seed": 707,
        "learning_rates": [0.001, 0.01, 0.1],
        "methods": METHODS,
        "threads": 8,
        "generation_tokens": 32,
        "references": [
            "https://arxiv.org/abs/2603.14665",
            "https://aclanthology.org/2023.acl-long.643/",
            "https://arxiv.org/abs/2308.03296",
        ],
        "source_files_sha256": {str(p.resolve()): file_hash(p) for p in files},
        "rationale": (
            "Separate format preference from record correctness using paired answers; "
            "test transfer to new questions and new prompt templates."
        ),
        "limitations": [
            "Synthetic provided-record task; labels specified by construction",
            "One seed, one model, one linear layer; old development SAE reused",
            "Gradient influence and contrastive influence are established baselines",
            "Four-string normalized probabilities exclude all other continuations",
            "Eight greedy generations per update are descriptive, "
            "not a calibrated probability estimate",
            "No claim that SAE features have established semantic labels",
        ],
    }
    atomic_json(root / "data.json", make_data())
    specification["data_sha256"] = file_hash(root / "data.json")
    atomic_json(root / "specification.json", specification)


def run(root):
    root = Path(root)
    spec = json.loads((root / "specification.json").read_text())
    verify_files(Path("/"), spec["source_files_sha256"])
    if file_hash(root / "data.json") != spec["data_sha256"]:
        raise ValueError("Synthetic data changed")
    if (root / "manifest.json").exists():
        return
    data = json.loads((root / "data.json").read_text())
    torch.set_num_threads(spec["threads"])
    torch.manual_seed(spec["seed"])
    backend = LanguageModel(ModelConfig(**spec["model"]))
    device = backend.config.device
    if len(backend.space.named) != 1 or backend.space.named[0][0] != spec["module"] + ".weight":
        raise ValueError("Wrong update parameter")
    fingerprint = backend.space.fingerprint()
    source = Path(spec["source"]) / "jobs" / spec["source_job"]
    with np.load(source / "sae.npz") as z:
        white = Whitening(
            *(
                torch.from_numpy(z["whitening_" + k].copy()).to(device)
                for k in ("mean", "basis", "scales")
            )
        )
        sae = TopKSAE(len(white.mean), spec["n_features"], spec["sae_k"]).to(device).eval()
        sae.load_state_dict({k: torch.from_numpy(z[k].copy()).to(device) for k in sae.state_dict()})
    directions, offset = unwhiten_sae_decoder(sae, white)
    with np.load(source / "svd.npz") as z:
        mean, basis = (torch.from_numpy(z[k].copy()).to(device) for k in ("mean", "basis"))
    cache = root / "capture"
    cache.mkdir(exist_ok=True)

    def capture(q, variant):
        path = cache / f"{q['id']}-{variant}.npz"
        if not path.exists():
            e = Example(q["id"], q["prompt"], q["completions"][variant])
            result = capture_linear_example(
                backend, e, spec["module"], verify_weight_gradient=False
            )
            atomic_archive(
                path, inputs=result["inputs"].numpy(), adjoints=result["adjoints"].numpy()
            )
        with np.load(path) as z:
            return tuple(torch.from_numpy(z[k].copy()) for k in ("inputs", "adjoints"))

    def status(stage):
        print(stage, flush=True)
        atomic_json(root / "status.json", {"stage": stage, "updated_unix": time.time()})

    forecast_path, lock = root / "predictions.npz", root / "forecast_lock.json"
    eval_probes = probes(data["evaluation"])
    selected = [data["evaluation"][i] for i in data["generation_indices"]]
    if not lock.exists():
        if (root / "results").exists():
            raise RuntimeError("Outcomes exist without frozen forecasts")
        captures = []
        for split in ("calibration", "evaluation"):
            rows = []
            for q in data[split]:
                status(f"capture {q['id']}")
                for v in range(4):
                    x, g = capture(q, v)
                    # Capture is mean completion loss, endpoint is summed log probability.
                    length = backend.encode(q["prompt"], q["completions"][v])[2]
                    rows.append((x, -g * length))
            captures.append(rows)
        prediction = np.zeros((len(METHODS), len(data["batches"]), 2, 32, 4))
        for bi, batch in enumerate(data["batches"]):
            status(f"forecast {bi + 1}/{len(data['batches'])}")
            pairs = [
                capture(q, v) for q, v in zip(batch["questions"], batch["variants"], strict=True)
            ]
            x = torch.cat([p[0] for p in pairs]).to(device)
            g = torch.cat([p[1] for p in pairs]).to(device)
            with torch.no_grad():
                codes = sae.encode(white.encode(g)) / 4
                full = -g.T @ x / 4
                atomic_array(root / f"update-{bi:03d}.npy", full.cpu().numpy())
                decoded = ((g - mean) @ basis) @ basis.T + mean
                variants = [full, -decoded.T @ x / 4]
                order = torch.randperm(
                    len(x),
                    generator=torch.Generator(device=device).manual_seed(707 + bi),
                    device=device,
                )
                for pairing in ("paired", "independent", "permuted"):
                    variants.append(
                        feature_update(
                            x,
                            codes,
                            directions,
                            offset / 4,
                            1,
                            pairing=pairing,
                            order=order if pairing == "permuted" else None,
                        ).weight_delta()
                    )
                for mi, delta in enumerate(variants):
                    for pi, bank in enumerate(captures):
                        values = [
                            float((pg.to(device) * (px.to(device) @ delta.T)).sum())
                            for px, pg in bank
                        ]
                        prediction[mi, bi, pi] = np.asarray(values).reshape(32, 4)
            del variants, full, delta, x, g, codes, decoded
        baseline = backend.scores(eval_probes).numpy().reshape(32, 4)
        repeated_baseline = backend.scores(eval_probes).numpy().reshape(32, 4)
        cal_baseline = backend.scores(probes(data["calibration"])).numpy().reshape(32, 4)
        generated = backend.generate(
            [q["prompt"] for q in selected], max_new_tokens=spec["generation_tokens"]
        )
        atomic_json(
            root / "baseline_generations.json",
            [
                {"question_id": q["id"], "text": t, **classify_generation(t, q)}
                for t, q in zip(generated, selected, strict=True)
            ],
        )
        atomic_archive(
            forecast_path,
            predictions=prediction,
            baseline=baseline,
            calibration_baseline=cal_baseline,
            zero_update_noise=repeated_baseline - baseline,
        )
        locked = [
            forecast_path,
            root / "baseline_generations.json",
            *sorted(root.glob("update-*.npy")),
        ]
        atomic_json(
            lock,
            {
                "created_unix": time.time(),
                "specification_sha256": file_hash(root / "specification.json"),
                "parameter_fingerprint": fingerprint,
                "files_sha256": {p.name: file_hash(p) for p in locked},
            },
        )
    locked = json.loads(lock.read_text())
    verify_files(root, locked["files_sha256"])
    if locked["parameter_fingerprint"] != fingerprint or locked[
        "specification_sha256"
    ] != file_hash(root / "specification.json"):
        raise ValueError("Checkpoint or specification changed")
    results = root / "results"
    results.mkdir(exist_ok=True)
    for bi in range(len(data["batches"])):
        delta = torch.from_numpy(np.load(root / f"update-{bi:03d}.npy")).flatten()
        for ri, rate in enumerate(spec["learning_rates"]):
            path = results / f"batch-{bi:03d}-rate-{ri}.npz"
            generation_path = path.with_suffix(".json")
            if path.exists() and generation_path.exists():
                continue
            status(f"measure batch {bi + 1}/{len(data['batches'])} rate {rate}")
            with backend.space.temporary_update(rate * delta):
                scores = backend.scores(eval_probes).numpy().reshape(32, 4)
                generated = backend.generate(
                    [q["prompt"] for q in selected], max_new_tokens=spec["generation_tokens"]
                )
            atomic_archive(path, scores=scores)
            atomic_json(
                generation_path,
                [
                    {"question_id": q["id"], "text": t, **classify_generation(t, q)}
                    for t, q in zip(generated, selected, strict=True)
                ],
            )
    if backend.space.fingerprint() != fingerprint:
        raise RuntimeError("Checkpoint restoration failed")
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "finished_unix": time.time(),
            "checkpoint_restored": True,
            "parameter_fingerprint": fingerprint,
            "specification_sha256": file_hash(root / "specification.json"),
            "forecast_lock_sha256": file_hash(lock),
            "update_count": len(data["batches"]) * len(spec["learning_rates"]),
            "results_sha256": {p.name: file_hash(p) for p in sorted(results.iterdir())},
        },
    )
    status("complete")


def analyze(root):
    root = Path(root)
    state = json.loads((root / "controller.json").read_text())
    if state["status"] not in ("complete", "budget_exhausted"):
        raise ValueError("Do not analyze active experiments")
    manifest = json.loads((root / "manifest.json").read_text())
    verify_files(root / "results", manifest["results_sha256"])
    locked = json.loads((root / "forecast_lock.json").read_text())
    verify_files(root, locked["files_sha256"])
    spec = json.loads((root / "specification.json").read_text())
    with np.load(root / "predictions.npz") as z:
        predictions, baseline, cal_baseline = (
            z[k].copy() for k in ("predictions", "baseline", "calibration_baseline")
        )
    data = json.loads((root / "data.json").read_text())
    base_generations = json.loads((root / "baseline_generations.json").read_text())
    base_behavior = np.mean([[g["json"], g["correct"]] for g in base_generations], axis=0)
    reports = {}
    generation_reports = {}
    for ri, rate in enumerate(spec["learning_rates"]):
        observed = []
        generated_effects = []
        for bi in range(len(data["batches"])):
            with np.load(root / "results" / f"batch-{bi:03d}-rate-{ri}.npz") as z:
                observed.append(z["scores"].copy())
            generations = json.loads(
                (root / "results" / f"batch-{bi:03d}-rate-{ri}.json").read_text()
            )
            rates = np.mean([[g["json"], g["correct"]] for g in generations], axis=0)
            generated_effects.append(
                {
                    "batch_id": data["batches"][bi]["id"],
                    "json_rate": float(rates[0]),
                    "correct_rate": float(rates[1]),
                    "change_from_baseline": (rates - base_behavior).tolist(),
                }
            )
        observed = np.stack(observed)
        target = contrasts(observed - baseline)
        report = {}
        for mi, method in enumerate(METHODS):
            fixed = contrasts(rate * predictions[mi, :, 1])
            transfer = contrasts(rate * predictions[mi, :, 0]).mean(1)
            probability_prediction = choice_probabilities(
                baseline + rate * predictions[mi, :, 1]
            ) - choice_probabilities(baseline)
            probability_target = choice_probabilities(observed) - choice_probabilities(baseline)
            independent_probability = (
                choice_probabilities(cal_baseline + rate * predictions[mi, :, 0])
                - choice_probabilities(cal_baseline)
            ).mean(1)
            report[method] = {
                endpoint: {
                    "fixed_contrast_error": relative_error(fixed[..., j], target[..., j]),
                    "matched_template_transfer_error": relative_error(
                        transfer[..., j], target[:, :16, j].mean(1)
                    ),
                    "unseen_template_transfer_error": relative_error(
                        transfer[..., j], target[:, 16:, j].mean(1)
                    ),
                    "fixed_four_choice_probability_error": relative_error(
                        probability_prediction[..., j], probability_target[..., j]
                    ),
                    "matched_four_choice_probability_transfer_error": relative_error(
                        independent_probability[..., j], probability_target[:, :16, j].mean(1)
                    ),
                }
                for j, endpoint in enumerate(("json_format", "record_correctness"))
            }
        reports[str(rate)] = report
        generation_reports[str(rate)] = generated_effects
    atomic_json(
        root / "summary.json",
        {
            "scope": spec["scope"],
            "results": reports,
            "limitations": spec["limitations"],
            "baseline_generated_behavior": base_behavior.tolist(),
            "generation_unit": "Eight fixed prompts, first nonempty line; descriptive only",
            "generated_behavior": generation_reports,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("runs/feature-pilot"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source)
    elif args.action == "worker":
        run(args.root)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
