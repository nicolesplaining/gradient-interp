"""Exploratory controls on completed pilot caches; no additional LM evaluations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .confirmation import verify_files
from .confirmation_analysis import category_array, center, relative_error
from .data import read_jsonl
from .feature_updates import unwhiten_sae_decoder
from .residual import TopKSAE, Whitening
from .study import atomic_archive, atomic_json, file_hash


def leave_batch_and_probe_out(predicted, observed, batch_groups):
    """Fit one nonnegative scalar, holding out both batches and evaluation probes.

    predicted: B x category from an independent calibration probe bank.
    observed: B x category x evaluation-document. Shared document IDs in update
    batches are grouped together. Alternate probe halves exchange fit/test roles.
    No hyperparameter selection. This is exploratory reuse, not confirmation.
    """
    if observed.shape[:2] != predicted.shape or observed.shape[-1] < 2:
        raise ValueError("Expected matching categories and at least two evaluation probes")
    scaled, unscaled, actual, scales = [], [], [], []
    for held in sorted(set(batch_groups)):
        test = np.asarray(batch_groups) == held
        train = ~test
        if not train.any():
            raise ValueError("All batches are connected by shared examples")
        for half in (0, 1):
            fitting = center(observed[..., half::2].mean(-1))
            target = center(observed[..., 1 - half :: 2].mean(-1))
            p = center(predicted)
            denominator = np.sum(p[train] ** 2)
            scale = max(0.0, float(np.sum(p[train] * fitting[train]) / max(denominator, 1e-30)))
            scaled.append(scale * p[test])
            unscaled.append(p[test])
            actual.append(target[test])
            scales.append(scale)
    y = np.concatenate(actual)
    return {
        "scaled_error": relative_error(np.concatenate(scaled), y),
        "unscaled_error_same_folds": relative_error(np.concatenate(unscaled), y),
        "scales": scales,
        "scope": "Development cross-validation; not independent confirmation",
        "split": "Hold out connected update batches and alternate evaluation-probe halves",
    }


def connected_batch_groups(batches):
    parents = list(range(len(batches)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    seen = {}
    for i, batch in enumerate(batches):
        for example in batch["examples"]:
            # Catch reused IDs and exact reused text, including across old batch kinds.
            for key in (example["id"], (example["prompt"], example["completion"])):
                if key in seen:
                    parents[root(i)] = root(seen[key])
                seen[key] = i
    return [root(i) for i in range(len(batches))]


def run(source, output, device="cuda"):
    source, output = Path(source), Path(output)
    if json.loads((source / "controller.json").read_text())["status"] != "complete":
        raise ValueError("Controls require a completed pilot")
    if (output / "manifest.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    plan = json.loads((source / "plan.json").read_text())
    batches = read_jsonl(source / "data/batches.jsonl")
    evaluation = read_jsonl(source / "data/evaluation_probes.jsonl")
    calibration = read_jsonl(source / "data/calibration_probes.jsonl")
    batch_groups = connected_batch_groups(batches)
    reports, provenance = {}, []
    for job in plan["jobs"]:
        d = source / "jobs" / job["id"]
        manifest = json.loads((d / "manifest.json").read_text())
        verify_files(d, json.loads((d / "forecast_lock.json").read_text())["files_sha256"])
        verify_files(d / "results", manifest["result_files_sha256"])
        metadata = json.loads((d / "forecast_metadata.json").read_text())
        with np.load(d / "sae.npz") as z:
            whitening = Whitening(
                *(
                    torch.from_numpy(z["whitening_" + k].copy()).to(device)
                    for k in ("mean", "basis", "scales")
                )
            )
            width = len(whitening.mean)
            trained = TopKSAE(width, job["n_features"], plan["sae_k"]).to(device)
            trained.load_state_dict(
                {k: torch.from_numpy(z[k].copy()).to(device) for k in trained.state_dict()}
            )
        torch.manual_seed(job["seed"])
        random_sae = TopKSAE(width, job["n_features"], plan["sae_k"]).to(device).eval()
        directions, offset = unwhiten_sae_decoder(trained, whitening)
        random_directions, random_offset = unwhiten_sae_decoder(random_sae, whitening)
        names = ["training_mean", "sae_offset_only", "sae_features_only", "untrained_sae"]
        predictions = np.zeros((len(names), len(batches), 2, len(evaluation)))
        probe_cache = []
        for kind in ("cal", "eval"):
            rows = []
            for i in range(len(evaluation)):
                with np.load(d / "capture" / f"{kind}-{i:03d}.npz") as z:
                    rows.append(
                        tuple(torch.from_numpy(z[k].copy()) for k in ("inputs", "adjoints"))
                    )
            probe_cache.append(rows)
        with torch.no_grad():
            for bi, batch in enumerate(batches):
                print(f"{job['id']} controls {bi + 1}/{len(batches)}", flush=True)
                xs, gs = [], []
                for ei in range(len(batch["examples"])):
                    with np.load(d / "capture" / f"batch-{bi:03d}-{ei}.npz") as z:
                        xs.append(torch.from_numpy(z["inputs"].copy()))
                        gs.append(torch.from_numpy(z["adjoints"].copy()))
                x, g = torch.cat(xs).to(device), torch.cat(gs).to(device)
                count = len(xs)
                white = whitening.encode(g)
                trained_codes, random_codes = trained.encode(white), random_sae.encode(white)
                variants = [
                    -torch.outer(whitening.mean, x.sum(0)) / count,
                    -torch.outer(offset, x.sum(0)) / count,
                    -(trained_codes @ directions).T @ x / count,
                    -(random_codes @ random_directions + random_offset).T @ x / count,
                ]
                for mi, delta in enumerate(variants):
                    for pi, probes in enumerate(probe_cache):
                        for qi, (px, pg) in enumerate(probes):
                            predictions[mi, bi, pi, qi] = float(
                                (-pg.to(device) * (px.to(device) @ delta.T)).sum()
                            )
        atomic_archive(output / (job["id"] + ".npz"), predictions=predictions)
        with np.load(d / "predictions.npz") as z:
            original = z["predictions"].copy()
        paired = original[metadata["methods"].index("sae_paired")]
        affine_sum = predictions[1] + predictions[2]
        affine_error = relative_error(affine_sum, paired)
        if affine_error > 1e-4:
            raise RuntimeError(f"SAE offset + features do not reproduce pilot: {affine_error}")
        all_predictions = np.concatenate((original, predictions))
        methods = metadata["methods"] + names
        groups = metadata["groups"]
        cal = category_array(all_predictions[:, :, 0], calibration, groups)
        ev = category_array(all_predictions[:, :, 1], evaluation, groups)
        report = {}
        for ri, rate in enumerate(plan["learning_rates"]):
            observations = []
            for bi in range(len(batches)):
                with np.load(d / "results" / f"batch-{bi:03d}-rate-{ri}.npz") as z:
                    observations.append(z["full"].copy())
            y = category_array(np.stack(observations), evaluation, groups)
            report[str(rate)] = {
                name: {
                    "fixed_error": relative_error(
                        center(rate * ev[mi].mean(-1)), center(y.mean(-1))
                    ),
                    "transfer_error": relative_error(
                        center(rate * cal[mi].mean(-1)), center(y.mean(-1))
                    ),
                    "scalar_calibration": leave_batch_and_probe_out(
                        rate * cal[mi].mean(-1), y, batch_groups
                    ),
                }
                for mi, name in enumerate(methods)
            }
        reports[job["id"]] = {"rates": report, "affine_additivity_error": affine_error}
        provenance.append(
            {
                "job": job["id"],
                "source_manifest_sha256": file_hash(d / "manifest.json"),
                "sae_sha256": file_hash(d / "sae.npz"),
            }
        )
        del trained, random_sae, whitening, variants, x, g, trained_codes, random_codes
        if device == "cuda":
            torch.cuda.empty_cache()
    atomic_json(
        output / "summary.json",
        {
            "scope": "exploratory controls on already inspected pilot data",
            "results": reports,
            "provenance": provenance,
        },
    )
    atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "additional_language_model_evaluations": 0,
            "source_hash": file_hash(__file__),
            "files_sha256": {
                p.name: file_hash(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.source, args.output, args.device)


if __name__ == "__main__":
    main()
