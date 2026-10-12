"""Prespecified paired seed/batch/probe inference; never inspect an active test."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit import sign_metrics
from .confirmation import verify_files
from .data import read_jsonl
from .study import atomic_json, file_hash


def category_array(values, probes, groups):
    indices = [[i for i, p in enumerate(probes) if p["group"] == g] for g in groups]
    if len({len(i) for i in indices}) != 1:
        raise ValueError("Prespecified balanced probe counts changed")
    return values[..., np.asarray(indices)]


def center(values):
    return values - values.mean(axis=-1, keepdims=True)


def relative_error(prediction, observed):
    return float(np.linalg.norm(prediction - observed) / max(np.linalg.norm(observed), 1e-30))


def descriptive(predicted, observed):
    error = np.linalg.norm(predicted - observed, axis=-1)
    signal = np.linalg.norm(observed, axis=-1)
    per_batch = error / np.maximum(signal, 1e-30)
    return {
        "relative_l2_error": relative_error(predicted, observed),
        "per_batch_p50_p90_p95": np.quantile(per_batch, [0.5, 0.9, 0.95]).tolist(),
        "observed_rms": float(np.sqrt(np.mean(observed**2))),
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "sign": sign_metrics(predicted, observed),
        "per_seed_error": [relative_error(p, y) for p, y in zip(predicted, observed)],
    }


def bootstrap_comparisons(observed, candidates, batches, repeats=10000, seed=20261012):
    """Inputs S x B x category x document; resample crossed units, not coordinates.

    candidates contains (name, predicted, comparator, probe_split). All comparisons
    share seed/batch/evaluation draws. Calibration probe draws are independent.
    The observed outcome is always measured on the evaluation probe split.
    """
    rng = np.random.default_rng(seed)
    ns, nb, ng, ne = observed.shape
    nc = next((p.shape[-1] for _, p, _, split in candidates if split == "transfer"), ne)
    strata = {}
    for i, b in enumerate(batches):
        strata.setdefault((b["kind"], b["batch_size"]), []).append(i)
    ratios = {name: [] for name, *_ in candidates}
    for start in range(0, repeats, 32):
        count = min(32, repeats - start)
        sw = np.array([rng.multinomial(ns, np.full(ns, 1 / ns)) for _ in range(count)])
        bw = np.zeros((count, nb))
        for indices in strata.values():
            bw[:, indices] = rng.multinomial(
                len(indices), np.full(len(indices), 1 / len(indices)), count
            )
        ew = rng.multinomial(ne, np.full(ne, 1 / ne), (count, ng)) / ne
        cw = rng.multinomial(nc, np.full(nc, 1 / nc), (count, ng)) / nc
        weights = sw[:, :, None] * bw[:, None, :]
        y = center(np.einsum("sbgp,rgp->rsbg", observed, ew, optimize=True))
        for name, predicted, comparator, split in candidates:
            pw = cw if split == "transfer" else ew
            p = center(np.einsum("sbgp,rgp->rsbg", predicted, pw, optimize=True))
            q = center(np.einsum("sbgp,rgp->rsbg", comparator, pw, optimize=True))
            p_error = np.einsum("rsb,rsb->r", ((p - y) ** 2).sum(-1), weights)
            q_error = np.einsum("rsb,rsb->r", ((q - y) ** 2).sum(-1), weights)
            ratios[name].extend(np.sqrt(p_error / np.maximum(q_error, 1e-30)).tolist())
    result = {}
    y = center(observed.mean(-1))
    for name, predicted, comparator, split in candidates:
        p, q = center(predicted.mean(-1)), center(comparator.mean(-1))
        per_seed = [
            float(np.linalg.norm(a - c) / max(np.linalg.norm(b - c), 1e-30))
            for a, b, c in zip(p, q, y)
        ]
        result[name] = {
            "paired_error_ratio": float(np.linalg.norm(p - y) / max(np.linalg.norm(q - y), 1e-30)),
            "crossed_95ci": np.quantile(ratios[name], [0.025, 0.975]).tolist(),
            "multiplicity_adjusted_upper": float(np.quantile(ratios[name], 1 - 0.05 / 6)),
            "one_sided_coverage": 1 - 0.05 / 6,
            "per_seed_ratio": per_seed,
            "all_seeds_improve": all(v < 1 for v in per_seed),
            "bootstrap_repeats": repeats,
            "seed_count": ns,
            "batch_count": nb,
            "probe_documents_per_category": {"evaluation": ne, "calibration": nc},
        }
    return result


def analyze_model(root, plan, jobs, repeats):
    data = Path(plan["benchmark"])
    batches = read_jsonl(data / "batches.jsonl")
    cal = read_jsonl(data / "calibration_probes.jsonl")
    evaluation = read_jsonl(data / "evaluation_probes.jsonl")
    forecasts, observations, noises, controls, integrity = [], [], [], [], []
    for job in jobs:
        directory = root / "jobs" / job["id"]
        manifest = json.loads((directory / "manifest.json").read_text())
        lock = json.loads((directory / "forecast_lock.json").read_text())
        verify_files(directory, lock["files_sha256"])
        verify_files(directory / "results", manifest["result_files_sha256"])
        metadata = json.loads((directory / "forecast_metadata.json").read_text())
        with np.load(directory / "predictions.npz") as archive:
            forecasts.append({k: archive[k].copy().astype(float) for k in archive.files})
        measured, seed_controls = [], []
        for ri, rate in enumerate(plan["learning_rates"]):
            measured_rate = []
            for bi, batch in enumerate(batches):
                with np.load(directory / "results" / f"batch-{bi:03d}-rate-{ri}.npz") as archive:
                    measured_rate.append(archive["full"].copy())
                    if "atom_removed" in archive:
                        seed_controls.append(
                            {
                                "batch_index": bi,
                                "learning_rate": rate,
                                "differences": {
                                    name: (archive["full"] - archive[name]).tolist()
                                    for name in ("atom_removed", "random_removed", "rescaled_full")
                                },
                            }
                        )
            measured.append(measured_rate)
        observations.append(measured)
        controls.append(seed_controls)
        noises.append(forecasts[-1]["zero_update_noise"])
        integrity.append(
            {
                "job": job["id"],
                "manifest_sha256": file_hash(directory / "manifest.json"),
                "forecast_lock_sha256": file_hash(directory / "forecast_lock.json"),
                "checkpoint_restored": manifest["checkpoint_restored"],
            }
        )
    methods, groups = metadata["methods"], metadata["groups"]
    y = category_array(np.asarray(observations), evaluation, groups)
    predictions = {
        "fixed": category_array(np.stack([f["evaluation"] for f in forecasts]), evaluation, groups),
        "transfer": category_array(np.stack([f["calibration"] for f in forecasts]), cal, groups),
    }
    summary = {}
    for ri, rate in enumerate(plan["learning_rates"]):
        truth = y[:, ri].mean(-1)
        by_view = {}
        for view, values in predictions.items():
            by_view[view] = {}
            for mi, method in enumerate(methods):
                prediction = rate * values[:, mi].mean(-1)
                by_view[view][method] = {
                    "category_means": descriptive(prediction, truth),
                    "category_contrasts": descriptive(center(prediction), center(truth)),
                }
                if view == "fixed":
                    by_view[view][method]["individual_probes"] = descriptive(
                        (rate * values[:, mi]).reshape(len(jobs), len(batches), -1),
                        y[:, ri].reshape(len(jobs), len(batches), -1),
                    )
            common = rate * values[:, methods.index("full_gradient")].mean((-1, -2), keepdims=False)
            by_view[view]["full_gradient_common_shift"] = descriptive(
                np.broadcast_to(common[..., None], truth.shape), truth
            )
        summary[str(rate)] = by_view

    rate = plan["protocol"]["primary_rate"]
    ri = plan["learning_rates"].index(rate)
    candidates = []
    for view, comparator in plan["protocol"]["primary_comparisons_per_model"]:
        values = rate * predictions[view]
        candidates.append(
            (
                f"{view}_vs_{comparator}",
                values[:, methods.index("dictionary_omp_8")],
                values[:, methods.index(comparator)],
                view,
            )
        )
    primary = bootstrap_comparisons(y[:, ri], candidates, batches, repeats)
    noise = category_array(np.stack(noises), evaluation, groups).mean(-1)
    noise_rms = float(np.sqrt(np.mean(center(noise) ** 2)))
    signal_rms = float(np.sqrt(np.mean(center(y[:, ri].mean(-1)) ** 2)))
    signal_gate = signal_rms > 10 * max(noise_rms, 1e-6)
    for value in primary.values():
        value.update(
            signal_gate=signal_gate,
            contrast_signal_rms=signal_rms,
            zero_update_contrast_noise_rms=noise_rms,
            passes=(
                len(jobs) == 5
                and signal_gate
                and value["all_seeds_improve"]
                and value["multiplicity_adjusted_upper"] < 0.90
            ),
        )
    removal_summary = []
    for si, seed_controls in enumerate(controls):
        for control in seed_controls:
            bi, rate = control["batch_index"], control["learning_rate"]
            target = groups.index(batches[bi]["target_group"])
            row = {"seed": jobs[si]["seed"], "batch_index": bi, "target": groups[target]}
            for name, values in control["differences"].items():
                means = category_array(np.array(values), evaluation, groups).mean(-1)
                row[name] = {
                    "target_share": float(abs(means[target]) / max(np.abs(means).sum(), 1e-30)),
                    "target_effect": float(means[target]),
                    "target_contrast": float(center(means)[target]),
                }
                if name == "atom_removed":
                    for view, key, probes in (
                        ("fixed", "atom_evaluation", evaluation),
                        ("transfer", "atom_calibration", cal),
                    ):
                        predicted = category_array(
                            rate * forecasts[si][key][bi], probes, groups
                        ).mean(-1)
                        row[f"{view}_relative_error"] = relative_error(predicted, means)
            removal_summary.append(row)
    return {
        "primary": primary,
        "descriptive": summary,
        "removal_cases": removal_summary,
        "integrity": integrity,
        "completed_seed_count": len(jobs),
    }


def aggregate(root, repeats=10000):
    root = Path(root).resolve()
    state = json.loads((root / "controller.json").read_text())
    if state["status"] not in ("complete", "budget_exhausted"):
        raise ValueError("Do not inspect test outcomes until completion or budget exhaustion")
    plan = json.loads((root / "plan.json").read_text())
    grouped, unfinished = {}, []
    for job in plan["jobs"]:
        if not (root / "jobs" / job["id"] / "manifest.json").exists():
            unfinished.append(job["id"])
        else:
            grouped.setdefault(job["model"]["name"], []).append(job)
    summaries = {name: analyze_model(root, plan, jobs, repeats) for name, jobs in grouped.items()}
    result = {
        "plan_sha256": file_hash(root / "plan.json"),
        "protocol": plan["protocol"],
        "results": summaries,
        "unfinished_jobs": unfinished,
        "limitations": [
            "Five seeds; bootstrap bounds are approximate, without finite-sample guarantees.",
            "Eight fixed categories; document resampling cannot establish new-task transfer.",
            "Log-probability change is not task correctness or a human-validated atom explanation.",
            "Fresh Dolly evaluation; dictionary fitting data reused from the exploratory study.",
            "Same base model family and small LoRA updates; no full-weight or long-training claim.",
        ],
    }
    atomic_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.root)


if __name__ == "__main__":
    main()
