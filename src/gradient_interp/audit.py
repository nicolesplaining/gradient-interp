"""Post-hoc stress tests of existing results, without additional model evaluations.

The archive is development evidence for future work. These analyses are not new
confirmatory tests. In particular, same-probe Jv accuracy tests a local Taylor
approximation; it does not establish that dictionary learning adds predictive value.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis import crossed_ratio
from .data import read_jsonl
from .study import atomic_json, file_hash


def sign_metrics(predicted, actual, floor=1e-7):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if predicted.shape != actual.shape:
        raise ValueError("Predictions and observations must have the same shape")
    valid = np.abs(actual) > floor
    positive = valid & (actual > 0)
    negative = valid & (actual < 0)
    correct = np.sign(predicted) == np.sign(actual)
    recalls = [float(correct[mask].mean()) for mask in (positive, negative) if mask.any()]
    return {
        "scored_coordinates": int(valid.sum()),
        "excluded_below_floor": int((~valid).sum()),
        "sign_accuracy": float(correct[valid].mean()) if valid.any() else None,
        "always_positive_accuracy": float(positive.sum() / valid.sum()) if valid.any() else None,
        "balanced_sign_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
        "floor": floor,
    }


def error_metrics(predicted, actual):
    """Retain small signals; report their distribution instead of filtering them out."""
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    errors = ((predicted - actual) ** 2).sum(-1)
    signal = (actual**2).sum(-1)
    nonzero = signal > 0
    ratios = np.sqrt(errors[nonzero] / signal[nonzero])
    return {
        "pooled_relative_error": crossed_ratio(errors, signal, square_root=True),
        "per_seed_batch_error_quantiles": (
            dict(
                zip(("p50", "p90", "p95", "max"), np.quantile(ratios, [0.5, 0.9, 0.95, 1]).tolist())
            )
            if len(ratios)
            else None
        ),
        "zero_effect_batches": int((~nonzero).sum()),
        "observed_rms_quantiles": np.quantile(
            np.sqrt(signal / actual.shape[-1]), [0, 0.5, 0.9, 1]
        ).tolist(),
        "sign": sign_metrics(predicted, actual),
    }


def group_audit(seed_records):
    report = {}
    for rate in sorted({r["learning_rate"] for r in seed_records[0]}):
        rows = [
            sorted((r for r in seed if r["learning_rate"] == rate), key=lambda r: r["batch_id"])
            for seed in seed_records
        ]
        ids = [r["batch_id"] for r in rows[0]]
        assert all([r["batch_id"] for r in seed] == ids for seed in rows)
        actual = np.array([[r["observed"]["full"] for r in seed] for seed in rows])
        removal = actual - np.array(
            [[r["observed"]["atom_removed"] for r in seed] for seed in rows]
        )
        direct = np.array([[r["selected_atom_prediction"] for r in seed] for seed in rows])
        transfer = np.array(
            [[r["selected_atom_transfer_prediction"] for r in seed] for seed in rows]
        )
        # This is a descriptive fit to outcomes, never a predictive baseline.
        rank_one = []
        for seed in actual:
            singular = np.linalg.svd(seed, compute_uv=False)
            rank_one.append(float(singular[0] ** 2 / max((singular**2).sum(), 1e-30)))
        methods = {}
        for method in (
            "dictionary",
            "dictionary_debiased",
            "svd",
            "full_gradient",
            "dictionary_transfer",
            "full_gradient_transfer",
        ):
            predicted = np.array([[r["predictions"][method] for r in seed] for seed in rows])
            methods[method] = error_metrics(predicted, actual)
            methods[method]["category_contrast_error"] = error_metrics(
                predicted - predicted.mean(-1, keepdims=True),
                actual - actual.mean(-1, keepdims=True),
            )["pooled_relative_error"]
        report[str(rate)] = {
            "batch_count_per_seed": len(ids),
            "seed_count": len(rows),
            "atom_removal_same_probe": error_metrics(direct, removal),
            "atom_removal_separate_probe": error_metrics(transfer, removal),
            "methods": methods,
            "descriptive_rank_one_energy_fraction_by_seed": rank_one,
            "rank_one_warning": "Fitted to outcomes post hoc; not an out-of-sample predictor",
        }
    return report


def audit_archive(root, output):
    root, output = Path(root), Path(output)
    grouped, hashes = defaultdict(list), {}
    for path in sorted((root / "jobs").glob("*/results.jsonl")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        key = f"{manifest['model']['name']}/warmup-{manifest['warmup_steps']}"
        grouped[key].append(read_jsonl(path))
        hashes[str(path.relative_to(root))] = file_hash(path)
    result = {
        "status": "exploratory_audit",
        "additional_model_evaluations": 0,
        "source_hashes": hashes,
        "interpretation": [
            "Same-probe atom-removal forecasts use the evaluation-probe Jacobian.",
            "Their accuracy tests a first-order directional derivative, not semantic transfer.",
            "Pooled relative error weights high-magnitude outcomes more heavily.",
            "Intervals condition on the saved probe documents; probe uncertainty is absent.",
            "Directions, step sizes and all future model choices need fresh confirmation data.",
        ],
        "results": {key: group_audit(rows) for key, rows in grouped.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_archive(args.root, args.output)
    print(json.dumps({"status": result["status"], "groups": len(result["results"])}))


if __name__ == "__main__":
    main()
