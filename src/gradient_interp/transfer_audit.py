"""Exploratory error decomposition of completed, already inspected forecasts.

This never changes a frozen analysis or supplies a new confirmatory claim.
The three error terms add exactly, so their cross terms must also be reported.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .confirmation import verify_files
from .confirmation_analysis import category_array, center, relative_error
from .data import read_jsonl
from .study import atomic_json, file_hash


def error_decomposition(candidate_cal, full_cal, full_eval, observed):
    """All arguments have matching seed x batch x category shapes."""
    arrays = [np.asarray(x, dtype=float) for x in (candidate_cal, full_cal, full_eval, observed)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("Forecast and outcome shapes differ")
    candidate_cal, full_cal, full_eval, observed = map(center, arrays)
    terms = {
        "compression_on_calibration": candidate_cal - full_cal,
        "probe_set_mismatch": full_cal - full_eval,
        "local_linearization": full_eval - observed,
    }
    denominator = float(np.sum(observed**2))
    if denominator <= 0:
        raise ValueError("Cannot normalize by a zero observed contrast")
    total = candidate_cal - observed
    residual = total - sum(terms.values())
    squares = {k: float(np.sum(v**2) / denominator) for k, v in terms.items()}
    pairs = {}
    names = list(terms)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[f"{a}__{b}"] = float(2 * np.sum(terms[a] * terms[b]) / denominator)
    return {
        "relative_l2_error": relative_error(candidate_cal, observed),
        "normalized_squared_error": float(np.sum(total**2) / denominator),
        "term_squared_norms": squares,
        "twice_cross_products": pairs,
        "reconstruction_squared_error": sum(squares.values()) + sum(pairs.values()),
        "max_additive_residual": float(np.max(np.abs(residual))),
        "interpretation": "Signed algebraic decomposition, not causal variance attribution",
    }


def probe_variance_diagnostic(cal, evaluation):
    """Plug-in independent sample-mean variance under within-category iid sampling.

    Inputs are seed x batch x category x document. The category centering matrix
    has diagonal 1 - 1/G, so trace(C diag(v) C) = (1 - 1/G) sum(v).
    Estimates are conditional on these samples and can miss rare heavy tails.
    """
    if cal.shape[:-1] != evaluation.shape[:-1] or min(cal.shape[-1], evaluation.shape[-1]) < 2:
        raise ValueError("Need matching groups and at least two probes in each split")
    variance = cal.var(-1, ddof=1) / cal.shape[-1]
    variance += evaluation.var(-1, ddof=1) / evaluation.shape[-1]
    expected_sse = float((1 - 1 / cal.shape[-2]) * variance.sum())
    mismatch_sse = float(np.sum(center(cal.mean(-1) - evaluation.mean(-1)) ** 2))
    return {
        "observed_probe_mismatch_squared_norm": mismatch_sse,
        "iid_plugin_expected_squared_norm": expected_sse,
        "observed_over_plugin": mismatch_sse / max(expected_sse, 1e-30),
        "assumption": "Independent probes within each category; no distribution-shift claim",
    }


def subsample_curve(cal, evaluation, observed, repeats=256, seed=20261014):
    """Repeated calibration subsamples versus the fixed evaluation set.

    Bands describe subsampling only, not uncertainty over future datasets.
    Never use evaluation values to choose the sampled calibration documents.
    """
    rng = np.random.default_rng(seed)
    ng, nd = cal.shape[-2:]
    target, linear_target = center(observed.mean(-1)), center(evaluation.mean(-1))
    result = []
    for count in (4, 8, 16, 32):
        if count > nd:
            continue
        empirical, linear = [], []
        for _ in range(repeats):
            weights = np.zeros((ng, nd))
            for group in range(ng):
                weights[group, rng.choice(nd, count, replace=False)] = 1 / count
            prediction = center(np.einsum("sbgp,gp->sbg", cal, weights))
            empirical.append(relative_error(prediction, target))
            linear.append(relative_error(prediction, linear_target))
        result.append(
            {
                "calibration_probes_per_category": count,
                "actual_outcome_error_p10_p50_p90": np.quantile(
                    empirical, [0.1, 0.5, 0.9]
                ).tolist(),
                "evaluation_derivative_error_p10_p50_p90": np.quantile(
                    linear, [0.1, 0.5, 0.9]
                ).tolist(),
            }
        )
    return result


def audit(root, output):
    root, output = Path(root), Path(output)
    controller = json.loads((root / "controller.json").read_text())
    if controller.get("status") != "complete":
        raise ValueError("Only completed studies with already released outcomes may be audited")
    plan = json.loads((root / "plan.json").read_text())
    cal_probes = read_jsonl(root / "data/calibration_probes.jsonl")
    eval_probes = read_jsonl(root / "data/evaluation_probes.jsonl")
    batches = read_jsonl(root / "data/batches.jsonl")
    results, provenance = {}, []
    for model in sorted({j["model"]["name"] for j in plan["jobs"]}):
        cal, evaluation, actual = [], [], []
        for job in (j for j in plan["jobs"] if j["model"]["name"] == model):
            directory = root / "jobs" / job["id"]
            metadata = json.loads((directory / "forecast_metadata.json").read_text())
            lock = json.loads((directory / "forecast_lock.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            verify_files(directory, {"predictions.npz": lock["files_sha256"]["predictions.npz"]})
            verify_files(directory / "results", manifest["result_files_sha256"])
            with np.load(directory / "predictions.npz") as archive:
                cal.append(archive["calibration"].astype(float))
                evaluation.append(archive["evaluation"].astype(float))
            rows = []
            for ri in range(len(plan["learning_rates"])):
                measured = []
                for bi in range(len(batches)):
                    with np.load(directory / "results" / f"batch-{bi:03d}-rate-{ri}.npz") as z:
                        measured.append(z["full"].copy())
                rows.append(measured)
            actual.append(rows)
            provenance.append(
                {"job": job["id"], "manifest_sha256": file_hash(directory / "manifest.json")}
            )
        groups, methods = metadata["groups"], metadata["methods"]
        cal = category_array(np.stack(cal), cal_probes, groups)
        evaluation = category_array(np.stack(evaluation), eval_probes, groups)
        actual = category_array(np.array(actual), eval_probes, groups)
        fi = methods.index("full_gradient")
        model_results = {}
        for ri, rate in enumerate(plan["learning_rates"]):
            fc, fe, y = rate * cal[:, fi], rate * evaluation[:, fi], actual[:, ri]
            model_results[str(rate)] = {
                "decompositions": {
                    name: error_decomposition(
                        rate * cal[:, methods.index(name)].mean(-1),
                        fc.mean(-1),
                        fe.mean(-1),
                        y.mean(-1),
                    )
                    for name in ("full_gradient", "svd_12", "dictionary_omp_8")
                },
                "probe_variance": probe_variance_diagnostic(fc, fe),
                "subsampling": subsample_curve(fc, fe, y),
            }
        results[model] = model_results
    payload = {
        "status": "exploratory_only",
        "scope": "Previously inspected confirmation outputs; no new model evaluations",
        "source_root": str(root.resolve()),
        "source_plan_sha256": file_hash(root / "plan.json"),
        "analysis_sha256": file_hash(__file__),
        "provenance": provenance,
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "transfer-audit.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-exploratory-reuse", action="store_true", required=True)
    args = parser.parse_args()
    audit(args.root, args.output)


if __name__ == "__main__":
    main()
