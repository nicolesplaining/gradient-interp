"""Guarded analysis of prospective teacher-forced loss-effect magnitude forecasts."""

import argparse
import csv
import html
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .paper_magnitude import (
    ARMS,
    METHODS,
    RATES,
    bootstrap_errors,
    bootstrap_interval,
    error_metrics,
)
from .paper_setup import read_json
from .study import atomic_json, file_hash

FROZEN_COMMIT = "e12ecc1989531ce48b1352e2f76985c6a7cf1a23"
PRELIMINARY_QUOTE = "wait so we have the magnitude run now right? can we analyze those results?"
SCOPES = {"evaluation": slice(16, 80), "calibration": slice(0, 16)}
GATE_LIMITS = {"actual_rms": 0.01, "nrmse": 0.5, "mae": 0.01}


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(clean(value))
                    if isinstance(value, (dict, list))
                    else clean(value)
                    for key, value in row.items()
                }
            )


def require_complete_exports(exports, analysis_plan):
    """Validate both metadata receipts before opening any compact numeric array."""
    if len(exports) != 2:
        raise ValueError("Both compact exports are required before analysis")
    plan = read_json(analysis_plan)
    if (
        plan.get("scientific_outcomes_read") is not False
        or plan.get("source_commit") != FROZEN_COMMIT
    ):
        raise ValueError("Incorrect frozen pre-reading analysis plan")
    metadata = []
    for directory in exports:
        directory = Path(directory)
        manifest, integrity = [
            read_json(directory / name) for name in ("export.json", "integrity.json")
        ]
        if (
            manifest.get("type") != "paper_magnitude_compact_export"
            or not integrity.get("complete")
            or integrity.get("source_commit") != FROZEN_COMMIT
            or manifest["analysis_plan_sha256"] != file_hash(analysis_plan)
            or integrity["analysis_plan_sha256"] != file_hash(analysis_plan)
        ):
            raise ValueError("Export is not a complete audit tied to this analysis plan")
        required = (
            "source_input_output_hashes_verified",
            "full_grid_verified",
            "locks_restores_verified",
            "original_deadline_verified",
            "tokenization_verified",
            "cache_recomputed",
        )
        if any(integrity.get(key) is not True for key in required):
            raise ValueError("A scientific integrity assertion is missing")
        receipts = integrity["completion_receipts"]
        if {row["seed"] for row in receipts} != {42, 43} or len(receipts) != 2:
            raise ValueError("Audit does not contain both explicit completion receipts")
        for row in receipts:
            if (
                row["type"] != "paper_magnitude_completion_receipt"
                or row["release_reason"] != "complete"
                or row["status"] != "complete"
                or not row["all_processes_dead"]
                or not row["completed_unix"] <= row["observed_unix"]
                or not row["started_unix"] < row["completed_unix"] <= row["deadline_unix"]
                or plan["written_unix"] >= row["observed_unix"]
                or abs(row["deadline_unix"] - row["started_unix"] - 43200) > 0.01
            ):
                raise ValueError("Completion receipt does not release a complete result grid")
        if (
            manifest["seed"] != integrity["seed"]
            or manifest["methods"] != list(METHODS)
            or manifest["rates"] != list(RATES)
            or set(manifest["files_sha256"]) != {"arrays.npz", "data.json", "integrity.json"}
            or integrity["updates"] != 64
            or integrity["probes"] != 80
        ):
            raise ValueError("Export seed, methods or complete grid differs")
        metadata.append((directory, manifest, integrity))
    if {m[1]["seed"] for m in metadata} != {42, 43}:
        raise ValueError("Exports must cover both distinct seeds")
    # Only metadata has been read above. Both barriers passed before these hashes.
    for directory, manifest, _ in metadata:
        for filename, digest in manifest["files_sha256"].items():
            if file_hash(directory / filename) != digest:
                raise ValueError(f"Compact export changed: {filename}")
    return sorted(metadata, key=lambda row: row[1]["seed"])


def require_preliminary_export(exports, analysis_plan, amendment_path):
    """Explicit one-seed exception; normal complete-export guards remain unchanged."""
    if len(exports) != 1:
        raise ValueError("Preliminary analysis requires exactly one seed43 export")
    plan, amendment = read_json(analysis_plan), read_json(amendment_path)
    if (
        plan.get("scientific_outcomes_read") is not False
        or plan.get("source_commit") != FROZEN_COMMIT
        or amendment.get("type") != "paper_magnitude_preliminary_authorization"
        or amendment.get("scope") != "preliminary_one_seed"
        or amendment.get("seed") != 43
        or amendment.get("user_quote") != PRELIMINARY_QUOTE
        or amendment.get("source_commit") != FROZEN_COMMIT
        or amendment.get("analysis_plan_sha256") != file_hash(analysis_plan)
        or amendment.get("scientific_outcomes_read_before_amendment") is not False
        or not plan["written_unix"] < amendment.get("written_unix", 0) <= time.time()
        or any(
            amendment.get(key) is not True
            for key in (
                "criteria_unchanged",
                "workers_unchanged",
                "remaining_seed_required_for_full_conclusion",
            )
        )
    ):
        raise ValueError("Explicit dated preliminary authorization is required")
    directory = Path(exports[0])
    manifest, integrity = [
        read_json(directory / name) for name in ("export.json", "integrity.json")
    ]
    if (
        manifest.get("type") != "paper_magnitude_preliminary_compact_export"
        or manifest.get("seed") != 43
        or integrity.get("seed") != 43
        or integrity.get("source_commit") != FROZEN_COMMIT
        or manifest.get("analysis_plan_sha256") != file_hash(analysis_plan)
        or integrity.get("analysis_plan_sha256") != file_hash(analysis_plan)
        or manifest.get("preliminary_amendment_sha256") != file_hash(amendment_path)
        or integrity.get("preliminary_amendment_sha256") != file_hash(amendment_path)
        or integrity.get("preliminary_amendment") != amendment
        or integrity.get("complete_two_seed_claim") is not False
        or any(
            integrity.get(key) is not True
            for key in (
                "complete",
                "preliminary",
                "source_input_output_hashes_verified",
                "full_grid_verified",
                "locks_restores_verified",
                "original_deadline_verified",
                "tokenization_verified",
                "cache_recomputed",
            )
        )
        or manifest.get("methods") != list(METHODS)
        or manifest.get("rates") != list(RATES)
        or set(manifest.get("files_sha256", {})) != {"arrays.npz", "data.json", "integrity.json"}
        or integrity.get("updates") != 64
        or integrity.get("probes") != 80
    ):
        raise ValueError("Incomplete preliminary audit or authorization mismatch")
    receipts = integrity.get("completion_receipts", [])
    if len(receipts) != 1:
        raise ValueError("Preliminary export must contain its one genuine completion receipt")
    row = receipts[0]
    if (
        row.get("type") != "paper_magnitude_completion_receipt"
        or row.get("seed") != 43
        or row.get("release_reason") != "complete"
        or row.get("status") != "complete"
        or row.get("all_processes_dead") is not True
        or not row["started_unix"] < row["completed_unix"] <= row["deadline_unix"]
        or row["completed_unix"] > row["observed_unix"]
        or abs(row["deadline_unix"] - row["started_unix"] - 43200) > 0.01
    ):
        raise ValueError("Preliminary seed43 was not complete with its processes dead")
    for filename, digest in manifest["files_sha256"].items():
        if file_hash(directory / filename) != digest:
            raise ValueError(f"Compact preliminary export changed: {filename}")
    return [(directory, manifest, integrity)]


def validate_arrays(arrays):
    shapes = {
        "predicted_change": (32, 2, 4, 80),
        "actual_mean_ce": (32, 2, 80),
        "baseline_mean_ce": (80,),
        "completion_tokens": (80,),
        "coefficients": (32, 2, 500),
        "batch_gradient_norm": (32,),
        "reconstructed_gradient_norm": (32, 4),
        "physical_relative_error": (32, 4),
        "physical_cosine": (32, 4),
        "projected_norm": (32,),
        "projected_relative_error": (32, 2),
        "projected_target_gain": (32, 2),
        "coefficient_l0": (32, 2),
    }
    if set(arrays) != set(shapes):
        raise ValueError("Unknown compact export schema")
    undefined_allowed = {
        "physical_relative_error",
        "physical_cosine",
        "projected_relative_error",
        "projected_target_gain",
    }
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Compact array shape differs: {key}")
        if key not in undefined_allowed and not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite compact numeric array: {key}")
        if np.isinf(arrays[key]).any():
            raise ValueError(f"Infinite compact diagnostic: {key}")
    if np.count_nonzero(arrays["predicted_change"][:, :, -1]):
        raise ValueError("No-change forecast is not exactly zero")
    if np.any(arrays["actual_mean_ce"] < 0) or np.any(arrays["baseline_mean_ce"] < 0):
        raise ValueError("Reference CE cannot be negative")


def detailed_metrics(predicted, actual, baseline):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    result = error_metrics(predicted, actual)
    absolute = abs(predicted - actual)
    eligible = (abs(predicted) > 0.001) & (abs(actual) > 0.001)
    matched = eligible & (np.sign(predicted) == np.sign(actual))
    result.update(
        error_p90_nats_per_token=float(np.quantile(absolute, 0.9)),
        error_p95_nats_per_token=float(np.quantile(absolute, 0.95)),
        direction_agreement=float(matched.sum() / eligible.sum()) if eligible.any() else None,
        direction_eligible_pairs=int(eligible.sum()),
        direction_matched_pairs=int(matched.sum()),
        total_pairs=int(predicted.size),
        predicted_negative_ce_pairs=int(np.count_nonzero(predicted + baseline < 0)),
    )
    return result


def gate_status(record):
    bounds = record["simultaneous"]
    signal = bounds["actual_rms"]["interval"]
    nrmse, mae = bounds["nrmse"]["interval"], bounds["mae"]["interval"]
    reasons = []
    if signal is None or signal[0] < GATE_LIMITS["actual_rms"]:
        reasons.append("signal_floor_not_cleared")
    if nrmse is None:
        reasons.append("relative_error_interval_undefined")
    elif nrmse[1] > GATE_LIMITS["nrmse"]:
        reasons.append("nrmse_limit_not_cleared")
    if mae is None or mae[1] > GATE_LIMITS["mae"]:
        reasons.append("mae_limit_not_cleared")
    return {
        "cleared": not reasons,
        "reasons": reasons,
        "status": "cleared"
        if not reasons
        else "inconclusive_signal"
        if "signal_floor_not_cleared" in reasons
        else "failed_accuracy",
    }


def summarize(seed, arrays, data, workroot, replicates=1000):
    predicted = arrays["predicted_change"]
    actual = arrays["actual_mean_ce"] - arrays["baseline_mean_ce"][None, None]
    metrics, comparisons, batch_rows, probe_rows, pairs, examples, reconstruction = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    all_probes = data["calibration"] + data["evaluation"]
    for scope, subset in SCOPES.items():
        p, y = predicted[..., subset], actual[..., subset]
        baseline = arrays["baseline_mean_ce"][subset]
        draws = bootstrap_errors(p, y, replicates=replicates, seed=2026101409)
        np.savez_compressed(workroot / f"seed{seed}-{scope}-bootstrap.npz", **draws)
        for ri, rate in enumerate(RATES):
            for mi, method in enumerate(METHODS):
                record = dict(
                    seed=seed,
                    scope=scope,
                    rate=rate,
                    method=method,
                    **detailed_metrics(p[:, ri, mi], y[:, ri], baseline),
                )
                record["pointwise"] = {
                    key: bootstrap_interval(
                        draws[key][:, ri] if key == "actual_rms" else draws[key][:, ri, mi]
                    )
                    for key in ("nrmse", "mae", "actual_rms", "mse_difference_vs_zero")
                }
                record["simultaneous"] = {
                    key: bootstrap_interval(
                        draws[key][:, ri] if key == "actual_rms" else draws[key][:, ri, mi],
                        family_size=4,
                    )
                    for key in ("nrmse", "mae", "actual_rms")
                }
                record["gate"] = (
                    gate_status(record) if scope == "evaluation" and method in ARMS else None
                )
                metrics.append(record)
                for b in range(32):
                    batch_rows.append(
                        dict(
                            seed=seed,
                            scope=scope,
                            rate=rate,
                            method=method,
                            batch=b,
                            batch_id=data["batches"][b]["id"],
                            **detailed_metrics(p[b, ri, mi], y[b, ri], baseline),
                        )
                    )
                scope_probes = data[scope]
                for q, probe in enumerate(scope_probes):
                    probe_rows.append(
                        dict(
                            seed=seed,
                            scope=scope,
                            rate=rate,
                            method=method,
                            probe=q,
                            probe_id=probe["id"],
                            source_id=probe["source_id"],
                            cluster_id=probe["cluster_id"],
                            **detailed_metrics(p[:, ri, mi, q], y[:, ri, q], baseline[q]),
                        )
                    )
            difference = float(
                np.mean((p[:, ri, 1] - y[:, ri]) ** 2) - np.mean((p[:, ri, 0] - y[:, ri]) ** 2)
            )
            comparison_draws = draws["mse_difference_magnitude_minus_normalized"][:, ri]
            comparisons.append(
                dict(
                    seed=seed,
                    scope=scope,
                    rate=rate,
                    magnitude_minus_normalized_mse=difference,
                    pointwise=bootstrap_interval(comparison_draws),
                    simultaneous=bootstrap_interval(comparison_draws, family_size=4),
                )
            )
    for b, batch in enumerate(data["batches"]):
        for ri, rate in enumerate(RATES):
            for mi, method in enumerate(METHODS):
                for qi, probe in enumerate(all_probes):
                    base = float(arrays["baseline_mean_ce"][qi])
                    prediction = float(predicted[b, ri, mi, qi])
                    target = float(actual[b, ri, qi])
                    row = dict(
                        seed=seed,
                        batch=b,
                        batch_id=batch["id"],
                        rate=rate,
                        method=method,
                        split="calibration" if qi < 16 else "evaluation",
                        probe_index=qi,
                        probe_id=probe["id"],
                        source_id=probe["source_id"],
                        cluster_id=probe["cluster_id"],
                        completion_tokens=probe["completion_tokens"],
                        baseline_ce=base,
                        actual_ce=base + target,
                        predicted_ce=base + prediction,
                        actual_change=target,
                        predicted_change=prediction,
                        residual=prediction - target,
                        absolute_error=abs(prediction - target),
                        negative_predicted_ce=base + prediction < 0,
                        direction_eligible=abs(prediction) > 0.001 and abs(target) > 0.001,
                    )
                    pairs.append(row)
                    if b < 8 and 16 <= qi < 24:
                        examples.append(
                            {
                                **row,
                                "prompt_excerpt": probe["prompt"][:500],
                                "reference_excerpt": probe["answer"][:500],
                            }
                        )
        for mi, method in enumerate(METHODS):
            reconstruction.append(
                dict(
                    seed=seed,
                    batch=b,
                    batch_id=batch["id"],
                    method=method,
                    physical_gradient_norm=float(arrays["batch_gradient_norm"][b]),
                    reconstructed_gradient_norm=float(arrays["reconstructed_gradient_norm"][b, mi]),
                    physical_relative_error=float(arrays["physical_relative_error"][b, mi]),
                    physical_cosine=float(arrays["physical_cosine"][b, mi]),
                    projected_gradient_norm=float(arrays["projected_norm"][b]),
                    projected_relative_error=float(arrays["projected_relative_error"][b, mi])
                    if mi < 2
                    else None,
                    projected_target_gain=float(arrays["projected_target_gain"][b, mi])
                    if mi < 2
                    else None,
                    active_coefficients=int(arrays["coefficient_l0"][b, mi]) if mi < 2 else None,
                )
            )
    return dict(
        metrics=metrics,
        comparisons=comparisons,
        batches=batch_rows,
        probes=probe_rows,
        pairs=pairs,
        examples=examples,
        reconstruction=reconstruction,
    )


def combined_gates(metrics):
    result = {}
    for arm in ARMS:
        records = [row for row in metrics if row["scope"] == "evaluation" and row["method"] == arm]
        if len(records) != 4 or {(row["seed"], row["rate"]) for row in records} != {
            (s, r) for s in (42, 43) for r in RATES
        }:
            raise ValueError("Incomplete seed/rate family for the development gate")
        result[arm] = dict(
            cleared_every_seed_and_rate=all(row["gate"]["cleared"] for row in records),
            conditions=[dict(seed=row["seed"], rate=row["rate"], **row["gate"]) for row in records],
        )
    return result


def preliminary_gates(metrics):
    result = {}
    for arm in ARMS:
        rows = [row for row in metrics if row["scope"] == "evaluation" and row["method"] == arm]
        if len(rows) != 2 or {(row["seed"], row["rate"]) for row in rows} != {
            (43, rate) for rate in RATES
        }:
            raise ValueError("Preliminary gates require both rates of seed43 only")
        result[arm] = dict(
            cleared_every_seed_and_rate=None,
            full_two_seed_conclusion_available=False,
            cleared_both_rates_this_seed=all(row["gate"]["cleared"] for row in rows),
            conditions=[dict(seed=row["seed"], rate=row["rate"], **row["gate"]) for row in rows],
        )
    return result


def plots(report, summary, array_sets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#1f77b4", "#d66100", "#2c8c54", "#555555"]
    seeds = sorted(array_sets)
    prefix = "PRELIMINARY seed43 only — " if summary.get("preliminary") else ""
    figure, axes = plt.subplots(
        len(seeds), 2, figsize=(11, 4 * len(seeds)), constrained_layout=True, squeeze=False
    )
    for si, seed in enumerate(seeds):
        for ri, rate in enumerate(RATES):
            ax = axes[si, ri]
            rows = [
                r
                for r in summary["metrics"]
                if r["seed"] == seed and r["rate"] == rate and r["scope"] == "evaluation"
            ]
            for mi, method in enumerate(METHODS):
                row = next(r for r in rows if r["method"] == method)
                point = row["nrmse_vs_zero"]
                ci = row["pointwise"]["nrmse"]["interval"]
                if point is not None:
                    ax.scatter(mi, point, color=colors[mi], zorder=3)
                if ci is not None:
                    ax.vlines(mi, ci[0], ci[1], color=colors[mi], linewidth=2)
            ax.axhline(1, color="#888888", linestyle="--", linewidth=1)
            ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1)
            ax.set_xticks(
                range(4), ["Normalized\nlearning", "Magnitude\nlearning", "Full\ngradient", "Zero"]
            )
            ax.set_title(f"Seed {seed}, rate {rate:g}")
            ax.set_ylabel("RMSE / no-change RMSE")
    figure.suptitle(
        prefix + "Teacher-forced CE forecasts, evaluation probes\n"
        "Pointwise 95% intervals; strong gates use simultaneous intervals"
    )
    figure.savefig(report / "forecast-errors.png", dpi=170)
    figure.savefig(report / "forecast-errors.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(
        len(seeds), 4, figsize=(15, 3.5 * len(seeds)), constrained_layout=True, squeeze=False
    )
    for si, seed in enumerate(seeds):
        arrays = array_sets[seed]
        actual = arrays["actual_mean_ce"] - arrays["baseline_mean_ce"][None, None]
        for ri, rate in enumerate(RATES):
            for mi, arm in enumerate(ARMS):
                ax = axes[si, 2 * ri + mi]
                y = actual[:, ri, 16:].reshape(-1)
                p = arrays["predicted_change"][:, ri, mi, 16:].reshape(-1)
                ax.scatter(y, p, s=6, alpha=0.2, color=colors[mi], rasterized=True)
                lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
                ax.plot([lo, hi], [lo, hi], color="#777777", linewidth=1)
                ax.axhline(0, color="#bbbbbb", linewidth=0.5)
                ax.axvline(0, color="#bbbbbb", linewidth=0.5)
                ax.set_title(f"Seed {seed}, {arm}\nrate {rate:g}", fontsize=10)
                ax.set_xlabel("Actual CE change, nats/token")
                ax.set_ylabel("Predicted CE change")
    figure.suptitle(prefix + "Every evaluation batch/probe pair, without clipping or selection")
    figure.savefig(report / "effect-size-scatter.png", dpi=170)
    figure.savefig(report / "effect-size-scatter.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(seeds), figsize=(5.5 * len(seeds), 4), constrained_layout=True, squeeze=False
    )
    axes = axes[0]
    for si, seed in enumerate(seeds):
        for mi, arm in enumerate(ARMS):
            rows = [
                r for r in summary["reconstruction"] if r["seed"] == seed and r["method"] == arm
            ]
            axes[si].plot(
                [r["batch"] for r in rows],
                [r["physical_relative_error"] for r in rows],
                marker=".",
                color=colors[mi],
                label=arm,
            )
        axes[si].set_title(f"Seed {seed}")
        axes[si].set_xlabel("Batch index, stored order")
        axes[si].set_ylabel("Physical gradient relative reconstruction error")
        axes[si].legend(fontsize=8)
    figure.suptitle(prefix + "Gradient reconstruction")
    figure.savefig(report / "reconstruction.png", dpi=170)
    figure.savefig(report / "reconstruction.pdf")
    plt.close(figure)


def cell(value):
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, (dict, list)):
        return html.escape(json.dumps(value))
    return html.escape(str(value))


def table(rows, columns):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(k)}</th>" for k in columns)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell(row.get(k))}</td>" for k in columns) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def render_report(report, summary):
    preliminary = summary.get("preliminary", False)
    title = (
        "PRELIMINARY seed43: dictionary training magnitude and loss effects"
        if preliminary
        else "Dictionary training magnitude and prospective loss effects"
    )
    pieces = [
        f"<!doctype html><meta charset='utf-8'><title>{title}</title>",
        "<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;"
        "padding:0 24px;line-height:1.5}"
        "table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:7px;"
        "border:1px solid #ddd;vertical-align:top}th{background:#eef2f5}img{max-width:100%}"
        "details{margin:20px 0}code{font-size:13px}</style>",
        f"<h1>{title}</h1>",
    ]
    if preliminary:
        pieces.append(
            "<p><b>User-requested preliminary reading of completed seed43 only.</b> "
            "A dated authorization amended the original wait-for-both-seeds reading rule. "
            "The scientific workers, forecasts, methods, rates and criteria remain frozen. "
            "Seed42 is still required before any full two-seed conclusion; no method or rate "
            "is selected from this preliminary result.</p>"
        )
    pieces += [
        "<p>This is a development study of teacher-forced reference completion CE on reused "
        "Gemma-3 4B q/v LoRA checkpoints. It tests normalized-row dictionary learning against "
        "learning that retains fitting-row magnitudes after one global scale. Both arms encode "
        "the normalized batch direction and restore its norm. Every forecast was locked before "
        "the same actual full-gradient SGD updates. These results do not establish interpretable "
        "components or unrestricted generated behavior.</p>",
    ]
    for arm, result in summary["gates"].items():
        if preliminary:
            verdict = "cleared" if result["cleared_both_rates_this_seed"] else "did not clear"
            pieces.append(
                f"<p><b>{html.escape(arm)}</b> {verdict} both rate thresholds in seed43. "
                "The full two-seed gate is unresolved.</p>"
            )
            pieces.append(table(result["conditions"], ["seed", "rate", "status", "reasons"]))
            continue
        verdict = (
            "cleared all prespecified development gates"
            if result["cleared_every_seed_and_rate"]
            else "did not clear all prespecified development gates"
        )
        pieces.append(
            f"<p><b>{html.escape(arm)}</b> {verdict}. Both rates and both seeds are required.</p>"
        )
        pieces.append(table(result["conditions"], ["seed", "rate", "status", "reasons"]))
    pieces += [
        "<p>The fixed gates require actual RMS change ≥0.01 nats/token, nRMSE ≤0.5 and MAE "
        "≤0.01 nats/token with simultaneous interval bounds clearing each threshold. A weak "
        "effect is inconclusive even when absolute error is small. We used 1,000 paired batch "
        "and probe resamples. Pointwise intervals are 95%; four-comparison Bonferroni intervals "
        "are 98.75% within each seed. Undefined bootstrap draws remain visible and prevent a "
        "relative-error interval. Calibration is descriptive and supplies no fitted scale.</p>",
        "<h2>Evaluation forecasts</h2><img src='forecast-errors.png' alt='Forecast errors'>",
    ]
    rows = [row for row in summary["metrics"] if row["scope"] == "evaluation"]
    pieces.append(
        table(
            rows,
            [
                "seed",
                "rate",
                "method",
                "rmse_nats_per_token",
                "mae_nats_per_token",
                "nrmse_vs_zero",
                "actual_rms_nats_per_token",
                "mean_predicted_change",
                "mean_actual_change",
                "actual_on_predicted_slope_through_origin",
                "error_p90_nats_per_token",
                "error_p95_nats_per_token",
                "worst_absolute_error",
                "direction_agreement",
                "direction_eligible_pairs",
                "total_pairs",
                "predicted_negative_ce_pairs",
                "pointwise",
                "simultaneous",
            ],
        )
    )
    pieces += [
        "<img src='effect-size-scatter.png' alt='All individual loss changes'>",
        "<h2>Paired learning-geometry comparison</h2>",
        "<p>Negative MSE differences favor magnitude learning. These comparisons do not "
        "replace the fixed absolute accuracy and signal criteria.</p>",
        table(
            summary["comparisons"],
            [
                "seed",
                "scope",
                "rate",
                "magnitude_minus_normalized_mse",
                "pointwise",
                "simultaneous",
            ],
        ),
        "<h2>Calibration probes</h2>",
        table(
            [r for r in summary["metrics"] if r["scope"] == "calibration"],
            [
                "seed",
                "rate",
                "method",
                "rmse_nats_per_token",
                "mae_nats_per_token",
                "nrmse_vs_zero",
                "actual_rms_nats_per_token",
                "pointwise",
            ],
        ),
        "<h2>Reconstruction and runtime</h2>"
        "<img src='reconstruction.png' alt='Gradient reconstruction'>",
        table(
            summary["runtime"],
            [
                "seed",
                "magnitude_fit_seconds",
                "worker_seconds",
                "peak_gpu_gb",
                "normalized_dictionary_reused",
                "global_fitting_scale",
            ],
        ),
        "<p>The normalized dictionary was reused. Fitting runtime is recorded for the new "
        "magnitude arm only and is not a controlled efficiency comparison.</p>",
        "<h2>Complete machine-readable results</h2><ul>",
    ]
    for filename in (
        "metrics.csv",
        "comparisons.csv",
        "batches.csv",
        "probes.csv",
        "pairs.csv",
        "examples.csv",
        "reconstruction.csv",
        "summary.json",
    ):
        pieces.append(f"<li><a href='{filename}'>{filename}</a></li>")
    pieces += [
        "</ul><h2>Fixed examples</h2><p>First eight batches and first eight evaluation probes, "
        "every method, rate and available seed. Excerpts are capped at 500 characters; "
        "the input data "
        "and full per-pair numeric table are retained. No examples were chosen by error.</p>"
    ]
    for seed in sorted({row["seed"] for row in summary["metrics"]}):
        pieces.append(f"<details><summary>All fixed examples, seed {seed}</summary>")
        pieces.append(
            table(
                [r for r in summary["examples"] if r["seed"] == seed],
                [
                    "batch",
                    "rate",
                    "method",
                    "probe_id",
                    "prompt_excerpt",
                    "reference_excerpt",
                    "baseline_ce",
                    "actual_ce",
                    "predicted_ce",
                    "actual_change",
                    "predicted_change",
                    "residual",
                ],
            )
        )
        pieces.append("</details>")
    pieces.append(
        "<p>The available seed's full source/input/cache/output audit passed before numerical "
        "analysis. "
        if preliminary
        else "<p>Both original source/input/cache/output audits passed before these outputs were "
        "read. "
    )
    pieces.append(
        "Stored gradients were reprojected and contractions recomputed without loading "
        "the model; they were not independently redifferentiated. Bootstrap intervals describe "
        "this development sample. New checkpoints and a separately frozen confirmation are "
        "required for a broader reliability claim.</p>"
    )
    (report / "report.html").write_text("\n".join(pieces))


def run(exports, analysis_plan, report, workroot):
    metadata = require_complete_exports(exports, analysis_plan)
    return run_validated(metadata, analysis_plan, report, workroot)


def run_preliminary(exports, analysis_plan, report, workroot, amendment_path):
    metadata = require_preliminary_export(exports, analysis_plan, amendment_path)
    return run_validated(metadata, analysis_plan, report, workroot, amendment_path=amendment_path)


def run_validated(metadata, analysis_plan, report, workroot, amendment_path=None):
    preliminary = amendment_path is not None
    report, workroot = Path(report), Path(workroot)
    if (report / "summary.json").exists():
        raise ValueError("Analysis already exists; render it or preserve an explicit repair")
    workroot.mkdir(parents=True, exist_ok=True)
    first = workroot / "first-reading.json"
    if first.exists():
        raise ValueError("Preserve the previous reading attempt before an explicit analysis repair")
    atomic_json(
        first,
        dict(
            written_unix=time.time(),
            analysis_plan_sha256=file_hash(analysis_plan),
            source_sha256=file_hash(Path(__file__)),
            both_complete_exports_verified=not preliminary,
            preliminary_user_authorized=preliminary,
            preliminary_amendment_sha256=file_hash(amendment_path) if preliminary else None,
            export_manifest_hashes={str(d): file_hash(d / "export.json") for d, _, _ in metadata},
        ),
    )
    shutil.copyfile(Path(__file__), workroot / "analysis-source.py")
    combined = {
        key: []
        for key in (
            "metrics",
            "comparisons",
            "batches",
            "probes",
            "pairs",
            "examples",
            "reconstruction",
        )
    }
    array_sets, runtime, provenance = {}, [], []
    for directory, manifest, integrity in metadata:
        seed = manifest["seed"]
        arrays = dict(np.load(directory / "arrays.npz", allow_pickle=False))
        validate_arrays(arrays)
        data = read_json(directory / "data.json")
        result = summarize(seed, arrays, data, workroot)
        for key, value in result.items():
            combined[key].extend(value)
        array_sets[seed] = arrays
        fit = integrity["fit"]
        runtime.append(
            dict(
                seed=seed,
                magnitude_fit_seconds=fit["completed_unix"] - fit["started_unix"],
                worker_seconds=integrity["worker_completed_unix"]
                - integrity["worker_started_unix"],
                peak_gpu_gb=integrity["peak_memory_bytes"] / 1e9,
                normalized_dictionary_reused=fit["original_dictionary_reused"],
                global_fitting_scale=fit["magnitude_global_scale"],
            )
        )
        provenance.append(integrity)
    combined["gates"] = (
        preliminary_gates(combined["metrics"])
        if preliminary
        else combined_gates(combined["metrics"])
    )
    combined["runtime"] = runtime
    report.mkdir(parents=True, exist_ok=True)
    for key in (
        "metrics",
        "comparisons",
        "batches",
        "probes",
        "pairs",
        "examples",
        "reconstruction",
    ):
        write_csv(report / f"{key}.csv", combined[key])
    # Keep the large complete per-pair table in CSV while retaining all fixed examples in JSON/HTML.
    summary = clean(
        {key: value for key, value in combined.items() if key not in ("pairs", "batches", "probes")}
    )
    summary.update(
        scope="development_only",
        preliminary=preliminary,
        available_seeds=sorted(array_sets),
        missing_seeds=[42] if preliminary else [],
        complete_two_seed_claim=not preliminary,
        preliminary_amendment_sha256=file_hash(amendment_path) if preliminary else None,
        claim="teacher-forced completion CE only",
        analysis_plan_sha256=file_hash(analysis_plan),
        source_sha256=file_hash(Path(__file__)),
        bootstrap_replicates=1000,
        bootstrap_seed=2026101409,
        counts={
            key: len(combined[key])
            for key in (
                "metrics",
                "comparisons",
                "batches",
                "probes",
                "pairs",
                "examples",
                "reconstruction",
            )
        },
    )
    atomic_json(report / "summary.json", summary)
    atomic_json(report / "integrity-summary.json", provenance)
    plots(report, summary, array_sets)
    render_report(report, summary)
    atomic_json(
        workroot / "analysis-complete.json",
        dict(
            completed_unix=time.time(),
            files_sha256={p.name: file_hash(p) for p in report.iterdir() if p.is_file()},
            source_sha256=file_hash(Path(__file__)),
            analysis_plan_sha256=file_hash(analysis_plan),
            preliminary=preliminary,
            preliminary_amendment_sha256=file_hash(amendment_path) if preliminary else None,
            counts=summary["counts"],
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "run-preliminary", "render"))
    parser.add_argument("--export", type=Path, action="append", default=[])
    parser.add_argument("--analysis-plan", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workroot", type=Path)
    parser.add_argument("--amendment", type=Path)
    args = parser.parse_args()
    if args.command == "render":
        render_report(args.report, read_json(args.report / "summary.json"))
    else:
        if args.analysis_plan is None or args.workroot is None:
            parser.error("run requires --analysis-plan and --workroot")
        if args.command == "run-preliminary":
            if args.amendment is None:
                parser.error("run-preliminary requires --amendment")
            result = run_preliminary(
                args.export, args.analysis_plan, args.report, args.workroot, args.amendment
            )
        else:
            result = run(args.export, args.analysis_plan, args.report, args.workroot)
        print(json.dumps(result["gates"]))


if __name__ == "__main__":
    main()
