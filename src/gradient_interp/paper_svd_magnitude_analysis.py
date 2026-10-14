"""Guarded analysis of the frozen prospective SVD loss-effect study."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .paper_magnitude import bootstrap_interval
from .paper_magnitude_analysis import clean, detailed_metrics, table, write_csv
from .paper_setup import read_json
from .paper_svd_magnitude import (
    BOOTSTRAP_SEED,
    METHODS,
    RATES,
    bootstrap_metrics,
    combined_primary_gates,
    primary_gate,
    validate_selection,
)
from .paper_svd_magnitude import (
    REPRESENTATIONS as ARMS,
)
from .study import atomic_json, file_hash

FROZEN_COMMIT = "a6b79c5d1a3f7acccf856694ac09fd99b1e575ee"
SCOPES = {"evaluation": slice(16, 80), "calibration": slice(0, 16)}


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
            manifest.get("type") != "paper_svd_magnitude_compact_export"
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
            "unscaled_components_recomputed",
        )
        if any(integrity.get(key) is not True for key in required):
            raise ValueError("A scientific integrity assertion is missing")
        receipts = integrity["completion_receipts"]
        if {row["seed"] for row in receipts} != {42, 43} or len(receipts) != 2:
            raise ValueError("Audit does not contain both explicit completion receipts")
        for row in receipts:
            if (
                row["type"] != "paper_svd_magnitude_completion_receipt"
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
            or set(manifest["files_sha256"])
            != {"arrays.npz", "data.json", "integrity.json", "named-selections.json"}
            or integrity["updates"] != 64
            or integrity["probes"] != 80
        ):
            raise ValueError("Export seed, methods or complete grid differs")
        metadata.append((directory, manifest, integrity))
    if {m[1]["seed"] for m in metadata} != {42, 43}:
        raise ValueError("Exports must cover both distinct seeds")
    if metadata[0][2]["completion_receipts"] != metadata[1][2]["completion_receipts"]:
        raise ValueError("Exports do not share the same complete receipt barrier")
    # Only metadata has been read above. Both barriers passed before these hashes.
    for directory, manifest, _ in metadata:
        for filename, digest in manifest["files_sha256"].items():
            if file_hash(directory / filename) != digest:
                raise ValueError(f"Compact export changed: {filename}")
    return sorted(metadata, key=lambda row: row[1]["seed"])


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
        "coefficient_active_above_1e_minus6": (32, 2),
        "component_loss_profiles": (2, 500, 80),
        "component_loss_contributions": (32, 2, 2, 500, 80),
        "component_physical_norms": (2, 500),
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

    np.testing.assert_array_equal(
        arrays["coefficient_l0"], np.count_nonzero(arrays["coefficients"], axis=2)
    )
    np.testing.assert_array_equal(
        arrays["coefficient_active_above_1e_minus6"],
        np.count_nonzero(abs(arrays["coefficients"]) > 1e-6, axis=2),
    )
    np.testing.assert_allclose(
        arrays["component_loss_contributions"].sum(3),
        arrays["predicted_change"][:, :, :2],
        rtol=1e-4,
        atol=1e-7,
    )


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
        draws = bootstrap_metrics(p, y, replicates=replicates, seed=BOOTSTRAP_SEED)
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
                    for key in ("nrmse", "mae", "actual_rms")
                }
                record["simultaneous"] = {
                    key: bootstrap_interval(
                        draws[key][:, ri] if key == "actual_rms" else draws[key][:, ri, mi],
                        family_size=8,
                    )
                    for key in ("nrmse", "mae", "actual_rms")
                }
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
            signal = np.mean(y[:, ri] ** 2)
            difference = (
                (
                    np.sqrt(np.mean((p[:, ri, 1] - y[:, ri]) ** 2))
                    - np.sqrt(np.mean((p[:, ri, 0] - y[:, ri]) ** 2))
                )
                / np.sqrt(signal)
                if signal > 0
                else None
            )
            comparison_draws = draws["dictionary_minus_svd_nrmse"][:, ri]
            comparisons.append(
                dict(
                    seed=seed,
                    scope=scope,
                    rate=rate,
                    dictionary_minus_svd_nrmse=difference,
                    pointwise=bootstrap_interval(comparison_draws),
                    simultaneous=bootstrap_interval(comparison_draws, family_size=8),
                )
            )
        if scope == "evaluation":
            gate = primary_gate(draws)
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
                    active_above_1e_minus6=int(arrays["coefficient_active_above_1e_minus6"][b, mi])
                    if mi < 2
                    else None,
                )
            )
    return dict(
        gate=gate,
        metrics=metrics,
        comparisons=comparisons,
        batches=batch_rows,
        probes=probe_rows,
        pairs=pairs,
        examples=examples,
        reconstruction=reconstruction,
    )


def plots(report, summary, array_sets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#1f77b4", "#d66100", "#2c8c54", "#555555"]
    seeds = sorted(array_sets)
    prefix = ""
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
            ax.set_xticks(range(4), ["SVD500", "Dictionary", "Full\ngradient", "Zero"])
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


def component_tables(seed, arrays, selections):
    """Report every component and fixed named share, without selecting on outcomes."""
    validate_selection(selections)
    component_rows, named_rows = [], []
    c = arrays["component_loss_contributions"]
    for mi, method in enumerate(ARMS):
        selected = sorted({card["component"] for card in selections[method].values()})
        labels = {k: [] for k in range(500)}
        for label, card in selections[method].items():
            labels[card["component"]].append(label)
        for scope, subset in SCOPES.items():
            for ri, rate in enumerate(RATES):
                values = c[:, ri, mi, :, subset]
                for k in range(500):
                    component_rows.append(
                        dict(
                            seed=seed,
                            method=method,
                            scope=scope,
                            rate=rate,
                            component=k,
                            candidate_labels=labels[k],
                            mean_signed_contribution=float(values[:, k].mean()),
                            mean_absolute_contribution=float(abs(values[:, k]).mean()),
                            contribution_rms=float(np.sqrt(np.mean(values[:, k] ** 2))),
                            active_batches=int(np.count_nonzero(arrays["coefficients"][:, mi, k])),
                            active_above_1e_minus6=int(
                                np.count_nonzero(abs(arrays["coefficients"][:, mi, k]) > 1e-6)
                            ),
                            physical_component_norm=float(
                                arrays["component_physical_norms"][mi, k]
                            ),
                        )
                    )
                total, named = values.sum(1), values[:, selected].sum(1)
                absolute_total = abs(values).sum(1)
                for b in range(32):
                    for q in range(total.shape[1]):
                        denom, signed = absolute_total[b, q], total[b, q]
                        named_rows.append(
                            dict(
                                seed=seed,
                                method=method,
                                scope=scope,
                                rate=rate,
                                batch=b,
                                probe_index=q + (16 if scope == "evaluation" else 0),
                                named_components=selected,
                                total_effect=float(signed),
                                named_effect=float(named[b, q]),
                                absolute_contribution_sum=float(denom),
                                named_absolute_share=float(
                                    abs(values[b, selected, q]).sum() / denom
                                )
                                if denom
                                else None,
                                named_signed_share=float(named[b, q] / signed) if signed else None,
                                cancellation_fraction=float(1 - abs(signed) / denom)
                                if denom
                                else None,
                                zero_absolute_denominator=bool(denom == 0),
                                zero_signed_denominator=bool(signed == 0),
                            )
                        )
    return component_rows, named_rows


def render_report(report, summary):
    report = Path(report)
    verdict = "cleared" if summary["gates"]["cleared_both_seeds"] else "did not clear"
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Prospective SVD effect-size forecasts</title>",
        "<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 "
        "24px;line-height:1.5}"
        "table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{"
        "padding:7px;border:1px solid #ddd;vertical-align:top}"
        "th{background:#eef2f5}img{max-width:100%}details{margin:20px 0}</style>",
        "<h1>Prospective SVD effect-size forecasts</h1>",
        f"<p>SVD {verdict} all four prespecified primary criteria in both seeds.</p>",
        "<p>We forecast teacher-forced reference completion loss changes on fresh natural "
        "prompts, using reused Gemma-3 4B q/v LoRA adapters. "
        "The exact training-fitted SVD500 basis and original dictionary were reused. All "
        "forecasts were locked before the same 64 full-gradient SGD updates per seed. "
        "SVD retains the projected gradient magnitude directly; dictionary coding restores "
        "the projected norm after original alpha .1 sparse coding. No fitted scale.</p>",
        "<p>The primary rate is .003. Required bounds: actual RMS at least .01 nats/token, "
        "SVD nRMSE at most .5, MAE at most .01, "
        "and dictionary-minus-SVD nRMSE at least .10, in both seeds. The .0003 rate is "
        "descriptive and cannot rescue failure. "
        "We use 1,000 paired batch/probe resamples, shared across methods and rates. Eight "
        "primary quantities use 99.375% marginal Bonferroni percentile intervals "
        "targeting 95% family coverage. This bootstrap is approximate development "
        "uncertainty, not an exact finite-sample guarantee. Undefined draws prevent success. "
        "Calibration remains descriptive.</p>",
        table(
            [dict(seed=seed, **gate) for seed, gate in summary["gates"]["seeds"].items()],
            ["seed", "cleared", "signal_inconclusive", "failed_quantities", "intervals"],
        ),
        "<h2>Evaluation forecasts</h2><img src='forecast-errors.png' alt='Forecast errors'>",
    ]
    columns = [
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
    ]
    parts += [
        table([r for r in summary["metrics"] if r["scope"] == "evaluation"], columns),
        "<img src='effect-size-scatter.png' alt='All evaluation pairs'>",
        "<h2>Paired relative-error difference</h2><p>Positive values favor SVD. All rates and "
        "splits retained.</p>",
        table(
            summary["comparisons"],
            ["seed", "scope", "rate", "dictionary_minus_svd_nrmse", "pointwise", "simultaneous"],
        ),
        "<h2>Descriptive calibration</h2>",
        table([r for r in summary["metrics"] if r["scope"] == "calibration"], columns),
        "<h2>Reconstruction and component contributions</h2><img src='reconstruction.png' "
        "alt='Physical reconstruction'>",
        "<p>All 500 components are retained in the numeric export. The component table covers "
        "every component, method, rate, split and seed. "
        "Named candidates are the previously frozen training-only descriptions. Signed shares "
        "can exceed one or be undefined through cancellation. "
        "Their presence does not validate a compact interpretable explanation. Full "
        "contribution arrays remain in the audited compact archives; coefficients, profiles "
        "and summaries are provided below.</p>",
        table(summary["runtime"], ["seed", "worker_seconds", "peak_gpu_gb"]),
        "<p>No fitting was performed and no efficiency comparison is claimed.</p><h2>Complete "
        "numeric tables</h2><ul>",
    ]
    for name in (
        "metrics",
        "comparisons",
        "batches",
        "probes",
        "pairs",
        "examples",
        "reconstruction",
        "components",
        "named-shares",
        "coefficients",
        "profiles",
    ):
        parts.append(f"<li><a href='{name}.csv'>{name}.csv</a></li>")
    parts.append(
        "</ul><h2>Fixed examples</h2><p>First eight batches and first eight evaluation "
        "probes, every method and rate. Text excerpts capped at 500 characters; complete "
        "reference inputs preserved in exports.</p>"
    )
    for seed in (42, 43):
        parts += [
            f"<details><summary>All fixed examples, seed {seed}</summary>",
            table(
                [r for r in summary["examples"] if r["seed"] == seed],
                [
                    "batch",
                    "rate",
                    "method",
                    "probe_id",
                    "prompt_excerpt",
                    "reference_excerpt",
                    "actual_change",
                    "predicted_change",
                    "absolute_error",
                ],
            ),
            "</details>",
        ]
    parts.append(
        "<h2>Scope and audit</h2><p>This is prospective development on two reused adapters, "
        "not fresh-seed confirmation. "
        "Alpaca references are model-generated and not asserted correct. Exact/source/lexical "
        "filtering is not full semantic independence. "
        "Numerical reference CE accuracy does not establish generated-behavior forecasting or "
        "component interpretability. The completed steering/description study remains "
        "separate. "
        "Both source/input/cache/output audits passed before numerical analysis. Saved "
        "gradients were reprojected, all physical component profiles and loss contractions "
        "recomputed, "
        "and every lock/restore/deadline verified. Model gradients were not independently "
        "redifferentiated. "
        "An initial training-only preflight failed at a float64-to-float32 inverse boundary. "
        "Its failure was preserved and the explicit cast fixed before scientific forecasts or "
        "updates, without changing data or criteria.</p>"
    )
    (report / "report.html").write_text("\n".join(parts))


def run(exports, analysis_plan, report, workroot):
    metadata = require_complete_exports(exports, analysis_plan)
    report, workroot = Path(report), Path(workroot)
    if (report / "summary.json").exists() or (workroot / "first-reading.json").exists():
        raise ValueError("Preserve existing analysis before any repair")
    report.mkdir(parents=True, exist_ok=True)
    workroot.mkdir(parents=True, exist_ok=True)
    atomic_json(
        workroot / "first-reading.json",
        dict(
            written_unix=time.time(),
            source_sha256=file_hash(Path(__file__)),
            analysis_plan_sha256=file_hash(analysis_plan),
            both_complete_exports_verified=True,
            export_manifest_hashes={str(d): file_hash(d / "export.json") for d, _, _ in metadata},
        ),
    )
    shutil.copyfile(Path(__file__), workroot / "analysis-source.py")
    keys = (
        "metrics",
        "comparisons",
        "batches",
        "probes",
        "pairs",
        "examples",
        "reconstruction",
        "components",
        "named-shares",
        "coefficients",
        "profiles",
    )
    combined = {k: [] for k in keys}
    gates = {}
    arrays_by_seed = {}
    runtime = []
    provenance = []
    for directory, manifest, integrity in metadata:
        seed = manifest["seed"]
        arrays = dict(np.load(directory / "arrays.npz", allow_pickle=False))
        validate_arrays(arrays)
        data = read_json(directory / "data.json")
        selections = read_json(directory / "named-selections.json")
        result = summarize(seed, arrays, data, workroot)
        gates[seed] = result.pop("gate")
        for k, v in result.items():
            combined[k].extend(v)
        components, named = component_tables(seed, arrays, selections)
        combined["components"].extend(components)
        combined["named-shares"].extend(named)
        for mi, method in enumerate(ARMS):
            for k in range(500):
                for b in range(32):
                    combined["coefficients"].append(
                        dict(
                            seed=seed,
                            method=method,
                            batch=b,
                            component=k,
                            coefficient=float(arrays["coefficients"][b, mi, k]),
                        )
                    )
                for qi in range(80):
                    combined["profiles"].append(
                        dict(
                            seed=seed,
                            method=method,
                            component=k,
                            probe_index=qi,
                            split="calibration" if qi < 16 else "evaluation",
                            loss_profile=float(arrays["component_loss_profiles"][mi, k, qi]),
                        )
                    )
        arrays_by_seed[seed] = arrays
        provenance.append(integrity)
        runtime.append(
            dict(
                seed=seed,
                worker_seconds=integrity["worker_completed_unix"]
                - integrity["worker_started_unix"],
                peak_gpu_gb=integrity["peak_memory_bytes"] / 1e9,
            )
        )
    for key in keys:
        write_csv(report / f"{key}.csv", combined[key])
    summary = clean(
        {
            k: v
            for k, v in combined.items()
            if k in ("metrics", "comparisons", "examples", "reconstruction")
        }
    )
    summary.update(
        gates=clean(combined_primary_gates(gates)),
        runtime=runtime,
        scope="prospective_development_only",
        available_seeds=[42, 43],
        analysis_plan_sha256=file_hash(analysis_plan),
        source_sha256=file_hash(Path(__file__)),
        bootstrap_replicates=1000,
        bootstrap_seed=BOOTSTRAP_SEED,
        counts={k: len(v) for k, v in combined.items()},
    )
    atomic_json(report / "summary.json", summary)
    atomic_json(report / "integrity-summary.json", provenance)
    plots(report, summary, arrays_by_seed)
    render_report(report, summary)
    atomic_json(
        workroot / "analysis-complete.json",
        dict(
            completed_unix=time.time(),
            files_sha256={p.name: file_hash(p) for p in report.iterdir() if p.is_file()},
            source_sha256=file_hash(Path(__file__)),
            analysis_plan_sha256=file_hash(analysis_plan),
            counts=summary["counts"],
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "render"))
    parser.add_argument("--export", type=Path, action="append", default=[])
    parser.add_argument("--analysis-plan", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workroot", type=Path)
    args = parser.parse_args()
    if args.command == "render":
        render_report(args.report, read_json(args.report / "summary.json"))
    else:
        if args.analysis_plan is None or args.workroot is None:
            parser.error("run requires analysis plan and workroot")
        result = run(args.export, args.analysis_plan, args.report, args.workroot)
        print(json.dumps(result["gates"]))


if __name__ == "__main__":
    main()
