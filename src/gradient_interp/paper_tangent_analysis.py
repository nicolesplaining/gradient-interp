"""Guarded descriptive analysis of the completed autoregressive tangent experiment."""

# ruff: noqa: E501
import argparse
import html
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .paper_batch import STYLES
from .paper_batch_analysis import write_csv
from .paper_sample import ENDPOINTS
from .paper_sample_analysis import EXTRA, stats, values
from .paper_tangent import TANGENT_METHODS
from .study import atomic_json, file_hash

MODES = ("forecast_mc", "evaluation_mc", "both_mc", "both_mc_and_questions")
SCOPES = (
    "evaluation_same_question",
    "evaluation_calibration_transfer",
    "calibration_same_question",
)


def read(path):
    return json.loads(path.read_text())


def guard(roots, plan):
    if not plan.exists() or read(plan)["outcomes_read_before_plan"]:
        raise ValueError("A pre-reading analysis plan is required")
    for root in roots:
        c, audit, manifest = [
            read(root / n) for n in ("controller.json", "integrity.json", "manifest.json")
        ]
        if (
            c["status"] != "complete"
            or not audit["complete"]
            or not audit["source_input_output_hashes_verified"]
        ):
            raise ValueError("Both tangent runs must be complete and audited before analysis")
        assert manifest["completed_unix"] <= c["deadline_unix"]
        for n in ("plan", "specification", "manifest", "forecast-lock"):
            assert file_hash(root / f"{n}.json") == audit[n.replace("-", "_") + "_sha256"]
        for n, digest in manifest["files_sha256"].items():
            assert file_hash(root / n) == digest, n


def load(root):
    spec, data, lock, manifest = [
        read(root / n)
        for n in ("specification.json", "data.json", "forecast-lock.json", "manifest.json")
    ]
    records = {
        name: read(root / name) for name in manifest["files_sha256"] if name != "forecast-lock.json"
    }
    fbase = values(records["forecasts/baseline.json"]["rows"], data["probes"], 32, 96)
    ybase = values(records["outcomes/baseline.json"]["rows"], data["probes"], 128, 96)
    forecast = np.empty((4, 3, 5, 4, 32, 8))
    observed = np.empty((4, 3, 4, 128, 8))
    for b, ri in itertools.product(range(4), range(3)):
        rate = spec["rates"][ri]
        for mi, method in enumerate(TANGENT_METHODS):
            name = (
                "forecasts/baseline.json"
                if method == "zero"
                else f"forecasts/batch-{b:02d}-rate-{rate:g}-{method}.json"
            )
            r = records[name]
            assert r["original_adapter_unchanged"] and r["completed_unix"] <= lock["locked_unix"]
            forecast[b, ri, mi] = values(r["rows"], data["probes"], 32, 96)
        name = f"outcomes/batch-{b:02d}-rate-{rate:g}.json"
        r = records[name]
        assert r["adapter_restored"] and lock["locked_unix"] < r["update_started_unix"]
        observed[b, ri] = values(r["rows"], data["probes"], 128, 96)
    return spec, data, records, forecast, fbase, observed, ybase


def draw_bootstrap(fdiff, ydiff, reps=1000):
    """Pair each response stream across conditions; never resample the four fixed batches."""
    rng = np.random.default_rng(2026101406)
    fw = rng.multinomial(32, np.full(32, 1 / 32), size=(reps, 4)) / 32
    yw = rng.multinomial(128, np.full(128, 1 / 128), size=(reps, 4)) / 128
    fp = np.einsum("brmqsk,tqs->tbrmqk", fdiff, fw, optimize=True)
    yt = np.einsum("brqsk,tqs->tbrqk", ydiff, yw, optimize=True)
    cq = rng.multinomial(2, [0.5, 0.5], size=reps) / 2
    eq = rng.multinomial(2, [0.5, 0.5], size=reps) / 2
    return fp, yt, cq, eq


def scoped(pred, target, scope, calibration_weights=None):
    """Inputs T,B,R,M,Q,K and T,B,R,Q,K; retain the two selected question positions."""
    if scope == "calibration_same_question":
        return pred[..., :2, :], target[..., :2, :]
    same = pred[..., 2:, :]
    if scope == "evaluation_same_question":
        return same, target[..., 2:, :]
    if scope != "evaluation_calibration_transfer":
        raise ValueError(scope)
    if calibration_weights is None:
        transferred = pred[..., :2, :].mean(-2)
    else:
        transferred = np.einsum("tbrmqk,tq->tbrmk", pred[..., :2, :], calibration_weights)
    return np.broadcast_to(transferred[..., None, :], same.shape), target[..., 2:, :]


def boot_statistics(pred, target, question_weights):
    """Return replicate R,M,K metrics, with undefined relative errors kept explicit."""
    error = pred - target[:, :, :, None]
    mse = np.einsum("tbrmqk,tq->trmk", error**2, question_weights) / pred.shape[1]
    zero_mse = np.einsum("tbrqk,tq->trk", target**2, question_weights) / pred.shape[1]
    denom = np.broadcast_to(zero_mse[:, :, None], mse.shape)
    ratio = np.divide(mse, denom, out=np.full_like(mse, np.nan), where=denom > 0)
    batch_rmse = np.sqrt(np.einsum("tbrmqk,tq->tbrmk", error**2, question_weights))
    pred_square = np.einsum("tbrmqk,tq->trmk", pred**2, question_weights)
    cross = np.einsum("tbrmqk,tq->trmk", pred * target[:, :, :, None], question_weights)
    return dict(
        rmse=np.sqrt(mse),
        nrmse=np.sqrt(ratio),
        mse_difference_vs_zero=mse - denom,
        predicted_mean=np.einsum("tbrmqk,tq->trmk", pred, question_weights) / pred.shape[1],
        actual_mean=np.einsum("tbrqk,tq->trk", target, question_weights) / pred.shape[1],
        actual_on_prediction_slope=np.divide(
            cross, pred_square, out=np.full_like(cross, np.nan), where=pred_square > 1e-30
        ),
        batch_rmse_max=batch_rmse.max(1),
        batch_rmse_p90=np.quantile(batch_rmse, 0.9, axis=1),
        per_batch_rmse=batch_rmse,
    )


def interval_fields(samples, name):
    finite = np.isfinite(samples)
    # Never silently drop degenerate bootstrap cases. Absolute error remains available.
    bounds = np.quantile(samples, [0.025, 0.975]).tolist() if finite.all() else [None, None]
    return {
        f"{name}_lo": bounds[0],
        f"{name}_hi": bounds[1],
        f"{name}_defined_replicates": int(finite.sum()),
    }


def summarize(root, out, reps):
    spec, data, records, forecast, fbase, observed, ybase = load(root)
    seed = spec["training_seed"]
    fdiff, ydiff = forecast - fbase, observed - ybase
    pred, target = fdiff.mean(-2), ydiff.mean(-2)
    assert np.count_nonzero(pred[:, :, -1]) == 0
    fp, yt, cq, eq = draw_bootstrap(fdiff[..., :4], ydiff[..., :4], reps)
    rows, batches, details, generated, examples, costs = [], [], [], [], [], []
    all_boot = {}
    for scope in SCOPES:
        p, y = scoped(pred[None], target[None], scope)
        p, y = p[0], y[0]
        boot = {}
        for mode in MODES:
            pp = np.broadcast_to(pred[None, ..., :4], fp.shape) if mode == "evaluation_mc" else fp
            yy = np.broadcast_to(target[None, ..., :4], yt.shape) if mode == "forecast_mc" else yt
            question_mc = mode == "both_mc_and_questions"
            sp, sy = scoped(pp, yy, scope, cq if question_mc else None)
            qw = (
                (cq if scope == "calibration_same_question" else eq)
                if question_mc
                else np.full((reps, 2), 0.5)
            )
            boot[mode] = boot_statistics(sp, sy, qw)
            for key, val in boot[mode].items():
                all_boot[f"{scope}_{mode}_{key}"] = val
        for ri, mi, ki in itertools.product(range(3), range(5), range(4)):
            a, b = p[:, ri, mi, :, ki], y[:, ri, :, ki]
            record = dict(
                seed=seed,
                scope=scope,
                rate=spec["rates"][ri],
                method=TANGENT_METHODS[mi],
                endpoint=ENDPOINTS[ki],
                **stats(a, b),
            )
            record["mse_difference_vs_zero"] = float(np.mean((a - b) ** 2) - np.mean(b**2))
            for mode in MODES:
                for key in (
                    "rmse",
                    "nrmse",
                    "mse_difference_vs_zero",
                    "predicted_mean",
                    "actual_on_prediction_slope",
                    "batch_rmse_max",
                    "batch_rmse_p90",
                ):
                    record.update(interval_fields(boot[mode][key][:, ri, mi, ki], f"{mode}_{key}"))
            rows.append(record)
            for mode in MODES:
                record.update(
                    interval_fields(boot[mode]["actual_mean"][:, ri, ki], f"{mode}_actual_mean")
                )
            for bi in range(4):
                batches.append(
                    dict(
                        seed=seed,
                        scope=scope,
                        rate=spec["rates"][ri],
                        method=TANGENT_METHODS[mi],
                        endpoint=ENDPOINTS[ki],
                        batch=bi,
                        style=STYLES[bi],
                        **stats(a[bi : bi + 1], b[bi : bi + 1]),
                    )
                )
                for mode in MODES:
                    batches[-1].update(
                        interval_fields(
                            boot[mode]["per_batch_rmse"][:, bi, ri, mi, ki], f"{mode}_rmse"
                        )
                    )
                for qi in range(2):
                    details.append(
                        dict(
                            seed=seed,
                            scope=scope,
                            rate=spec["rates"][ri],
                            method=TANGENT_METHODS[mi],
                            endpoint=ENDPOINTS[ki],
                            batch=bi,
                            style=STYLES[bi],
                            question_index=qi + (0 if scope == "calibration_same_question" else 2),
                            predicted_change=float(a[bi, qi]),
                            actual_change=float(b[bi, qi]),
                            error=float(a[bi, qi] - b[bi, qi]),
                        )
                    )
    np.savez_compressed(out / f"seed{seed}-bootstrap.npz", **all_boot)
    for stream, arrays, baseline in (
        ("forecast", forecast, fbase),
        ("evaluation", observed[:, :, None], ybase),
    ):
        methods = TANGENT_METHODS if stream == "forecast" else ("actual_update",)
        for bi, ri, mi, split in itertools.product(
            range(4), range(3), range(len(methods)), ("calibration", "evaluation")
        ):
            ix = slice(0, 2) if split == "calibration" else slice(2, 4)
            current, original = arrays[bi, ri, mi, ix], baseline[ix]
            for ki, endpoint in enumerate((*ENDPOINTS, *EXTRA)):
                generated.append(
                    dict(
                        seed=seed,
                        stream=stream,
                        batch=bi,
                        style=STYLES[bi],
                        rate=spec["rates"][ri],
                        method=methods[mi],
                        split=split,
                        endpoint=endpoint,
                        baseline=float(original[..., ki].mean()),
                        updated=float(current[..., ki].mean()),
                        change=float(current[..., ki].mean() - original[..., ki].mean()),
                        responses=int(np.prod(current.shape[:2])),
                    )
                )
    for bi, qi, ri, mi in itertools.product(range(4), (0, 2), range(3), range(5)):
        rate, method = spec["rates"][ri], TANGENT_METHODS[mi]
        fn = (
            "forecasts/baseline.json"
            if method == "zero"
            else f"forecasts/batch-{bi:02d}-rate-{rate:g}-{method}.json"
        )
        yn = f"outcomes/batch-{bi:02d}-rate-{rate:g}.json"
        f, y = records[fn]["rows"][qi * 32], records[yn]["rows"][qi * 128]
        examples.append(
            dict(
                seed=seed,
                batch=bi,
                style=STYLES[bi],
                question=qi,
                rate=rate,
                method=method,
                prompt=data["probes"][qi]["prompt"],
                forecast_response=f["response"],
                actual_response=y["response"],
                forecast_checks=f["checks"],
                actual_checks=y["checks"],
                forecast_quality=f["quality"],
                actual_quality=y["quality"],
            )
        )
    for name, record in records.items():
        costs.append(
            dict(
                seed=seed,
                condition=name,
                stream="forecast" if name.startswith("forecasts/") else "evaluation",
                responses=len(record["rows"]),
                tokens=sum(r["quality"]["tokens"] for r in record["rows"]),
                **record["cost"],
            )
        )
    manifest = read(root / "manifest.json")
    provenance = dict(
        seed=seed,
        root=str(root),
        source_snapshot_sha256=read(root / "integrity.json")["source_snapshot_sha256"],
        started_unix=manifest["started_unix"],
        completed_unix=manifest["completed_unix"],
        elapsed_seconds=manifest["completed_unix"] - manifest["started_unix"],
        peak_memory_bytes=manifest["peak_memory_bytes"],
        forecast_responses=6272,
        evaluation_responses=6656,
        updates=12,
        analysis_plan_sha256=file_hash(out / "analysis-plan.json"),
    )
    return dict(
        metrics=rows,
        batches=batches,
        questions=details,
        generated=generated,
        examples=examples,
        costs=costs,
        provenance=provenance,
    )


def plots(out, rows, questions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(
        zip(TANGENT_METHODS, ("#2364aa", "#2e8b57", "#c34d21", "#8153a5", "#777777"), strict=True)
    )
    labels = dict(
        zip(
            ENDPOINTS,
            ("Numbered formatting", "Exact content and order", "Response tokens", "Truncation"),
            strict=True,
        )
    )
    for scope in SCOPES[:2]:
        fig, axes = plt.subplots(4, 2, figsize=(12, 12), squeeze=False)
        for col, seed in enumerate((42, 43)):
            for ki, endpoint in enumerate(ENDPOINTS):
                ax = axes[ki, col]
                for mi, method in enumerate(TANGENT_METHODS):
                    rr = sorted(
                        (
                            r
                            for r in rows
                            if r["seed"] == seed
                            and r["scope"] == scope
                            and r["method"] == method
                            and r["endpoint"] == endpoint
                        ),
                        key=lambda r: r["rate"],
                    )
                    xx = np.arange(3) + (mi - 2) * 0.045
                    yy = [r["nrmse"] if r["nrmse"] is not None else np.nan for r in rr]
                    ax.plot(xx, yy, "o-", ms=4, lw=1, color=colors[method], label=method)
                ax.axhline(1, color="black", lw=0.6, ls=":")
                ax.set_xticks(range(3), (".0003", ".003", ".03"))
                ax.set_title(f"Seed {seed}: {labels[endpoint]}", fontsize=10)
                ax.set_ylabel("RMSE / no-change RMSE")
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.15)
        axes[0, 0].legend(fontsize=8, loc="best")
        fig.suptitle(
            f"Autoregressive tangent | {scope.replace('_', ' ')}\nFixed development tasks; lower is better; missing points have zero observed effect",
            fontsize=12,
        )
        fig.supxlabel("SGD learning rate")
        fig.tight_layout(rect=(0, 0.025, 1, 0.95))
        fig.savefig(out / f"{scope}.png", dpi=170)
        fig.savefig(out / f"{scope}.pdf")
        plt.close(fig)
    # Absolute errors stay visible even when a relative error is undefined.
    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    markers = ("o", "s", "^")
    for col, seed in enumerate((42, 43)):
        for ki, endpoint in enumerate(ENDPOINTS):
            ax = axes[ki, col]
            rr = [
                r
                for r in questions
                if r["seed"] == seed
                and r["scope"] == "evaluation_same_question"
                and r["method"] == "dictionary"
                and r["endpoint"] == endpoint
            ]
            scale = 1 if endpoint == "tokens" else 100
            for rate, marker in zip((0.0003, 0.003, 0.03), markers, strict=True):
                ss = [r for r in rr if r["rate"] == rate]
                ax.scatter(
                    [r["actual_change"] * scale for r in ss],
                    [r["predicted_change"] * scale for r in ss],
                    marker=marker,
                    s=40,
                    alpha=0.75,
                    label=f"rate {rate:g}",
                )
            vals = [r[k] * scale for r in rr for k in ("actual_change", "predicted_change")]
            lo, hi = min([0, *vals]), max([0, *vals])
            pad = max((hi - lo) * 0.1, 0.01)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=0.8, ls=":")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"Seed {seed}: {labels[endpoint]}", fontsize=10)
            unit = "tokens" if endpoint == "tokens" else "percentage points"
            ax.set_xlabel(f"Actual change ({unit})")
            ax.set_ylabel("Atom forecast change")
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Dictionary forecasts: all fixed batch × evaluation-question pairs\nSame-question surrogate generations; independent evaluation samples",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "dictionary-calibration.png", dpi=170)
    fig.savefig(out / "dictionary-calibration.pdf")
    plt.close(fig)


def formatted(value):
    if value is None:
        return "undefined"
    return f"{value:.3g}" if isinstance(value, float) else html.escape(str(value))


def table(rows, fields):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(k.replace('_', ' '))}</th>" for k in fields)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{formatted(r.get(k))}</td>" for k in fields) + "</tr>"
            for r in rows
        )
        + "</tbody></table>"
    )


def report(out, summary):
    rows = summary["metrics"]
    atom_rows = [
        r for r in rows if r["method"] == "dictionary" and r["scope"].startswith("evaluation")
    ]
    atom_table = []
    for r in atom_rows:
        prefix = "both_mc_and_questions_nrmse"
        atom_table.append(
            {
                k: r[k]
                for k in (
                    "seed",
                    "scope",
                    "rate",
                    "endpoint",
                    "rmse",
                    "nrmse",
                    "actual_mean",
                    "predicted_mean",
                    "batch_rmse_max",
                )
            }
            | {
                "relative_error_interval": f"{formatted(r[prefix + '_lo'])} to {formatted(r[prefix + '_hi'])}",
                "defined_replicates": r[prefix + "_defined_replicates"],
            }
        )
    cost_summary = []
    for seed, stream in itertools.product((42, 43), ("forecast", "evaluation")):
        rr = [r for r in summary["costs"] if r["seed"] == seed and r["stream"] == stream]
        cost_summary.append(
            dict(
                seed=seed,
                stream=stream,
                conditions=len(rr),
                responses=sum(r["responses"] for r in rr),
                generated_tokens=sum(r["tokens"] for r in rr),
                forward_calls=sum(r["forward_calls"] for r in rr),
                jvp_calls=sum(r["jvp_calls"] for r in rr),
                hours=sum(r["elapsed_seconds"] for r in rr) / 3600,
            )
        )
    sections = [
        """<!doctype html><html><head><meta charset="utf-8"><title>Autoregressive tangent development results</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#172335;max-width:1250px;margin:auto;padding:36px;background:#fafbfc}h1,h2,h3{line-height:1.2}p{max-width:1000px}table{border-collapse:collapse;font-size:12px;width:100%;margin:18px 0}th,td{padding:7px;border-bottom:1px solid #cdd6df;text-align:left}th{background:#e6ecf2}figure{margin:30px 0}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f6;padding:14px}details{margin:15px 0}summary{cursor:pointer;font-weight:600}.scroll{overflow:auto}.note{border-left:4px solid #c34d21;padding-left:18px}</style></head><body>
<h1>Autoregressive tangent development results</h1>
<p class="note">This completed experiment tests numerical forecasts from atom reconstructions. It does not test whether the atoms have valid behavioral descriptions or selective causal effects. The separately running atom-centered study addresses those questions. Full gradients and SVD measure approximation loss; outperforming them is not a success requirement for atoms.</p>
<h2>What was predicted</h2><p>At every newly sampled prefix, the original model's final logits and their directional derivative produce a first-order approximation to logits after a batch update. Responses are sampled from that approximation at temperature 0.8, without applying the update. All forecasts were locked before any actual update. The test uses reconstructed Gemma-3 4B rank-8 query/value adapters, all 136 LoRA tensors, four old synthetic catalog-copying batches and four old questions. Rates are .0003, .003 and .03. Dictionary predictions reconstruct the batch gradient from sparse atom coefficients while restoring its projected magnitude. Full gradients, EKFAC projection, SVD16 and no change are retained.</p>
<p>Each seed has 32 independent forecast-stream samples per question and condition, and 128 independent evaluation-stream samples after the actual update. Each stream uses common randomness across conditions. Evaluation uses the same custom float32 uncached math-SDPA sampler as forecasting. The cap is 96 tokens. There is one batch per style and only two evaluation questions. The adapters, task families and questions were previously inspected. These are development results, not untouched-data confirmation.</p>
<h2>Atom forecast errors</h2><p>Relative RMSE compares forecast error with predicting no change: one is equal error; below one is lower error. Probability changes and their absolute errors below are fractions; multiply by 100 for percentage points. Token errors are in tokens. The table includes every rate and endpoint, both same-question prediction and transfer of the mean calibration-question forecast to each evaluation question. Same-question access is not template transfer. No estimates are clipped.</p>"""
    ]

    def row(seed, rate, endpoint, scope="evaluation_same_question", method="dictionary"):
        return next(
            r
            for r in rows
            if r["seed"] == seed
            and r["rate"] == rate
            and r["endpoint"] == endpoint
            and r["scope"] == scope
            and r["method"] == method
        )

    middle = [row(s, 0.003, "exact_content_and_order") for s in (42, 43)]
    large = [row(s, 0.03, "exact_content_and_order") for s in (42, 43)]
    length = [row(s, 0.03, "tokens") for s in (42, 43)]
    transferred = [row(s, 0.03, "tokens", "evaluation_calibration_transfer") for s in (42, 43)]

    def pair(rr, key):
        return " / ".join(f"{r[key]:.3f}" for r in rr)

    sections.append(
        f"<p><strong>The atom reconstruction shows a limited numerical signal, with substantial failures.</strong> At rate .003, same-question copying nRMSE is {pair(middle, 'nrmse')} for seeds 42 / 43. The favorable seed-43 point estimate has absolute error {100 * middle[1]['rmse']:.2f} percentage points, but its paired bootstrap MSE-difference interval spans zero. Transferring the mean calibration prediction gives zero copying change at this rate in both seeds. At .03, same-question copying nRMSE is {pair(large, 'nrmse')}. Thus this run does not establish reliable copying forecasts across seeds or templates.</p>"
    )
    sections.append(
        f"<p>Large-step length forecasts do reduce error against no change: nRMSE {pair(length, 'nrmse')} on the same questions and {pair(transferred, 'nrmse')} after calibration-template transfer. Their pointwise MC-plus-question intervals are below one, but the size of the change is still underestimated. The same-question mean predictions are +{length[0]['predicted_mean']:.2f} / +{length[1]['predicted_mean']:.2f} tokens versus actual +{length[0]['actual_mean']:.2f} / +{length[1]['actual_mean']:.2f}. This is a descriptive length result on four fixed development batches, not validation of atom meanings or a promoted primary success.</p>"
    )
    sections.append(
        "<p>Numbered formatting remains a failure for the dictionary: no numbered responses appear in its evaluation-question forecasts, while actual large updates produce them. It also misses most severe truncation in seed 42. For the fixed plain-short batch at .03, actual copying falls from 97.27% to 31.64%, and 80.86% of responses truncate. The dictionary stream goes from 98.44% to 95.31% copying and predicts 1.56% truncation. Those are separate-stream baselines. The fixed first example preserves correct copying in the forecast while the actual response repeats and reorders tags until the cap. All other fixed examples and quality checks remain below and in the downloadable tables.</p>"
    )
    sections.append('<div class="scroll">' + table(atom_table, list(atom_table[0])) + "</div>")
    sections.append(
        "<h2>Uncertainty and limits</h2><p>There are 1,000 descriptive paired bootstrap replicates. Forecast response indices and evaluation response indices are sampled independently within question, with each set shared across conditions. Additional question resampling draws two questions within each template; calibration questions and evaluation questions are sampled separately. The four fixed batches are not resampled because each style has only one batch. The CSV and bootstrap arrays separate forecast Monte Carlo variation, evaluation Monte Carlo variation, both streams, and both streams plus questions. These are pointwise intervals, with no multiplicity correction or two-seed population claim. A zero observed effect makes relative error undefined. If any replicate has an undefined relative error, its interval is withheld and the number of defined replicates is reported; absolute RMSE and MSE difference remain available. An empirical bootstrap cannot recover events absent from the saved samples. Thirty-two forecast responses make small effects coarsely resolved; identical paired samples can yield a degenerate interval without proving that the true effect is zero. There is no claim of success from selecting a favorable endpoint or seed.</p>"
    )
    for name, caption in (
        ("evaluation_same_question", "All methods: same-question error"),
        ("evaluation_calibration_transfer", "All methods: calibration-template transfer"),
        ("dictionary-calibration", "Every dictionary forecast and observed change"),
    ):
        sections.append(
            f'<figure><img src="{name}.png" alt="{caption}"><figcaption>{caption}. Plots show point estimates; uncertainty is in the tables and machine-readable results.</figcaption></figure>'
        )
    sections.append(
        "<h2>Computation and audit</h2><p>This surrogate is computationally expensive. The forecast total includes four representations at each batch/rate; the no-change method aliases a shared baseline. Evaluation runs one actual update per batch/rate with four times as many samples per question. A raw total-time ratio is therefore not an equal-work speed comparison. No efficiency or compression claim is made. Prefixes use full uncached forwards and JVPs. Recorded forward counts include the primal forward inside a JVP; JVP counts identify those calls rather than an additional independent primal forward.</p>"
    )
    sections.append(table(cost_summary, list(cost_summary[0])))
    sections.append(
        table(
            summary["provenance"],
            (
                "seed",
                "elapsed_seconds",
                "peak_memory_bytes",
                "forecast_responses",
                "evaluation_responses",
                "updates",
                "source_snapshot_sha256",
            ),
        )
    )
    sections.append(
        "<p>Both source archives, all source/input/cache/forecast/output hashes, every lock-before-update, unchanged parameters throughout forecasts, all restores and original deadlines were verified. The initial chunk-16 memory failure on A100 was preserved, and chunk size was changed to four on both hosts before freezing. The tested scientific kernel stayed unchanged through deployment; the freeze-only contract change and the initial interpreter-symlink launcher failure are audited in provenance. The dated analysis plan preceded outcome reading. Older reweighting experiments are context only because the sampler, numerical kernel and task subset changed.</p>"
    )
    sections.append(
        '<h2>Complete results</h2><p><a href="metrics.csv">All pooled metrics and uncertainty</a> · <a href="batches.csv">Every batch and tails</a> · <a href="questions.csv">Every batch/question forecast</a> · <a href="generated.csv">All generation quality checks</a> · <a href="costs.csv">Per-condition cost</a> · <a href="summary.json">Complete JSON</a> · <a href="analysis-plan.json">Pre-reading analysis plan</a> · <a href="seed42-bootstrap.npz">Seed 42 bootstrap arrays</a> · <a href="seed43-bootstrap.npz">Seed 43 bootstrap arrays</a></p>'
    )
    for scope in SCOPES:
        rr = [r for r in rows if r["scope"] == scope]
        fields = (
            "seed",
            "rate",
            "method",
            "endpoint",
            "rmse",
            "nrmse",
            "actual_mean",
            "predicted_mean",
            "actual_on_prediction_slope",
            "batch_rmse_p90",
            "batch_rmse_max",
        )
        sections.append(
            f'<details><summary>{html.escape(scope.replace("_", " "))}: all methods</summary><div class="scroll">'
            + table(rr, fields)
            + "</div></details>"
        )
    sections.append(
        "<h2>Fixed generation examples</h2><p>Every batch, question 0 and 2, sample 0, rate, method and seed is shown below. Forecast and evaluation streams are independent; a single pair is illustrative, not an effect estimate. Content/order, format, length, truncation, empty outputs and possible degeneration are retained. No examples were chosen for favorable behavior.</p>"
    )
    for ex in summary["examples"]:
        title = f"Seed {ex['seed']}, {ex['style']}, question {ex['question']}, rate {ex['rate']}, {ex['method']}"
        sections.append(
            f"<details><summary>{html.escape(title)}</summary><p>{html.escape(ex['prompt'])}</p><h3>Forecast sample</h3><pre>{html.escape(ex['forecast_response'])}</pre><pre>{html.escape(json.dumps(dict(checks=ex['forecast_checks'], quality=ex['forecast_quality'])))}</pre><h3>Actual-update sample</h3><pre>{html.escape(ex['actual_response'])}</pre><pre>{html.escape(json.dumps(dict(checks=ex['actual_checks'], quality=ex['actual_quality'])))}</pre></details>"
        )
    sections.append("</body></html>")
    (out / "report.html").write_text("\n".join(sections))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        type=Path,
        nargs=2,
        default=[Path("runs/paper-tangent-seed42"), Path("runs/paper-tangent-seed43")],
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("runs/paper-tangent-analysis/analysis-plan.json")
    )
    parser.add_argument("--out", type=Path, default=Path("results/paper-tangent"))
    args = parser.parse_args()
    guard(args.roots, args.plan)
    stamp = args.plan.parent / "first-outcome-reading.json"
    if not stamp.exists():
        atomic_json(
            stamp,
            dict(
                started_unix=time.time(),
                analysis_plan_sha256=file_hash(args.plan),
                analysis_source_sha256=file_hash(Path(__file__)),
                both_completed_audits_verified=True,
            ),
        )
    assert read(stamp)["started_unix"] > read(args.plan)["written_unix"]
    assert read(stamp)["analysis_plan_sha256"] == file_hash(args.plan)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.plan, args.out / "analysis-plan.json")
    results = [summarize(root, args.out, 1000) for root in args.roots]
    summary = {
        k: [row for result in results for row in result[k]]
        for k in ("metrics", "batches", "questions", "generated", "examples", "costs")
    }
    summary["provenance"] = [result["provenance"] for result in results]
    summary["analysis"] = dict(
        analysis_plan_sha256=file_hash(args.plan),
        analysis_source_sha256=file_hash(Path(__file__)),
        bootstrap_replicates=1000,
        bootstrap_seed=2026101406,
        scope="development_only",
        computed_unix=time.time(),
    )
    atomic_json(args.out / "summary.json", summary)
    for key in ("metrics", "batches", "questions", "generated", "costs"):
        write_csv(args.out / f"{key}.csv", summary[key])
    plots(args.out, summary["metrics"], summary["questions"])
    report(args.out, summary)
    print(
        json.dumps(
            dict(
                report=str(args.out / "report.html"),
                metrics=len(summary["metrics"]),
                fixed_examples=len(summary["examples"]),
                complete=True,
            )
        )
    )


if __name__ == "__main__":
    main()
