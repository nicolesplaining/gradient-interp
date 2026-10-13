"""Descriptive analysis with proposal/evaluation sampling uncertainty."""

# ruff: noqa: E501

import argparse
import html
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from .paper_batch import METHODS, STYLES, generated_checks
from .paper_batch_analysis import LABELS, write_csv
from .paper_sample import ENDPOINTS, estimators
from .paper_steering_analysis import quality
from .study import atomic_json, file_hash

EXTRA = ("all_tags_present", "words", "empty", "possible_degeneration")


def read(path):
    return json.loads(path.read_text())


def values(rows, probes, n, cap):
    if [(r["id"], r["sample"]) for r in rows] != [(q["id"], s) for q in probes for s in range(n)]:
        raise ValueError("Unpaired response IDs")
    output = []
    for i, row in enumerate(rows):
        checks = generated_checks(row["response"], probes[i // n]["tags"])
        diagnostic = quality(row, cap)
        if checks != row["checks"] or diagnostic != row["quality"]:
            raise ValueError("Response checks do not reproduce")
        merged = {**checks, **diagnostic}
        output.append([merged[k] for k in (*ENDPOINTS, *EXTRA)])
    return np.array(output, dtype=float).reshape(len(probes), n, -1)


def load_complete(root):
    state, audit, manifest, spec, data = [
        read(root / n)
        for n in (
            "controller.json",
            "integrity.json",
            "manifest.json",
            "specification.json",
            "data.json",
        )
    ]
    if (
        state["status"] != "complete"
        or not audit["complete"]
        or not audit["source_input_output_hashes_verified"]
    ):
        raise ValueError("Completed run and remote integrity audit required")
    for name, key in (
        ("plan.json", "plan_sha256"),
        ("manifest.json", "manifest_sha256"),
        ("specification.json", "specification_sha256"),
        ("forecast-lock.json", "forecast_lock_sha256"),
    ):
        if file_hash(root / name) != audit[key]:
            raise ValueError(f"Provenance mismatch: {name}")
    for name, expected in manifest["files_sha256"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Output hash mismatch: {name}")
    if manifest["completed_unix"] > state["deadline_unix"] or not manifest["adapter_restored"]:
        raise ValueError("Invalid deadline/restoration")
    lock = read(root / "forecast-lock.json")
    proposals = read(root / "proposals.json")
    baseline = read(root / "outcomes/baseline.json")
    cache = dict(np.load(root / "forecasts.npz"))
    rewards = values(proposals["rows"], data["probes"], 64, spec["max_new_tokens"])[..., :4]
    np.testing.assert_array_equal(rewards, cache["rewards"])
    linear, snis, ess = estimators(cache["unit_rate_logp_changes"], rewards, spec["rates"])
    for name, expected in (("linear", linear), ("snis", snis), ("ess", ess)):
        np.testing.assert_allclose(cache[name], expected, rtol=1e-10, atol=1e-10)
    bvalues = values(baseline["rows"], data["probes"], 32, spec["max_new_tokens"])
    outcomes, arrays = [], []
    for bi in range(16):
        records, array = [], []
        for rate in spec["rates"]:
            value = read(root / f"outcomes/batch-{bi:02d}-rate-{rate:g}.json")
            if not value["adapter_restored"] or not (
                lock["locked_unix"]
                < value["update_started_unix"]
                < value["completed_unix"]
                < manifest["completed_unix"]
            ):
                raise ValueError("Update ordering or restoration mismatch")
            if (
                value["batch_index"] != bi
                or value["rate"] != rate
                or value["plan_sha256"] != audit["plan_sha256"]
                or value["forecast_lock_sha256"] != audit["forecast_lock_sha256"]
            ):
                raise ValueError("Update metadata mismatch")
            array.append(values(value["rows"], data["probes"], 32, spec["max_new_tokens"]))
            records.append(value)
        arrays.append(array)
        outcomes.append(records)
    return spec, data, cache, np.asarray(arrays), bvalues, proposals, baseline, outcomes


def stats(pred, actual):
    error = pred - actual
    mse = np.mean(error**2)
    actual_rms = np.sqrt(np.mean(actual**2))
    batch_rmse = np.sqrt(np.mean(error**2, axis=-1))
    correlation = None
    if pred.std() > 1e-15 and actual.std() > 1e-15:
        correlation = float(np.corrcoef(pred.ravel(), actual.ravel())[0, 1])
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "nrmse": float(np.sqrt(mse) / actual_rms) if actual_rms > 0 else None,
        "actual_rms": float(actual_rms),
        "predicted_rms": float(np.sqrt(np.mean(pred**2))),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(pred.mean()),
        "actual_on_prediction_slope": float(np.sum(actual * pred) / np.sum(pred**2))
        if np.sum(pred**2) > 1e-30
        else None,
        "correlation": correlation,
        "batch_rmse_median": float(np.median(batch_rmse)),
        "batch_rmse_p90": float(np.quantile(batch_rmse, 0.9)),
        "batch_rmse_max": float(batch_rmse.max()),
    }


def resampled_forecasts(unit, rewards, normalized_weights, proposal_weights, rates):
    """Resample complete trajectories jointly for their score gradients and rewards."""
    reward_mean = np.einsum("qs,qsk->qk", proposal_weights, rewards)
    unit_mean = np.einsum("bmqs,qs->bmq", unit, proposal_weights)
    cross = np.einsum("bmqs,qs,qsk->bmqk", unit, proposal_weights, rewards, optimize=True)
    derivative = (
        (cross - unit_mean[..., None] * reward_mean) * rewards.shape[1] / (rewards.shape[1] - 1)
    )
    linear = derivative[:, None] * rates[None, :, None, None, None]
    normalizer = np.einsum("brmqs,qs->brmq", normalized_weights, proposal_weights)
    numerator = np.einsum(
        "brmqs,qs,qsk->brmqk", normalized_weights, proposal_weights, rewards, optimize=True
    )
    if np.any(normalizer < 1e-250):
        # A bootstrap can omit the highest-weight trajectory. Recenter on its actual support.
        log_counts = np.full_like(proposal_weights, -np.inf)
        np.log(proposal_weights, out=log_counts, where=proposal_weights > 0)
        logw = unit[:, None] * rates[None, :, None, None, None] + log_counts
        weights = np.exp(logw - logw.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        snis = np.einsum("brmqs,qsk->brmqk", weights, rewards) - reward_mean
    else:
        snis = numerator / normalizer[..., None] - reward_mean
    return np.stack((linear, snis))


def bootstrap(cache, observed, baseline, rates, repetitions=1000):
    rng = np.random.default_rng(8173)
    unit, rewards = cache["unit_rate_logp_changes"].astype(float), cache["rewards"]
    logw = unit[:, None] * np.asarray(rates)[None, :, None, None, None]
    weights = np.exp(logw - logw.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    differences = observed[..., :4] - baseline[..., :4]
    all_mse, all_actual, all_predicted = [], [], []
    for i in range(repetitions):
        pw = rng.multinomial(64, np.full(64, 1 / 64), size=16) / 64
        ew = rng.multinomial(32, np.full(32, 1 / 32), size=16) / 32
        bw = np.zeros(16)
        for j in range(4):
            bw[j::4] = rng.multinomial(4, np.full(4, 0.25)) / 16
        cq = rng.multinomial(8, np.full(8, 1 / 8)) / 8
        eq = rng.multinomial(8, np.full(8, 1 / 8)) / 8
        forecast = resampled_forecasts(unit, rewards, weights, pw, np.asarray(rates))
        target = np.einsum("brqsk,qs->brqk", differences, ew, optimize=True)[:, :, 8:]
        same = forecast[:, :, :, :, 8:]
        transfer = np.broadcast_to(
            np.einsum("ebrmqk,q->ebrmk", forecast[:, :, :, :, :8], cq)[..., None, :], same.shape
        )
        preds = np.stack(
            (same, transfer), axis=1
        )  # estimator, mode, batch, rate, method, question, endpoint
        error = preds - target[None, None, :, :, None]
        all_mse.append(np.einsum("eobrmqk,b,q->eormk", error**2, bw, eq, optimize=True))
        all_actual.append(
            np.stack(
                (
                    np.einsum("brqk,b,q->rk", target, bw, eq),
                    np.einsum("brqk,b,q->rk", target**2, bw, eq),
                )
            )
        )
        all_predicted.append(np.einsum("eobrmqk,b,q->eormk", preds, bw, eq, optimize=True))
        if (i + 1) % 100 == 0:
            print(f"bootstrap {i + 1}/{repetitions}", flush=True)
    return np.asarray(all_mse), np.asarray(all_actual), np.asarray(all_predicted)


def interval(x):
    x = np.asarray(x)
    finite = x[np.isfinite(x)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None


def summarize(root, repetitions):
    spec, data, cache, observed, baseline, proposals, base_record, outcomes = load_complete(root)
    seed = spec["training_seed"]
    actual = (observed - baseline).mean(axis=-2)
    predictions = np.stack((cache["linear"], cache["snis"]))
    boot_mse, boot_actual, boot_predicted = bootstrap(
        cache, observed, baseline, spec["rates"], repetitions
    )
    rows, comparisons, batches, coverage, overlap, generated, examples = [], [], [], [], [], [], []
    for ei, oi, ri, mi, ki in itertools.product(range(2), range(2), range(3), range(6), range(4)):
        target = actual[:, ri, 8:, ki]
        pred = (
            predictions[ei, :, ri, mi, 8:, ki]
            if oi == 0
            else np.broadcast_to(
                predictions[ei, :, ri, mi, :8, ki].mean(1, keepdims=True), target.shape
            )
        )
        header = {
            "seed": seed,
            "estimator": ("linear", "snis")[ei],
            "mode": ("same_probe", "calibration_mean_transfer")[oi],
            "rate": spec["rates"][ri],
            "method": METHODS[mi],
            "endpoint": ENDPOINTS[ki],
        }
        for group in ("all", *STYLES):
            idx = np.arange(16) if group == "all" else np.arange(STYLES.index(group), 16, 4)
            row = {**header, "style": group, **stats(pred[idx], target[idx])}
            if group == "all":
                mse = boot_mse[:, ei, oi, ri, mi, ki]
                denom = boot_actual[:, 1, ri, ki]
                normalized = np.divide(mse, denom, out=np.full_like(mse, np.nan), where=denom > 0)
                row.update(
                    rmse_ci95=interval(np.sqrt(mse)),
                    nrmse_ci95=interval(np.sqrt(normalized)),
                    nrmse_bootstrap_defined_fraction=float(np.mean(denom > 0)),
                    actual_mean_ci95=interval(boot_actual[:, 0, ri, ki]),
                    predicted_mean_ci95=interval(boot_predicted[:, ei, oi, ri, mi, ki]),
                )
                difference_samples = observed[:, ri, 8:, :, ki] - baseline[8:, :, ki]
                row["evaluation_mean_noise_rms"] = float(
                    np.sqrt(np.mean(difference_samples.var(axis=-1, ddof=1) / 32))
                )
                if ki in (0, 1, 3):
                    levels = cache["rewards"][8:, :, ki].mean(axis=1) + pred
                    row["outside_probability_range_fraction"] = float(
                        np.mean((levels < 0) | (levels > 1))
                    )
                for comparator in ("zero", "svd16"):
                    ci = METHODS.index(comparator)
                    cpred = (
                        predictions[ei, :, ri, ci, 8:, ki]
                        if oi == 0
                        else np.broadcast_to(
                            predictions[ei, :, ri, ci, :8, ki].mean(1, keepdims=True), target.shape
                        )
                    )
                    comparisons.append(
                        {
                            **header,
                            "comparator": comparator,
                            "mse_difference": float(
                                np.mean((pred - target) ** 2 - (cpred - target) ** 2)
                            ),
                            "ci95": interval(mse - boot_mse[:, ei, oi, ri, ci, ki]),
                        }
                    )
            rows.append(row)
        for bi in range(16):
            batches.append(
                {
                    **header,
                    "batch": bi,
                    "style": data["batches"][bi]["style"],
                    **stats(pred[bi : bi + 1], target[bi : bi + 1]),
                }
            )
    for qi, probe in enumerate(data["probes"]):
        for ki, endpoint in enumerate(ENDPOINTS):
            v = cache["rewards"][qi, :, ki]
            coverage.append(
                {
                    "seed": seed,
                    "question": probe["id"],
                    "split": probe["split"],
                    "endpoint": endpoint,
                    "proposal_mean": float(v.mean()),
                    "proposal_min": float(v.min()),
                    "proposal_max": float(v.max()),
                    "positive_samples": int((v > 0).sum()),
                    "samples": 64,
                    "independent_baseline_mean": float(baseline[qi, :, ki].mean()),
                    "any_post_update_positive": bool(np.any(observed[:, :, qi, :, ki] > 0)),
                    "zero_events_pointwise_binomial_upper95": float(1 - 0.05 ** (1 / 64))
                    if ki in (0, 1, 3) and not v.any()
                    else None,
                }
            )
    for ri, mi, split in itertools.product(range(3), range(6), ("calibration", "evaluation")):
        qs = slice(0, 8) if split == "calibration" else slice(8, 16)
        ess = cache["ess"][:, ri, mi, qs]
        overlap.append(
            {
                "seed": seed,
                "rate": spec["rates"][ri],
                "method": METHODS[mi],
                "split": split,
                "ess_min": float(ess.min()),
                "ess_median": float(np.median(ess)),
                "fraction_below16": float(np.mean(ess < 16)),
            }
        )
    for ri, split, style, ki in itertools.product(
        range(3), ("calibration", "evaluation"), ("all", *STYLES), range(8)
    ):
        qs = slice(0, 8) if split == "calibration" else slice(8, 16)
        bs = np.arange(16) if style == "all" else np.arange(STYLES.index(style), 16, 4)
        generated.append(
            {
                "seed": seed,
                "rate": spec["rates"][ri],
                "split": split,
                "style": style,
                "endpoint": (*ENDPOINTS, *EXTRA)[ki],
                "baseline": float(baseline[qs, :, ki].mean()),
                "after": float(observed[bs, ri, qs, :, ki].mean()),
                "change": float(actual[bs, ri, qs, ki].mean()),
            }
        )
    for bi, qi, ri in itertools.product(range(4), (0, 8), range(3)):
        examples.append(
            {
                "seed": seed,
                "batch": bi,
                "style": data["batches"][bi]["style"],
                "rate": spec["rates"][ri],
                "prompt": data["probes"][qi]["prompt"],
                "split": data["probes"][qi]["split"],
                "baseline": base_record["rows"][qi * 32],
                "updated": outcomes[bi][ri]["rows"][qi * 32],
            }
        )
    return {
        "metrics": rows,
        "comparisons": comparisons,
        "batch_metrics": batches,
        "coverage": coverage,
        "overlap": overlap,
        "generated": generated,
        "examples": examples,
    }


def plots(out, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#222222", "#4590b9", "#c15e2b", "#518745", "#9e5ea6", "#999999")
    for endpoint in ENDPOINTS:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        for si, seed in enumerate((42, 43)):
            for col, (estimator, mode) in enumerate(
                itertools.product(("linear", "snis"), ("same_probe", "calibration_mean_transfer"))
            ):
                ax = axes[si, col]
                for mi, method in enumerate(METHODS):
                    rr = [
                        r
                        for r in summary["metrics"]
                        if r["seed"] == seed
                        and r["estimator"] == estimator
                        and r["mode"] == mode
                        and r["endpoint"] == endpoint
                        and r["method"] == method
                        and r["style"] == "all"
                    ]
                    scale = 1 if endpoint == "tokens" else 100
                    ax.plot(
                        [r["rate"] for r in rr],
                        [r["rmse"] * scale for r in rr],
                        marker="o",
                        color=colors[mi],
                        label=LABELS[mi],
                    )
                ax.set_xscale("log")
                ax.set_title(
                    f"Seed {seed} · {estimator.upper()}\n{'Same probes' if mode == 'same_probe' else 'Calibration → evaluation'}"
                )
                ax.grid(alpha=0.15)
                if col == 0:
                    ax.set_ylabel(
                        "RMSE (tokens)" if endpoint == "tokens" else "RMSE (percentage points)"
                    )
                if si == 1:
                    ax.set_xlabel("SGD learning rate")
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False
        )
        fig.suptitle(
            f"Forecasting actual sampled behavior · {endpoint.replace('_', ' ')}\nAll frozen methods and rates; lower error is better",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.91))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"error_{endpoint}.{ext}", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for si, seed in enumerate((42, 43)):
        for ki, endpoint in enumerate(ENDPOINTS):
            ax = axes[si, ki]
            for style in STYLES:
                rr = [
                    r
                    for r in summary["generated"]
                    if r["seed"] == seed
                    and r["split"] == "evaluation"
                    and r["style"] == style
                    and r["endpoint"] == endpoint
                ]
                ax.plot(
                    [r["rate"] for r in rr],
                    [r["after"] for r in rr],
                    marker="o",
                    label=style.replace("_", " "),
                )
            ax.axhline(rr[0]["baseline"], color="black", ls=":", label="Baseline")
            ax.set_xscale("log")
            ax.set_title(f"Seed {seed} · {endpoint.replace('_', ' ')}")
            ax.grid(alpha=0.15)
            if endpoint != "tokens":
                ax.set_ylim(-0.03, 1.03)
            if si == 1:
                ax.set_xlabel("SGD learning rate")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Measured sampled-response behavior on independent-template questions\nEach style: four batches × eight prompts × 32 samples; temperature 0.8",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.91))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"generated_behavior.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, seed in zip(axes, (42, 43), strict=True):
        for mi, method in enumerate(METHODS):
            rr = [
                r
                for r in summary["overlap"]
                if r["seed"] == seed and r["method"] == method and r["split"] == "evaluation"
            ]
            ax.plot(
                [r["rate"] for r in rr],
                [r["ess_median"] for r in rr],
                marker="o",
                label=LABELS[mi],
                color=colors[mi],
            )
        ax.axhline(16, color="#777", ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 66)
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("SGD learning rate")
        ax.set_ylabel("Median effective samples / 64")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False, fontsize=9
    )
    fig.suptitle("Reweighting overlap: median effective sample size; tails are in the full tables")
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"overlap.{ext}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, seed in zip(axes, (42, 43), strict=True):
        for mi, method in enumerate(METHODS[:-1]):
            r = next(
                r
                for r in summary["metrics"]
                if r["seed"] == seed
                and r["method"] == method
                and r["estimator"] == "linear"
                and r["mode"] == "same_probe"
                and r["rate"] == 0.003
                and r["endpoint"] == "exact_content_and_order"
                and r["style"] == "all"
            )
            lo, hi = r["nrmse_ci95"]
            ax.plot([lo, hi], [mi, mi], color=colors[mi], lw=2)
            ax.scatter([r["nrmse"]], [mi], color=colors[mi], s=40)
        ax.axvline(1, color="#777", ls=":")
        ax.set_yticks(range(5), LABELS[:5])
        ax.invert_yaxis()
        ax.set_xlabel("RMSE / error of predicting no change")
        ax.set_title(f"Seed {seed}")
        ax.set_xlim(0, 1.8)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Copying accuracy forecasts at the middle step (0.003)\nPointwise 95% bootstrap intervals; exploratory detail, all rates reported separately"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"accuracy_uncertainty.{ext}", dpi=160)
    plt.close(fig)


def findings(summary):
    def metric(seed, method, rate=0.003, mode="same_probe"):
        return next(
            r
            for r in summary["metrics"]
            if r["seed"] == seed
            and r["method"] == method
            and r["rate"] == rate
            and r["mode"] == mode
            and r["estimator"] == "linear"
            and r["endpoint"] == "exact_content_and_order"
            and r["style"] == "all"
        )

    parts = [
        "<h2>What we learned</h2><p><b>Local changes in copying accuracy are partly predictable, but this is not yet a reliable general predictor.</b> At the middle step, the full-gradient linear forecast has about 44–45% of the error of predicting no change. Its pointwise uncertainty intervals still include no improvement in both seeds. Dictionary forecasts are worse than SVD16 on this comparison. Moving forecasts between prompt templates loses most of the gain.</p>",
        "<p>The table reports normalized error for exact tag content and order on the evaluation questions: lower is better, and 1 means the error of predicting no change. The middle-rate detail is exploratory; no rate was selected as a confirmatory success.</p>",
        "<table><tr><th>Seed</th><th>Step</th><th>Full gradient</th><th>Dictionary</th><th>SVD16</th><th>Full gradient, template transfer</th></tr>",
    ]
    for seed, rate in itertools.product((42, 43), (0.0003, 0.003, 0.03)):
        rr = [metric(seed, m, rate) for m in ("full_gradient", "dictionary", "svd16")]
        transfer = metric(seed, "full_gradient", rate, "calibration_mean_transfer")
        parts.append(
            f"<tr><td>{seed}</td><td>{rate:g}</td>"
            + "".join(f"<td>{r['nrmse']:.3f}</td>" for r in (*rr, transfer))
            + "</tr>"
        )
    parts.extend(
        [
            '</table><img src="accuracy_uncertainty.png">',
            "<p>At step 0.003, full-gradient accuracy RMSE is 1.72 and 4.40 percentage points (seeds 42 and 43). The worst batch errors are 2.91 and 6.94 points; template transfer raises those worst-batch errors to 7.17 and 17.04 points. The pooled result is driven largely by plain-short batches: their measured accuracy gains are 3.61 and 14.84 points, versus forecasts of 3.60 and 17.56 points. The other three styles have smaller effects and much poorer relative accuracy. These are descriptive comparisons on reused synthetic tasks.</p>",
            "<p><b>Absent baseline examples cause a separate failure.</b> Neither seed has a numbered or truncated response among the 64 proposals on any of the eight evaluation questions. All same-question estimators therefore predict zero change for these endpoints. At the largest step, both behaviors appear. More accurate weights alone cannot repair a sample that contains no examples of the event.</p>",
            "<p>Large updates defeat the forecasts and concentrate the approximate importance weights. At step 0.03, the full-gradient linear accuracy error exceeds the no-change baseline in both seeds; SNIS also fails to beat it. Full-gradient ESS falls below 16/64 for 52% and 44% of evaluation batch–question pairs. Dictionary weights remain more uniform partly because their predicted changes are small; higher ESS is not evidence of a better predictor.</p>",
            "<p><b>Actual response quality must be checked alongside formatting.</b> For plain-short batches at the largest step, seed 42 copying accuracy falls from 94.1% to 44.0%, with 70.0% of responses hitting the length cap. Seed 43 rises from 74.6% to 92.0% accuracy but has 15.3% truncation. At the middle step, these same groups improve without truncation. The fixed first-example panel shows seed 42 repeating and adding incorrect tags at the large step; the corresponding seed 43 example stays correct. Neither example was selected for its outcome.</p>",
            "<p>The next development diagnostic separates approximate gradient likelihood ratios from limited proposal coverage by scoring the same saved responses under each updated adapter. That calculation uses post-update access, so it will be an error diagnosis, not a prospective forecast or a new positive result.</p>",
        ]
    )
    return parts


def report(out, summary):
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Sampled behavior forecasts</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#202428}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f3f5f7;padding:14px}details{margin:18px 0}a{color:#176b9a}</style>',
        "<h1>Forecasting what the model actually samples</h1><p>Development study · paper-architecture LoRA · two reconstructed Gemma-3 4B adapters · 96 updates · 50,176 evaluation responses and 2,048 baseline proposal responses.</p>",
        "<p>The previous pilot predicted scores for four hand-written answers. Those candidates missed common outputs, and their probability changes did not track generated formatting. This pilot instead samples responses from the unchanged model, measures their formatting and content, and predicts how an update would change their prevalence.</p>",
        *findings(summary),
        "<h2>Method, step by step</h2><ol><li>Reuse the same 16 synthetic copying batches and LoRA gradients. Keep the three step sizes fixed.</li><li>At each of 16 probes, draw 64 baseline responses using unrestricted vocabulary sampling at temperature 0.8, stopping at EOS or 96 tokens.</li><li>Differentiate each sampled response’s exact temperature-adjusted log probability. Contract those gradients with each full or compressed batch gradient.</li><li>Turn those predicted likelihood changes into expected behavior changes using either a centered linear score-function estimator or normalized exponential reweighting (SNIS). Save every forecast before any update.</li><li>Draw a separate set of 32 responses per probe before and after each update. Measure numbering, exact copied tags/order, length and truncation. Restore the adapter after each of 48 updates per seed.</li><li>Evaluate the full fixed grid after both runs finish. Compare forecasts on the same questions and forecasts transferred from a different prompt template.</li></ol>",
        "<p>These are established score-function and importance-sampling estimators, not new mathematical identities. Sampling differs from the previous greedy endpoint. The data and model seeds are reused development material.</p>",
        '<h2>Actual behavioral changes</h2><img src="generated_behavior.png">',
        "<h2>Forecast errors</h2><p>Absolute errors below use percentage points for binary outcomes and tokens for length. The zero method predicts no change. The full tables also give normalized errors, uncertainty intervals, calibration slopes, per-batch tails and paired differences against SVD16 and zero. No low-overlap conditions are dropped.</p>",
    ]
    for endpoint in ENDPOINTS:
        parts.append(
            f'<h3>{html.escape(endpoint.replace("_", " "))}</h3><img src="error_{endpoint}.png">'
        )
    parts.extend(
        [
            '<h2>Coverage and uncertainty</h2><img src="overlap.png">',
            "<p>Each plotted estimate is based on finite baseline and evaluation samples. The 1,000 bootstrap replicates resample batches within each style, calibration and evaluation questions separately, and proposal and evaluation responses separately. Responses are paired across methods, rates, baseline and updates where the random streams were shared. Both forecast estimators are recomputed in every replicate. Intervals are pointwise and descriptive; two reused model seeds and eight evaluation prompts do not support broad population claims.</p>",
            "<p>Effective sample size below 16 of 64 was flagged before results were inspected. A zero-event proposal cannot predict emergence of an unseen response mode by reweighting. Bootstrap intervals cannot fix absent support: for zero observed events in 64 independent proposals, a pointwise binomial upper bound is still about 4.6%. Linear probability forecasts outside [0,1] are flagged rather than clipped. Measured effect estimates also have evaluation sampling noise; headline errors retain that noise.</p>",
            "<h2>Fixed examples</h2><p>Batch indices 0–3, probes 0 and 8, sample 0; all rates and both seeds, selected before outcome inspection.</p>",
        ]
    )
    for ex in summary["examples"]:
        parts.append(
            f"<details><summary>Seed {ex['seed']}, {ex['style']}, rate {ex['rate']:g}, {ex['split']}</summary><p>{html.escape(ex['prompt'])}</p><b>Baseline</b><pre>{html.escape(ex['baseline']['response'])}</pre><b>After update</b><pre>{html.escape(ex['updated']['response'])}</pre><p>{html.escape(json.dumps(ex['updated']['checks']))}</p></details>"
        )
    parts.extend(
        [
            "<h2>Integrity and limits</h2><p>All forecast/result/source/input hashes, 48-update grids per seed, checkpoint restoration and original deadlines were checked. A library sampling-default override was detected in logs before any forecast lock or update. Both initial workers were intentionally stopped; the original freeze and logs remain archived. The implementation was corrected and preflight-tested, with the original specification and deadlines unchanged. The amended source hash and both attempts are recorded in the repair audit.</p>",
            "<p>This is synthetic copying, two reused adapters, two different GPUs and a small set of prompts. It does not validate atom descriptions, selective semantic interventions, general correctness forecasting or novelty. The original paper’s rank-8 checkpoint was unavailable: these are architecture-matched reconstructions. SVD16 is a simple baseline, not an equal-storage claim. All unsuccessful methods and extreme forecasts are included.</p>",
            '<p><a href="https://arxiv.org/abs/2603.14665">Gradient Atoms</a> · <a href="https://doi.org/10.1007/BF00992696">Williams (1992)</a> · <a href="https://proceedings.mlr.press/v119/huang20b.html">Huang &amp; Jiang (2020)</a> · <a href="https://papers.neurips.cc/paper/7789-policy-optimization-via-importance-sampling.pdf">Metelli et al. (2018)</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/40274e4c7a4752c59d617d327c9a387d2280ffa6/src/gradient_interp/paper_sample.py">Frozen generation/forecast code</a></p>',
            '<p><a href="summary.json">All results</a> · <a href="metrics.csv">Prediction metrics</a> · <a href="comparisons.csv">Paired comparisons</a> · <a href="batch_metrics.csv">Batch errors</a> · <a href="coverage.csv">Proposal coverage</a> · <a href="overlap.csv">Effective sample size</a> · <a href="generated.csv">Generation checks</a> · <a href="analysis-plan.json">Pre-reading analysis plan</a></p>',
        ]
    )
    (out / "report.html").write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    args = parser.parse_args()
    for root in args.roots:
        if read(root / "controller.json")["status"] != "complete":
            raise ValueError("Both runs must be complete before analysis")
    plan = read(args.analysis_plan)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"analysis_plan": plan, "scope": "synthetic_development_only"}
    for root in args.roots:
        result = summarize(root, 1000)
        for key, value in result.items():
            summary.setdefault(key, []).extend(value)
        dest = args.out / f"seed{read(root / 'specification.json')['training_seed']}"
        dest.mkdir(exist_ok=True)
        for name in (
            "controller.json",
            "integrity.json",
            "manifest.json",
            "plan.json",
            "specification.json",
            "forecast-lock.json",
            "preflight.json",
            "data.json",
            "forecasts.npz",
        ):
            shutil.copy2(root / name, dest / name)
        shutil.copytree(
            root / "repairs",
            dest / "repairs",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.tar", "*.log", "*.py"),
        )
    shutil.copy2(args.analysis_plan, args.out / "analysis-plan.json")
    atomic_json(args.out / "summary.json", summary)
    for key in ("metrics", "comparisons", "batch_metrics", "coverage", "overlap", "generated"):
        write_csv(args.out / f"{key}.csv", summary[key])
    plots(args.out, summary)
    report(args.out, summary)
    print(args.out / "report.html")


if __name__ == "__main__":
    main()
