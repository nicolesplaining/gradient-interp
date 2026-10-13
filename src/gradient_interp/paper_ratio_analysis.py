"""Separate direct likelihood-ratio diagnostics from prospective forecasts."""

# ruff: noqa: E501

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from .paper_batch import METHODS, STYLES
from .paper_batch_analysis import write_csv
from .paper_ratio import direct_reweight
from .paper_sample import ENDPOINTS
from .paper_sample_analysis import interval, load_complete, resampled_forecasts, stats
from .study import atomic_json, file_hash

METHOD_NAMES = ("direct_snis", *[m + "_snis" for m in METHODS], *[m + "_linear" for m in METHODS])


def read(p):
    return json.loads(p.read_text())


def load(root):
    state, audit, manifest, spec = [
        read(root / n)
        for n in ("controller.json", "integrity.json", "manifest.json", "specification.json")
    ]
    assert (
        state["status"] == "complete"
        and audit["complete"]
        and audit["source_input_output_hashes_verified"]
    )
    for name in ("manifest", "plan", "specification"):
        assert file_hash(root / f"{name}.json") == audit[f"{name}_sha256"]
    for name, value in manifest["files_sha256"].items():
        assert file_hash(root / name) == value
    assert manifest["completed_unix"] < state["deadline_unix"] and manifest["adapter_restored"]
    prior = root.parent / Path(spec["prior_root"]).name
    old = load_complete(prior)
    ratio = []
    for b in range(16):
        rows = []
        for rate in spec["rates"]:
            path = root / f"likelihoods/batch-{b:02d}-rate-{rate:g}.npz"
            record = dict(np.load(path))
            reproduced = direct_reweight(record["log_ratio"], old[2]["rewards"])
            for name, value in reproduced.items():
                np.testing.assert_allclose(record[name], value, rtol=1e-10, atol=1e-10)
            rows.append(record["log_ratio"])
        ratio.append(rows)
    base = dict(np.load(root / "likelihoods/baseline.npz"))
    np.testing.assert_array_equal(base["original_cache_logp"], old[2]["baseline_logp"])
    return spec, old, np.asarray(ratio), base


def importance_predictions(log_ratio, rewards, proposal_weights=None):
    if proposal_weights is None:
        proposal_weights = np.full(rewards.shape[:2], 1 / rewards.shape[1])
    counts = np.full_like(proposal_weights, -np.inf)
    np.log(proposal_weights, out=counts, where=proposal_weights > 0)
    lw = log_ratio + counts
    weights = np.exp(lw - lw.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    base = np.einsum("qs,qsk->qk", proposal_weights, rewards)
    return np.einsum("brqs,qsk->brqk", weights, rewards) - base


def bootstrap(cache, ratio, observed, baseline, rates, reps):
    rng = np.random.default_rng(9173)
    unit, rewards = cache["unit_rate_logp_changes"], cache["rewards"]
    lw = unit[:, None] * np.array(rates)[None, :, None, None, None]
    weights = np.exp(lw - lw.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    differences = observed[..., :4] - baseline[..., :4]
    errors, actuals, means = [], [], []
    for i in range(reps):
        pw = rng.multinomial(64, np.full(64, 1 / 64), size=16) / 64
        ew = rng.multinomial(32, np.full(32, 1 / 32), size=16) / 32
        bw = np.zeros(16)
        for j in range(4):
            bw[j::4] = rng.multinomial(4, np.full(4, 0.25)) / 16
        cq = rng.multinomial(8, np.full(8, 1 / 8)) / 8
        eq = rng.multinomial(8, np.full(8, 1 / 8)) / 8
        old = resampled_forecasts(unit, rewards, weights, pw, np.asarray(rates))
        direct = importance_predictions(ratio, rewards, pw)
        pred = np.concatenate((direct[:, :, None], old[1], old[0]), axis=2)
        same = pred[:, :, :, 8:]
        transfer = np.broadcast_to(
            np.einsum("brmqk,q->brmk", pred[:, :, :, :8], cq)[..., None, :], same.shape
        )
        preds = np.stack((same, transfer))
        target = np.einsum("brqsk,qs->brqk", differences, ew)[:, :, 8:]
        error = preds - target[None, :, :, None]
        errors.append(np.einsum("obrmqk,b,q->ormk", error**2, bw, eq, optimize=True))
        actuals.append(
            np.stack(
                (
                    np.einsum("brqk,b,q->rk", target, bw, eq),
                    np.einsum("brqk,b,q->rk", target**2, bw, eq),
                )
            )
        )
        means.append(np.einsum("obrmqk,b,q->ormk", preds, bw, eq, optimize=True))
        if (i + 1) % 100 == 0:
            print(f"bootstrap {i + 1}/{reps}", flush=True)
    return np.asarray(errors), np.asarray(actuals), np.asarray(means)


def summarize(root, reps=1000):
    spec, old, ratio, base = load(root)
    oldspec, data, cache, observed, baseline, proposals, _, _ = old
    seed, rates = spec["training_seed"], spec["rates"]
    rewards = cache["rewards"]
    pred = np.concatenate(
        (importance_predictions(ratio, rewards)[:, :, None], cache["snis"], cache["linear"]), axis=2
    )
    actual = (observed[..., :4] - baseline[..., :4]).mean(-2)
    boot, ba, bp = bootstrap(cache, ratio, observed, baseline, rates, reps)
    rows, batches, comparisons, scoring, drift = [], [], [], [], []
    for oi, ri, mi, ki in itertools.product(range(2), range(3), range(len(METHOD_NAMES)), range(4)):
        target = actual[:, ri, 8:, ki]
        p = (
            pred[:, ri, mi, 8:, ki]
            if oi == 0
            else np.broadcast_to(pred[:, ri, mi, :8, ki].mean(-1, keepdims=True), target.shape)
        )
        head = {
            "seed": seed,
            "mode": ("same_probe", "calibration_mean_transfer")[oi],
            "rate": rates[ri],
            "method": METHOD_NAMES[mi],
            "endpoint": ENDPOINTS[ki],
        }
        for style in ("all", *STYLES):
            ids = np.arange(16) if style == "all" else np.arange(STYLES.index(style), 16, 4)
            row = {**head, "style": style, **stats(p[ids], target[ids])}
            if style == "all":
                mse, denom = boot[:, oi, ri, mi, ki], ba[:, 1, ri, ki]
                normalized = np.divide(mse, denom, out=np.full_like(mse, np.nan), where=denom > 0)
                row.update(
                    rmse_ci95=interval(np.sqrt(mse)),
                    nrmse_ci95=interval(np.sqrt(normalized)),
                    nrmse_defined_fraction=float(np.mean(denom > 0)),
                    actual_mean_ci95=interval(ba[:, 0, ri, ki]),
                    predicted_mean_ci95=interval(bp[:, oi, ri, mi, ki]),
                )
                if mi == 0:
                    for ci in range(1, len(METHOD_NAMES)):
                        cp = (
                            pred[:, ri, ci, 8:, ki]
                            if oi == 0
                            else np.broadcast_to(
                                pred[:, ri, ci, :8, ki].mean(-1, keepdims=True), target.shape
                            )
                        )
                        comparisons.append(
                            {
                                **head,
                                "comparator": METHOD_NAMES[ci],
                                "mse_difference": float(
                                    np.mean((p - target) ** 2 - (cp - target) ** 2)
                                ),
                                "ci95": interval(mse - boot[:, oi, ri, ci, ki]),
                            }
                        )
            rows.append(row)
        for b in range(16):
            batches.append(
                {
                    **head,
                    "batch": b,
                    "style": data["batches"][b]["style"],
                    **stats(p[b : b + 1], target[b : b + 1]),
                }
            )
    for qi, q in enumerate(data["probes"]):
        difference = base["logp"][qi] - base["original_cache_logp"][qi]
        lengths = np.array(
            [len(r["token_ids"]) for r in proposals["rows"][64 * qi : 64 * (qi + 1)]]
        )
        drift.append(
            {
                "seed": seed,
                "question": q["id"],
                "split": q["split"],
                "max_abs_logp_drift": float(np.abs(difference).max()),
                "max_abs_per_token_drift": float((np.abs(difference) / lengths).max()),
            }
        )
    for ri, mi, split, style in itertools.product(
        range(3), range(7), ("calibration", "evaluation"), ("all", *STYLES)
    ):
        idx = np.arange(16) if style == "all" else np.arange(STYLES.index(style), 16, 4)
        qs = slice(0, 8) if split == "calibration" else slice(8, 16)
        exact = ratio[idx, ri, qs]
        lw = exact if mi == 0 else rates[ri] * cache["unit_rate_logp_changes"][idx, mi - 1, qs]
        shift = lw.max(-1, keepdims=True)
        w = np.exp(lw - shift)
        logmean = shift[..., 0] + np.log(w.mean(-1))
        w /= w.sum(-1, keepdims=True)
        ess = 1 / (w**2).sum(-1)
        rawdiff = lw - exact
        center_diff = (lw - lw.mean(-1, keepdims=True)) - (exact - exact.mean(-1, keepdims=True))
        scoring.append(
            {
                "seed": seed,
                "rate": rates[ri],
                "method": ("direct", *METHODS)[mi],
                "split": split,
                "style": style,
                "raw_ratio_rmse_vs_direct": float(np.sqrt(np.mean(rawdiff**2))),
                "centered_ratio_rmse_vs_direct": float(np.sqrt(np.mean(center_diff**2))),
                "direct_ratio_rms": float(np.sqrt(np.mean(exact**2))),
                "direct_centered_ratio_rms": float(
                    np.sqrt(np.mean((exact - exact.mean(-1, keepdims=True)) ** 2))
                ),
                "log_mean_weight_median": float(np.median(logmean)),
                "log_mean_weight_min": float(logmean.min()),
                "log_mean_weight_max": float(logmean.max()),
                "ess_median": float(np.median(ess)),
                "ess_min": float(ess.min()),
                "fraction_ess_below16": float(np.mean(ess < 16)),
            }
        )
    return {
        "metrics": rows,
        "comparisons": comparisons,
        "batch_metrics": batches,
        "scoring": scoring,
        "baseline_drift": drift,
    }


def plots(out, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#b24646", "#222222", "#777777")
    selected = ("direct_snis", "full_gradient_snis", "zero_snis")
    labels = (
        "Direct likelihood (post-update)",
        "Taylor likelihood (pre-update)",
        "Predict no change",
    )
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for si, seed in enumerate((42, 43)):
        for ki, endpoint in enumerate(ENDPOINTS):
            ax = axes[si, ki]
            for method, color, label in zip(selected, colors, labels, strict=True):
                rr = [
                    r
                    for r in summary["metrics"]
                    if r["seed"] == seed
                    and r["endpoint"] == endpoint
                    and r["method"] == method
                    and r["mode"] == "same_probe"
                    and r["style"] == "all"
                ]
                scale = 1 if endpoint == "tokens" else 100
                ax.plot(
                    [r["rate"] for r in rr],
                    [scale * r["rmse"] for r in rr],
                    "o-",
                    color=color,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_title(f"Seed {seed} · {endpoint.replace('_', ' ')}")
            ax.set_ylabel("RMSE (tokens)" if endpoint == "tokens" else "RMSE (percentage points)")
            ax.grid(alpha=0.15)
            if si == 1:
                ax.set_xlabel("SGD learning rate")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Does direct post-update likelihood fix the behavioral estimate?\nSame proposals and independent observed responses; this is an oracle diagnostic",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"direct_comparison.{ext}", dpi=160)
    plt.close(fig)


def report(out, summary):
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Likelihood diagnostic</title><style>body{max-width:1150px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#202428}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}a{color:#176b9a}</style>',
        "<h1>Where do the behavior forecasts fail?</h1><p>Post-update diagnostic · two Gemma-3 4B LoRA reconstructions · all 96 updates · same saved baseline responses.</p>",
        "<p>We replaced the gradient approximation of a response’s likelihood change with a direct forward computation under the updated model. Everything else stays fixed: the proposal samples, behavior labels, step sizes, and independent evaluation responses. This uses access to the updated model, so it is an error diagnosis rather than a usable pre-update predictor.</p>",
        "<h2>Result</h2><p><b>Direct likelihoods do not rescue the large-update estimates.</b> At step 0.03, copying-accuracy normalized error is 1.025 and 1.933 with direct ratios, versus 1.193 and 1.026 with full-gradient Taylor reweighting. A value of 1 is the no-change baseline. Direct ratios modestly improve one run and substantially worsen the other. Length estimates are worse than no change in both runs, and both methods miss new numbered and truncated responses entirely.</p>",
        "<p>At the middle step, direct-ratio accuracy errors are 0.565 and 0.431, compared with Taylor reweighting errors of 0.537 and 0.451. Similar performance suggests that eliminating Taylor error alone is not enough on these proposal samples. The original linear gradient estimates remain 0.439 and 0.448. These are descriptive development comparisons, not independent confirmation. Template transfer is also inconsistent: at the large step its direct-ratio errors are 0.547 and 2.165; at the middle step they are 0.916 and 0.961.</p>",
        "<p>The direct importance weights also reveal missing probability mass. At the largest step, the minimum empirical mean weight is about exp(−15.4) and exp(−25.0), even though the population expectation of a valid likelihood ratio is 1. This is consistent with severe finite-sample coverage failure. Direct ESS falls below 16/64 for 55% and 30% of evaluation batch–question pairs. Normalizing the remaining samples cannot invent absent response modes.</p>",
        "<p>Baseline forward scores reproduce the old saved scores exactly. The largest-step Taylor log-ratio RMSE is 9.16 and 12.05 nats; after removing each question’s mean shift, it is still 2.18 and 4.64 nats. At the smallest step, discrepancies are large relative to the small actual changes; this diagnostic does not isolate floating-point effects from curvature. The next development comparison will test broader proposals and larger samples with explicit proposal-density correction.</p>",
        "<h2>Method</h2><ol><li>Re-score each saved response at baseline, with the original temperature, actual stop token and length cap.</li><li>Apply each saved batch gradient to all 136 LoRA tensors. Score those same responses and restore the checkpoint.</li><li>Use the direct probability ratios to reweight the same 64 samples per question. Repeated samples retain their multiplicities.</li><li>Compare predicted changes with the old independent generation measurements, every original Taylor method and no-change baseline.</li></ol>",
        '<img src="direct_comparison.png"><h2>Copying accuracy on evaluation questions</h2><p>Normalized RMSE: lower is better; 1 is the error of predicting no change. Every frozen rate appears below. All endpoints, methods, styles and batch tails are in the linked tables.</p><table><tr><th>Seed</th><th>Step</th><th>Direct SNIS</th><th>95% descriptive interval</th><th>Taylor SNIS</th><th>Linear full gradient</th><th>Direct transfer</th></tr>',
    ]
    for seed, rate in itertools.product((42, 43), (0.0003, 0.003, 0.03)):

        def get(method, mode="same_probe"):
            return next(
                r
                for r in summary["metrics"]
                if r["seed"] == seed
                and r["rate"] == rate
                and r["method"] == method
                and r["mode"] == mode
                and r["style"] == "all"
                and r["endpoint"] == "exact_content_and_order"
            )

        r = get("direct_snis")
        lo, hi = r["nrmse_ci95"]
        parts.append(
            f"<tr><td>{seed}</td><td>{rate:g}</td><td>{r['nrmse']:.3f}</td><td>{lo:.3f}–{hi:.3f}</td><td>{get('full_gradient_snis')['nrmse']:.3f}</td><td>{get('full_gradient_linear')['nrmse']:.3f}</td><td>{get('direct_snis', 'calibration_mean_transfer')['nrmse']:.3f}</td></tr>"
        )
    parts.extend(
        [
            "</table><h2>Uncertainty and limits</h2><p>The 1,000 pointwise bootstrap replicates resample complete proposal responses and independent evaluation responses separately, pair common evaluation streams across baseline and updates, resample questions within each template, and resample batches within style. Both direct and approximate estimators are recomputed. These intervals describe this reused development dataset; two model seeds and eight questions per template cannot establish broad reliability. Unsampled response modes remain absent in every bootstrap.</p>",
            "<p>Zero numbered and truncated evaluation proposals mean all reweighting methods predict zero change for those events, including the direct oracle. A high effective sample size alone does not establish overlap with updated behaviors absent from the sample. Finite-sample normalization and response sampling noise remain even when likelihood ratios are exact within floating-point precision. Approximate Taylor likelihoods are not normalized distributions.</p>",
            "<h2>Integrity</h2><p>Both completed runs passed source, input, result, checkpoint-restoration and original-deadline checks. Direct scores use one response per forward call to match the original scoring procedure. The initial batched scoring preflight failed the unchanged numerical tolerance; its specification, code and logs were preserved, and a bounded correction was made before the plan was frozen. Duplicate trajectories are scored once per question and expanded to their original sample multiplicities.</p>",
            '<p><a href="metrics.csv">All behavior errors and intervals</a> · <a href="comparisons.csv">Paired comparisons</a> · <a href="batch_metrics.csv">Batch tails</a> · <a href="scoring.csv">Likelihood error, overlap and normalization</a> · <a href="baseline_drift.csv">Baseline scoring drift</a> · <a href="summary.json">Full results</a> · <a href="analysis-plan.json">Pre-reading analysis plan</a> · <a href="../paper-sample/report.html">Original prospective forecasts and fixed generation examples</a></p>',
            '<p><a href="https://arxiv.org/abs/2603.14665">Gradient Atoms</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/67900b80f1d3783b4ef51933cf7b1dbf6197c23c/src/gradient_interp/paper_ratio.py">Frozen diagnostic code</a> · <a href="https://proceedings.mlr.press/v119/huang20b.html">From Importance Sampling to Doubly Robust Policy Gradient</a>. Direct likelihood-ratio reweighting is standard importance sampling, not a novel interpretation method.</p>',
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
        assert read(root / "controller.json")["status"] == "complete"
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "posthoc oracle diagnostic", "analysis_plan": read(args.analysis_plan)}
    for root in args.roots:
        for key, value in summarize(root).items():
            summary.setdefault(key, []).extend(value)
        dest = args.out / root.name
        shutil.copytree(
            root,
            dest,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.tar", "*.log", "*.py", "*.jsonl"),
        )
    shutil.copy2(args.analysis_plan, args.out / "analysis-plan.json")
    atomic_json(args.out / "summary.json", summary)
    for key in ("metrics", "comparisons", "batch_metrics", "scoring", "baseline_drift"):
        write_csv(args.out / f"{key}.csv", summary[key])
    plots(args.out, summary)
    report(args.out, summary)
    print(args.out / "report.html")


if __name__ == "__main__":
    main()
