"""Guarded oracle analysis of all frozen float32 mixture-bank likelihoods."""

# ruff: noqa: E501
import argparse
import html
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from .paper_batch import METHODS, STYLES
from .paper_batch_analysis import write_csv
from .paper_mixture import BANKS, normalized
from .paper_mixture_analysis import load as load_mixture
from .paper_mixture_analysis import nested_counts, resampled_bank
from .paper_mixture_ratio import direct_banks
from .paper_sample import ENDPOINTS
from .paper_sample_analysis import interval, stats
from .study import atomic_json, file_hash

METHOD_NAMES = ("direct_snis", *[m + "_snis" for m in METHODS], *[m + "_linear" for m in METHODS])


def read(path):
    return json.loads(path.read_text())


def load(root):
    state, audit, manifest, spec = [
        read(root / n)
        for n in ("controller.json", "integrity.json", "manifest.json", "specification.json")
    ]
    if (
        state["status"] != "complete"
        or not audit["complete"]
        or not audit["source_input_output_hashes_verified"]
    ):
        raise ValueError("Complete original-host audit required")
    for name in ("plan", "specification", "manifest"):
        assert file_hash(root / f"{name}.json") == audit[f"{name}_sha256"]
    for name, expected in manifest["files_sha256"].items():
        assert file_hash(root / name) == expected
    assert manifest["completed_unix"] < state["deadline_unix"] and manifest["adapter_restored"]
    assert manifest["conditions"] == 49 and manifest["updates"] == 48
    assert manifest["proposal_trajectories_per_condition"] == 8192
    prior = root.parent / Path(spec["prior_root"]).name
    for name in (
        "forecasts.npz",
        "proposals.json",
        "batch-representations.pt",
        "specification.json",
        "manifest.json",
        "integrity.json",
        "forecast-lock.json",
        "data.json",
    ):
        assert file_hash(prior / name) == spec["input_hashes"][str(Path(spec["prior_root"]) / name)]
    old = load_mixture(prior)
    cache = old[2]
    base = dict(np.load(root / "likelihoods/baseline.npz"))
    np.testing.assert_array_equal(base["original_cache_logp"], cache["baseline_logp"])
    np.testing.assert_array_equal(base["original_warm_logp"], cache["warm_logp"])
    records, logps = [], []
    for b in range(16):
        br, bl = [], []
        for rate in spec["rates"]:
            path = root / f"likelihoods/batch-{b:02d}-rate-{rate:g}.npz"
            record = dict(np.load(path))
            meta = read(path.with_suffix(".json"))
            assert meta["adapter_restored"] and meta["batch_index"] == b and meta["rate"] == rate
            assert meta["sha256"] == file_hash(path) and meta["plan_sha256"] == audit["plan_sha256"]
            np.testing.assert_array_equal(record["log_ratio"], record["logp"] - base["logp"])
            for n, v in direct_banks(
                record["logp"], cache["baseline_logp"], cache["warm_logp"], cache["rewards"]
            ).items():
                np.testing.assert_allclose(record[n], v, rtol=1e-10, atol=1e-10)
            br.append(record)
            bl.append(record["logp"])
        records.append(br)
        logps.append(bl)
    return spec, old, np.asarray(logps), base, records


def resampled_direct(log_weights, log_base_weights, rewards, counts):
    """Return [replicate,batch,rate,question,endpoint]; retain original q."""
    t, q, s = counts.shape
    b, r = log_weights.shape[:2]
    out = np.empty((t, b, r, q, rewards.shape[-1]))
    for qi in range(q):
        c, y = counts[:, qi], rewards[qi]
        w0 = c * normalized(log_base_weights[qi])
        w0 /= w0.sum(-1, keepdims=True)
        base = w0 @ y
        lw = log_weights[:, :, qi].reshape(b * r, s)
        w = normalized(lw)
        den = c @ w.T
        num = np.stack([c @ (w * y[:, k]).T for k in range(y.shape[-1])], axis=-1)
        safe = den > 1e-250
        updated = np.divide(num, den[..., None], out=np.zeros_like(num), where=safe[..., None])
        for ti, di in zip(*np.where(~safe), strict=True):
            logc = np.full(s, -np.inf)
            np.log(c[ti], out=logc, where=c[ti] > 0)
            updated[ti, di] = normalized(lw[di] + logc) @ y
        out[:, :, :, qi] = (updated - base[:, None]).reshape(t, b, r, -1)
    return out


def bootstrap(cache, logp, observed, baseline, rates, repetitions):
    rng = np.random.default_rng(102713)
    pc = nested_counts(rng, repetitions)
    ew = rng.multinomial(128, np.full(128, 1 / 128), size=(repetitions, 16)) / 128
    bw = np.zeros((repetitions, 16))
    for j in range(4):
        bw[:, j::4] = rng.multinomial(4, np.full(4, 0.25), size=repetitions) / 16
    cq = rng.multinomial(8, np.full(8, 1 / 8), size=repetitions) / 8
    eq = rng.multinomial(8, np.full(8, 1 / 8), size=repetitions) / 8
    target = np.einsum("brqsk,tqs->tbrqk", (observed - baseline)[..., :4], ew, optimize=True)[
        :, :, :, 8:
    ]
    actual = np.stack(
        [np.einsum("tbrqk,tb,tq->trk", v, bw, eq, optimize=True) for v in (target, target**2)],
        axis=1,
    )
    errors, means = [], []
    for bank, (idx, _) in BANKS.items():
        w0 = cache[f"{bank}_log_base_weights"]
        lw = (logp - cache["baseline_logp"])[..., idx] + w0
        direct = resampled_direct(lw, w0, cache["rewards"][:, idx], pc[..., idx])
        old = resampled_bank(
            cache["unit_rate_logp_changes"][..., idx],
            cache["rewards"][:, idx],
            w0,
            rates,
            pc[..., idx],
        )
        pred = np.concatenate((direct[:, :, :, None], old[:, 1], old[:, 0]), axis=3)
        same = pred[..., 8:, :]
        transfer = np.broadcast_to(
            np.einsum("tbrmqk,tq->tbrmk", pred[..., :8, :], cq, optimize=True)[..., None, :],
            same.shape,
        )
        preds = np.stack((same, transfer), axis=1)
        error = preds - target[:, None, :, :, None]
        errors.append(np.einsum("tobrmqk,tb,tq->tormk", error**2, bw, eq, optimize=True))
        means.append(np.einsum("tobrmqk,tb,tq->tormk", preds, bw, eq, optimize=True))
        print(f"bootstrap {bank}: {repetitions}", flush=True)
    return np.stack(errors, axis=1), actual, np.stack(means, axis=1)


def summarize(root, repetitions):
    spec, old, logp, base, records = load(root)
    _, data, cache, observed, baseline, proposals, base_record, outcomes = old
    seed, rates = spec["training_seed"], spec["rates"]
    actual = (observed - baseline).mean(-2)
    pred = np.stack(
        [
            np.concatenate(
                (
                    np.array([[rec[f"{bank}_change"] for rec in row] for row in records])[
                        :, :, None
                    ],
                    cache[f"{bank}_snis"],
                    cache[f"{bank}_linear"],
                ),
                axis=2,
            )
            for bank in BANKS
        ]
    )
    boot, ba, bp = bootstrap(cache, logp, observed, baseline, rates, repetitions)
    metrics, batches, comparisons, scoring, overlap, drift, examples = [], [], [], [], [], [], []
    for ni, oi, ri, mi, ki in itertools.product(range(5), range(2), range(3), range(13), range(4)):
        bank = list(BANKS)[ni]
        target = actual[:, ri, 8:, ki]
        p = (
            pred[ni, :, ri, mi, 8:, ki]
            if oi == 0
            else np.broadcast_to(pred[ni, :, ri, mi, :8, ki].mean(-1, keepdims=True), target.shape)
        )
        head = dict(
            seed=seed,
            bank=bank,
            mode=("same_probe", "calibration_mean_transfer")[oi],
            rate=rates[ri],
            method=METHOD_NAMES[mi],
            endpoint=ENDPOINTS[ki],
        )
        for style in ("all", *STYLES):
            idx = np.arange(16) if style == "all" else np.arange(STYLES.index(style), 16, 4)
            row = dict(**head, style=style, **stats(p[idx], target[idx]))
            if style == "all":
                mse, denominator = boot[:, ni, oi, ri, mi, ki], ba[:, 1, ri, ki]
                n = np.divide(
                    mse, denominator, out=np.full_like(mse, np.nan), where=denominator > 0
                )
                row.update(
                    rmse_ci95=interval(np.sqrt(mse)),
                    nrmse_ci95=interval(np.sqrt(n)),
                    nrmse_defined_fraction=float(np.mean(denominator > 0)),
                    actual_mean_ci95=interval(ba[:, 0, ri, ki]),
                    predicted_mean_ci95=interval(bp[:, ni, oi, ri, mi, ki]),
                )
                if mi == 0:
                    for ci in range(1, 13):
                        cp = (
                            pred[ni, :, ri, ci, 8:, ki]
                            if oi == 0
                            else np.broadcast_to(
                                pred[ni, :, ri, ci, :8, ki].mean(-1, keepdims=True), target.shape
                            )
                        )
                        comparisons.append(
                            dict(
                                **head,
                                comparator=METHOD_NAMES[ci],
                                mse_difference=float(
                                    np.mean((p - target) ** 2 - (cp - target) ** 2)
                                ),
                                ci95=interval(mse - boot[:, ni, oi, ri, ci, ki]),
                            )
                        )
            metrics.append(row)
        for b in range(16):
            batches.append(
                dict(
                    **head,
                    batch=b,
                    style=data["batches"][b]["style"],
                    **stats(p[b : b + 1], target[b : b + 1]),
                )
            )
    # Baseline rescoring is diagnostic only; all importance densities remain frozen.
    for qi, probe in enumerate(data["probes"]):
        lengths = np.array(
            [len(r["token_ids"]) for r in proposals["rows"][512 * qi : 512 * (qi + 1)]]
        )
        for temp, key, cached in (
            (0.8, "logp", "original_cache_logp"),
            (1.2, "warm_logp", "original_warm_logp"),
        ):
            diff = base[key][qi] - base[cached][qi]
            drift.append(
                dict(
                    seed=seed,
                    question=probe["id"],
                    split=probe["split"],
                    temperature=temp,
                    max_abs_logp_drift=float(np.abs(diff).max()),
                    max_abs_per_token_drift=float((np.abs(diff) / lengths).max()),
                    rms_logp_drift=float(np.sqrt(np.mean(diff**2))),
                )
            )
    ratio = logp - base["logp"]
    for bank, (idx, _) in BANKS.items():
        w0 = cache[f"{bank}_log_base_weights"]
        exact = ratio[..., idx]
        for ri, mi, split, style in itertools.product(
            range(3), range(7), ("calibration", "evaluation"), ("all", *STYLES)
        ):
            qs = slice(0, 8) if split == "calibration" else slice(8, 16)
            bs = np.arange(16) if style == "all" else np.arange(STYLES.index(style), 16, 4)
            er = exact[bs, ri, qs]
            lr = (
                er
                if mi == 0
                else rates[ri] * cache["unit_rate_logp_changes"][bs, mi - 1, qs][..., idx]
            )
            lw = (
                (logp - cache["baseline_logp"])[bs, ri, qs][..., idx] + w0[qs]
                if mi == 0
                else lr + w0[qs]
            )
            w = normalized(lw)
            ess = 1 / (w * w).sum(-1)
            lm = np.logaddexp.reduce(lw, axis=-1) - np.log(len(idx))
            difference = lr - er
            cd = difference - difference.mean(-1, keepdims=True)
            scoring.append(
                dict(
                    seed=seed,
                    bank=bank,
                    rate=rates[ri],
                    method=("direct", *METHODS)[mi],
                    split=split,
                    style=style,
                    samples=len(idx),
                    raw_ratio_rmse_vs_direct=float(np.sqrt(np.mean(difference**2))),
                    centered_ratio_rmse_vs_direct=float(np.sqrt(np.mean(cd**2))),
                    direct_ratio_rms=float(np.sqrt(np.mean(er**2))),
                    ess_min=float(ess.min()),
                    ess_p10=float(np.quantile(ess, 0.1)),
                    ess_median=float(np.median(ess)),
                    fraction_below_quarter=float(np.mean(ess < len(idx) / 4)),
                    log_mean_weight_min=float(lm.min()),
                    log_mean_weight_p10=float(np.quantile(lm, 0.1)),
                    log_mean_weight_median=float(np.median(lm)),
                    log_mean_weight_max=float(lm.max()),
                    max_weight_median=float(np.median(w.max(-1))),
                    max_weight_max=float(w.max()),
                )
            )
        for b, ri, qi in itertools.product(range(16), range(3), range(16)):
            r = records[b][ri]
            overlap.append(
                dict(
                    seed=seed,
                    bank=bank,
                    batch=b,
                    style=data["batches"][b]["style"],
                    rate=rates[ri],
                    question=data["probes"][qi]["id"],
                    split=data["probes"][qi]["split"],
                    ess=float(r[f"{bank}_ess"][qi]),
                    log_mean_weight=float(r[f"{bank}_log_mean_weight"][qi]),
                    max_weight=float(r[f"{bank}_max_weight"][qi]),
                )
            )
    for b, qi, ri in itertools.product(range(4), (0, 8), range(3)):
        examples.append(
            dict(
                seed=seed,
                batch=b,
                style=data["batches"][b]["style"],
                rate=rates[ri],
                prompt=data["probes"][qi]["prompt"],
                split=data["probes"][qi]["split"],
                baseline=base_record["rows"][qi * 128],
                updated=outcomes[b][ri]["rows"][qi * 128],
            )
        )
    costs = dict(
        seed=seed,
        **read(root / "integrity.json"),
        no_new_generations=True,
        scoring_forward_calls=50 * read(root / "manifest.json")["unique_sequences_per_condition"],
    )
    return dict(
        metrics=metrics,
        batch_metrics=batches,
        comparisons=comparisons,
        scoring=scoring,
        overlap=overlap,
        baseline_drift=drift,
        examples=examples,
        costs=[costs],
    )


def plots(out, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    for endpoint in ENDPOINTS:
        fig, axes = plt.subplots(2, 5, figsize=(17, 7))
        for si, seed in enumerate((42, 43)):
            for ni, bank in enumerate(BANKS):
                ax = axes[si, ni]
                for method, color, label in [
                    ("direct_snis", "#b64b43", "Direct (post-update)"),
                    ("full_gradient_snis", "#285fad", "Taylor SNIS"),
                    ("full_gradient_linear", "#21846f", "Linear"),
                    ("zero_snis", "#888888", "No change"),
                ]:
                    rows = [
                        r
                        for r in summary["metrics"]
                        if (r["seed"], r["bank"], r["method"], r["endpoint"], r["mode"], r["style"])
                        == (seed, bank, method, endpoint, "same_probe", "all")
                    ]
                    ax.plot(
                        [r["rate"] for r in rows],
                        [r["rmse"] * (1 if endpoint == "tokens" else 100) for r in rows],
                        "o-",
                        label=label,
                        color=color,
                        linewidth=1.4,
                    )
                ax.set_xscale("log")
                ax.grid(alpha=0.15)
                ax.set_title(f"Seed {seed} · {bank}")
                if ni == 0:
                    ax.set_ylabel(
                        "RMSE (tokens)" if endpoint == "tokens" else "RMSE (percentage points)"
                    )
                if si == 1:
                    ax.set_xlabel("SGD learning rate")
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False
        )
        fig.suptitle(
            f"{endpoint.replace('_', ' ').capitalize()}: does actual likelihood fix the estimate?\nAll sample banks and rates; direct likelihood uses post-update model access"
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.9))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"direct_{endpoint}.{ext}", dpi=160)
        plt.close(fig)


def report(out, s):
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Float32 likelihood diagnostic</title><style>body{max-width:1300px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#202428}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}pre{white-space:pre-wrap;background:#f3f5f7;padding:14px}details{margin:16px 0}a{color:#176b9a}</style>',
        "<h1>Can actual likelihoods recover changed behavior?</h1><p>Post-update diagnostic on two Gemma-3 4B LoRA reconstructions. All 96 updates and five frozen proposal banks. No new responses were generated.</p>",
    ]
    finding = out / "findings.html"
    if finding.exists():
        parts.append(finding.read_text())
    parts.extend(
        [
            "<h2>What this tests</h2><p>We directly score the same saved ordinary and warm responses under every actual updated adapter. Importance weights use the original frozen proposal probabilities. We compare their weighted changes with all six prospective gradient representations, both estimators and the same independent generated responses. Direct scores are an oracle error diagnosis: obtaining them requires the updated model.</p>",
            "<p>All 136 rank-8 query/value LoRA tensors across 34 language layers are updated and restored. This matches the Gradient Atoms architecture, but the unavailable original rank-8 checkpoint prevents an exact checkpoint replication. Float32 SDPA, disabled TF32 and the same recomputed training gradients are used throughout this comparison.</p>",
            "<h2>Copying accuracy across the complete grid</h2><p>Normalized RMSE: one is the error of predicting no change. Lower is better. Intervals are pointwise descriptive bootstrap intervals.</p><table><tr><th>Seed</th><th>Rate</th><th>Bank</th><th>Direct</th><th>95% interval</th><th>Taylor SNIS</th><th>Linear</th><th>Direct transfer</th></tr>",
        ]
    )
    for seed, rate, bank in itertools.product((42, 43), (0.0003, 0.003, 0.03), BANKS):

        def get(method, mode="same_probe"):
            return next(
                r
                for r in s["metrics"]
                if (
                    r["seed"],
                    r["rate"],
                    r["bank"],
                    r["method"],
                    r["mode"],
                    r["style"],
                    r["endpoint"],
                )
                == (seed, rate, bank, method, mode, "all", "exact_content_and_order")
            )

        r = get("direct_snis")
        lo, hi = r["nrmse_ci95"]
        parts.append(
            f"<tr><td>{seed}</td><td>{rate:g}</td><td>{bank}</td><td>{r['nrmse']:.3f}</td><td>{lo:.3f}–{hi:.3f}</td><td>{get('full_gradient_snis')['nrmse']:.3f}</td><td>{get('full_gradient_linear')['nrmse']:.3f}</td><td>{get('direct_snis', 'calibration_mean_transfer')['nrmse']:.3f}</td></tr>"
        )
    parts.append("</table><h2>Every endpoint and sample bank</h2>")
    for e in ENDPOINTS:
        parts.append(f'<img src="direct_{e}.png">')
    parts.append(
        "<h2>Uncertainty and limits</h2><p>1,000 paired bootstrap replicates independently resample proposal and evaluation trajectories. Proposal counts are preserved within each temperature and nested fixed subset; all banks share the same resampled trajectories. Baseline and updated evaluations share resampling indices, with crossed questions and batches sampled within style. The subset partition is fixed. Intervals are descriptive and pointwise, with no adjustment for selecting comparisons. Two reused adapters, hardware and seed confounding, synthetic tasks and eight evaluation questions do not establish general reliability. A bootstrap cannot represent events missing from the original samples.</p><p>All styles, batches, error tails and low-overlap cases are retained. A high ESS alone does not establish coverage. Weighted sample probabilities are estimates, and exact log likelihoods do not remove finite-sample error. Ordinary and warm baseline rescoring drift is reported separately; newly rescored probabilities never replace the frozen proposal q.</p><h2>Fixed examples from the completed mixture experiment</h2><p>Batches 0–3, questions 0 and 8, sample 0, all three rates. These are the previously generated evaluation responses, not new diagnostic generations. Exact tag content/order, format, length, truncation, emptiness and repetition checks remain available in the original report and linked tables.</p>"
    )
    for ex in s["examples"]:
        parts.append(
            f"<details><summary>Seed {ex['seed']}, {ex['style']}, {ex['split']}, rate {ex['rate']:g}</summary><p>{html.escape(ex['prompt'])}</p><b>Before</b><pre>{html.escape(ex['baseline']['response'])}</pre><b>After</b><pre>{html.escape(ex['updated']['response'])}</pre><p>{html.escape(json.dumps(ex['updated']['checks']))}</p></details>"
        )
    parts.append(
        "<h2>Integrity and cost</h2><p>Original-host audits verify all source/input/result hashes, all 49 conditions and 48 updates per seed, all 8,192 scores per condition, float32 training-gradient caches, adapter restoration and original deadlines. Original source archives are retained. Duplicate token sequences share computation only within question; all multiplicities are restored.</p>"
    )
    for c in s["costs"]:
        parts.append(
            f"<p>Seed {c['seed']}: {c['elapsed_seconds'] / 60:.1f} minutes, {c['scoring_forward_calls']:,} single-sequence scoring forwards (two baseline temperatures plus 48 updates). No new generated answers and no backward calls in this diagnostic.</p>"
        )
    parts.append(
        "<p>"
        + " · ".join(
            f'<a href="{k}.csv">{k.replace("_", " ")}</a>'
            for k in (
                "metrics",
                "comparisons",
                "batch_metrics",
                "scoring",
                "overlap",
                "baseline_drift",
                "coverage",
                "generated",
            )
        )
        + ' · <a href="summary.json">Full results</a> · <a href="analysis-plan.json">Pre-reading plan</a> · <a href="../paper-mixture/report.html">Original prospective experiment</a></p>'
    )
    parts.append(
        '<p><a href="https://arxiv.org/abs/2603.14665">Gradient Atoms</a> · <a href="https://arxiv.org/abs/1511.03095">Generalized multiple importance sampling</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/e031b89dc2b24f11ed880c36cc44d74103c809b7/src/gradient_interp/paper_mixture_ratio.py">Frozen diagnostic code</a>. No novelty claim for standard importance sampling or gradient dot products.</p>'
    )
    (out / "report.html").write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", type=Path, nargs=2, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--analysis-plan", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=1000)
    a = p.parse_args()
    if any(read(r / "controller.json")["status"] != "complete" for r in a.roots):
        raise ValueError("Both runs must finish before reading outcomes")
    plan = read(a.analysis_plan)
    a.out.mkdir(parents=True, exist_ok=True)
    s = {"scope": "post-update oracle diagnostic, development only", "analysis_plan": plan}
    for root in a.roots:
        for k, v in summarize(root, a.repetitions).items():
            s.setdefault(k, []).extend(v)
        dest = a.out / root.name
        dest.mkdir(exist_ok=True)
        for n in (
            "controller.json",
            "integrity.json",
            "manifest.json",
            "plan.json",
            "specification.json",
            "preflight.json",
        ):
            shutil.copy2(root / n, dest / n)
    original = read(a.out.parent / "paper-mixture/summary.json")
    for k in ("coverage", "generated"):
        s[k] = original[k]
    atomic_json(a.out / "summary.json", s)
    for k in (
        "metrics",
        "batch_metrics",
        "comparisons",
        "scoring",
        "overlap",
        "baseline_drift",
        "coverage",
        "generated",
        "costs",
    ):
        write_csv(a.out / f"{k}.csv", s[k])
    shutil.copy2(a.analysis_plan, a.out / "analysis-plan.json")
    plots(a.out, s)
    report(a.out, s)
    print(a.out / "report.html", flush=True)


if __name__ == "__main__":
    main()
