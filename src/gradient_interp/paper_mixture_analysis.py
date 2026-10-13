"""Guarded, descriptive analysis of frozen temperature-mixture forecasts."""

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
from .paper_mixture import BANKS, normalized, weighted_estimators
from .paper_sample import ENDPOINTS
from .paper_sample_analysis import EXTRA, interval, stats, values
from .study import atomic_json, file_hash


def read(p):
    return json.loads(p.read_text())


def load(root):
    state, audit, manifest, spec, data, lock = [
        read(root / n)
        for n in (
            "controller.json",
            "integrity.json",
            "manifest.json",
            "specification.json",
            "data.json",
            "forecast-lock.json",
        )
    ]
    if (
        state["status"] != "complete"
        or not audit["complete"]
        or not audit["source_input_output_hashes_verified"]
    ):
        raise ValueError("Completed run and original-host integrity audit required")
    for n in ("plan", "manifest", "specification"):
        assert file_hash(root / f"{n}.json") == audit[f"{n}_sha256"]
    assert file_hash(root / "forecast-lock.json") == audit["forecast_lock_sha256"]
    for n, expected in manifest["files_sha256"].items():
        assert file_hash(root / n) == expected, n
    assert manifest["completed_unix"] < state["deadline_unix"] and manifest["adapter_restored"]
    proposals = read(root / "proposals.json")
    cache = dict(np.load(root / "forecasts.npz"))
    rewards = values(proposals["rows"], data["probes"], 512, spec["max_new_tokens"])[..., :4]
    np.testing.assert_array_equal(rewards, cache["rewards"])
    for bank, (indices, proposal) in BANKS.items():
        recalculated = weighted_estimators(
            cache["unit_rate_logp_changes"],
            rewards,
            cache["baseline_logp"],
            cache["warm_logp"],
            spec["rates"],
            indices,
            proposal,
        )
        for name, value in recalculated.items():
            np.testing.assert_allclose(cache[f"{bank}_{name}"], value, rtol=1e-10, atol=1e-10)
    base_record = read(root / "outcomes/baseline.json")
    baseline = values(base_record["rows"], data["probes"], 128, spec["max_new_tokens"])
    arrays, outcomes = [], []
    for b in range(16):
        aa, oo = [], []
        for rate in spec["rates"]:
            rec = read(root / f"outcomes/batch-{b:02d}-rate-{rate:g}.json")
            assert (
                lock["locked_unix"]
                < rec["update_started_unix"]
                < rec["completed_unix"]
                < manifest["completed_unix"]
            )
            assert rec["adapter_restored"] and rec["batch_index"] == b and rec["rate"] == rate
            assert (
                rec["plan_sha256"] == audit["plan_sha256"]
                and rec["forecast_lock_sha256"] == audit["forecast_lock_sha256"]
            )
            aa.append(values(rec["rows"], data["probes"], 128, spec["max_new_tokens"]))
            oo.append(rec)
        arrays.append(aa)
        outcomes.append(oo)
    return spec, data, cache, np.asarray(arrays), baseline, proposals, base_record, outcomes


def nested_counts(rng, repetitions, questions=16):
    """Condition on fixed nested subsets, preserving each bank's component counts."""
    out = np.zeros((repetitions, questions, 512))
    for start, stop in ((0, 64), (64, 128), (128, 256), (256, 320), (320, 384), (384, 512)):
        n = stop - start
        out[..., start:stop] = rng.multinomial(n, np.full(n, 1 / n), size=(repetitions, questions))
    return out


def resampled_bank(unit, rewards, log_base_weights, rates, counts):
    """Vectorized paired bootstrap, with a stable fallback for omitted dominant modes.

    Returns [replicate, estimator, batch, rate, method, question, endpoint].
    Counts refer to complete trajectories and are shared across every update/method.
    """
    t, q, s = counts.shape
    b, m = unit.shape[:2]
    rates = np.asarray(rates)
    result = np.empty((t, 2, b, len(rates), m, q, rewards.shape[-1]))
    for qi in range(q):
        c = counts[:, qi]
        w0 = normalized(log_base_weights[qi])
        baseline_weights = c * w0
        baseline_weights /= baseline_weights.sum(-1, keepdims=True)
        y = rewards[qi]
        base = baseline_weights @ y
        u = unit[:, :, qi].reshape(b * m, s)
        mean_u = baseline_weights @ u.T
        cross = np.stack([baseline_weights @ (u * y[:, k]).T for k in range(y.shape[-1])], axis=-1)
        deriv = (cross - mean_u[..., None] * base[:, None]).reshape(t, b, m, -1)
        result[:, 0, :, :, :, qi] = deriv[:, :, None] * rates[None, None, :, None, None]
        lw = (log_base_weights[qi] + unit[:, None, :, qi] * rates[None, :, None, None]).reshape(
            b * len(rates) * m, s
        )
        w = normalized(lw)
        denom = c @ w.T
        numerator = np.stack([c @ (w * y[:, k]).T for k in range(y.shape[-1])], axis=-1)
        safe = denom > 1e-250
        updated = np.divide(
            numerator, denom[..., None], out=np.zeros_like(numerator), where=safe[..., None]
        )
        for ti, di in zip(*np.where(~safe), strict=True):
            logc = np.full(s, -np.inf)
            np.log(c[ti], out=logc, where=c[ti] > 0)
            updated[ti, di] = normalized(lw[di] + logc) @ y
        result[:, 1, :, :, :, qi] = (updated - base[:, None]).reshape(t, b, len(rates), m, -1)
    return result


def bootstrap(cache, observed, baseline, rates, repetitions=1000):
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
    ba = np.stack(
        [np.einsum("tbrqk,tb,tq->trk", v, bw, eq, optimize=True) for v in (target, target**2)],
        axis=1,
    )
    errors, means = [], []
    for bank, (idx, _) in BANKS.items():
        pred = resampled_bank(
            cache["unit_rate_logp_changes"][..., idx],
            cache["rewards"][:, idx],
            cache[f"{bank}_log_base_weights"],
            rates,
            pc[..., idx],
        )
        same = pred[..., 8:, :]
        transfer = np.broadcast_to(
            np.einsum("tebrmqk,tq->tebrmk", pred[..., :8, :], cq, optimize=True)[..., None, :],
            same.shape,
        )
        preds = np.stack((same, transfer), axis=2)
        err = preds - target[:, None, None, :, :, None]
        errors.append(np.einsum("teobrmqk,tb,tq->teormk", err**2, bw, eq, optimize=True))
        means.append(np.einsum("teobrmqk,tb,tq->teormk", preds, bw, eq, optimize=True))
        print(f"bootstrap {bank}: {repetitions} replicates", flush=True)
    return np.stack(errors, axis=1), ba, np.stack(means, axis=1)


def summarize(root, reps):
    spec, data, cache, observed, baseline, proposals, base_record, outcomes = load(root)
    seed, rates = spec["training_seed"], spec["rates"]
    actual = (observed - baseline).mean(-2)
    predictions = np.stack(
        [np.stack([cache[f"{bank}_{est}"] for est in ("linear", "snis")]) for bank in BANKS]
    )
    boot, ba, bp = bootstrap(cache, observed, baseline, rates, reps)
    rows, batches, comparisons, coverage, overlap, generated, examples = [], [], [], [], [], [], []
    for ni, ei, oi, ri, mi, ki in itertools.product(
        range(5), range(2), range(2), range(3), range(6), range(4)
    ):
        bank = list(BANKS)[ni]
        target = actual[:, ri, 8:, ki]
        pred = (
            predictions[ni, ei, :, ri, mi, 8:, ki]
            if oi == 0
            else np.broadcast_to(
                predictions[ni, ei, :, ri, mi, :8, ki].mean(-1, keepdims=True), target.shape
            )
        )
        header = dict(
            seed=seed,
            bank=bank,
            estimator=("linear", "snis")[ei],
            mode=("same_probe", "calibration_mean_transfer")[oi],
            rate=rates[ri],
            method=METHODS[mi],
            endpoint=ENDPOINTS[ki],
        )
        for style in ("all", *STYLES):
            ids = np.arange(16) if style == "all" else np.arange(STYLES.index(style), 16, 4)
            row = {**header, "style": style, **stats(pred[ids], target[ids])}
            if style == "all":
                mse, denom = boot[:, ni, ei, oi, ri, mi, ki], ba[:, 1, ri, ki]
                ratio = np.divide(mse, denom, out=np.full_like(mse, np.nan), where=denom > 0)
                row.update(
                    rmse_ci95=interval(np.sqrt(mse)),
                    nrmse_ci95=interval(np.sqrt(ratio)),
                    nrmse_defined_fraction=float(np.mean(denom > 0)),
                    actual_mean_ci95=interval(ba[:, 0, ri, ki]),
                    predicted_mean_ci95=interval(bp[:, ni, ei, oi, ri, mi, ki]),
                )
                diff = observed[:, ri, 8:, :, ki] - baseline[8:, :, ki]
                row["evaluation_mean_noise_rms"] = float(
                    np.sqrt(np.mean(diff.var(-1, ddof=1) / 128))
                )
                if ki in (0, 1, 3):
                    base = (
                        cache[f"{bank}_baseline"][8:, ki]
                        if oi == 0
                        else cache[f"{bank}_baseline"][:8, ki].mean()
                    )
                    row["outside_probability_range_fraction"] = float(
                        np.mean(((base + pred) < 0) | ((base + pred) > 1))
                    )
                controls = [(ni, METHODS.index(m)) for m in ("zero", "svd16")]
                if bank in ("mixture256", "base256", "mixture512"):
                    controls.append(
                        (
                            list(BANKS).index(
                                {
                                    "mixture256": "base256",
                                    "base256": "base64",
                                    "mixture512": "mixture256",
                                }[bank]
                            ),
                            mi,
                        )
                    )
                for ci, cm in controls:
                    cp = (
                        predictions[ci, ei, :, ri, cm, 8:, ki]
                        if oi == 0
                        else np.broadcast_to(
                            predictions[ci, ei, :, ri, cm, :8, ki].mean(-1, keepdims=True),
                            target.shape,
                        )
                    )
                    comparisons.append(
                        {
                            **header,
                            "comparator_bank": list(BANKS)[ci],
                            "comparator_method": METHODS[cm],
                            "mse_difference": float(
                                np.mean((pred - target) ** 2 - (cp - target) ** 2)
                            ),
                            "ci95": interval(mse - boot[:, ci, ei, oi, ri, cm, ki]),
                        }
                    )
            rows.append(row)
        for b in range(16):
            batches.append(
                {
                    **header,
                    "batch": b,
                    "style": data["batches"][b]["style"],
                    **stats(pred[b : b + 1], target[b : b + 1]),
                }
            )
    for bank, (idx, _) in BANKS.items():
        w0 = normalized(cache[f"{bank}_log_base_weights"])
        for qi, probe in enumerate(data["probes"]):
            for ki, endpoint in enumerate(ENDPOINTS):
                v = cache["rewards"][qi, idx, ki]
                coverage.append(
                    dict(
                        seed=seed,
                        bank=bank,
                        question=probe["id"],
                        split=probe["split"],
                        endpoint=endpoint,
                        samples=len(idx),
                        positive_samples=int((v > 0).sum()),
                        raw_mean=float(v.mean()),
                        weighted_mean=float(cache[f"{bank}_baseline"][qi, ki]),
                        independent_baseline_mean=float(baseline[qi, :, ki].mean()),
                        any_post_update_positive=bool(np.any(observed[:, :, qi, :, ki] > 0)),
                        baseline_ess=float(cache[f"{bank}_baseline_ess"][qi]),
                        max_baseline_weight=float(w0[qi].max()),
                        log_mean_baseline_weight=float(
                            np.logaddexp.reduce(cache[f"{bank}_log_base_weights"][qi])
                            - np.log(len(idx))
                        ),
                    )
                )
        for ri, mi, split, style in itertools.product(
            range(3), range(6), ("calibration", "evaluation"), ("all", *STYLES)
        ):
            qs = slice(0, 8) if split == "calibration" else slice(8, 16)
            bs = np.arange(16) if style == "all" else np.arange(STYLES.index(style), 16, 4)
            ess = cache[f"{bank}_ess"][bs, ri, mi, qs]
            lw = (
                cache[f"{bank}_log_base_weights"][qs]
                + rates[ri] * cache["unit_rate_logp_changes"][bs, mi, qs][..., idx]
            )
            maxw = normalized(lw).max(-1)
            overlap.append(
                dict(
                    seed=seed,
                    bank=bank,
                    rate=rates[ri],
                    method=METHODS[mi],
                    split=split,
                    style=style,
                    samples=len(idx),
                    ess_min=float(ess.min()),
                    ess_median=float(np.median(ess)),
                    fraction_below_quarter=float(np.mean(ess < len(idx) / 4)),
                    max_weight_median=float(np.median(maxw)),
                    max_weight_max=float(maxw.max()),
                )
            )
    for ri, split, style, ki in itertools.product(
        range(3), ("calibration", "evaluation"), ("all", *STYLES), range(8)
    ):
        qs = slice(0, 8) if split == "calibration" else slice(8, 16)
        bs = np.arange(16) if style == "all" else np.arange(STYLES.index(style), 16, 4)
        generated.append(
            dict(
                seed=seed,
                rate=rates[ri],
                split=split,
                style=style,
                endpoint=(*ENDPOINTS, *EXTRA)[ki],
                baseline=float(baseline[qs, :, ki].mean()),
                after=float(observed[bs, ri, qs, :, ki].mean()),
                change=float(actual[bs, ri, qs, ki].mean()),
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
    unique_per_question = [
        len({tuple(r["token_ids"]) for r in proposals["rows"][q * 512 : (q + 1) * 512]})
        for q in range(16)
    ]
    assert sum(unique_per_question) == proposals["unique_gradient_forwards"]
    costs = {
        "seed": seed,
        **read(root / "integrity.json"),
        "unique_proposal_forward_backward_calls": sum(unique_per_question),
        "training_gradient_forward_backward_calls": 64,
        "proposal_generation_chunks": 128,
        "evaluation_generation_chunks": 3136,
        "proposal_generated_tokens": sum(len(r["token_ids"]) for r in proposals["rows"]),
        "evaluation_generated_tokens": sum(len(r["token_ids"]) for r in base_record["rows"])
        + sum(len(r["token_ids"]) for br in outcomes for rec in br for r in rec["rows"]),
        "cost_note": "Each unique trajectory has one model forward and one backward, two temperature loss reductions, and contractions against all six banks. Token-generation calls also require prefill/decode forwards, not counted as gradient forwards. Runtime is joint, not per method.",
    }
    return dict(
        metrics=rows,
        comparisons=comparisons,
        batch_metrics=batches,
        coverage=coverage,
        overlap=overlap,
        generated=generated,
        examples=examples,
        costs=[costs],
    )


def plots(out, s):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#8a8a8a", "#235ba8", "#c58424", "#11856c", "#a54c7c")
    for endpoint in ENDPOINTS:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        for si, seed in enumerate((42, 43)):
            for col, (est, mode) in enumerate(
                itertools.product(("linear", "snis"), ("same_probe", "calibration_mean_transfer"))
            ):
                ax = axes[si, col]
                for bank, color in zip(BANKS, colors, strict=True):
                    rr = [
                        r
                        for r in s["metrics"]
                        if (
                            r["seed"],
                            r["bank"],
                            r["estimator"],
                            r["mode"],
                            r["method"],
                            r["endpoint"],
                            r["style"],
                        )
                        == (seed, bank, est, mode, "full_gradient", endpoint, "all")
                    ]
                    ax.plot(
                        [r["rate"] for r in rr],
                        [r["rmse"] * (1 if endpoint == "tokens" else 100) for r in rr],
                        marker="o",
                        label=bank,
                        color=color,
                    )
                zz = [
                    r
                    for r in s["metrics"]
                    if (
                        r["seed"],
                        r["bank"],
                        r["estimator"],
                        r["mode"],
                        r["method"],
                        r["endpoint"],
                        r["style"],
                    )
                    == (seed, "base256", est, mode, "zero", endpoint, "all")
                ]
                ax.plot(
                    [r["rate"] for r in zz],
                    [r["rmse"] * (1 if endpoint == "tokens" else 100) for r in zz],
                    ls=":",
                    color="black",
                    label="no change",
                )
                ax.set_xscale("log")
                ax.grid(alpha=0.15)
                ax.set_title(
                    f"Seed {seed}, {est}\n{'same questions' if mode == 'same_probe' else 'template transfer'}"
                )
                if col == 0:
                    ax.set_ylabel(
                        "RMSE in tokens" if endpoint == "tokens" else "RMSE in percentage points"
                    )
                if si == 1:
                    ax.set_xlabel("SGD learning rate")
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False
        )
        fig.suptitle(
            f"Full-gradient forecasts: {endpoint.replace('_', ' ')}\nAll proposal banks and fixed rates; lower error is better"
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.90))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"banks_{endpoint}.{ext}", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    methods = ("full_gradient", "ekfac_projection", "dictionary", "svd16", "mean_direction")
    for si, seed in enumerate((42, 43)):
        for ri, rate in enumerate((0.0003, 0.003, 0.03)):
            ax = axes[si, ri]
            for mi, method in enumerate(methods):
                for bank, offset, color in (
                    ("base256", -0.12, colors[1]),
                    ("mixture256", 0.12, colors[3]),
                ):
                    r = next(
                        r
                        for r in s["metrics"]
                        if (
                            r["seed"],
                            r["bank"],
                            r["estimator"],
                            r["mode"],
                            r["method"],
                            r["endpoint"],
                            r["style"],
                            r["rate"],
                        )
                        == (
                            seed,
                            bank,
                            "linear",
                            "same_probe",
                            method,
                            "exact_content_and_order",
                            "all",
                            rate,
                        )
                    )
                    lo, hi = r["rmse_ci95"]
                    y = mi + offset
                    ax.plot([lo * 100, hi * 100], [y, y], color=color, lw=1.2)
                    ax.scatter(
                        [r["rmse"] * 100], [y], color=color, s=20, label=bank if mi == 0 else None
                    )
            ax.set_yticks(range(5), methods if ri == 0 else [""] * 5)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.15)
            ax.set_title(f"Seed {seed}, rate {rate:g}")
            ax.set_xlabel("Accuracy RMSE, percentage points")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Equal-count comparison: 256 proposals, linear estimator\nPointwise 95% intervals include proposal and evaluation sampling uncertainty"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"equal_count_uncertainty.{ext}", dpi=160)
    plt.close(fig)


def findings(s):
    return """<h2>What we learned</h2>
<p><b>Broader samples did not consistently improve forecasts.</b> At equal count, mixture256 versus base256 copying-error differences have pointwise intervals spanning zero at every fixed rate for both estimators and seeds. This does not establish equality, but it gives no reliable advantage for the warm mixture in this small development study.</p>
<p>At rate 0.003, ordinary base256 samples with full-gradient SNIS give normalized copying errors of 0.416 and 0.257. Absolute RMSE is 1.12 and 2.11 percentage points; worst-batch RMSE is 2.14 and 4.23 points. The pointwise normalized-error intervals are 0.321 to 1.597 and 0.157 to 0.858. Only one seed excludes the no-change reference, and these are reused development tasks with many comparisons. This is not a confirmatory positive result.</p>
<p>The dictionary remains worse than SVD16 on that comparison: normalized errors 0.623/0.548 versus 0.524/0.447. Transferring full-gradient predictions across prompt templates raises error to 0.948/0.910 and worst-batch RMSE to 4.70/14.87 percentage points. Question-specific forecasting is much stronger than template transfer.</p>
<p>Warmer proposals found a few numbered and truncated responses that all ordinary proposal banks missed. But coverage is still sparse: even mixture512 has numbered examples on only three of eight evaluation questions for seed 42 and two of eight for seed 43. Truncated warm responses have very small corrected weights. At the large step, all banks' full-gradient SNIS truncation errors round to the no-change error. Mixed sampling did not solve this failure.</p>
<p>Large-step weights concentrate sharply. For mixture256, 61.7% and 75.0% of evaluation batch-question pairs have ESS below 64 of 256. A single trajectory can carry almost all the weight. Its large-step copying nRMSE is 1.153/1.262, compared with base256's 1.036/0.741. More diverse samples are not automatically more useful after weighting.</p>
<p>Actual generations reveal serious large-step failures. For plain-short batches, seed 42 copying accuracy falls from 94.6% to 46.7%, with 72.1% hitting the length cap. Seed 43 rises from 76.6% to 87.7%, but 20.4% hit the cap. The fixed first evaluation sample repeats tags for seed 42 and reorders them incorrectly for seed 43. At the middle rate those groups improve to 96.9%/88.9% without truncation. We retain both favorable and unfavorable cases.</p>
<p>The next bounded diagnostic will directly score this same broader proposal bank under the actual updated float32 adapters. It uses post-update access, so it cannot count as forecasting. It will test whether inaccurate likelihood approximations or inadequate weighted coverage explain the remaining failures before investing in another predictive method.</p>"""


def report(out, s):
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Broader proposals for behavior forecasts</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#202428}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}pre{white-space:pre-wrap;background:#f3f5f7;padding:14px}details{margin:18px 0}a{color:#176b9a}</style>',
        "<h1>Does broader sampling improve behavior forecasts?</h1><p>Development study. Two reconstructed Gemma-3 4B LoRA adapters, 96 updates, 200,704 evaluation responses and 16,384 proposal responses. All predictions were saved before the updates.</p>",
        findings(s),
        "<h2>Method</h2><ol><li>Use rank-8 query/value LoRA in all 34 language layers, matching the Gradient Atoms architecture. Reuse the 16 synthetic copying batches and two previously studied adapters.</li><li>Recompute batch gradients in float32, keeping the original dictionary and projection bases fixed.</li><li>Draw 256 ordinary and 256 warmer answers per question. Score each sequence at both temperatures and differentiate its ordinary-temperature log probability.</li><li>Forecast changes in numbering, exact copied content/order, length and truncation. Compare ordinary, warm and mixed sample banks, correcting for their sampling probabilities. Test both linear changes and normalized exponential reweighting.</li><li>Save the entire forecast grid, then apply one actual LoRA SGD update for each batch and fixed learning rate. Generate 128 independent responses per question and restore the adapter after every update.</li><li>Compare all forecasts on the same questions and after transferring average predictions across prompt templates.</li></ol>",
        "<p>These use established importance-sampling estimators. More responses or successful local Taylor approximations are not themselves a novel research result.</p>",
        '<h2>Equal proposal count</h2><p>base256 uses 256 ordinary-temperature samples. mixture256 uses 128 ordinary plus 128 warmer samples. Both total 256; mixture512 uses twice as many. The two-temperature scoring and importance weights add work, so equal sample counts are not exactly equal compute.</p><img src="equal_count_uncertainty.png">',
        "<h2>Copying accuracy, every fixed rate</h2><p>Normalized RMSE for full-gradient same-question predictions. One equals the error of predicting no change; lower is better. All representations and endpoints appear in the linked tables.</p><table><tr><th>Seed</th><th>Rate</th><th>Estimator</th>"
        + "".join(f"<th>{n}</th>" for n in BANKS)
        + "</tr>",
    ]
    for seed, rate, est in itertools.product((42, 43), (0.0003, 0.003, 0.03), ("linear", "snis")):
        parts.append(
            f"<tr><td>{seed}</td><td>{rate:g}</td><td>{est}</td>"
            + "".join(
                "<td>"
                + format(
                    next(
                        r
                        for r in s["metrics"]
                        if (
                            r["seed"],
                            r["rate"],
                            r["estimator"],
                            r["bank"],
                            r["method"],
                            r["mode"],
                            r["endpoint"],
                            r["style"],
                        )
                        == (
                            seed,
                            rate,
                            est,
                            bank,
                            "full_gradient",
                            "same_probe",
                            "exact_content_and_order",
                            "all",
                        )
                    )["nrmse"],
                    ".3f",
                )
                + "</td>"
                for bank in BANKS
            )
            + "</tr>"
        )
    parts.append("</table><h2>All endpoints and transfer</h2>")
    for endpoint in ENDPOINTS:
        parts.append(f'<img src="banks_{endpoint}.png">')
    parts.extend(
        [
            "<h2>Uncertainty and failure checks</h2><p>1,000 bootstrap replicates resample complete proposal trajectories within each temperature component and fixed nested sample subset, preserving shared samples across bank comparisons. Evaluation responses are resampled separately but paired across baseline and updates. Questions and style-stratified batches are also resampled. Intervals are pointwise and descriptive. The fixed-subset bootstrap conditions on that partition; it does not include variation from choosing different subset boundaries. Two reused seeds and eight evaluation questions do not support population-level confirmation.</p>",
            "<p>Measured effects contain Monte Carlo noise, which remains in the reported errors. Event counts, weighted baseline calibration, ESS below one-quarter of the bank size, maximum weight, probability-range violations and per-batch tails are reported without excluding bad cases. Reweighting still cannot create an event absent from its proposal bank; bootstrap intervals cannot repair absent support.</p>",
            "<h2>Fixed generated examples</h2><p>Batch indices 0 through 3, question 0 and 8, sample 0, every rate and both seeds. Selected before outcome inspection.</p>",
        ]
    )
    for ex in s["examples"]:
        parts.append(
            f"<details><summary>Seed {ex['seed']}, {ex['style']}, rate {ex['rate']:g}, {ex['split']}</summary><p>{html.escape(ex['prompt'])}</p><b>Before</b><pre>{html.escape(ex['baseline']['response'])}</pre><b>After</b><pre>{html.escape(ex['updated']['response'])}</pre><p>{html.escape(json.dumps(ex['updated']['checks']))}</p></details>"
        )
    parts.append(
        "<h2>Costs and provenance</h2><table><tr><th>Seed</th><th>Runtime minutes</th><th>Forecast minutes</th><th>Unique proposal gradient calls</th><th>Generated proposal tokens</th><th>Evaluation tokens</th></tr>"
    )
    for c in s["costs"]:
        parts.append(
            f"<tr><td>{c['seed']}</td><td>{c['successful_attempt_seconds'] / 60:.1f}</td><td>{c['forecast_stage_seconds'] / 60:.1f}</td><td>{c['unique_proposal_forward_backward_calls']}</td><td>{c['proposal_generated_tokens']}</td><td>{c['evaluation_generated_tokens']}</td></tr>"
        )
    parts.extend(
        [
            "</table><p>Original-host audits verified every input/source/forecast/result hash, all 48 updates per seed, forecast locks before every update, restored adapters and unchanged phase deadlines. Source archives remain on both hosts. Every unique proposal uses one model forward/backward and two temperature loss reductions. Generation requires additional token decoding forwards. The shared calculation includes all representations; no per-method runtime claim is made.</p>",
            "<p>A warm-sampling preflight failed the original numerical tolerance before this phase was frozen. Float32 SDPA with TF32 disabled passed that same tolerance. Original failed code/specification/logs and the explicit amendment remain archived. Checkpoint values were exactly upcast, and training gradients were recomputed. Older bfloat16 results are context, not a controlled sample-count comparison.</p>",
            "<p>These are architecture-matched reconstructions because the paper rank-8 checkpoint was unavailable. Data and adapters are reused synthetic development material, with hardware and seed confounded. There is no validation of atom semantics, general correctness, untouched-data confirmation or a new importance-sampling identity. SVD16 is a simple baseline, not an equal-storage claim.</p>",
            '<p><a href="https://arxiv.org/abs/2603.14665">Gradient Atoms</a> · <a href="https://github.com/jrosseruk/gradient_atoms/tree/a83858557b411249097861f25d57eb13020eb334">Pinned author code</a> · <a href="https://arxiv.org/abs/1511.03095">Generalized multiple importance sampling</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/a73d781445c5d0fb493d71e9816e967e8267473e/src/gradient_interp/paper_mixture.py">Frozen experiment code</a></p>',
            "<p>"
            + " · ".join(
                f'<a href="{n}.csv">{n.replace("_", " ")}</a>'
                for n in (
                    "metrics",
                    "comparisons",
                    "batch_metrics",
                    "coverage",
                    "overlap",
                    "generated",
                    "costs",
                )
            )
            + ' · <a href="analysis-plan.json">Analysis plan</a> · <a href="summary.json">Complete results</a></p>',
        ]
    )
    (out / "report.html").write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", type=Path, nargs=2, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--analysis-plan", type=Path, required=True)
    args = p.parse_args()
    if any(read(r / "controller.json")["status"] != "complete" for r in args.roots):
        raise ValueError("Both runs must be complete")
    plan = read(args.analysis_plan)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "synthetic_development_only", "analysis_plan": plan}
    for root in args.roots:
        result = summarize(root, 1000)
        for k, v in result.items():
            summary.setdefault(k, []).extend(v)
        dest = args.out / f"seed{read(root / 'specification.json')['training_seed']}"
        dest.mkdir(exist_ok=True)
        for n in (
            "controller.json",
            "integrity.json",
            "manifest.json",
            "plan.json",
            "specification.json",
            "forecast-lock.json",
            "preflight.json",
            "data.json",
        ):
            shutil.copy2(root / n, dest / n)
        shutil.copytree(
            root / "repairs",
            dest / "repairs",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.tar", "*.log", "*.py"),
        )
        atomic_json(args.out / "partial-summary.json", summary)
    shutil.copy2(args.analysis_plan, args.out / "analysis-plan.json")
    atomic_json(args.out / "summary.json", summary)
    (args.out / "partial-summary.json").unlink()
    for key in (
        "metrics",
        "comparisons",
        "batch_metrics",
        "coverage",
        "overlap",
        "generated",
        "costs",
    ):
        write_csv(args.out / f"{key}.csv", summary[key])
    plots(args.out, summary)
    report(args.out, summary)
    print(args.out / "report.html", flush=True)


if __name__ == "__main__":
    main()
