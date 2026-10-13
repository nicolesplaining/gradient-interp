"""Guarded descriptive analysis of completed prospective LoRA batch pilots."""

# ruff: noqa: E501

import argparse
import csv
import html
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from .paper_batch import METHODS, STYLES, generated_checks
from .paper_steering_analysis import quality
from .study import atomic_json, file_hash

LABELS = ("Full gradient", "EKFAC 6800", "Dictionary 500", "SVD 16", "Mean direction", "Zero")
CONTRAST = np.array([[-0.5, 0.5, -0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]])


def read(path):
    return json.loads(path.read_text())


def endpoint(scores, lengths, kind):
    if kind == "four_choice_probability":
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        return np.stack((p[..., 1] + p[..., 3], p[..., 2] + p[..., 3]), axis=-1)
    if kind == "mean_token_logp":
        scores = scores / lengths
    return scores @ CONTRAST.T


def bootstrap_weights(styles, probes, repetitions=2000):
    """Crossed resampling: batches within style, independently sampled probes."""
    rng = np.random.default_rng(8173)
    batches = np.zeros((repetitions, len(styles)))
    for style in sorted(set(styles)):
        idx = np.flatnonzero(np.asarray(styles) == style)
        draws = rng.choice(idx, (repetitions, len(idx)))
        for r, draw in enumerate(draws):
            batches[r] += np.bincount(draw, minlength=len(styles)) / len(styles)
    questions = rng.multinomial(probes, np.full(probes, 1 / probes), size=repetitions) / probes
    return batches, questions


def bootstrap_mean(values, weights):
    return np.einsum("ij,jk,ik->i", weights[0], values, weights[1], optimize=True)


def finite(value):
    return float(value) if np.isfinite(value) else None


def ratio(a, b):
    return finite(a / b) if b > 1e-30 else None


def metrics(pred, actual):
    error = pred - actual
    rmse = np.sqrt(np.mean(error**2))
    rms = np.sqrt(np.mean(actual**2))
    batch_denominator = np.mean(actual**2, axis=1)
    ratios = np.sqrt(np.mean(error**2, axis=1) / np.maximum(batch_denominator, 1e-30))
    corr = None
    if pred.std() > 1e-15 and actual.std() > 1e-15:
        corr = float(np.corrcoef(pred.ravel(), actual.ravel())[0, 1])
    return {
        "rmse": float(rmse),
        "nrmse": ratio(rmse, rms),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "actual_rms": float(rms),
        "predicted_rms": float(np.sqrt(np.mean(pred**2))),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(pred.mean()),
        "magnitude_ratio": ratio(np.sqrt(np.mean(pred**2)), rms),
        "actual_on_prediction_slope": ratio(np.sum(pred * actual), np.sum(pred**2)),
        "correlation": corr,
        "batch_nrmse_median": float(np.median(ratios)),
        "batch_nrmse_p90": float(np.quantile(ratios, 0.9)),
        "batch_nrmse_max": float(ratios.max()),
    }


def load_complete(root):
    state, audit, manifest, spec, plan, lock, data = [
        read(root / n)
        for n in (
            "controller.json",
            "integrity.json",
            "manifest.json",
            "specification.json",
            "plan.json",
            "forecast-lock.json",
            "data.json",
        )
    ]
    if (
        state["status"] != "complete"
        or not audit["complete"]
        or not audit["source_input_output_hashes_verified"]
    ):
        raise ValueError("Completed runs and remote integrity verification required")
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("specification.json", "specification_sha256"),
        ("plan.json", "plan_sha256"),
        ("compact_forecasts.npz", "compact_forecasts_sha256"),
    ):
        if file_hash(root / name) != audit[key]:
            raise ValueError(f"Provenance mismatch: {name}")
    if manifest["completed_unix"] > state["deadline_unix"] or not manifest["adapter_restored"]:
        raise ValueError("Deadline or restoration violation")
    if lock["plan_sha256"] != file_hash(root / "plan.json") or manifest[
        "forecast_lock_sha256"
    ] != file_hash(root / "forecast-lock.json"):
        raise ValueError("Forecast lock mismatch")
    for filename, expected in manifest["files_sha256"].items():
        if filename == "forecasts.pt":
            if expected != audit["original_forecast_sha256"]:
                raise ValueError("Remote forecast attestation mismatch")
            continue
        if file_hash(root / filename) != expected:
            raise ValueError(f"Output hash mismatch: {filename}")
    saved = dict(np.load(root / "compact_forecasts.npz"))
    outcomes = []
    baseline = read(root / "outcomes/baseline.json")
    for i in range(16):
        rows = []
        for rate in spec["rates"]:
            value = read(root / f"outcomes/batch-{i:02d}-rate-{rate:g}.json")
            if value["batch_index"] != i or value["rate"] != rate or not value["adapter_restored"]:
                raise ValueError("Condition mismatch")
            if value["plan_sha256"] != audit["plan_sha256"] or value[
                "forecast_lock_sha256"
            ] != file_hash(root / "forecast-lock.json"):
                raise ValueError("Outcome provenance mismatch")
            rows.append(value)
        outcomes.append(rows)
    for condition in [baseline] + [o for batch in outcomes for o in batch]:
        if [r["id"] for r in condition["generations"]] != [r["id"] for r in data["evaluation"]]:
            raise ValueError("Mismatched generation IDs")
        for row, question in zip(condition["generations"], data["evaluation"], strict=True):
            if row["checks"] != generated_checks(row["response"], question["tags"]):
                raise ValueError("Generation checks did not reproduce")
            row["quality"] = quality(row, spec["max_new_tokens"])
    return spec, data, saved, outcomes, baseline


def summarize(root):
    spec, data, saved, outcomes, baseline = load_complete(root)
    seed = spec["training_seed"]
    styles = [r["style"] for r in data["batches"]]
    weights = bootstrap_weights(styles, 16)
    actual_scores = np.asarray([[o["actual_logp"] for o in batch] for batch in outcomes])
    base = saved["baseline_logp"].astype(float)
    predicted_scores = base + (
        saved["unit_rate_forecasts"].astype(float)[:, None]
        * np.array(spec["rates"])[None, :, None, None, None]
    )
    scores, comparisons, tails, mismatches, associations = [], [], [], [], []
    for kind in ("summed_logp", "mean_token_logp", "four_choice_probability"):
        base_end = endpoint(base, saved["completion_tokens"], kind)
        actual = endpoint(actual_scores, saved["completion_tokens"], kind) - base_end
        pred = endpoint(predicted_scores, saved["completion_tokens"], kind) - base_end
        for ri, rate in enumerate(spec["rates"]):
            for ei, name in enumerate(("format", "verbosity")):
                target = actual[:, ri, 16:, ei]
                cal_actual = actual[:, ri, :16, ei].mean(1, keepdims=True)
                mismatches.append(
                    {
                        "seed": seed,
                        "rate": rate,
                        "score": kind,
                        "endpoint": name,
                        "note": "Observed calibration effect transferred to evaluation; non-prospective diagnostic",
                        **metrics(np.broadcast_to(cal_actual, target.shape), target),
                    }
                )
                for mode in ("same_probe", "calibration_mean_transfer"):
                    predictions = (
                        pred[:, ri, :, 16:, ei]
                        if mode == "same_probe"
                        else np.broadcast_to(
                            pred[:, ri, :, :16, ei].mean(2, keepdims=True), (16, 6, 16)
                        )
                    )
                    for mi, method in enumerate(METHODS):
                        prediction = predictions[:, mi]
                        header = {
                            "seed": seed,
                            "rate": rate,
                            "score": kind,
                            "endpoint": name,
                            "mode": mode,
                            "method": method,
                        }
                        for group in ("all", *STYLES):
                            idx = (
                                np.arange(16)
                                if group == "all"
                                else np.flatnonzero(np.array(styles) == group)
                            )
                            row = {
                                **header,
                                "style": group,
                                **metrics(prediction[idx], target[idx]),
                            }
                            if group == "all":
                                errors = (prediction - target) ** 2
                                denominator = bootstrap_mean(target**2, weights)
                                ci = np.quantile(
                                    np.sqrt(
                                        bootstrap_mean(errors, weights)
                                        / np.maximum(denominator, 1e-30)
                                    ),
                                    [0.025, 0.975],
                                )
                                row["nrmse_ci95"] = ci.tolist()
                                for comparator in ("svd16", "zero"):
                                    reference_errors = (
                                        predictions[:, METHODS.index(comparator)] - target
                                    ) ** 2
                                    delta = errors - reference_errors
                                    comparisons.append(
                                        {
                                            **header,
                                            "comparator": comparator,
                                            "mse_difference": float(delta.mean()),
                                            "ci95": np.quantile(
                                                bootstrap_mean(delta, weights), [0.025, 0.975]
                                            ).tolist(),
                                        }
                                    )
                            scores.append(row)
                        for bi in range(16):
                            tails.append(
                                {
                                    **header,
                                    "batch": bi,
                                    "style": styles[bi],
                                    **metrics(prediction[bi : bi + 1], target[bi : bi + 1]),
                                }
                            )
                if kind == "four_choice_probability" and name == "format":
                    for bi in range(16):
                        gen_delta = np.mean(
                            [o["checks"]["numbered"] for o in outcomes[bi][ri]["generations"]]
                        ) - np.mean([o["checks"]["numbered"] for o in baseline["generations"]])
                        associations.append(
                            {
                                "seed": seed,
                                "batch": bi,
                                "style": styles[bi],
                                "rate": rate,
                                "actual_conditional_probability_change": float(target[bi].mean()),
                                "generated_numbered_change": float(gen_delta),
                                "full_gradient_same_probe_forecast": float(
                                    pred[bi, ri, 0, 16:, ei].mean()
                                ),
                                "dictionary_same_probe_forecast": float(
                                    pred[bi, ri, 2, 16:, ei].mean()
                                ),
                            }
                        )
    generation_rows, examples = [], []
    generation_metrics = (
        "exact_content_and_order",
        "all_tags_present",
        "numbered",
        "words",
        "empty",
        "tokens",
        "truncated",
        "possible_degeneration",
    )

    def genvalues(rows):
        return np.array(
            [[{**row["checks"], **row["quality"]}[k] for k in generation_metrics] for row in rows],
            dtype=float,
        )

    bg = genvalues(baseline["generations"])
    gg = np.asarray([[genvalues(o["generations"]) for o in batch] for batch in outcomes])
    for ri, rate in enumerate(spec["rates"]):
        for style in ("all", *STYLES):
            idx = np.arange(16) if style == "all" else np.flatnonzero(np.array(styles) == style)
            for ki, key in enumerate(generation_metrics):
                difference = gg[idx, ri, :, ki] - bg[:, ki]
                ci_weights = (
                    weights if style == "all" else bootstrap_weights([style] * len(idx), 16)
                )
                generation_rows.append(
                    {
                        "seed": seed,
                        "rate": rate,
                        "style": style,
                        "metric": key,
                        "baseline": float(bg[:, ki].mean()),
                        "after": float(gg[idx, ri, :, ki].mean()),
                        "change": float(difference.mean()),
                        "ci95": np.quantile(
                            bootstrap_mean(difference, ci_weights), [0.025, 0.975]
                        ).tolist(),
                    }
                )
        for bi, qi in itertools.product(range(4), range(2)):
            examples.append(
                {
                    "seed": seed,
                    "rate": rate,
                    "batch": bi,
                    "style": styles[bi],
                    "prompt": data["evaluation"][qi]["prompt"],
                    "tags": data["evaluation"][qi]["tags"],
                    "baseline": baseline["generations"][qi],
                    "updated": outcomes[bi][ri]["generations"][qi],
                }
            )
    coding = {
        "seed": seed,
        "active_atoms_per_batch": (np.abs(saved["sparse_codes"]) > 1e-8).sum(1).tolist(),
        "batch_gradient_norms": saved["batch_gradient_norms"].tolist(),
        "projected_norms": saved["projected_norms"].ravel().tolist(),
    }
    return {
        "metrics": scores,
        "comparisons": comparisons,
        "batch_metrics": tails,
        "probe_mismatch": mismatches,
        "generation_metrics": generation_rows,
        "examples": examples,
        "surrogate_generation_associations": associations,
        "coding": coding,
    }


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            )


def plots(out, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#222222", "#4590b9", "#c15e2b", "#518745", "#9e5ea6", "#999999"]
    rows = summary["metrics"]
    for kind in ("summed_logp", "mean_token_logp", "four_choice_probability"):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
        for si, seed in enumerate((42, 43)):
            for col, (mode, name) in enumerate(
                itertools.product(
                    ("same_probe", "calibration_mean_transfer"), ("format", "verbosity")
                )
            ):
                ax = axes[si, col]
                for mi, method in enumerate(METHODS):
                    rr = [
                        r
                        for r in rows
                        if r["seed"] == seed
                        and r["score"] == kind
                        and r["endpoint"] == name
                        and r["mode"] == mode
                        and r["method"] == method
                        and r["style"] == "all"
                    ]
                    y = [r["nrmse"] for r in rr]
                    ax.plot(
                        [r["rate"] for r in rr], y, marker="o", color=colors[mi], label=LABELS[mi]
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(
                    f"Seed {seed} · {name}\n{'Same probes' if mode == 'same_probe' else 'Calibration → evaluation'}"
                )
                ax.axhline(1, color="#aaa", ls=":", lw=1)
                if col == 0:
                    ax.set_ylabel("Error / zero-forecast error (lower better)")
                if si == 1:
                    ax.set_xlabel("SGD learning rate")
                ax.grid(alpha=0.15)
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False
        )
        fig.suptitle(
            f"LoRA batch forecasts · {kind.replace('_', ' ')}\nSynthetic development; curves include all frozen methods and rates",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.91))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"forecast_{kind}.{extension}", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for si, seed in enumerate((42, 43)):
        for ki, metric in enumerate(("numbered", "exact_content_and_order", "tokens", "truncated")):
            ax = axes[si, ki]
            for style in STYLES:
                rr = [
                    r
                    for r in summary["generation_metrics"]
                    if r["seed"] == seed and r["style"] == style and r["metric"] == metric
                ]
                ax.plot(
                    [r["rate"] for r in rr],
                    [r["after"] for r in rr],
                    marker="o",
                    label=style.replace("_", " "),
                )
            ax.axhline(rr[0]["baseline"], color="black", ls=":", label="Baseline")
            ax.set_xscale("log")
            ax.set_title(f"Seed {seed} · {metric.replace('_', ' ')}")
            ax.grid(alpha=0.15)
            if si == 1:
                ax.set_xlabel("SGD learning rate")
            if metric != "tokens":
                ax.set_ylim(-0.04, 1.04)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Actual free generations: presentation and content are separate checks\nFour batches per style × 16 evaluation prompts; no predictor was fitted to these outcomes",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"generated_behavior.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, seed in zip(axes, (42, 43), strict=True):
        for style in STYLES:
            rr = [
                r
                for r in summary["surrogate_generation_associations"]
                if r["seed"] == seed and r["style"] == style
            ]
            ax.scatter(
                [r["actual_conditional_probability_change"] for r in rr],
                [r["generated_numbered_change"] for r in rr],
                label=style.replace("_", " "),
                alpha=0.75,
            )
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Measured change in four-choice P(numbered)")
        ax.set_ylabel("Measured change in free-generation numbered rate")
        ax.axhline(0, color="#bbb", lw=1)
        ax.axvline(0, color="#bbb", lw=1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Conditional completion scores are not free-generation probabilities")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"surrogate_vs_generation.{extension}", dpi=160)
    plt.close(fig)


def report(out, summary):
    rows = summary["metrics"]
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Prospective LoRA batch pilot</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#202428}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f3f5f7;padding:14px}details{margin:18px 0}h1,h2{line-height:1.2}a{color:#176b9a}</style>',
        "<h1>Can a batch gradient forecast a LoRA update?</h1><p>Synthetic development pilot · two reconstructed Gemma-3 4B adapters · 96 prospective updates · 1,568 free generations.</p>",
        "<p>The model was asked to copy three unfamiliar tags. Training batches varied two things independently: plain versus numbered presentation, and short versus long answers. We predicted changes before applying each batch, then measured both completion scores and freely generated answers. This separates score prediction from behavioral validation.</p>",
        "<h2>What we learned</h2><p><b>The full gradient forecasts answer-score changes, but this pilot does not establish behavior forecasting.</b> At the middle learning rate, 0.003, same-probe format error was 0.140 and 0.112 times the no-change error (seeds 42 and 43). Dictionary errors were 0.845 and 0.814, versus 0.814 and 0.767 for SVD16. The dictionary predicted only 16% and 19% of the measured format-change RMS. It did not consistently improve on the simpler baseline.</p>",
        "<p><b>Aggregated accuracy hides difficult batches and transfer failures.</b> Full-gradient format error rose to 0.585 and 0.456 when transferring mean calibration forecasts to evaluation questions. Median batch errors for that transfer were 1.38 and 1.26 times the zero baseline; large-effect batches dominate the pooled metric. At the largest rate, summed-score full-gradient forecasts substantially overshot. Some compressed forecasts improved there, consistent with attenuation helping an overlarge linear forecast; this is not evidence of better atom semantics.</p>",
        "<p><b>The score changes did not produce the expected formatting change.</b> Only one of 1,536 post-update free responses was numbered. The four candidate answers omit common generated forms, including plain newline-separated tags. Their normalized probabilities can change sharply while actual generation remains unchanged. Long candidates also have near-zero conditional mass, so large relative errors for their probability contrast have tiny absolute denominators.</p>",
        "<p><b>Content checks catch a failure.</b> At rate 0.03, plain-short training on seed 42 reduced exact tag/order accuracy from 100% to 37.5%; 73.4% of those responses hit the generation cap. Seed 43 did not show that failure. At rate 0.003, seed 43 plain-short batches improved tag accuracy from 81.25% to 100%, but this was not forecast by a trained correctness predictor. These are descriptive development results.</p>",
        "<h2>What was fixed before running</h2><ol><li>Use rank-8 q/v LoRA in all 34 language layers, 136 A/B tensors, with the reconstructed Gradient Atoms checkpoints.</li><li>Use 16 batches of four documents, four batches for each format/length combination. Calibration and evaluation have 16 questions each, with different templates.</li><li>Save full-gradient, EKFAC, dictionary, SVD16, mean-direction and zero forecasts for all probes before any update. Restore gradient magnitude after sparse coding.</li><li>Apply one SGD step at each of three fixed learning rates, reset the adapter after every step, then score four content-matched completions and generate freely on the evaluation prompts.</li><li>Analyze every method, rate, seed and batch family after both runs finish. Pointwise crossed batch/probe bootstrap intervals are descriptive; two seeds do not establish seed-population robustness.</li></ol>",
        "<h2>Prediction errors</h2><p>Error is divided by the error of predicting no change. Below 1 beats that baseline. Same-probe forecasts use gradients of the actual scored questions. Transfer uses the mean forecast from a separate calibration template. SVD16 is a simple low-rank baseline, not an equal-storage or equal-compute comparison.</p>",
    ]
    for kind in ("summed_logp", "mean_token_logp", "four_choice_probability"):
        parts.append(
            f'<h3>{html.escape(kind.replace("_", " "))}</h3><img src="forecast_{kind}.png">'
        )
        if kind == "four_choice_probability":
            parts.append(
                "<p>Probabilities here are normalized over exactly four predefined completions, all with correct tags. They are not probabilities of unrestricted model behavior. Summed log probabilities also favor shorter sequences; per-token scores answer a different question and are not sequence probabilities.</p>"
            )
    parts.append(
        "<h3>All rates: same-probe format contrast, summed log probabilities</h3><table><tr><th>Seed / rate</th>"
        + "".join(f"<th>{label}</th>" for label in LABELS)
        + "</tr>"
    )
    for seed, rate in itertools.product((42, 43), (0.0003, 0.003, 0.03)):
        selected = [
            next(
                r
                for r in rows
                if r["seed"] == seed
                and r["rate"] == rate
                and r["score"] == "summed_logp"
                and r["endpoint"] == "format"
                and r["mode"] == "same_probe"
                and r["style"] == "all"
                and r["method"] == m
            )
            for m in METHODS
        ]
        parts.append(
            f"<tr><td>{seed} / {rate:g}</td>"
            + "".join(f"<td>{r['nrmse']:.3f}</td>" for r in selected)
            + "</tr>"
        )
    parts.extend(
        [
            "</table><p>Full machine-readable tables include each batch, median/p90/worst batch errors, calibration slopes and magnitude ratios, all style groups and both endpoints, and paired error differences against SVD16 and zero.</p>",
            '<h2>Actual generated behavior</h2><img src="generated_behavior.png"><img src="surrogate_vs_generation.png">',
            "<p>Exact content/order means the extracted tags exactly match the three requested tags in order, without extra tag occurrences. Numbering and output length are scored separately. Truncation, empty responses and repeated-token/4-gram flags are also retained. These checks do not measure general answer quality or task competence.</p>",
            "<h2>Fixed examples</h2><p>First batch of each style and first two evaluation prompts, all rates and both seeds; selection was written before outcome inspection.</p>",
        ]
    )
    for ex in summary["examples"]:
        parts.append(
            f"<details><summary>Seed {ex['seed']}, {ex['style']}, rate {ex['rate']:g}, {ex['updated']['id']}</summary><p>{html.escape(ex['prompt'])}</p><b>Baseline</b><pre>{html.escape(ex['baseline']['response'])}</pre><b>After update</b><pre>{html.escape(ex['updated']['response'])}</pre><p>Checks: {html.escape(json.dumps(ex['updated']['checks']))}</p></details>"
        )
    parts.extend(
        [
            "<h2>Limits and provenance</h2><p>This is development on synthetic copying prompts, selected after the steering reproduction. Seeds 42 and 43 reuse discovery checkpoints and differ in GPU hardware as well as training randomness. The original rank-8 checkpoint was unavailable: these are paper-architecture reconstructions, not exact checkpoint replicas. No model was fitted to predict free-generation frequencies. Same-probe Taylor accuracy, reconstruction and controlled response styles do not establish learned atom semantics or a novel research contribution.</p>",
            "<p>Both full manifests and forecast hashes were verified remotely; all original outputs remain there. Compact forecasts and their hashes are copied here. Source snapshots preserve the completed generation code. Forecast ordering is enforced by code and supported by lock and file timestamps; individual update timestamps were not embedded in each outcome.</p>",
            '<p><a href="https://arxiv.org/abs/2603.14665">Gradient Atoms paper</a> · <a href="https://github.com/jrosseruk/gradient_atoms/tree/a83858557b411249097861f25d57eb13020eb334">Pinned author implementation</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/c9835c0f1d551384f07e6028e0bf81b682ee5d85/src/gradient_interp/paper_batch.py">Frozen pilot code</a></p>',
            '<p><a href="summary.json">All results JSON</a> · <a href="metrics.csv">Prediction metrics</a> · <a href="batch_metrics.csv">Per-batch errors</a> · <a href="comparisons.csv">Paired comparisons</a> · <a href="generation_metrics.csv">Generation checks</a> · <a href="probe_mismatch.csv">Probe mismatch</a> · <a href="analysis-plan.json">Pre-reading analysis plan</a></p>',
        ]
    )
    (out / "report.html").write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    args = parser.parse_args()
    # Check both lifecycle states before opening either outcome collection.
    for root in args.roots:
        if read(root / "controller.json")["status"] != "complete":
            raise ValueError("Both runs must complete")
    plan = read(args.analysis_plan)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"analysis_plan": plan, "scope": "synthetic_development_only"}
    for root in args.roots:
        result = summarize(root)
        for key, values in result.items():
            summary.setdefault(key, []).extend(values if isinstance(values, list) else [values])
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
            "compact_forecasts.npz",
        ):
            shutil.copy2(root / name, dest / name)
    shutil.copy2(args.analysis_plan, args.out / "analysis-plan.json")
    atomic_json(args.out / "summary.json", summary)
    for key in (
        "metrics",
        "batch_metrics",
        "comparisons",
        "generation_metrics",
        "probe_mismatch",
        "surrogate_generation_associations",
    ):
        write_csv(args.out / f"{key}.csv", summary[key])
    plots(args.out, summary)
    report(args.out, summary)
    print(args.out / "report.html")


if __name__ == "__main__":
    main()
