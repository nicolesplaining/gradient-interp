"""Read-only reporting of frozen confirmation plus labeled descriptive diagnostics.

The primary analysis remains in confirmation_analysis.py at its frozen revision.
No methods are selected here and no model updates are made. Additional removal
intervals, outlier attribution and generation summaries are descriptive only.
"""
# ruff: noqa: E501

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np

from .confirmation import verify_files
from .confirmation_analysis import category_array, center, descriptive
from .data import read_jsonl
from .study import atomic_json, file_hash


def control_intervals(changes, targets, batches, repeats=5000):
    """Paired seed/batch/probe bootstrap for descriptive target-share differences."""
    rng = np.random.default_rng(20261013)
    ns, nb, ng, nd = changes["atom_removed"].shape
    groups = {}
    for i, b in enumerate(batches):
        groups.setdefault((b["kind"], b["batch_size"]), []).append(i)

    def shares(means):
        target = np.take_along_axis(means, np.asarray(targets)[None, :, None], axis=-1)[..., 0]
        return np.abs(target) / np.maximum(np.abs(means).sum(-1), 1e-30)

    point = {k: shares(v.mean(-1)) for k, v in changes.items()}
    samples = {k: [] for k in changes if k != "atom_removed"}
    for start in range(0, repeats, 32):
        count = min(32, repeats - start)
        sw = rng.multinomial(ns, np.full(ns, 1 / ns), count)
        bw = np.zeros((count, nb))
        for indices in groups.values():
            bw[:, indices] = rng.multinomial(
                len(indices), np.full(len(indices), 1 / len(indices)), count
            )
        pw = rng.multinomial(nd, np.full(nd, 1 / nd), (count, ng)) / nd
        weights = sw[:, :, None] * bw[:, None, :]
        sampled = {}
        for k, v in changes.items():
            means = np.einsum("sbgp,rgp->rsbg", v, pw, optimize=True)
            target = np.take_along_axis(means, np.asarray(targets)[None, None, :, None], axis=-1)[
                ..., 0
            ]
            sampled[k] = np.abs(target) / np.maximum(np.abs(means).sum(-1), 1e-30)
        for k in samples:
            difference = sampled["atom_removed"] - sampled[k]
            samples[k].extend((np.sum(difference * weights, axis=(1, 2)) / (ns * nb)).tolist())
    return {
        "scope": "Descriptive follow-up; not a new confirmatory test",
        "definition": "Mean abs(target change) / sum(abs(change in each category)) per seed/batch",
        "mean_target_share": {k: float(v.mean()) for k, v in point.items()},
        "atom_minus_control": {
            k: {
                "difference": float((point["atom_removed"] - point[k]).mean()),
                "crossed_seed_batch_probe_95ci": np.quantile(values, [0.025, 0.975]).tolist(),
            }
            for k, values in samples.items()
        },
        "bootstrap_repeats": repeats,
    }


def diagnose(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    batches = read_jsonl(root / "data/batches.jsonl")
    evaluation = read_jsonl(root / "data/evaluation_probes.jsonl")
    calibration = read_jsonl(root / "data/calibration_probes.jsonl")
    selected = plan["control_batch_indices"]
    groups = sorted({p["group"] for p in evaluation})
    model_results, generation_examples = {}, []
    for model in summary["results"]:
        jobs = [j for j in plan["jobs"] if j["model"]["name"] == model]
        actual, forecasts = [], []
        controls = {k: [] for k in ("atom_removed", "random_removed", "rescaled_full")}
        atom_predictions = {k: [] for k in ("fixed", "transfer")}
        generation_counts = {
            "comparisons": 0,
            "full_differs_from_baseline": 0,
            "removed_differs_from_full": 0,
        }
        for job in jobs:
            d = root / "jobs" / job["id"]
            lock = json.loads((d / "forecast_lock.json").read_text())
            assert file_hash(d / "predictions.npz") == lock["files_sha256"]["predictions.npz"]
            manifest = json.loads((d / "manifest.json").read_text())
            verify_files(d / "results", manifest["result_files_sha256"])
            metadata = json.loads((d / "forecast_metadata.json").read_text())
            with np.load(d / "predictions.npz") as z:
                forecasts.append({k: z[k].astype(float) for k in ("calibration", "evaluation")})
                for view, key, probes in (
                    ("fixed", "atom_evaluation", evaluation),
                    ("transfer", "atom_calibration", calibration),
                ):
                    atom_predictions[view].append(
                        category_array(z[key][selected] * 0.0001, probes, groups)
                    )
            seed_y, seed_controls = [], {k: [] for k in controls}
            for ri in range(len(plan["learning_rates"])):
                values = []
                for bi in range(len(batches)):
                    with np.load(d / "results" / f"batch-{bi:03d}-rate-{ri}.npz") as z:
                        values.append(z["full"].copy())
                        if ri == 0 and bi in selected:
                            for k in controls:
                                seed_controls[k].append(z["full"] - z[k])
                seed_y.append(values)
            actual.append(seed_y)
            for k in controls:
                controls[k].append(seed_controls[k])
            gen = json.loads((d / "generations.json").read_text())
            for case in gen["results"]:
                for pi, (base, full, removed) in enumerate(
                    zip(gen["baseline"], case["full"], case["atom_removed"])
                ):
                    generation_counts["comparisons"] += 1
                    generation_counts["full_differs_from_baseline"] += base != full
                    generation_counts["removed_differs_from_full"] += full != removed
                    if base != full or full != removed:
                        generation_examples.append(
                            {
                                "job": job["id"],
                                "batch_id": case["batch_id"],
                                "probe_group": groups[pi],
                                "prompt": gen["prompts"][pi],
                                "baseline": base,
                                "full": full,
                                "atom_removed": removed,
                            }
                        )
        actual = category_array(np.asarray(actual), evaluation, groups)
        predictions = {
            view: category_array(np.stack([v[key] for v in forecasts]), probes, groups)
            for view, key, probes in (
                ("fixed", "evaluation", evaluation),
                ("transfer", "calibration", calibration),
            )
        }
        methods = metadata["methods"]
        diagnostic = {"seeds": [j["seed"] for j in jobs], "rates": {}}
        for ri, rate in enumerate(plan["learning_rates"]):
            y = actual[:, ri].mean(-1)
            contrasts = center(y)
            by_method = {}
            for method in ("full_gradient", "svd_12", "svd_128", "dictionary_omp_8"):
                by_view = {}
                for view, values in predictions.items():
                    predicted = rate * values[:, methods.index(method)].mean(-1)
                    errors = center(predicted) - contrasts
                    squared = errors**2
                    shares = squared.sum((0, 2)) / max(squared.sum(), 1e-30)
                    by_view[view] = {
                        "per_seed_batch_relative_errors": (
                            np.linalg.norm(errors, axis=-1)
                            / np.maximum(np.linalg.norm(contrasts, axis=-1), 1e-30)
                        ).tolist(),
                        "error_share_by_seed": (
                            squared.sum((1, 2)) / max(squared.sum(), 1e-30)
                        ).tolist(),
                        "largest_error_batches": [
                            {"batch_id": batches[i]["id"], "error_share": float(shares[i])}
                            for i in np.argsort(shares)[-5:][::-1]
                        ],
                        "by_batch_kind_and_size": {
                            f"{kind}-{size}": descriptive(
                                center(predicted[:, indices]), contrasts[:, indices]
                            )
                            for kind, size, indices in (
                                (
                                    kind,
                                    size,
                                    [
                                        i
                                        for i, b in enumerate(batches)
                                        if b["kind"] == kind and b["batch_size"] == size
                                    ],
                                )
                                for kind, size in sorted(
                                    {(b["kind"], b["batch_size"]) for b in batches}
                                )
                            )
                        },
                    }
                by_method[method] = by_view
            diagnostic["rates"][str(rate)] = {
                "common_shift_energy_fraction": float(1 - np.sum(contrasts**2) / np.sum(y**2)),
                "category_contrast_rms": float(np.sqrt(np.mean(contrasts**2))),
                "methods": by_method,
            }
        controls = {
            k: category_array(np.asarray(v), evaluation, groups) for k, v in controls.items()
        }
        targets = [groups.index(batches[i]["target_group"]) for i in selected]
        diagnostic["removal"] = control_intervals(controls, targets, [batches[i] for i in selected])
        diagnostic["removal"]["prediction_error"] = {
            view: descriptive(np.asarray(value).mean(-1), controls["atom_removed"].mean(-1))
            for view, value in atom_predictions.items()
        }
        diagnostic["removal"]["mean_absolute_target_effect"] = {
            k: float(
                np.abs(
                    np.take_along_axis(v.mean(-1), np.asarray(targets)[None, :, None], -1)
                ).mean()
            )
            for k, v in controls.items()
        }
        diagnostic["generations"] = {
            **generation_counts,
            "batch_ids": [
                b["id"] for b in batches if b["kind"] == "pure" and b["batch_size"] == 16
            ][:2],
            "scope": "Two prespecified brainstorming batches; eight prompts, five seeds; exact text changes only",
        }
        model_results[model] = diagnostic
    diagnostics = {
        "scope": "Descriptive diagnostics; primary outcomes and decision rules are unchanged",
        "summary_sha256": file_hash(root / "summary.json"),
        "results": model_results,
        "changed_generation_count": len(generation_examples),
    }
    atomic_json(root / "diagnostics.json", diagnostics)
    atomic_json(
        root / "generation-examples.json",
        {
            "selection": "Every saved pair with any exact-text change; no semantic scoring",
            "examples": generation_examples,
        },
    )
    return diagnostics


def table(headers, rows):
    return (
        "<div class='table'><table><thead><tr>"
        + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def pct(value):
    return f"{100 * value:.1f}%"


def render(root, diagnostics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, PercentFormatter

    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    integrity = json.loads((root / "integrity-check.json").read_text())
    models = list(summary["results"])
    out = root / "report"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((out / f"{name}.png").read_bytes()).decode()
        return f"<img src='data:image/png;base64,{encoded}' alt='{name.replace('_', ' ')}'>"

    comparisons = ["fixed_vs_svd_12", "fixed_vs_rotated_dictionary_omp_8", "transfer_vs_svd_12"]
    labels = [
        "Fixed probes vs SVD12",
        "Fixed probes vs rotated dictionary",
        "Separate probes vs SVD12",
    ]
    fig, ax = plt.subplots(figsize=(11, 5.1))
    ticks, primary_rows = [], []
    for mi, model in enumerate(models):
        for ci, (key, label) in enumerate(zip(comparisons, labels)):
            row = summary["results"][model]["primary"][key]
            y = 5 - (mi * 3 + ci)
            point, bounds, upper = (
                row["paired_error_ratio"],
                row["crossed_95ci"],
                row["multiplicity_adjusted_upper"],
            )
            color = "#187e58" if row["passes"] else "#4f6573"
            ax.plot(bounds, [y, y], color=color, linewidth=2)
            ax.scatter(point, y, color=color, s=55, zorder=3)
            ax.scatter(upper, y, color=color, marker=">", s=65, zorder=3)
            ticks.append(f"{model.split('-')[-1]}   {label}")
            primary_rows.append(
                [
                    model.split("-")[-1],
                    label,
                    f"{point:.3f}",
                    f"[{bounds[0]:.3f}, {bounds[1]:.3f}]",
                    f"{upper:.3f}",
                    "Yes" if row["all_seeds_improve"] else "No",
                    "PASS" if row["passes"] else "Did not meet criterion",
                ]
            )
    ax.axvline(0.9, color="#b24b35", linestyle="--", label="Required upper bound <0.90")
    ax.axvline(1, color="#888888", linewidth=0.8)
    ax.axhline(2.5, color="#dddddd", linewidth=0.8)
    ax.set(
        yticks=np.arange(5, -1, -1),
        yticklabels=ticks,
        xscale="log",
        xlim=(0.23, 5.5),
        ylim=(-0.65, 5.65),
        xlabel="Dictionary OMP8 error / comparator error; lower is better",
    )
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.50", "1.0", "2.0", "4.0"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", alpha=0.15)
    fig.suptitle("One of six prespecified comparisons passes", fontsize=16)
    ax.set_title(
        "Circles: estimates. Lines: 95% intervals. Triangles: multiplicity-adjusted upper bounds.",
        fontsize=9,
        pad=16,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    primary_plot = save(fig, "primary_comparisons")

    names = ["full_gradient", "svd_12", "svd_128", "dictionary_omp_8", "rotated_dictionary_omp_8"]
    names_short = ["Full gradient", "SVD12", "SVD128", "Dictionary8", "Rotated8"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharey=True)
    for mi, model in enumerate(models):
        for ri, rate in enumerate(("0.0001", "0.001")):
            ax = axes[mi, ri]
            for vi, (view, label, color) in enumerate(
                (("fixed", "Fixed probes", "#23779c"), ("transfer", "Separate probes", "#d38335"))
            ):
                values = [
                    summary["results"][model]["descriptive"][rate][view][m]["category_contrasts"][
                        "relative_l2_error"
                    ]
                    for m in names
                ]
                ax.bar(
                    np.arange(len(names)) + (vi - 0.5) * 0.35,
                    values,
                    width=0.33,
                    label=label,
                    color=color,
                )
            ax.axhline(1, color="#777777", linestyle=":")
            ax.set_xticks(range(len(names)), names_short, fontsize=8)
            ax.set_title(f"{model.split('-')[-1]} | learning rate {rate}")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(0, 1.55)
            ax.grid(axis="y", alpha=0.15)
    axes[0, 0].set_ylabel("Category-contrast relative error")
    axes[1, 0].set_ylabel("Category-contrast relative error")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Forecasting deteriorates on independent probes", fontsize=16)
    fig.text(
        0.5,
        0.008,
        "Descriptive estimates. 100% is the zero-change baseline. Scores measure reference-answer log probability.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    errors_plot = save(fig, "forecast_errors")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    for ax, model in zip(axes, models):
        for method, label, color in (
            ("full_gradient", "Full gradient", "#42494c"),
            ("svd_12", "SVD12", "#23779c"),
            ("dictionary_omp_8", "Dictionary8", "#bf592f"),
        ):
            for view, linestyle in (("fixed", "-"), ("transfer", "--")):
                values = np.sort(
                    np.ravel(
                        diagnostics["results"][model]["rates"]["0.0001"]["methods"][method][view][
                            "per_seed_batch_relative_errors"
                        ]
                    )
                )
                ax.plot(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    label=f"{label}, {'fixed' if view == 'fixed' else 'separate'}",
                    color=color,
                    linestyle=linestyle,
                )
        ax.axvline(1, color="#777777", linewidth=0.8)
        ax.set(
            xscale="log",
            xlim=(0.001, 10),
            ylim=(0, 1.01),
            title=model.split("-")[-1],
            xlabel="Per-batch category-contrast relative error",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Fraction of seed/batch cases")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Pooled errors hide large differences between batches", fontsize=16)
    fig.text(
        0.5,
        0.005,
        "Learning rate 0.0001; 480 seed/batch cases per model, sharing source batches across seeds. Descriptive ECDFs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    tails_plot = save(fig, "batch_error_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, model in enumerate(models):
        stats = diagnostics["results"][model]["removal"]
        for ci, (control, label, color) in enumerate(
            (
                ("random_removed", "Random removal", "#23779c"),
                ("rescaled_full", "Norm-matched rescaling", "#bf592f"),
            )
        ):
            v = stats["atom_minus_control"][control]
            y = i + (ci - 0.5) * 0.22
            axes[0].plot(v["crossed_seed_batch_probe_95ci"], [y, y], color=color, linewidth=2)
            axes[0].scatter(v["difference"], y, color=color, label=label if i == 0 else None)
        for vi, (view, label, color) in enumerate(
            (("fixed", "Fixed probes", "#23779c"), ("transfer", "Separate probes", "#d38335"))
        ):
            axes[1].bar(
                i + (vi - 0.5) * 0.3,
                stats["prediction_error"][view]["relative_l2_error"],
                width=0.28,
                color=color,
                label=label if i == 0 else None,
            )
    axes[0].axvline(0, color="#777777", linestyle=":")
    axes[0].set(
        yticks=[0, 1],
        yticklabels=[m.split("-")[-1] for m in models],
        xlabel="Atom minus control target share",
        title="No clear target-share advantage",
    )
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="center left", frameon=False, fontsize=8)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=[m.split("-")[-1] for m in models],
        ylabel="Relative forecast error",
        title="Forecasting the effect of atom removal",
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, fontsize=8)
    fig.text(
        0.5,
        0.005,
        "Descriptive follow-up, 95% crossed seed/batch/probe intervals; 24 batches per seed, learning rate 0.0001.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    removal_plot = save(fig, "atom_removal")

    small = summary["results"][models[0]]["primary"]["fixed_vs_rotated_dictionary_omp_8"]
    body = [
        "<h1>One narrow positive result, with clear limits</h1>",
        f"<p class='subtitle'>Fresh-source confirmation · 10/10 jobs complete · {integrity['wall_clock_hours']:.2f} hours on one A100</p>",
        "<p class='lead'>On Qwen2.5-0.5B, the learned gradient dictionary predicts category-specific score changes better than its rotated control. "
        "This passes the frozen success rule. We did not establish that dictionaries beat equal-storage SVD or transfer effect sizes reliably to separate probe documents.</p>",
        f"<p>The positive comparison has an error ratio of <strong>{small['paired_error_ratio']:.3f}</strong>, or {pct(1 - small['paired_error_ratio'])} lower error. "
        f"Its 95% interval is [{small['crossed_95ci'][0]:.3f}, {small['crossed_95ci'][1]:.3f}]; the multiplicity-adjusted upper bound is {small['multiplicity_adjusted_upper']:.3f}, below the required 0.90. "
        "All five seeds improve. This supports a claim about learned gradient structure. It does not establish human-interpretable atoms or a general behavior-change predictor.</p>",
        "<h2>What was frozen before testing</h2>",
        "<p>Two Qwen base models, 0.5B and 3B, with new seeds 101, 202, 303, 404 and 505. Each uses rank-4 q/v LoRA and 64 fitting-data AdamW steps. "
        "The 5,000 No Robots fitting documents are reused from development; evaluation moves to 816 fresh Dolly source documents in 96 disjoint batches, sizes 1, 4 and 16. "
        "Calibration and evaluation each have 32 independent documents per category, eight categories and 256 probes total. Prompt and source-ID overlap checks passed. "
        "This tests a new dataset, not unseen task concepts or absence from model pretraining.</p>",
        "<p>Each test applies one stateless SGD step at learning rate 0.0001 or 0.001 and restores the checkpoint. There are 1,920 full-update cases, plus 720 removal/control evaluations. "
        "All forecasts were saved and hashed before test updates within each job. All source, data, forecast and result hashes passed verification, as did checkpoint restoration. No jobs are unfinished.</p>",
        "<p>The primary target is the change in mean reference-answer log probability for each category after subtracting that batch's average change across categories. "
        "This removes the common shift. Relative error divides total prediction-error norm by total observed-change norm; zero-change prediction has 100% error. "
        "The measured category-contrast RMS is about 0.00022 nats per token at the primary rate, well above the prespecified numerical floor, but small in practical terms.</p>",
        "<h2>All six primary comparisons</h2>",
        primary_plot,
        table(
            [
                "Model",
                "Comparison",
                "Error ratio",
                "95% interval",
                "Adjusted upper",
                "All seeds improve",
                "Frozen decision",
            ],
            primary_rows,
        ),
        "<p class='caption'>10,000 paired bootstrap replicates resample seeds, batches within kind/size strata, and probe documents within categories. "
        "Transfer resamples calibration and evaluation probes independently. Success requires an upper bound below 0.90 at one-sided 1−0.05/6 coverage, improvement in every seed and signal above the numerical threshold. "
        "With only five seeds, tail bounds are approximate. One of six passes; no comparison replicates the full success rule across both models.</p>",
        "<p>The fixed dictionary has 512 atoms in a rank-128 whitened gradient SVD space. OMP uses up to eight atoms. Eight float32 coefficients plus uint16 indices cost 48 bytes, equal to twelve float32 SVD coefficients. "
        "This is a per-update code comparison; shared bases, dictionary storage, probe calibration and computation are excluded. The rotated control preserves the dictionary's atom Gram matrix while rotating it in the whitened space. "
        "On 0.5B the dictionary's point estimate also beats SVD12, but its interval includes both improvement and degradation. On 3B it is worse than SVD12 at the point estimate.</p>",
        "<h2>Fixed probes and independent probes</h2>",
        errors_plot,
        "<p>Fixed-probe methods use the evaluation probes' derivatives before applying an update. This tests local forecasting and compression of gradients. "
        "Transfer methods use a separate calibration set to predict category means on unseen probe documents. These are different claims. "
        "At the primary rate, dictionary category-contrast errors are 25.7% and 44.6% on fixed probes, but 52.5% and 75.6% on separate probes for 0.5B and 3B. "
        "SVD12 transfer errors are 41.8% and 78.4%. Neither dictionary-versus-SVD transfer comparison passes.</p>",
        "<p>Descriptively, 128 SVD coordinates forecast fixed-probe changes with 8.8% error on 0.5B and 29.6% on 3B at the primary rate. "
        "This is evidence that compressed gradients can retain useful forecasting information under a specified checkpoint and probe set. It is not a new confirmed superiority claim. "
        "At the larger step, errors grow sharply, and shrinkage can appear beneficial by damping nonlinear overprediction.</p>",
        "<h2>Batch variability and sign predictions</h2>",
        tails_plot,
        "<p>For dictionary OMP8 at the primary rate, median fixed-probe category-contrast error is 46.6% on 0.5B and 35.7% on 3B. The 90th percentiles are 86.2% and 106.8%. "
        "Separate-probe medians are 81.2% and 98.7%, with 90th percentiles of 128.7% and 213.1%. A good pooled error does not imply accurate prediction for a typical or difficult batch.</p>",
        "<p>Balanced sign accuracy on category contrasts is 88.5% and 89.6% for fixed-probe dictionaries, versus 90.6% and 92.5% for SVD12. "
        "On separate probes the dictionary reaches 73.6% and 70.4%, versus 75.8% and 73.2% for SVD12. These are descriptive results; sign prediction is easier than calibrated magnitude prediction. "
        "The common category shift accounts for 39.3% and 37.5% of observed squared change in this fresh dataset; it was explicitly removed from primary scoring.</p>",
        "<p>At the primary rate, one closed-QA batch accounts for 97.0% of the 0.5B full-gradient forecast's squared contrast error. "
        "For 3B, three batches account for 98.6%. These are error-concentration diagnostics on all retained cases; no outlier was removed or used to retune the experiment.</p>",
        "<h2>Do atoms isolate a behavior?</h2>",
        removal_plot,
        "<p>For each of 24 preset pure batches, we remove the OMP8 atom with the largest predicted target-category contrast on calibration probes. "
        "We compare this with an equal-norm random component and a rescaling of the full update to the same remaining norm. "
        "Atom removal does not have a clear target-share advantage over rescaling. The random-control effects are extremely small, around 3×10⁻⁷ nats per token in mean absolute target change, so their share ratios are sensitive to numerical precision.</p>",
    ]
    removal_rows = []
    for model in models:
        rem = diagnostics["results"][model]["removal"]
        for control, label in (
            ("random_removed", "Random removal"),
            ("rescaled_full", "Norm-matched rescaling"),
        ):
            v = rem["atom_minus_control"][control]
            bounds = v["crossed_seed_batch_probe_95ci"]
            removal_rows.append(
                [
                    model.split("-")[-1],
                    label,
                    f"{100 * v['difference']:+.2f} pp",
                    f"[{100 * bounds[0]:+.2f}, {100 * bounds[1]:+.2f}] pp",
                ]
            )
    body += [
        table(
            ["Model", "Comparator", "Atom target-share difference", "Descriptive 95% interval"],
            removal_rows,
        ),
        "<p>The small fixed-probe errors for predicting atom-removal effects should be read as local derivative accuracy, not evidence of semantic selectivity. "
        "Separate-probe prediction is much weaker. The secondary intervals above were computed after completion for diagnosis; they do not alter the six primary decisions.</p>",
        "<h2>Saved generation examples</h2>",
        "<p>The generation check used two prespecified brainstorming batches, eight prompts, five seeds and learning rate 0.001 per model. "
        "Only 2/80 full-update continuations changed on 0.5B and 4/80 on 3B. Atom removal changed 0/80 and 2/80 relative to the full update. "
        "These are exact-text comparisons, with no blinded semantic grading. They do not show a reliable improvement or a controllable semantic behavior.</p>",
    ]
    examples = json.loads((root / "generation-examples.json").read_text())["examples"]
    for e in examples:
        body.append(
            "<details><summary>"
            + html.escape(f"{e['job']} | {e['batch_id']} | {e['probe_group']}")
            + "</summary>"
        )
        for name in ("prompt", "baseline", "full", "atom_removed"):
            body.append(
                f"<h3>{name.replace('_', ' ').capitalize()}</h3><pre>{html.escape(e[name])}</pre>"
            )
        body.append("</details>")
    body += [
        "<h2>What this adds to the papers</h2>",
        "<p><a href='https://arxiv.org/abs/2603.14665'>Gradient Atoms</a> motivates unsupervised sparse decomposition of training gradients and weight-space steering. "
        "Our extension forecasts the magnitude of a batch update, locks the predictions before intervention and compares with matched compression and independent probes. "
        "It uses SVD rather than the paper's EKFAC preconditioning and is not an exact replication. The normalization and dictionary estimator follow "
        "<a href='https://github.com/jrosseruk/gradient_atoms/tree/a83858557b411249097861f25d57eb13020eb334'>the pinned upstream implementation</a>. "
        "The full-gradient local derivative baseline is related to the gradient inner products used in <a href='https://arxiv.org/abs/2002.08484'>TracIn</a>, without claiming multi-checkpoint influence estimation.</p>",
        "<p>The fresh evaluation source is <a href='https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a'>Databricks Dolly 15k</a>, "
        "CC-BY-SA-3.0. Fitting data are <a href='https://huggingface.co/datasets/HuggingFaceH4/no_robots'>No Robots</a>, CC-BY-NC-4.0. Source texts and complete qualitative outputs remain in ignored local/remote artifacts; published numeric results contain no dataset text.</p>",
        "<p>The earlier LoRAcle comparison measured source-task decoding, and the residual SAE comparisons measured reconstruction. Neither is evidence of predicting behavioral effect magnitude. "
        "They were not repeated in this confirmation. Old-study results informed the frozen methods and are development evidence, not an additional confirmation.</p>",
        "<h2>Scope and reproducibility</h2>",
        "<p>The defensible positive claim is narrow: this learned dictionary outperforms its rotated control on category-contrast forecasting for one model. "
        "The evidence does not support dictionary superiority over SVD, robust probe transfer, semantic selectivity, full-weight updates, other model families or longer training trajectories. "
        "All effects use short reference-answer prefixes and a role-labelled prompt template. Five seeds and one fresh dataset limit the population claims.</p>",
        "<p>Run code: <a href='https://github.com/nicolesplaining/gradient-interp/tree/6b48a38f4ba6c4c8ba85a051471e77c5c2b0679b/src/gradient_interp'>6b48a38</a>. "
        "Frozen protocol: <a href='https://github.com/nicolesplaining/gradient-interp/tree/0d91184/results/confirmation-protocol'>0d91184</a>. "
        "Primary analysis: <a href='https://github.com/nicolesplaining/gradient-interp/blob/6b48a38/src/gradient_interp/confirmation_analysis.py'>confirmation_analysis.py</a>. "
        "The original experiment files and frozen analysis were not edited after results. The remote archive remains at /home/ubuntu/gradient-interp/runs/confirmation.</p>",
    ]
    css = """body{font-family:system-ui,sans-serif;color:#24323b;max-width:1120px;margin:40px auto;padding:0 28px;line-height:1.6}h1{font-size:38px;line-height:1.15}h2{margin-top:44px;border-top:1px solid #dde3e7;padding-top:22px}h3{font-size:15px}.subtitle,.caption{color:#66757f}.lead{font-size:21px;background:#eef5f1;padding:22px;border-left:4px solid #187e58}img{width:100%;height:auto;margin:20px 0}.table{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3e6}th{background:#edf2f5}a{color:#166994}details{border:1px solid #dce3e6;padding:12px;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;font-family:inherit;background:#f5f7f8;padding:12px;font-size:14px} @media print{body{margin:0}details{break-inside:avoid}}"""
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Gradient forecasting confirmation</title><style>"
        + css
        + "</style><body>"
        + "\n".join(body)
        + "</body></html>"
    )
    (out / "results.html").write_text(document)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = diagnose(args.root)
    print(render(args.root, diagnostics))


if __name__ == "__main__":
    main()
