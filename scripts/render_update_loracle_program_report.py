"""Present completed development evidence without changing any scientific analysis."""

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/update-loracle-program"


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def main():
    roots = {v: ROOT / f"results/update-loracle-development-v{v}" for v in range(1, 6)}
    summaries = {v: read(r / "summary.json") for v, r in roots.items()}
    v6 = read(ROOT / "results/update-loracle-development-v6/summary.json")
    stability = read(OUT / "label-stability.json")["summary"]
    s = summaries[5]
    cases = read(roots[5] / "all-development-cases.json")
    actual = [c for c in cases if not c["control"]]
    assert len(actual) == 64 and len(cases) == 72
    assert all(not x["all_development_gates_pass"] for x in summaries.values())
    methods = [
        ("Released LoRAcle", "released", "Weights and physical norm"),
        ("Training majority", "factorized_majority", "Training labels only"),
        ("Fine-tuned LoRAcle, v2", "epoch1", "Weights and physical norm"),
        ("Physical-update regression, v3", "physical_profile", "Weight changes"),
        ("Linear calibration, v4", "sensitivity", "Base model and fixed prompt library"),
        ("Exact calibration, v5", "functional", "Base model and fixed prompt library"),
        ("Exact calibration, shuffled input", "functional_shuffled", "Norm-matched donor update"),
    ]
    values = [100 * s["metrics"][k]["actual_update_accuracy"] for _, k, _ in methods]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(
        np.arange(len(methods)), values, color=["#8796a2"] * 4 + ["#347a9b", "#20617f", "#b87c39"]
    )
    ax.set(
        yticks=np.arange(len(methods)),
        yticklabels=[m[0] for m in methods],
        xlim=(0, 100),
        xlabel="Full format + case + order correct (%)",
    )
    ax.invert_yaxis()
    ax.axvline(70, color="#555", linestyle="--", label="Original development criterion: 70%")
    for i, (_, key, _) in enumerate(methods):
        n = s["metrics"][key]["actual_update_correct"]
        ax.text(values[i] + 1, i, f"{n}/64", va="center")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Same 64 development updates; no method meets the full criterion")
    fig.tight_layout()
    fig.savefig(OUT / "development-comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [
        "All four prompts agree",
        "Two-prompt halves agree",
        "One prompt agrees with four-prompt label",
    ]
    for offset, split, name, color in (
        (-0.18, "train", "Training", "#8796a2"),
        (0.18, "development", "Development", "#21658c"),
    ):
        st = stability[split]
        counts = [
            st["all_four_singletons_agree"],
            st["half_pair_agreements"],
            st["singleton_gold_matches"],
        ]
        denominators = [
            st["actual_cases"],
            st["half_pair_denominator"],
            st["singleton_gold_denominator"],
        ]
        y = [100 * n / d for n, d in zip(counts, denominators, strict=True)]
        ax.barh(np.arange(3) + offset, y, height=0.34, label=name, color=color)
        for i, (n, d, v) in enumerate(zip(counts, denominators, y, strict=True)):
            ax.text(v + 1, i + offset, f"{n}/{d}", va="center", fontsize=10)
    ax.set(
        yticks=np.arange(3),
        yticklabels=labels,
        xlim=(0, 100),
        xlabel="Agreement (%)",
        title="Same update, different evaluation prompts",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "label-stability.png", dpi=180)
    plt.close(fig)

    target = np.asarray([c["measured"]["effect"]["centered_improvements"] for c in actual])
    profiles = [
        np.asarray([c[k]["profile"] for c in actual])
        for k in ("sensitivity_decoder", "functional_decoder")
    ]
    low = min(target.min(), *(p.min() for p in profiles))
    high = max(target.max(), *(p.max() for p in profiles))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, predicted, name in zip(
        axes, profiles, ("Linear calibration", "Exact calibration"), strict=True
    ):
        nr = np.sqrt(np.mean((predicted - target) ** 2) / np.mean(target**2))
        ax.scatter(target.ravel(), predicted.ravel(), s=6, alpha=0.25)
        ax.plot([low, high], [low, high], "--", color="#555")
        ax.set(
            xlim=(low, high),
            ylim=(low, high),
            xlabel="Measured effect on separate probes",
            title=f"{name}: relative RMSE {nr:.3f}",
        )
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted centered loss improvement")
    fig.suptitle("All 2,048 pattern/update pairs; nats per token; shared axes")
    fig.tight_layout()
    fig.savefig(OUT / "effect-size-comparison.png", dpi=180)
    plt.close(fig)

    rows = [
        [
            name,
            f"{s['metrics'][key]['actual_update_correct']}/64",
            f"{s['metrics'][key]['zero_correct']}/8",
            access,
        ]
        for name, key, access in methods
    ]
    c = actual[0]
    example = table(
        ["Method", "Predicted complete pattern"],
        [[name, c["exact_comparison_predictions"][key]] for name, key, _ in methods]
        + [["Measured on four separate probes", c["label"]]],
    )
    groups = []
    for pattern, data in s["strata"]["source_pattern"].items():
        groups.append(
            [pattern]
            + [
                f"{data[k]['correct']}/{data[k]['n']}"
                for k in ("released", "epoch1", "sensitivity", "functional")
            ]
        )
    fixed = []
    for pattern in sorted(s["strata"]["source_pattern"]):
        x = next(c for c in actual if c["source_pattern"] == pattern)
        fixed.append([x["id"], pattern, x["functional_decoder"]["prediction"], x["label"]])
    links = " ".join(
        f"<a href='../update-loracle-development-v{v}/report.html'>v{v} full report</a>"
        for v in range(1, 6)
    )
    report = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interpreting one optimizer update: evidence so far</title>
<style>body{font:17px/1.6 system-ui,sans-serif;color:#203039;background:#fafaf8;
max-width:1100px;margin:40px auto;padding:0 24px 70px}
h1{line-height:1.15;font-size:38px}
h2{margin-top:42px;font-size:25px}
img{width:100%;background:white}
table{border-collapse:collapse;width:100%;font-size:15px;margin:22px 0}
td,th{padding:10px;text-align:left;border-bottom:1px solid #d9dedf;vertical-align:top}
th{background:#eef2f3}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f3;
padding:20px;font:14px/1.5 ui-monospace,monospace}
a{color:#176288}
nav a{display:inline-block;margin-right:20px}
.status{border-left:5px solid #b87c39;padding:10px 20px;background:#fff5e5}
.small{font-size:14px;color:#4b5b65}
</style>
<h1>Can we describe what one optimizer update changes?</h1>
<p class="status">No strong generalization result yet. Six development recipes are complete.
The learned LoRAcle variants did not meet the original criterion. The latest base-assisted
method gets 16 of 64 full descriptions right. Reserved and fresh confirmation sets remain
untested.</p>
<p>We want to read a small weight change and say which behavior it makes more likely. The test uses
Qwen3-14B and one AdamW update on four examples, with learning rate 0.0002. We update a rank-16
LoRA adapter across all seven module types. There are 560 trainable tensors and 64,225,280
parameters.
These are first updates from fresh adapters, not arbitrary steps later in training.</p>
<h2>What counts as a correct description?</h2>
<p>We score 32 possible reference-answer patterns on four separate prompts, before and after
the update.
The patterns combine four layouts, two letter cases and four orderings. The target is the complete
pattern whose mean completion loss improves most relative to the other patterns. A correct layout
with the wrong case or order does not count as a full match. Loss includes the original end
token.</p>
<p>This is a measurement of likelihood assigned to fixed answers. It does not establish how often
the model would freely generate that behavior, or whether its answers are generally correct.</p>
<h2>All completed development methods</h2>
<img src="development-comparison.png" alt="All main methods fall below 70% complete-pattern
accuracy">
"""
        + table(
            ["Method", "Actual updates correct", "Zero controls correct", "Inference access"], rows
        )
        + """
<p>The first fine-tuned recipe, v1, reached 12/64 at its selected epoch. It used a single 33-choice
question. V2 uses four component questions, and all three of its saved epochs reached 13/64.
The plot uses the matched four-question released baseline for v2 onward. Choosing an epoch on
development is not independent confirmation.</p>
<p>The exact-calibration advantage over released LoRAcle is 23.44 percentage points, with a
descriptive
paired 95% interval of -1.56 to 54.69 points. All methods reuse the same 64 development updates,
so these are not five independent replications. All 64 updates exceed the fixed effect floor.</p>
<h2>What changed in the latest test?</h2>
<p>V4 estimates the loss change from a gradient dot product. V5 actually applies the saved update
and scores the same eight fixed calibration prompts. Both then predict effects on separate prompts.
The calibration prompts and words were fixed before v4 and are distinct from the evaluation
probes.</p>
<img src="effect-size-comparison.png" alt="Exact calibration reduces loss-profile error but
still makes substantial transfer errors">
<p>Exact scoring reduces relative RMSE from 1.817 to 0.734. One means the error of always predicting
zero effect. Full-description accuracy stays at 16/64: two errors become correct and two formerly
correct predictions become wrong. Every predicted category is lowercase, while 19 measured
categories
are uppercase. Removing the linear approximation alone does not solve transfer.</p>
<p>These calibration methods require the pinned base model and a task-specific prompt library.
They are separate base-assisted predictors, not improved learned LoRAcle checkpoints. Their extra
access and compute must accompany any comparison with LoRAcle.</p>
<h2>An actual input and output</h2>
<p>The first development case is shown without selecting it for success. One of its four training
examples was:</p><pre>"""
        + html.escape(c["batch"][0]["prompt"] + "\n\nTraining answer:\n" + c["batch"][0]["answer"])
        + """</pre>
<p>We took one optimizer step on that batch. The interpreter received the weight-change encoding,
its physical norm and the question. It did not receive those four training examples or their
task label.
Here is the actual formatting question:</p><pre>"""
        + html.escape(c["questions"][1]["text"])
        + """</pre>
<p>Released and fine-tuned LoRAcle receive 4,480 direction tokens of width 5,120, plus text.
We compare option-token likelihoods rather than judging freely written explanations.
The exact-calibration predictor instead scores its fixed library with the updated base model.</p>
<p>One of the four separate evaluation prompts was:</p><pre>"""
        + html.escape(c["probes"][0]["prompt"])
        + """</pre>
"""
        + example
        + """
<p>The measured bullet/lowercase/reverse pattern improves by 0.27469 nats per token relative to
the average pattern change. Its lead over the runner-up is only 0.01342. A small lead can change
the winning category even when the overall loss profile is reasonably close. We retain such
cases.</p>
<h2>Failures across every development family</h2>
"""
        + table(
            ["Training composition", "Released", "Fine-tuned v2", "Linear v4", "Exact v5"], groups
        )
        + """
<p>Composition names describe the training batch; measured effect labels may differ.
The following examples take the first case in each family, with no outcome-based selection.</p>
"""
        + table(
            ["Case", "Batch composition", "Exact-calibration prediction", "Measured effect"], fixed
        )
        + """
<h2>What remains unproven</h2>
<p>No method has passed the original 70% development accuracy criterion. We have therefore not
opened the reserved test or fresh confirmation run. A strong result still requires both independent
grids, complete-category accuracy, superiority over the same released baseline, zero and shuffled
controls, and the predeclared uncertainty bounds. Partial-axis gains do not replace that test.</p>
<p>The latest audit passed all 344 native-update records and 688 own/shuffled scoring conditions,
including four independent rescoring checks, source and output hashes, restoration and deadlines.
The scientific run and automatic audit are complete. No worker will be resumed.</p>
<nav>"""
        + links
        + """<a href="protocol.json">Original protocol</a>
<a href="../update-loracle-development-v5/interpretation.json">Latest interpretation</a></nav>
<p class="small">This page presents the existing audits and results. It introduces no new endpoint,
filter, fit or experiment. All raw predictions and full cases remain in the linked
reports.</p></html>"""
    )
    assert v6["metrics"]["transfer"]["actual_update_correct"] == 16
    assert not v6["all_development_gates_pass"]
    extra = """<h2>One learned transfer map also fails the full criterion</h2>
<p>V6 fits a fixed ridge regression from calibration loss profiles to evaluation loss profiles
using only the original 272 training cases. It improves relative RMSE from 0.734 to 0.703, but
the same 16/64 full descriptions remain correct. All eight zero controls are correct.
A norm-matched shuffled update gets 5/64. The penalty was fixed before fitting
and was not retuned.</p>
<p>This uses saved scores and required no new GPU work. It remains a base-assisted method.
<a href="../update-loracle-development-v6/report.html">Full v6 results and all cases</a>;
<a href="../update-loracle-development-v6/interpretation.json">v6 decision</a>.</p>
<h2>The measured label depends on the evaluation prompts</h2>
<img src="label-stability.png" alt="Low agreement among prompt subsets for the same update">
<p>In a retrospective check of all 64 development updates, the three complementary splits of
four probes into two-prompt halves agree on the full label only 75/192 times, or 39.1%.
All four individual probes agree for only 4/64 updates. Training probes show similar variation.</p>
<p>These overlapping comparisons are dependent. They are not a prediction ceiling or an excuse
to drop difficult cases. The original four-probe targets and success criteria stay unchanged.</p>
<p>The earlier interpreter did not receive those evaluation prompts. A new comparison is
running with the four prompt texts supplied to both the trained interpreter and released
LoRAcle. Reference answers, measured effects and training-batch labels remain hidden.
The training recipe and full-category success criteria are unchanged. There are no results
from this comparison yet. The evidence above establishes prompt dependence; it does not
establish that adding context will improve prediction.</p>
<p><a href="label-stability.json">Every diagnostic case and count</a>;
<a href="provenance/label-stability-diagnostic-plan.json">Diagnostic plan</a>;
<a href="next-development.json">Current study and its fixed comparison</a>.</p>
"""
    report = report.replace(
        "<h2>What remains unproven</h2>", extra + "<h2>What remains unproven</h2>"
    )
    (OUT / "report.html").write_text(report)


if __name__ == "__main__":
    main()
