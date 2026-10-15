"""Fixed descriptive analysis of the one-trajectory LoRAcle toy."""
# ruff: noqa: E501

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .loracle_trajectory_toy import INTERVALS, STYLES, parse_prediction
from .study import atomic_json, file_hash


def directional_labels(changes, threshold=0.20):
    labels = {}
    for key, sign in [("more", 1), ("less", -1)]:
        ranked = sorted((sign * float(changes[style]), style) for style in STYLES)
        largest, style = ranked[-1]
        labels[key] = (
            style if largest >= threshold and largest > ranked[-2][0] + 1e-12 else "unknown"
        )
    return labels


def summarize(behavior, predictions):
    summary = {}
    for step in [0, 10, 20]:
        evaluation = [r for r in behavior if r["step"] == step and r["id"].startswith("eval-")]
        controls = [r for r in behavior if r["step"] == step and r["id"].startswith("control-")]
        if len(evaluation) != 24 or len(controls) != 8:
            raise ValueError("Incomplete behavior grid")
        summary[step] = {
            name: float(np.mean([r["checks"][name] for r in evaluation]))
            for name in [*STYLES, "content_order", "empty", "truncated"]
        }
        summary[step]["control_exact"] = float(
            np.mean([r["checks"]["exact_word"] for r in controls])
        )
    intervals = {}
    for case, (before, after) in INTERVALS.items():
        changes = {k: summary[after][k] - summary[before][k] for k in STYLES}
        actual = directional_labels(changes)
        variants = {}
        for convention in ["correct_svd", "upstream_compatible"]:
            rows = [
                r
                for r in predictions
                if r["case"] == case
                and r["convention"] == convention
                and r["query"] == "structured"
            ]
            if len(rows) != 1:
                raise ValueError("Incomplete prediction grid")
            prediction = parse_prediction(rows[0]["response"])
            variants[convention] = {"prediction": prediction, "matches": prediction == actual}
        intervals[case] = {
            "before": before,
            "after": after,
            "changes": changes,
            "measured_labels": actual,
            "encodings": variants,
        }
    manipulation = (
        intervals["case-k"]["changes"]["numbered_list"] >= 0.20
        and intervals["case-r"]["changes"]["bulleted_list"] >= 0.20
        and intervals["case-r"]["changes"]["numbered_list"] <= -0.20
    )
    quality = all(
        summary[s]["content_order"] >= 23 / 24
        and summary[s]["truncated"] == 0
        and summary[s]["control_exact"] >= summary[0]["control_exact"] - 1 / 8
        for s in [10, 20]
    )
    zero = [
        p
        for p in predictions
        if p["case"] == "case-z" and p["convention"] == "correct_svd" and p["query"] == "structured"
    ]
    if len(zero) != 1:
        raise ValueError("Missing zero control")
    zero_ok = parse_prediction(zero[0]["response"]) == {"more": "unknown", "less": "unknown"}
    correct = all(intervals[c]["encodings"]["correct_svd"]["matches"] for c in ["case-k", "case-r"])
    return {
        "checkpoint_rates": summary,
        "intervals": intervals,
        "manipulation_check": manipulation,
        "quality_check": quality,
        "primary_adjacent_predictions_correct": correct,
        "zero_control_correct": zero_ok,
        "primary_toy_support": manipulation and quality and correct and zero_ok,
    }


def analyze(root, output):
    root, output = Path(root), Path(output)
    audit = json.loads((root / "integrity.json").read_text())
    if not audit["passed"] or not audit["controller_complete_and_dead"]:
        raise ValueError("A genuine completed integrity audit is required")
    for name, digest in audit["compact_files_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Export changed: {name}")
    behavior = json.loads((root / "behavior.json").read_text())
    predictions = json.loads((root / "predictions.json").read_text())
    scores = json.loads((root / "scores.json").read_text())
    data = json.loads((root / "data.json").read_text())
    result = summarize(behavior, predictions)
    result["scope"] = (
        "Single toy trajectory, descriptive only; no individual-update or magnitude claim."
    )
    result["reference_ce"] = {
        str(step): {
            style: float(
                np.mean(
                    [
                        r["mean_answer_ce"]
                        for r in scores
                        if r["step"] == step and r["style"] == style
                    ]
                )
            )
            for style in STYLES
        }
        for step in [0, 10, 20]
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "results.json", result)
    for name in ["predictions", "behavior", "scores", "data", "export-audit"]:
        (output / f"{name}.json").write_bytes((root / f"{name}.json").read_bytes())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    labels = ["Before training", "After numbered phase", "After bullet phase"]
    x = np.arange(3)
    for i, (style, color) in enumerate(zip(STYLES, ["#326bb0", "#ce713b", "#7b838d"])):
        axes[0].bar(
            x + (i - 1) * 0.24,
            [100 * result["checkpoint_rates"][s][style] for s in [0, 10, 20]],
            0.24,
            label=style.replace("_", " "),
            color=color,
        )
        axes[1].plot(
            x,
            [result["reference_ce"][str(s)][style] for s in [0, 10, 20]],
            "o-",
            label=style.replace("_", " "),
            color=color,
        )
    for ax in axes:
        ax.set_xticks(x, labels, rotation=16, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(
        ylabel="Responses with this format (%)",
        ylim=(0, 105),
        title="Actual responses on 24 unseen prompts",
    )
    axes[1].set(
        ylabel="Mean answer loss (nats/token)", title="Fixed reference answers; lower is better"
    )
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(output / "trajectory.png", dpi=180)
    fig.savefig(output / "trajectory.pdf")
    plt.close(fig)
    esc = html.escape
    verdict = (
        "The frozen toy criteria passed."
        if result["primary_toy_support"]
        else "The frozen toy criteria did not pass."
        if result["manipulation_check"]
        else "The two-phase training manipulation was inconclusive."
    )
    rows = ""
    for case, interval in result["intervals"].items():
        actual = interval["measured_labels"]
        for convention, outcome in interval["encodings"].items():
            rows += f"<tr><td>{case}: {interval['before']} → {interval['after']}</td><td>{convention}</td><td>{esc(json.dumps(actual))}</td><td>{esc(json.dumps(outcome['prediction']))}</td><td>{outcome['matches']}</td></tr>"
    descriptions = "".join(
        f"<details><summary>{esc(p['case'])} · {esc(p['convention'])} · {esc(p['query'])}</summary><p><b>Question</b> {esc(p['question'])}</p><pre>{esc(p['response'])}</pre></details>"
        for p in predictions
    )
    ids = [r["id"] for r in data["evaluation"] if int(r["id"].split("-")[1]) % 6 < 2]
    examples = "".join(
        f"<details><summary>{esc(r['id'])} · step {r['step']}</summary><p>{esc(r['prompt'])}</p><pre>{esc(r['response'])}</pre><p>{esc(json.dumps(r['checks']))}</p></details>"
        for r in behavior
        if r["id"] in ids or r["id"].startswith("control-")
    )
    sample = data["train"][0]
    from .loracle_trajectory_toy import answer

    text = f"""<!doctype html><html><head><meta charset="utf-8"><title>Reading a training trajectory with LoRAcles</title><style>body{{font:17px/1.6 system-ui;color:#233143;background:#f9fafb;max-width:1150px;margin:40px auto;padding:0 25px}}h1{{font-size:34px}}h2{{margin-top:35px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #cbd5e1}}pre{{white-space:pre-wrap;background:#edf1f6;padding:16px}}details{{border-bottom:1px solid #cbd5e1;padding:10px}}.lead{{font-size:21px}}a{{color:#235aa0}}</style></head><body>
    <h1>Can LoRAcles read changes during training?</h1><p class="lead">{verdict}</p>
    <p>One Qwen3-14B model, one 20-step run, and a frozen LoRAcle interpreter. Ten steps teach numbered lists; the next ten teach bullets using the same content. We save checkpoints at steps 0, 10 and 20. The interpreter receives their weight differences and a formatting question. It does not receive the training examples or phase labels. All descriptions were locked before evaluating the saved models.</p>
    <h2>What went in</h2><p>Example training prompt:</p><pre>{esc(sample["prompt"])}</pre><p>First-phase answer:</p><pre>{esc(answer(sample["values"], "numbered_list"))}</pre><p>Second-phase answer:</p><pre>{esc(answer(sample["values"], "bulleted_list"))}</pre>
    <p>Each step uses four examples. The 24 evaluation prompts have different wording and disjoint label vocabulary. Eight single-word controls check for formatting spillover.</p>
    <h2>What the model actually changed</h2><img src="trajectory.png" alt="Actual formatting and reference-answer losses across checkpoints"><p>Greedy responses, one per prompt. These are descriptive rates from this fixed toy, without population uncertainty estimates. Format detectors can both fire on a mixed response. Reference losses score specified answers, not generated-answer quality.</p>
    <pre>{esc(json.dumps(result["checkpoint_rates"], indent=2))}</pre>
    <h2>Did the weight descriptions match?</h2><table><tr><th>Interval</th><th>Encoding</th><th>Measured more / less</th><th>LoRAcle more / less</th><th>Match</th></tr>{rows}</table>
    <p>Labels use a frozen 20 percentage-point threshold. Ties and smaller changes are unknown. The mathematically correct SVD is primary. The second encoding retains a right-singular-vector convention in the pinned upstream code and is reported separately.</p>
    <p>Training manipulation passed: {result["manipulation_check"]}. Content/control quality passed: {result["quality_check"]}. Primary adjacent predictions both correct: {result["primary_adjacent_predictions_correct"]}. Zero-update control correct: {result["zero_control_correct"]}.</p>
    <h2>Every LoRAcle answer, including controls</h2><p>case-k = steps 0–10; case-r = 10–20; case-v = 0–20. case-z has zero direction tokens. case-p permutes the second interval's tokens while preserving their values. A permutation does not erase all information, so it is a layout control rather than a guaranteed meaningless update.</p>{descriptions}
    <h2>Fixed response examples</h2><p>First two evaluation prompts in each template, plus all single-word controls, at every checkpoint. The complete responses and scores are linked below.</p>{examples}
    <h2>What this can establish</h2><p>This is a test of two ten-step windows, not every gradient update. AdamW uses optimizer history. Adjacent effective matrix differences can have rank 32; the interpreter receives only the leading 16 directions per module. The interpreter does not receive the starting checkpoint, and the same update can affect different starting models differently. Its normalization removes absolute positive scaling, so this does not test effect-size prediction. Even a successful toy would need broader validation.</p>
    <p><a href="results.json">All summaries</a> · <a href="predictions.json">All interpreter responses</a> · <a href="behavior.json">All generated responses and token IDs</a> · <a href="scores.json">All reference losses</a> · <a href="data.json">All input examples</a> · <a href="export-audit.json">Exact-difference checks and retained energy</a></p></body></html>"""
    (output / "report.html").write_text(text)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)


if __name__ == "__main__":
    main()
