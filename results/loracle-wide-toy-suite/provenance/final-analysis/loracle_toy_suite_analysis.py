"""Guarded analysis of the fixed LoRAcle toy suite."""
# ruff: noqa: E501

import argparse
import csv
import html
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loracle_toy_suite_data import SEEDS, TASKS
from .study import atomic_json, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def label_permutation_test(rows, repetitions=10000, seed=2026101500):
    """Preserve predictions and eight-task balance within each initialization seed."""
    if len(rows) != 24 or any(
        sorted(r["target"] for r in rows if r["seed"] == s) != sorted(TASKS) for s in SEEDS
    ):
        raise ValueError("Permutation test requires all eight tasks in all three seeds")
    observed = sum(r["choice"] == r["target"] for r in rows)
    rng = np.random.default_rng(seed)
    null = np.zeros(repetitions, dtype=int)
    for i in range(repetitions):
        for s in SEEDS:
            mapping = dict(zip(TASKS, rng.permutation(TASKS), strict=True))
            null[i] += sum(r["choice"] == mapping[r["target"]] for r in rows if r["seed"] == s)
    return {
        "observed_correct": observed,
        "total": len(rows),
        "null_mean_correct": float(null.mean()),
        "null_quantiles": np.quantile(null, [0.025, 0.5, 0.975]).tolist(),
        "p_one_sided": (1 + int((null >= observed).sum())) / (repetitions + 1),
        "repetitions": repetitions,
        "seed": seed,
    }, null


def group_behavior(behavior):
    groups = {}
    for r in behavior:
        groups.setdefault((r["condition"], r["task"]), []).append(r)
    return {
        key: {
            "n": len(rows),
            "target": sum(r["checks"]["target"] for r in rows) / len(rows),
            "content": sum(r["checks"]["content"] for r in rows) / len(rows),
            "truncated": sum(r["truncated"] for r in rows) / len(rows),
            "empty": sum(r["checks"]["empty"] for r in rows) / len(rows),
            "mean_tokens": float(np.mean([len(r["token_ids"]) for r in rows])),
        }
        for key, rows in groups.items()
    }


def summarize(data, behavior, predictions):
    groups = group_behavior(behavior)
    choices = {
        (r["case"], r["encoding"]): r["choice"] for r in predictions if r["query"] == "choice"
    }
    rows = []
    for t in data["trajectories"]:
        for phase, window in [(1, "first"), (2, "adjacent")]:
            task = t["tasks"][phase - 1]
            before = f"initial-{t['seed']}" if phase == 1 else f"{t['id']}-1"
            after = f"{t['id']}-{phase}"
            b, a = groups[before, task], groups[after, task]
            strong = a["target"] >= 0.8 and a["target"] - b["target"] >= 0.5
            for encoding in ["released", "correct_svd"]:
                choice = choices[f"{t['id']}-{window}", encoding]
                rows.append(
                    {
                        "trajectory": t["id"],
                        "seed": t["seed"],
                        "pair": t["pair"],
                        "order": t["order"],
                        "window": window,
                        "target": task,
                        "encoding": encoding,
                        "choice": choice,
                        "correct": choice == task,
                        "before_target": b["target"],
                        "after_target": a["target"],
                        "target_change": a["target"] - b["target"],
                        "strong_manipulation": strong,
                        "before_content": b["content"],
                        "after_content": a["content"],
                        "after_truncated": a["truncated"],
                        "after_mean_tokens": a["mean_tokens"],
                        "control_before": groups[before, "control"]["target"],
                        "control_after": groups[after, "control"]["target"],
                        "control_truncated_after": groups[after, "control"]["truncated"],
                    }
                )
    aggregate = []
    for encoding in ["released", "correct_svd"]:
        for window in ["first", "adjacent"]:
            selected = [r for r in rows if r["encoding"] == encoding and r["window"] == window]
            strong = [r for r in selected if r["strong_manipulation"]]
            aggregate.append(
                {
                    "encoding": encoding,
                    "window": window,
                    "correct": sum(r["correct"] for r in selected),
                    "total": len(selected),
                    "strong_count": len(strong),
                    "correct_strong": sum(r["correct"] for r in strong),
                    "unknown": sum(r["choice"] == "unknown" for r in selected),
                    "malformed": sum(r["choice"] is None for r in selected),
                    "by_task": {
                        task: {
                            "correct": sum(r["correct"] for r in selected if r["target"] == task),
                            "strong": sum(
                                r["strong_manipulation"] for r in selected if r["target"] == task
                            ),
                            "n": 3,
                        }
                        for task in TASKS
                    },
                    "by_seed": {
                        str(s): sum(r["correct"] for r in selected if r["seed"] == s) for s in SEEDS
                    },
                }
            )
    primary = [r for r in rows if r["encoding"] == "released" and r["window"] == "first"]
    permutation, null = label_permutation_test(primary)
    controls = [
        r
        for r in predictions
        if r["query"] == "choice" and (r["case"] == "zero" or r["case"].endswith("permuted"))
    ]
    control_counts = {
        encoding: {
            window: dict(
                Counter(
                    str(r["choice"])
                    for r in controls
                    if r["encoding"] == encoding and ((r["case"] == "zero") == (window == "zero"))
                )
            )
            for window in ["zero", "permuted"]
        }
        for encoding in ["released", "correct_svd"]
    }
    return {
        "aggregate": aggregate,
        "primary_permutation": permutation,
        "controls": control_counts,
        "rows": rows,
    }, null


def verify_local(root):
    integrity = read(root / "integrity.json")
    if not integrity.get("passed") or not integrity.get("processes_dead"):
        raise ValueError("Require complete remote integrity audit before analysis")
    if file_hash(root / "controller.json") != integrity["controller_sha256"]:
        raise ValueError("Completion receipt changed")
    if read(root / "controller.json")["status"] != "complete":
        raise ValueError("Incomplete controller")
    if file_hash(root / "manifest.json") != integrity["manifest_sha256"]:
        raise ValueError("Manifest changed")
    for name, digest in integrity["output_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Audited output changed: {name}")
    return integrity


def plots(summary, output):
    rows = summary["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, window in zip(axes, ["first", "adjacent"], strict=True):
        selected = [r for r in rows if r["encoding"] == "released" and r["window"] == window]
        for ti, task in enumerate(TASKS):
            samples = [r for r in selected if r["target"] == task]
            for si, r in enumerate(samples):
                x = ti + (si - 1) * 0.14
                ax.plot(
                    [x, x],
                    [r["before_target"] * 100, r["after_target"] * 100],
                    color="#b7c9cc",
                    lw=2,
                )
                ax.scatter(x, r["before_target"] * 100, color="#7a858c", s=17, zorder=3)
                ax.scatter(
                    x,
                    r["after_target"] * 100,
                    color="#157d88" if r["strong_manipulation"] else "#bc6c25",
                    s=28,
                    zorder=4,
                )
        ax.set(
            xticks=range(8),
            xticklabels=TASKS,
            title="Base → first checkpoint" if window == "first" else "First → second checkpoint",
            ylim=(-4, 105),
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.axhline(80, color="#777", ls=":", lw=0.8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Target behavior on 16 held-out prompts (%)")
    fig.suptitle(
        "Did the target model learn the intended behavior?\nGray: before · teal: strong change · orange: did not meet the fixed change criterion",
        fontsize=12,
    )
    fig.savefig(output / "learning.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, window in zip(axes, ["first", "adjacent"], strict=True):
        for offset, encoding, color in [
            (-0.18, "released", "#157d88"),
            (0.18, "correct_svd", "#b56947"),
        ]:
            group = next(
                g
                for g in summary["aggregate"]
                if g["encoding"] == encoding and g["window"] == window
            )
            ax.bar(
                np.arange(8) + offset,
                [group["by_task"][t]["correct"] for t in TASKS],
                width=0.35,
                color=color,
                label=encoding,
            )
        ax.set(
            xticks=range(8),
            xticklabels=TASKS,
            yticks=[0, 1, 2, 3],
            ylim=(0, 3.4),
            title="Base → first checkpoint" if window == "first" else "Adjacent update",
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Correct multiple-choice answers out of 3 seeds")
    fig.suptitle(
        "Could LoRAcles identify the task from the weights?\nAll cases retained; malformed answers and abstentions count incorrect",
        fontsize=12,
    )
    fig.savefig(output / "identification.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    selected = [r for r in rows if r["encoding"] == "released" and r["window"] == "first"]
    labels = [*TASKS, "unknown", "invalid"]
    matrix = np.zeros((8, 10), dtype=int)
    for r in selected:
        matrix[TASKS.index(r["target"]), labels.index(r["choice"] or "invalid")] += 1
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=3, aspect="auto")
    for i in range(8):
        for j in range(10):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] >= 2 else "black",
            )
    ax.set(
        xticks=range(10),
        xticklabels=labels,
        yticks=range(8),
        yticklabels=TASKS,
        ylabel="Trained task",
        xlabel="LoRAcles' answer",
        title="Primary result: released input format, base → first checkpoint",
    )
    ax.tick_params(axis="x", labelrotation=45)
    fig.savefig(output / "confusion.png", dpi=160)
    plt.close(fig)


def report(summary, data, behavior, predictions, output):
    def esc(x):
        return html.escape(str(x))

    primary = summary["aggregate"][0]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>LoRAcles: eight clear toy tasks</title><style>body{font:16px/1.6 system-ui;color:#20313d;background:#f7f8fa;max-width:1120px;margin:40px auto;padding:0 24px}h1,h2,h3{line-height:1.25}section{background:white;padding:24px;margin:24px 0;border:1px solid #dde3e7;border-radius:12px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f6;padding:12px}summary{cursor:pointer;font-weight:600}.scroll{overflow:auto}</style></head><body>",
        "<h1>Can LoRAcles read a clear training change?</h1>",
        f"<p>We trained Qwen3-14B on eight toy behaviors, using three adapter initializations and both orders within four task pairs. All 24 trajectories completed. The primary test identified <b>{primary['correct']} of 24 tasks</b> from base-to-first-checkpoint weights. {primary['strong_count']} of those 24 runs met the fixed strong-learning criterion; LoRAcles identified {primary['correct_strong']} of those.</p>",
        f"<p>The prespecified within-seed label-permutation test gave p = {summary['primary_permutation']['p_one_sided']:.4g}. This tests association with the eight task labels. It is not evidence that performance generalizes to other models, interpreter checkpoints or real training runs.</p>",
        "<section><h2>What went in</h2><p>The target was Qwen3-14B with rank-16 LoRA on all seven attention/MLP projections in all 40 layers. Each trajectory used 20 updates on one task, followed by 20 on its paired task. Each phase saw 40 fixed examples twice, four documents per update. The interpreter was the pinned released LoRAcles checkpoint, using the same Qwen base.</p><p>We gave the interpreter weights only, without training examples or task names. The open question was <code>Describe this model's behavior.</code> A separate assisted question offered eight task descriptions plus “no clear change,” with letters shuffled for each case. That assisted score is the primary numerical endpoint; it is not an open-ended explanation score.</p><p>For base-to-checkpoint inputs, we used the original rank-16 adapter tensors and the released tokenizer directly. For adjacent checkpoint differences, we encoded the exact effective matrix difference with at most rank 32, retaining the first 16 components per module. A secondary encoding fixes the released helper's right-singular-vector transpose. Both are reported, without choosing the better one after seeing results.</p></section>",
        "<section><h2>Training and interpretation are separate tests</h2><img src='learning.png'><p>A strong manipulation required at least 80% target behavior after training and an increase of at least 50 percentage points. A task that was not learned does not establish an interpretation failure. Formatting checks are deliberately narrow; fact checks look for the expected invented association. They do not measure general answer quality.</p><img src='identification.png'><img src='confusion.png'></section>",
        "<section><h2>All aggregate results</h2><table><tr><th>Input format</th><th>Window</th><th>Correct / 24</th><th>Strong learning</th><th>Correct among strong</th><th>Abstentions</th><th>Malformed</th></tr>",
    ]
    for g in summary["aggregate"]:
        parts.append(
            "<tr>"
            + "".join(
                f"<td>{esc(v)}</td>"
                for v in [
                    g["encoding"],
                    g["window"],
                    g["correct"],
                    g["strong_count"],
                    f"{g['correct_strong']} / {g['strong_count']}",
                    g["unknown"],
                    g["malformed"],
                ]
            )
            + "</tr>"
        )
    parts.append(
        "</table><p>The first checkpoint contains one trained task and is the clean primary test. The adjacent-update test is secondary. The cumulative final checkpoint may contain both tasks, so it has no forced single-category ground truth.</p></section>"
    )
    pred = {(r["case"], r["encoding"], r["query"]): r for r in predictions}
    beh = {(r["condition"], r["id"]): r for r in behavior}
    parts.append(
        "<section><h2>Fixed examples: every task in the first seed</h2><p>These are the first held-out prompt for every task in seed 2026101471, fixed before results. Both windows and input encodings are shown.</p>"
    )
    for t in data["trajectories"]:
        if t["seed"] != SEEDS[0]:
            continue
        for phase, window in [(1, "first"), (2, "adjacent")]:
            task = t["tasks"][phase - 1]
            probe = data["tasks"][task]["evaluation"][0]
            before = f"initial-{t['seed']}" if phase == 1 else f"{t['id']}-1"
            after = f"{t['id']}-{phase}"
            parts.append(
                f"<details><summary>{esc(task)} · {window} · {esc(t['id'])}</summary><p>Training example</p><pre>{esc(data['tasks'][task]['train'][0]['prompt'])}\n→ {esc(data['tasks'][task]['train'][0]['answer'])}</pre><p>Held-out prompt</p><pre>{esc(probe['prompt'])}</pre><p>Before</p><pre>{esc(beh[before, probe['id']]['response'])}</pre><p>After</p><pre>{esc(beh[after, probe['id']]['response'])}</pre>"
            )
            for encoding in ["released", "correct_svd"]:
                desc = pred[f"{t['id']}-{window}", encoding, "description"]
                choice = pred[f"{t['id']}-{window}", encoding, "choice"]
                parts.append(
                    f"<p>LoRAcles, {encoding}</p><pre>{esc(desc['question'])}\n\n{esc(desc['response'])}</pre><p>Multiple-choice answer: {esc(choice['response'])} → {esc(choice['choice'])}</p>"
                )
            parts.append("</details>")
    parts.append(
        "</section><section><h2>Zero and shuffled-weight controls</h2><p>Zero inputs carry no update. Shuffling preserves the individual direction vectors and may retain some task evidence, so it is not a guaranteed semantic null. Both controls may be outside the interpreter's training distribution.</p>"
    )
    first_permuted = next(c["id"] for c in data["cases"] if c["window"] == "permuted")
    for case in ["zero", first_permuted]:
        for encoding in ["released", "correct_svd"]:
            row = pred[case, encoding, "description"]
            choice = pred[case, encoding, "choice"]
            parts.append(
                f"<h3>{esc(case)} · {encoding}</h3><pre>{esc(row['response'])}</pre><p>Choice: {esc(choice['response'])} → {esc(choice['choice'])}</p>"
            )
    parts.append(
        "</section><section><h2>Limits and complete records</h2><p>These are development toys on one base model and one interpreter checkpoint. Three initializations reuse the same tasks and data; the 24 trajectories are not 24 independent model replications. The eight offered descriptions make recognition easier than open-ended explanation. Conditional scores among strong training cases are descriptive, and no failed training was rerun. We have not reproduced a published positive LoRAcles benchmark. Failures here therefore do not prove that the entire method is bad.</p><p>Interpreter answers were locked before behavior evaluation. All 388 answers, 2,328 task-scored records from 1,512 unique generations, 960 training updates, 72 weight-difference exports and 51 checkpoints passed the integrity audit. Exact duplicate prompts share a generation within a checkpoint. Capped and empty outputs remain in the results.</p><p><a href='summary.json'>All summaries and case scores</a> · <a href='case-scores.csv'>Case scores CSV</a> · <a href='predictions.json'>All interpreter responses</a> · <a href='behavior.json'>All target outputs</a> · <a href='provenance/analysis-plan.json'>Frozen analysis plan</a></p></section><section><details><summary>All interpreter answers</summary>"
    )
    for r in predictions:
        parts.append(
            f"<h3>{esc(r['case'])} · {esc(r['encoding'])} · {esc(r['query'])}</h3><pre>{esc(r['question'])}\n\n{esc(r['response'])}</pre>"
        )
    parts.append("</details></section></body></html>")
    (output / "report.html").write_text("\n".join(parts))


def run(root, output):
    integrity = verify_local(root)
    if (root / "analysis-execution.json").exists():
        raise ValueError("Analysis already executed; preserve the first analysis before revisions")
    atomic_json(
        root / "analysis-execution.json",
        {
            "started_unix": time.time(),
            "analysis_source_sha256": file_hash(__file__),
            "integrity_sha256": file_hash(root / "integrity.json"),
            "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        },
    )
    data, behavior, predictions = [
        read(root / name) for name in ["data.json", "behavior.json", "predictions.json"]
    ]
    summary, null = summarize(data, behavior, predictions)
    summary["integrity"] = integrity
    summary["retained_energy"] = {}
    for window in ["first", "adjacent", "cumulative"]:
        energies = [
            r["retained_energy_fraction"]
            for case, modules in read(root / "export-audit.json").items()
            if case.endswith(window)
            for r in modules
            if r["retained_energy_fraction"] is not None
        ]
        summary["retained_energy"][window] = {
            "min": min(energies),
            "mean": float(np.mean(energies)),
        }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", summary)
    np.save(root / "permutation-null.npy", null)
    with (output / "case-scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    for name in ["predictions.json", "behavior.json"]:
        (output / name).write_bytes((root / name).read_bytes())
    plots(summary, output)
    report(summary, data, behavior, predictions, output)
    print(
        json.dumps(
            {
                "aggregate": summary["aggregate"],
                "primary_permutation": summary["primary_permutation"],
                "controls": summary["controls"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
