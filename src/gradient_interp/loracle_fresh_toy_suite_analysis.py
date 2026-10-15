"""Compare fresh segment adapters with locked prior checkpoint-difference judgments."""

import argparse
import itertools
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loracle_toy_suite_analysis import read, verify_local
from .loracle_toy_suite_chance import exact_null, holm, matrix_for
from .loracle_toy_suite_data import SEEDS, TASKS
from .loracle_toy_suite_llm_judge import counts, response_hash
from .loracle_wide_toy_suite_analysis import join_judgments, paired_gap_test
from .study import atomic_json, file_hash


def verify_baseline(root):
    plan = read(root / "analysis-plan.json")
    for name, digest in plan["baseline_files_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"Previously locked baseline changed: {name}")
    old = Path("runs/loracle-wide-toy-suite")
    verify_local(old)
    return old


def prepare_deck(root, work):
    verify_local(root)
    verify_baseline(root)
    if (work / "deck.json").exists():
        raise ValueError("Masked deck already exists")
    work.mkdir(parents=True, exist_ok=True)
    predictions = read(root / "predictions.json")
    cases = {c["id"]: c for c in read(root / "data.json")["cases"]}
    if len(predictions) != 292 or len(cases) != 73:
        raise ValueError("Require the full interpretation grid")
    order = list(range(len(predictions)))
    random.Random(2026101517).shuffle(order)
    deck, mapping = [], []
    for i, row_index in enumerate(order, 1):
        r = predictions[row_index]
        ident = f"F{i:03d}"
        deck.append(
            {
                "id": ident,
                "query": r["query"],
                "response": r["response"],
                "options": cases[r["case"]]["options"] if r["query"] == "choice" else None,
            }
        )
        mapping.append(
            {
                "id": ident,
                "row_index": row_index,
                **{k: r[k] for k in ("case", "encoding", "query")},
                "response_sha256": response_hash(r["response"]),
            }
        )
    shutil.copy2(root / "judge-rubric.json", work / "rubric.json")
    atomic_json(work / "deck.json", deck)
    atomic_json(work / "mapping.json", mapping)
    atomic_json(
        work / "preparation.json",
        {
            "created_unix": time.time(),
            "predictions_sha256": file_hash(root / "predictions.json"),
            "deck_sha256": file_hash(work / "deck.json"),
            "mapping_sha256": file_hash(work / "mapping.json"),
            "n": len(deck),
            "scope": "Fresh-adapter development; target, encoding and window hidden until lock.",
        },
    )


def select(rows, query="choice", encoding="released", window="adjacent"):
    return [
        r for r in rows if (r["query"], r["encoding"], r["window"]) == (query, encoding, window)
    ]


def paired_method_test(continued, fresh):
    result = paired_gap_test(continued, fresh)
    # Strengthen pairing beyond the eight-task/three-seed validation inherited above.
    previous = {(r["seed"], r["target"]): r for r in continued}
    for r in fresh:
        p = previous[r["seed"], r["target"]]
        if p["trajectory"] != r["trajectory"] or p["question"] != r["question"]:
            raise ValueError("The old and new cases do not share trajectory and question")
    result["null"] = "paired_method_swap_by_four_task_pairs"
    result["continued_correct"] = result.pop("short_correct")
    result["fresh_correct"] = result.pop("wide_correct")
    result["difference_percentage_points"] = result["observed_difference"] / 24 * 100
    result["discordant"] = {
        "new_only": sum(
            r["match"] == "full" and previous[r["seed"], r["target"]]["match"] != "full"
            for r in fresh
        ),
        "old_only": sum(
            r["match"] != "full" and previous[r["seed"], r["target"]]["match"] == "full"
            for r in fresh
        ),
    }
    return result


def summarize(fresh, continued):
    groups, controls = [], []
    for method, rows in (("fresh", fresh), ("continued", continued)):
        for query, encoding, window in itertools.product(
            ("choice", "description"), ("released", "correct_svd"), ("short", "adjacent")
        ):
            subset = select(rows, query, encoding, window)
            if len(subset) != 24:
                raise ValueError("Incomplete method/encoding/query/window grid")
            matrix_for(subset)
            groups.append(
                {
                    "method": method,
                    "query": query,
                    "encoding": encoding,
                    "window": window,
                    **counts(subset),
                    "strong": counts([r for r in subset if r["strong_manipulation"]]),
                    "by_task": {t: counts([r for r in subset if r["target"] == t]) for t in TASKS},
                    "by_seed": {
                        str(s): counts([r for r in subset if r["seed"] == s]) for s in SEEDS
                    },
                }
            )
        for window, encoding, query in itertools.product(
            ("permuted", "zero"), ("released", "correct_svd"), ("choice", "description")
        ):
            subset = select(rows, query, encoding, window)
            controls.append(
                {
                    "method": method,
                    "window": window,
                    "encoding": encoding,
                    "query": query,
                    "n": len(subset),
                    "statuses": dict(Counter(r["status"] for r in subset)),
                    "claimed_categories": dict(Counter(t for r in subset for t in r["labels"])),
                    "single_correct_target": None,
                }
            )
    matrix = matrix_for(select(fresh))
    tests = [exact_null(matrix, k) for k in ("all_categories", "within_training_pairs")]
    tests.append(paired_method_test(select(continued), select(fresh)))
    for t, p in zip(tests, holm([t["p_one_sided"] for t in tests]), strict=True):
        t["p_holm_three"] = p
    comparisons = []
    for query, encoding, window in itertools.product(
        ("choice", "description"), ("released", "correct_svd"), ("short", "adjacent")
    ):
        a, b = select(continued, query, encoding, window), select(fresh, query, encoding, window)
        comparisons.append(
            {
                "query": query,
                "encoding": encoding,
                "window": window,
                "primary": (query, encoding, window) == ("choice", "released", "adjacent"),
                "continued": counts(a),
                "fresh": counts(b),
                "difference_full": counts(b)["full"] - counts(a)["full"],
            }
        )
    return {"groups": groups, "tests": tests, "comparisons": comparisons, "controls": controls}


def figures(summary, fresh, continued, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained", sharey=True)
    for i, query in enumerate(("choice", "description")):
        for j, encoding in enumerate(("released", "correct_svd")):
            ax = axes[i, j]
            for shift, method, label, color in [
                (-0.19, "continued", "Checkpoint difference", "#71869b"),
                (0.19, "fresh", "Fresh task-B adapter", "#137e83"),
            ]:
                g = next(
                    g
                    for g in summary["groups"]
                    if (g["method"], g["query"], g["encoding"], g["window"])
                    == (method, query, encoding, "adjacent")
                )
                ax.bar(
                    np.arange(8) + shift,
                    [g["by_task"][t]["full"] for t in TASKS],
                    width=0.36,
                    color=color,
                    label=label,
                )
            ax.set(
                xticks=range(8),
                xticklabels=TASKS,
                yticks=[0, 1, 2, 3],
                ylim=(0, 3.5),
                title=(
                    f"{'Options supplied' if query == 'choice' else 'Open description'}"
                    f" · {encoding}"
                ),
            )
            ax.tick_params(axis="x", labelrotation=40)
            ax.set_ylabel("Full matches / 3 initializations")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("LoRAcle recognition after 100 task-B updates")
    fig.savefig(output / "recognition.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained", sharey=True)
    for i, (method, rows) in enumerate(
        (("Continued adapter", continued), ("Fresh adapter", fresh))
    ):
        for j, (window, steps) in enumerate((("short", 20), ("adjacent", 100))):
            ax = axes[i, j]
            selected = select(rows, window=window)
            for ti, task in enumerate(TASKS):
                for si, seed in enumerate(SEEDS):
                    r = next(r for r in selected if r["target"] == task and r["seed"] == seed)
                    x = ti + (si - 1) * 0.14
                    a, b = r["before_behavior"]["target"] * 100, r["after_behavior"]["target"] * 100
                    ax.plot([x, x], [a, b], color="#b5c3ce", lw=2)
                    ax.scatter(x, a, s=15, color="#71869b")
                    ax.scatter(
                        x, b, s=25, color="#137e83" if r["strong_manipulation"] else "#b66c39"
                    )
            ax.set(
                xticks=range(8),
                xticklabels=TASKS,
                ylim=(-4, 105),
                title=f"{method}, {steps} B updates",
            )
            ax.tick_params(axis="x", labelrotation=40)
            ax.grid(axis="y", alpha=0.2)
            ax.set_ylabel("Target task accuracy (%)")
    fig.suptitle("Actual learning; gray starting points share the same task-A checkpoint")
    fig.savefig(output / "learning.png", dpi=160)
    plt.close(fig)


def run(root, work, output):
    verify_local(root)
    old = verify_baseline(root)
    if (output / "execution.json").exists():
        raise ValueError("Analysis already executed; retain first outputs")
    output.mkdir(exist_ok=True, parents=True)
    # Joining labels requires the complete lock, before reading behavior in the inherited join.
    fresh, new_behavior = join_judgments(root, work)
    continued, old_behavior = join_judgments(old, old / "llm-judge")
    if len(fresh) != 292 or len(continued) != 484:
        raise ValueError("Incomplete judgment grids")
    atomic_json(
        output / "execution.json",
        {
            "started_unix": time.time(),
            "analysis_source_sha256": file_hash(__file__),
            "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
            "judge_lock_sha256": file_hash(work / "judgments-lock.json"),
            "baseline_judge_lock_sha256": file_hash(old / "llm-judge/judgments-lock.json"),
        },
    )
    result = summarize(fresh, continued)
    result["behavior"] = [
        {"method": method, "condition": c, "task": t, **v}
        for method, values in (("fresh", new_behavior), ("continued", old_behavior))
        for (c, t), v in values.items()
    ]
    atomic_json(output / "summary.json", result)
    atomic_json(output / "judged-records.json", fresh)
    atomic_json(output / "baseline-judged-records.json", continued)
    figures(result, fresh, continued, output)
    print(result["tests"][-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare-deck", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare-deck":
        prepare_deck(args.root, args.work)
    else:
        if args.output is None:
            parser.error("--output is required for analyze")
        run(args.root, args.work, args.output)


if __name__ == "__main__":
    main()
