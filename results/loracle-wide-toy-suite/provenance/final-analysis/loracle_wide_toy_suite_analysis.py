"""Guarded comparison of matched short and wide checkpoint differences.

Judgments are supplied by direct reading, never inferred by this module.
"""

import argparse
import itertools
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loracle_toy_suite_analysis import group_behavior, read, verify_local
from .loracle_toy_suite_chance import PAIRS, exact_null, holm, matrix_for
from .loracle_toy_suite_data import SEEDS, TASKS
from .loracle_toy_suite_llm_judge import counts, load_locked, response_hash, semantic_match
from .study import atomic_json, file_hash


def prepare_deck(root, work):
    verify_local(root)
    if (work / "deck.json").exists():
        raise ValueError("Masked deck already prepared")
    work.mkdir(parents=True, exist_ok=True)
    predictions = read(root / "predictions.json")
    cases = {c["id"]: c for c in read(root / "data.json")["cases"]}
    if len(predictions) != 484:
        raise ValueError("Require all484 interpretations")
    order = list(range(len(predictions)))
    random.Random(2026101511).shuffle(order)
    deck, mapping = [], []
    for i, row_index in enumerate(order, 1):
        r = predictions[row_index]
        ident = f"W{i:03d}"
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
            "scientific_scope": (
                "Human-directed wider-gap prospective development; "
                "judgments remain masked until lock."
            ),
        },
    )


def paired_gap_test(short, wide):
    # Keep all seeds and both orders in each task pair together.
    matrix_for(short)
    matrix_for(wide)

    def by_key(rows):
        for r in rows:
            if r["match"] != semantic_match(r["labels"], r["status"], r["target"]):
                raise ValueError("Semantic score disagrees with locked labels")
        return {(r["seed"], r["target"]): r for r in rows}

    a, b = by_key(short), by_key(wide)
    if a.keys() != b.keys():
        raise ValueError("Short and wide cases are not paired")
    diffs = [
        sum(
            int(b[s, t]["match"] == "full") - int(a[s, t]["match"] == "full")
            for s in SEEDS
            for t in pair
        )
        for pair in PAIRS
    ]
    observed = sum(diffs)
    values = [
        sum(sign * d for sign, d in zip(signs, diffs, strict=True))
        for signs in itertools.product((-1, 1), repeat=4)
    ]
    return {
        "null": "paired_gap_swap_by_four_task_pairs",
        "observed_difference": observed,
        "short_correct": sum(r["match"] == "full" for r in short),
        "wide_correct": sum(r["match"] == "full" for r in wide),
        "n": 24,
        "pair_differences": diffs,
        "null_values": values,
        "permutations": 16,
        "tail_count": sum(v >= observed for v in values),
        "p_one_sided": sum(v >= observed for v in values) / 16,
    }


def join_judgments(root, work):
    data, predictions = read(root / "data.json"), read(root / "predictions.json")
    cases = {c["id"]: c for c in data["cases"]}
    rows = load_locked(work, predictions, cases)
    groups = group_behavior(read(root / "behavior.json"))
    for r in rows:
        if r["window"] == "short":
            r["target"] = cases[r["case"]]["target"]
            r["match"] = semantic_match(r["labels"], r["status"], r["target"])
        if r["target"] is None:
            r["strong_manipulation"] = False
            continue
        start = f"initial-{r['seed']}" if r["window"] == "first" else r["trajectory"] + "-1"
        suffix = {"first": "1", "short": "short", "adjacent": "2"}[r["window"]]
        end = r["trajectory"] + "-" + suffix
        before, after = groups[start, r["target"]], groups[end, r["target"]]
        r.update(
            before_behavior=before,
            after_behavior=after,
            control_before=groups[start, "control"],
            control_after=groups[end, "control"],
            strong_manipulation=after["target"] >= 0.8
            and after["target"] - before["target"] >= 0.5,
        )
    return rows, groups


def summarize(rows):
    groups = []
    for query, encoding, window in itertools.product(
        ("choice", "description"), ("released", "correct_svd"), ("first", "short", "adjacent")
    ):
        subset = [
            r for r in rows if (r["query"], r["encoding"], r["window"]) == (query, encoding, window)
        ]
        groups.append(
            {
                "query": query,
                "encoding": encoding,
                "window": window,
                **counts(subset),
                "strong": counts([r for r in subset if r["strong_manipulation"]]),
                "by_task": {t: counts([r for r in subset if r["target"] == t]) for t in TASKS},
                "by_seed": {str(s): counts([r for r in subset if r["seed"] == s]) for s in SEEDS},
            }
        )
    selected = {
        w: [
            r for r in rows if (r["query"], r["encoding"], r["window"]) == ("choice", "released", w)
        ]
        for w in ("short", "adjacent")
    }
    matrix = matrix_for(selected["adjacent"])
    tests = [exact_null(matrix, kind) for kind in ("all_categories", "within_training_pairs")]
    tests.append(paired_gap_test(selected["short"], selected["adjacent"]))
    for t, p in zip(tests, holm([t["p_one_sided"] for t in tests]), strict=True):
        t["p_holm_three"] = p
    controls = []
    for window, encoding, query in itertools.product(
        ("cumulative", "permuted", "zero"), ("released", "correct_svd"), ("choice", "description")
    ):
        subset = [
            r for r in rows if (r["window"], r["encoding"], r["query"]) == (window, encoding, query)
        ]
        controls.append(
            {
                "window": window,
                "encoding": encoding,
                "query": query,
                "n": len(subset),
                "statuses": dict(Counter(r["status"] for r in subset)),
                "claimed_categories": dict(Counter(t for r in subset for t in r["labels"])),
                "single_correct_target": None,
            }
        )
    return {"groups": groups, "tests": tests, "controls": controls}


def figures(summary, rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained", sharey=True)
    for i, query in enumerate(("choice", "description")):
        for j, encoding in enumerate(("released", "correct_svd")):
            ax = axes[i, j]
            for shift, window, label, color in [
                (-0.19, "short", "20 updates", "#71869b"),
                (0.19, "adjacent", "100 updates", "#137e83"),
            ]:
                group = next(
                    g
                    for g in summary["groups"]
                    if (g["query"], g["encoding"], g["window"]) == (query, encoding, window)
                )
                ax.bar(
                    np.arange(8) + shift,
                    [group["by_task"][t]["full"] for t in TASKS],
                    width=0.36,
                    label=label,
                    color=color,
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
            ax.set_ylabel("Full category matches / 3 seeds")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Does a wider checkpoint gap help LoRAcle identify the new task?")
    fig.savefig(output / "recognition.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharey=True)
    for ax, window, name in zip(axes, ("short", "adjacent"), ("20 → 40", "20 → 120"), strict=True):
        selected = [
            r
            for r in rows
            if (r["query"], r["encoding"], r["window"]) == ("choice", "released", window)
        ]
        for ti, task in enumerate(TASKS):
            for si, s in enumerate(SEEDS):
                r = next(r for r in selected if r["target"] == task and r["seed"] == s)
                x = ti + (si - 1) * 0.14
                a, b = r["before_behavior"]["target"] * 100, r["after_behavior"]["target"] * 100
                ax.plot([x, x], [a, b], color="#b5c3ce", lw=2)
                ax.scatter(x, a, s=15, color="#71869b")
                ax.scatter(x, b, s=25, color="#137e83" if r["strong_manipulation"] else "#b66c39")
        ax.set(xticks=range(8), xticklabels=TASKS, ylim=(-4, 105), title=f"Steps {name}")
        ax.tick_params(axis="x", labelrotation=40)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Actual target behavior on 16 held-out prompts (%)")
    fig.suptitle(
        "What the target model actually learned; gray points are the same step-20 baseline"
    )
    fig.savefig(output / "learning.png", dpi=160)
    plt.close(fig)


def run(root, work, output):
    verify_local(root)
    if (output / "execution.json").exists():
        raise ValueError("Analysis already executed; preserve original results")
    output.mkdir(exist_ok=True, parents=True)
    atomic_json(
        output / "execution.json",
        {
            "started_unix": time.time(),
            "analysis_source_sha256": file_hash(__file__),
            "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
            "judge_lock_sha256": file_hash(work / "judgments-lock.json"),
        },
    )
    rows, behavior = join_judgments(root, work)
    result = summarize(rows)
    result["behavior"] = [{"condition": c, "task": t, **v} for (c, t), v in behavior.items()]
    atomic_json(output / "summary.json", result)
    atomic_json(output / "judged-records.json", rows)
    figures(result, rows, output)
    # Narrative/HTML is written after inspecting the entire locked result, not selected cases.
    print(
        json.dumps({"primary_comparison": result["tests"][-1], "tests": result["tests"]}, indent=2)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare-deck", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.action == "prepare-deck":
        prepare_deck(args.root, args.work)
    else:
        if args.output is None:
            p.error("--output is required for analyze")
        run(args.root, args.work, args.output)


if __name__ == "__main__":
    main()
