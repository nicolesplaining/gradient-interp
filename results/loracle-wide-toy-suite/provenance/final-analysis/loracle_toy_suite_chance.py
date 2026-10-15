"""Retrospective exact, task-clustered null tests of locked semantic scores."""

import argparse
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loracle_toy_suite_analysis import read, verify_local
from .loracle_toy_suite_data import SEEDS, TASKS
from .study import atomic_json, file_hash

PAIRS = (("numbered", "bullets"), ("json", "uppercase"), ("reverse", "sort"), ("capitals", "diets"))


def matrix_for(rows):
    """Invalid answers never match; keep all24 cases in the denominator."""
    if len(rows) != 24 or {r["seed"] for r in rows} != set(SEEDS):
        raise ValueError("Require24 cases from all three frozen seeds")
    if any(sorted(r["target"] for r in rows if r["seed"] == s) != sorted(TASKS) for s in SEEDS):
        raise ValueError("Require all eight task categories once per seed")
    matrix = np.zeros((8, 8), dtype=np.int64)
    for r in rows:
        if r["status"] == "specific" and len(r["labels"]) == 1:
            matrix[TASKS.index(r["target"]), TASKS.index(r["labels"][0])] += 1
    return matrix


def exact_null(matrix, kind):
    """One category mapping shared by all seeds, never independent seed shuffles."""
    if matrix.shape != (8, 8) or np.any(matrix < 0):
        raise ValueError("Expected nonnegative8x8 count matrix")
    if kind == "all_categories":
        mappings = itertools.permutations(range(8))
    elif kind == "within_training_pairs":
        mappings = []
        for bits in itertools.product((False, True), repeat=4):
            p = list(range(8))
            for swap, pair in zip(bits, PAIRS, strict=True):
                if swap:
                    a, b = (TASKS.index(t) for t in pair)
                    p[a], p[b] = p[b], p[a]
            mappings.append(p)
    else:
        raise ValueError("Unknown null")
    scores = np.array([sum(int(matrix[i, p[i]]) for i in range(8)) for p in mappings])
    observed = int(np.trace(matrix))
    tail = int(np.count_nonzero(scores >= observed))
    hist = np.bincount(scores, minlength=25)
    return {
        "null": kind,
        "observed_correct": observed,
        "n": 24,
        "null_mean_correct": float(scores.mean()),
        "null_95_percentile_range": np.quantile(scores, [0.025, 0.975]).tolist(),
        "tail_count": tail,
        "permutations": len(scores),
        "p_one_sided": tail / len(scores),
        "null_histogram_counts": hist.tolist(),
        "correct_by_task": dict(zip(TASKS, np.diag(matrix).tolist(), strict=True)),
        "confusion_counts": matrix.tolist(),
    }


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    result = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        if not 0 <= pvalues[i] <= 1:
            raise ValueError("Invalid probability")
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        result[i] = min(1.0, running)
    return result


def draw(tests, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for index, t in enumerate(tests):
        wi, ni = divmod(index, 2)
        ax = axes[ni, wi]
        y = np.array(t["null_histogram_counts"]) / t["permutations"]
        ax.bar(np.arange(len(y)), y, width=0.85, color="#879ca9", label="Shuffled labels")
        ax.axvline(
            t["observed_correct"],
            color="#bd5340",
            lw=2,
            label=f"Observed {t['observed_correct']}/24",
        )
        ax.set(
            xlim=(-0.6, 15.6),
            xticks=range(0, 16, 2),
            xlabel="Correct category matches out of 24",
            ylabel="Null probability",
        )
        window = "Initial training" if t["window"] == "first" else "Later change"
        null = "All eight category labels" if ni == 0 else "Only swap within training pairs"
        ax.set_title(f"{window} · {null}\nHolm-adjusted p = {t['p_holm_four']:.4g}", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Could these semantic scores arise by chance?\n"
        "Exact label shuffles shared across all three seeds",
        fontsize=13,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def section(tests):
    table = (
        "<table><tr><th>Window</th><th>Null</th><th>Observed</th><th>Null mean</th>"
        "<th>Raw p</th><th>Holm p, four tests</th></tr>"
    )
    for t in tests:
        name = "All eight categories" if t["null"] == "all_categories" else "Within training pairs"
        table += (
            f"<tr><td>{t['window']}</td><td>{name}</td>"
            f"<td>{t['observed_correct']}/24</td><td>{t['null_mean_correct']:.3g}/24</td>"
            f"<td>{t['p_one_sided']:.5g}</td><td>{t['p_holm_four']:.5g}</td></tr>"
        )
    return (
        """<section id='chance-check'><h2>Are the semantic scores above chance?</h2>
<p>This separate analysis was requested after the scores were known. All existing judgments stay
fixed. The main test keeps LoRAcles' exact answer preferences, shuffles the eight task labels,
and applies the same shuffle to all three seeds. It enumerates all 40,320 assignments. This
avoids treating three repetitions of the same toy task as independent problems.</p>
<p>The second test preserves each broad training pair and only swaps its two labels. For example,
it asks whether an answer distinguishes fictional capitals from fictional creature diets, beyond
recognizing that general domain. It enumerates 16 assignments. Both use full category matches;
vague, ambiguous and abstaining answers remain failures.</p>
"""
        + table
        + """</table><img src='chance-check.png' alt='Exact null distributions and observed scores'>
<p>Holm correction covers both windows under both nulls, four tests total. The small within-pair
permutation set limits power. The label-shuffle null assumes category exchangeability under no
association. These curated toy categories were not a random population sample or random assignment
of meaning. This is exploratory evidence conditional on that null, not independent confirmation.
The assistant had seen some earlier results before judging. A low p-value does not establish
faithful explanations, accurate details, or generalization to new training tasks.</p>
<p><a href='chance-check.json'>Exact counts and p-values</a> ·
<a href='provenance/chance-check/plan.json'>Plan recorded before new calculations</a> ·
<a href='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html'>
Exact permutation convention</a> ·
<a href='https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html'>
Holm correction</a></p></section>"""
    )


def run(root, output, work):
    verify_local(root)
    plan = read(work / "plan.json")
    records_path = output / "llm-judge-records.json"
    if file_hash(records_path) != plan["input_sha256"]:
        raise ValueError("Semantic records changed after analysis plan")
    if (work / "execution.json").exists():
        raise ValueError("Chance check already executed")
    history = work / "before-analysis"
    history.mkdir()
    for n in ("report.html", "interpretation.json"):
        shutil.copy2(output / n, history / n)
    atomic_json(
        work / "execution.json",
        {
            "started_unix": time.time(),
            "source_sha256": file_hash(__file__),
            "plan_sha256": file_hash(work / "plan.json"),
            "records_sha256": file_hash(records_path),
            "judge_lock_sha256": file_hash(output / "provenance/llm-judge/judgments-lock.json"),
        },
    )
    shutil.copy2(__file__, work / "executed-source.py")
    all_rows = read(records_path)
    tests, selected = [], []
    for window in ("first", "adjacent"):
        rows = [
            r
            for r in all_rows
            if (r["encoding"], r["query"], r["window"]) == ("released", "choice", window)
        ]
        matrix = matrix_for(rows)
        if int(np.trace(matrix)) != sum(r["match"] == "full" for r in rows):
            raise ValueError("Locked semantic score mismatch")
        selected.extend(rows)
        for kind in ("all_categories", "within_training_pairs"):
            tests.append(
                {
                    "window": window,
                    **exact_null(matrix, kind),
                    "correct_by_seed": {
                        str(s): sum(r["match"] == "full" for r in rows if r["seed"] == s)
                        for s in SEEDS
                    },
                }
            )
    for t, p in zip(tests, holm([t["p_one_sided"] for t in tests]), strict=True):
        t.update(p_holm_four=p, passes_exploratory_005=p <= 0.05)
    assert tests[0]["permutations"] == math.factorial(8) and tests[1]["permutations"] == 16
    result = {"created_unix": time.time(), "tests": tests, "plan": plan, "rows": selected}
    atomic_json(output / "chance-check.json", result)
    draw(tests, output / "chance-check.png")
    doc = (output / "report.html").read_text()
    doc = doc.replace(
        "<section><h2>What went in</h2>", section(tests) + "<section><h2>What went in</h2>"
    )
    doc = doc.replace(
        "These counts make no new claim about statistical significance.",
        "A separately requested exploratory chance check is reported below.",
    )
    (output / "report.html").write_text(doc)
    provenance = output / "provenance/chance-check"
    provenance.mkdir()
    for n in ("plan.json", "execution.json"):
        shutil.copy2(work / n, provenance / n)
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in t.items()
                    if k not in ("confusion_counts", "null_histogram_counts")
                }
                for t in tests
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.output, a.work)
