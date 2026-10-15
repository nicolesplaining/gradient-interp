"""Join locked, directly read semantic judgments to audited toy-suite targets.

This module never infers labels from response text. The assistant supplied every
label and rationale in the separate, locked judgment artifact.
"""

import argparse
import hashlib
import html
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loracle_toy_suite_analysis import read, verify_local
from .loracle_toy_suite_data import TASKS
from .study import atomic_json, file_hash


def response_hash(response):
    return hashlib.sha256(response.encode()).hexdigest()


def semantic_match(labels, status, target):
    if target is None:
        return None
    if target not in labels:
        return "none"
    return "full" if labels == [target] and status == "specific" else "partial"


def load_locked(judge_root, predictions, cases):
    lock = read(judge_root / "judgments-lock.json")
    for name, digest in lock["files"].items():
        if Path(name).name != name or file_hash(judge_root / name) != digest:
            raise ValueError(f"Locked judge artifact changed: {name}")
    required = {"rubric.json", "deck.json", "mapping.json", "judgments.json"}
    if not required <= set(lock["files"]):
        raise ValueError("Incomplete judgment lock")
    rubric = read(judge_root / "rubric.json")
    if not rubric["created_unix"] < lock["created_unix"] <= time.time():
        raise ValueError("Invalid rubric/judgment chronology")
    values = [read(judge_root / n) for n in ("deck.json", "mapping.json", "judgments.json")]
    ids = [set(x["id"] for x in rows) for rows in values]
    n = len(predictions)
    if any(len(rows) != n or len(i) != n for rows, i in zip(values, ids, strict=True)):
        raise ValueError("Incomplete or duplicated judgment IDs")
    if not ids[0] == ids[1] == ids[2] or lock["n_judgments"] != n:
        raise ValueError("Judgment ID coverage differs")
    deck, mapping, judgments = [{x["id"]: x for x in rows} for rows in values]
    if {x["row_index"] for x in mapping.values()} != set(range(n)):
        raise ValueError("Mapping is not a bijection over prediction rows")
    rows = []
    for ident in sorted(deck):
        d, m, j = deck[ident], mapping[ident], judgments[ident]
        p = predictions[m["row_index"]]
        c = cases[p["case"]]
        if any(m[key] != p[key] for key in ("case", "encoding", "query")):
            raise ValueError("Mapping identity differs from original prediction")
        if d["query"] != p["query"] or d["response"] != p["response"]:
            raise ValueError("Deck text differs from original prediction")
        if p["query"] == "choice" and d["options"] != c["options"]:
            raise ValueError("Choice options changed in masked deck")
        digest = response_hash(p["response"])
        if not m["response_sha256"] == j["response_sha256"] == digest:
            raise ValueError("Response digest differs")
        labels, status = j["labels"], j["status"]
        if not set(labels) <= set(TASKS) or len(labels) != len(set(labels)):
            raise ValueError("Invalid judge labels")
        if status not in rubric["statuses"] or not j["rationale"].strip():
            raise ValueError("Invalid judge status/rationale")
        if bool(labels) != (status in ("specific", "ambiguous")):
            raise ValueError("Label/status contradiction")
        if len(labels) > 1 and status != "ambiguous":
            raise ValueError("Multiple specific categories require ambiguous status")
        target = c["target"] if c["window"] in ("first", "adjacent") else None
        rows.append(
            {
                **p,
                **j,
                "row_index": m["row_index"],
                "window": c["window"],
                "trajectory": c["trajectory"],
                "seed": c["seed"],
                "target": target,
                "match": semantic_match(labels, status, target),
            }
        )
    return rows


def counts(rows):
    return {
        "n": len(rows),
        "full": sum(r["match"] == "full" for r in rows),
        "partial": sum(r["match"] == "partial" for r in rows),
        "any_target": sum(r["match"] in ("full", "partial") for r in rows),
        "none": sum(r["match"] == "none" for r in rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
    }


def summarize(rows):
    groups = []
    for query in ("choice", "description"):
        for encoding in ("released", "correct_svd"):
            for window in ("first", "adjacent"):
                subset = [
                    r
                    for r in rows
                    if (r["query"], r["encoding"], r["window"]) == (query, encoding, window)
                ]
                groups.append(
                    {
                        "query": query,
                        "encoding": encoding,
                        "window": window,
                        **counts(subset),
                        "strong": counts([r for r in subset if r["strong_manipulation"]]),
                        "by_task": {
                            t: counts([r for r in subset if r["target"] == t]) for t in TASKS
                        },
                        "by_seed": {
                            str(s): counts([r for r in subset if r["seed"] == s])
                            for s in sorted({r["seed"] for r in subset})
                        },
                    }
                )
    controls = []
    for window in ("cumulative", "permuted", "zero"):
        for encoding in ("released", "correct_svd"):
            for query in ("choice", "description"):
                subset = [
                    r
                    for r in rows
                    if (r["window"], r["encoding"], r["query"]) == (window, encoding, query)
                ]
                controls.append(
                    {
                        "window": window,
                        "encoding": encoding,
                        "query": query,
                        "n": len(subset),
                        "statuses": dict(Counter(r["status"] for r in subset)),
                        "claimed_categories": dict(Counter(t for r in subset for t in r["labels"])),
                        "specific_category_claims": sum(bool(r["labels"]) for r in subset),
                        "any_content_claim": sum(r["status"] != "abstain" for r in subset),
                        "single_task_accuracy": None,
                    }
                )
    return {"groups": groups, "controls": controls}


def draw(groups, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True, layout="constrained")
    for qi, query in enumerate(("choice", "description")):
        for wi, window in enumerate(("first", "adjacent")):
            ax = axes[qi, wi]
            for off, encoding, color in (
                (-0.18, "released", "#167c87"),
                (0.18, "correct_svd", "#b66742"),
            ):
                g = next(
                    g
                    for g in groups
                    if (g["query"], g["encoding"], g["window"]) == (query, encoding, window)
                )
                full = [g["by_task"][t]["full"] for t in TASKS]
                partial = [g["by_task"][t]["partial"] for t in TASKS]
                ax.bar(np.arange(8) + off, full, width=0.35, color=color, label=encoding)
                ax.bar(
                    np.arange(8) + off,
                    partial,
                    bottom=full,
                    width=0.35,
                    color="white",
                    hatch="///",
                    edgecolor=color,
                )
            label = "Options supplied" if query == "choice" else "Open description"
            stage = "First training phase" if window == "first" else "Adjacent checkpoint change"
            ax.set(
                title=f"{label} · {stage}",
                xticks=range(8),
                xticklabels=TASKS,
                yticks=[0, 1, 2, 3],
                ylim=(0, 3.5),
            )
            ax.tick_params(axis="x", labelrotation=40)
            ax.grid(axis="y", alpha=0.2)
            if qi == 0 and wi == 0:
                ax.legend(fontsize=9, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recognized training categories / 3 seeds")
    fig.suptitle(
        "Direct LLM judging of saved answers\n"
        "Solid: full category match. Hatched: target named with alternatives.",
        fontsize=13,
    )
    fig.savefig(output / "llm-judge.png", dpi=160)
    plt.close(fig)


def table(groups):
    result = (
        "<table><tr><th>Question</th><th>Encoding</th><th>Window</th>"
        "<th>Full matches</th><th>Partial</th><th>Full among strong learning</th></tr>"
    )
    for g in groups:
        cells = [
            "Options supplied" if g["query"] == "choice" else "Open description",
            g["encoding"],
            g["window"],
            f"{g['full']}/{g['n']}",
            g["partial"],
            f"{g['strong']['full']}/{g['strong']['n']}",
        ]
        result += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in cells) + "</tr>"
    return result + "</table>"


def report(result, rows, original):
    groups = result["groups"]
    a, b = groups[:2]
    open_a, open_b = groups[4:6]
    intro = f"""<h1>LoRAcles toy tasks, scored for meaning</h1>
<p>I directly read and judged all 388 saved replies, including answers written as sentences.
The released encoding recognizes <b>{a["full"]}/24 initial training categories</b> and
<b>{b["full"]}/24 adjacent changes</b> when the question supplies the task options.
With only “Describe this model's behavior,” the corresponding full matches are
<b>{open_a["full"]}/24</b> and <b>{open_b["full"]}/24</b>.</p>
<p>The target model clearly learned the intended behavior in 21/24 initial phases and
20/24 second phases. Category recognition remains uneven even after fixing the format penalty.
Recognizing a topic does not establish that the additional details in an answer are true.</p>
<section><h2>Sentences now count</h2>
<p>“The model learned to return labels as a JSON array” counts as JSON. Generic
“sorting and organizing” is too vague. Alphabetical and reverse-alphabetical order are different
rules. Answers naming the target alongside alternatives get partial credit, shown separately.</p>
<p>The assistant supplied every label and rationale by reading the saved text.
The scoring code only joins those locked labels to the known tasks; it does not classify text.
No model was retrained and no response was regenerated.</p>
<img src='llm-judge.png' alt='Full and partial category matches by task, question and encoding'>
{table(groups)}
<p>This is a user-requested retrospective evaluation. Targets, seeds, windows and encodings
were hidden during the shuffled review, but I had already seen some earlier answers and results.
It is not an independent blind evaluation. The two encodings use the same trained models and
are not independent replications. These counts make no new claim about statistical significance.</p>
<p><a href='llm-judge-records.html'>Read all 388 judgments and reasons</a> ·
<a href='llm-judge-summary.json'>Full results, including each seed and task</a> ·
<a href='provenance/llm-judge/rubric.json'>Fixed judging rubric</a></p></section>"""
    # Retain all original scientific content after the previous sentence-review section.
    boundary = "<section><h2>What went in</h2>"
    if boundary not in original:
        raise ValueError("Expected original report structure missing")
    doc = original[: original.index("<h1>")] + intro + original[original.index(boundary) :]
    doc = doc.replace(
        "See the retrospective sentence review above.", "See the direct LLM judgment results above."
    )
    history = """<section><h2>Scoring history</h2>
<p>The original exact-letter parser rejected every answer, including correct sentences.
Its 0/24 score measures format compliance, not zero semantic understanding; its degenerate
permutation test does not answer whether topic recognition exceeds chance.</p>
<p>The later phrase-matching review remains available. Direct LLM judging changes
{disagreements} correct/incorrect decisions among the 96 scored choice replies across both encodings
and both windows. All differences in recognized categories are retained in the result file.
The new open-description scores were not included in that earlier phrase-matching analysis.</p>
<p><a href='summary.json'>Original strict results</a> ·
<a href='sentence-review.json'>Earlier phrase-matching review</a> ·
<a href='provenance/llm-judge/judgments-lock.json'>Judgments locked before joining targets</a> ·
<a href='provenance/llm-judge/execution.json'>Input and source hashes</a></p></section>""".format(
        disagreements=len(result["correctness_disagreements"])
    )
    return doc.replace("</body>", history + "</body>")


def records_html(rows):
    text = (
        "<!doctype html><html><head><meta charset='utf-8'><title>All LLM judgments</title>"
        "<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:40px auto;padding:20px}"
        "details{padding:14px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap;"
        "background:#f4f6f7;padding:12px}summary{cursor:pointer}</style></head><body>"
        "<h1>All 388 direct judgments</h1><p><a href='report.html'>Back to report</a>. "
        "Full and partial describe category recognition, not factual faithfulness. "
        "Cumulative and control answers have no single-task accuracy score.</p>"
    )
    for r in rows:
        match = r["match"] or "not single-task scored"
        title = f"{r['id']} · {r['case']} · {r['encoding']} · {r['query']} · {match}"
        target = r["target"] or "no single target"
        text += f"<details><summary>{html.escape(title)}</summary>"
        text += f"<p>Target: {target}. Recognized: {', '.join(r['labels']) or 'none'}. "
        text += f"Status: {r['status']}.</p><p>{html.escape(r['rationale'])}</p>"
        text += f"<pre>{html.escape(r['question'])}</pre><pre>{html.escape(r['response'])}</pre>"
        text += "</details>"
    return text + "</body></html>"


def run(root, output, judge_root):
    verify_local(root)
    execution_path = judge_root / "execution.json"
    if execution_path.exists():
        raise ValueError("Judgment scoring already executed; preserve results before any repair")
    data, predictions = read(root / "data.json"), read(root / "predictions.json")
    if len(predictions) != 388:
        raise ValueError("Require all 388 saved answers")
    cases = {c["id"]: c for c in data["cases"]}
    expected = {
        (c, e, q)
        for c in cases
        for e in ("released", "correct_svd")
        for q in ("description", "choice")
    }
    observed = {(p["case"], p["encoding"], p["query"]) for p in predictions}
    if observed != expected or len(observed) != len(predictions):
        raise ValueError("Require exact case/encoding/question coverage")
    rows = load_locked(judge_root, predictions, cases)
    strict, previous = read(output / "summary.json"), read(output / "sentence-review.json")
    strict_map = {(f"{r['trajectory']}-{r['window']}", r["encoding"]): r for r in strict["rows"]}
    previous_map = {
        (f"{r['trajectory']}-{r['window']}", r["encoding"]): r for r in previous["rows"]
    }
    for r in rows:
        key = (r["case"], r["encoding"])
        if r["window"] in ("first", "adjacent"):
            s = strict_map[key]
            if s["target"] != r["target"]:
                raise ValueError("Original target mismatch")
            r.update(
                strong_manipulation=s["strong_manipulation"],
                before_target=s["before_target"],
                after_target=s["after_target"],
            )
            if r["query"] == "choice":
                r.update(
                    original_strict_correct=s["correct"],
                    previous_regex_choice=previous_map[key]["choice"],
                    previous_regex_correct=previous_map[key]["correct"],
                )
    result = summarize(rows)
    choices = [r for r in rows if r["query"] == "choice" and r["target"] is not None]
    result.update(
        scope="Retrospective user-requested direct LLM judging; no new significance claim.",
        n_judgments=len(rows),
        correctness_disagreements=[
            r for r in choices if r["previous_regex_correct"] != (r["match"] == "full")
        ],
        category_disagreements=[
            r
            for r in choices
            if set(r["labels"])
            != (
                {r["previous_regex_choice"]}
                if r["previous_regex_choice"] not in (None, "unknown")
                else set()
            )
        ],
    )
    # One immutable source record, plus a recoverable copy of the prior displayed report.
    execution = {
        "created_unix": time.time(),
        "source_sha256": file_hash(__file__),
        "judgments_lock_sha256": file_hash(judge_root / "judgments-lock.json"),
        "input_sha256": {
            str(p): file_hash(p)
            for p in [
                root / "data.json",
                root / "predictions.json",
                output / "summary.json",
                output / "sentence-review.json",
                root / "integrity.json",
            ]
        },
    }
    history = root / "analysis-history" / "pre-llm-judge"
    history.mkdir(parents=True, exist_ok=False)
    for name in (
        "report.html",
        "interpretation.json",
        "summary.json",
        "sentence-review.json",
        "sentence-review.png",
        "learning.png",
    ):
        shutil.copy2(output / name, history / name)
    atomic_json(execution_path, execution)
    shutil.copy2(__file__, judge_root / "executed-source.py")
    atomic_json(output / "llm-judge-summary.json", result)
    atomic_json(output / "llm-judge-records.json", rows)
    draw(result["groups"], output)
    (output / "report.html").write_text(report(result, rows, (history / "report.html").read_text()))
    (output / "llm-judge-records.html").write_text(records_html(rows))
    provenance = output / "provenance" / "llm-judge"
    provenance.mkdir(parents=True, exist_ok=False)
    for name in [
        "judgments-lock.json",
        *read(judge_root / "judgments-lock.json")["files"],
        "execution.json",
    ]:
        shutil.copy2(judge_root / name, provenance / name)
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-root", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.output, args.judge_root)
