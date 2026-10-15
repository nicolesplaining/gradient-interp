"""Explicitly retrospective reading of sentence-form multiple-choice replies."""
# ruff: noqa: E501

import argparse
import html
import json
import re
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loracle_toy_suite_analysis import read, verify_local
from .loracle_toy_suite_data import TASKS
from .study import atomic_json, file_hash


def sentence_choice(text, options):
    """Do not infer generic descriptions or repair conflicting letters and wording."""
    patterns = {
        "numbered": r"\bnumbered lists?\b",
        "bullets": r"\bbullet(?:ed)? lists?\b",
        "json": r"\bjson arrays?\b",
        "uppercase": r"\buppercase\b",
        "reverse": r"\breverse(?:d)? order\b",
        "sort": r"\balphabetical(?:ly)?\b",
        "capitals": r"\bcapital\w*\b.*\b(?:invented|fictional)\b|\b(?:invented|fictional)\b.*\bcapital\w*\b",
        "diets": r"\bcreatures?\b.*\b(?:eat\w*|diet\w*|food)\b|\b(?:diet\w*|food)\b.*\bcreatures?\b",
        "unknown": r"\bno clear change\b",
    }
    matches = {name for name, pattern in patterns.items() if re.search(pattern, text, re.I | re.S)}
    letter = re.match(r"\s*([A-I])[.)](?:\s|$)", text)
    if letter:
        matches.add(options[letter[1]])
    if len(matches) != 1:
        return None, "ambiguous" if matches else "unrecognized"
    return next(iter(matches)), "explicit_letter_or_phrase"


def review(root, output):
    verify_local(root)
    amendment = root / "response-format-analysis-amendment.json"
    if not amendment.exists():
        raise ValueError("Require dated post-hoc amendment")
    if (root / "semantic-review-execution.json").exists():
        raise ValueError("Review already executed")
    atomic_json(
        root / "semantic-review-execution.json",
        {
            "started_unix": time.time(),
            "source_sha256": file_hash(__file__),
            "amendment_sha256": file_hash(amendment),
            "strict_analyzer_sha256": file_hash(
                Path(__file__).with_name("loracle_toy_suite_analysis.py")
            ),
        },
    )
    data, predictions = read(root / "data.json"), read(root / "predictions.json")
    strict = read(output / "summary.json")
    cases = {c["id"]: c for c in data["cases"]}
    choices = {(r["case"], r["encoding"]): r for r in predictions if r["query"] == "choice"}
    rows = []
    for r in strict["rows"]:
        case = cases[f"{r['trajectory']}-{r['window']}"]
        reply = choices[case["id"], r["encoding"]]
        choice, status = sentence_choice(reply["response"], case["options"])
        rows.append(
            {
                **r,
                "strict_choice": r["choice"],
                "strict_correct": r["correct"],
                "choice": choice,
                "correct": choice == r["target"],
                "parse_status": status,
                "response": reply["response"],
            }
        )
    groups = []
    for encoding in ["released", "correct_svd"]:
        for window in ["first", "adjacent"]:
            selected = [r for r in rows if r["encoding"] == encoding and r["window"] == window]
            strong = [r for r in selected if r["strong_manipulation"]]
            groups.append(
                {
                    "encoding": encoding,
                    "window": window,
                    "correct": sum(r["correct"] for r in selected),
                    "total": len(selected),
                    "correct_strong": sum(r["correct"] for r in strong),
                    "strong": len(strong),
                    "unrecognized_or_ambiguous": sum(r["choice"] is None for r in selected),
                    "by_task": {
                        task: sum(r["correct"] for r in selected if r["target"] == task)
                        for task in TASKS
                    },
                    "by_seed": {
                        str(s): sum(r["correct"] for r in selected if r["seed"] == s)
                        for s in sorted({r["seed"] for r in selected})
                    },
                }
            )
    controls = []
    for r in predictions:
        if r["query"] != "choice" or not (r["case"] == "zero" or r["case"].endswith("permuted")):
            continue
        choice, status = sentence_choice(r["response"], cases[r["case"]]["options"])
        controls.append({**r, "sentence_choice": choice, "parse_status": status})
    result = {
        "scope": "Post-hoc descriptive interpretation of sentence replies; not the original primary endpoint; no new significance claim",
        "groups": groups,
        "rows": rows,
        "controls": controls,
    }
    atomic_json(output / "sentence-review.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, window in zip(axes, ["first", "adjacent"], strict=True):
        for offset, encoding, color in [
            (-0.18, "released", "#157d88"),
            (0.18, "correct_svd", "#b56947"),
        ]:
            g = next(g for g in groups if g["encoding"] == encoding and g["window"] == window)
            ax.bar(
                np.arange(8) + offset,
                [g["by_task"][t] for t in TASKS],
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
    axes[0].set_ylabel("Replies explicitly naming the trained task / 3 seeds")
    fig.suptitle(
        "Retrospective reading of sentence-form answers\nTask descriptions were offered in the question; this is assisted recognition",
        fontsize=12,
    )
    fig.savefig(output / "sentence-review.png", dpi=160)
    plt.close(fig)
    primary = groups[0]
    doc = (output / "report.html").read_text()
    start, end = doc.index("<h1>"), doc.index("<section>")
    intro = f"""<h1>LoRAcles on eight clear toy tasks: mixed recognition, unreliable details</h1>
<p>The target models learned clear changes in <b>21 of 24 initial phases</b> and <b>20 of 24 second phases</b>. LoRAcles' descriptions sometimes recognized the topic, but often invented unsupported content. Read the exact answers below.</p>
<p><b>The planned multiple-choice score failed as a measurement.</b> Every primary answer violated the exact single-letter format, including answers that plainly named the correct option. The original strict score remains 0/24. Its permutation p = 1 is degenerate and says nothing useful about chance-level task understanding.</p>
<p>A separately recorded, retrospective review accepts unambiguous task names or leading option letters in sentences. It identifies <b>{primary["correct"]}/24</b> initial tasks using the released input format, including <b>{primary["correct_strong"]}/{primary["strong"]}</b> cases with strong measured learning. These are descriptive results after inspecting the formatting failure, not a replacement primary result or a new significance claim.</p>
<p>This tests one released interpreter on Qwen3-14B toys. It supports neither “LoRAcles is useless” nor “it reliably explains every update.”</p>
<section><h2>What the sentences actually said</h2><img src='sentence-review.png'><table><tr><th>Encoding</th><th>Window</th><th>Explicit task matches / 24</th><th>Matches among strong learning</th><th>Unrecognized / ambiguous</th></tr>"""
    for g in groups:
        intro += (
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in [
                    g["encoding"],
                    g["window"],
                    g["correct"],
                    f"{g['correct_strong']}/{g['strong']}",
                    g["unrecognized_or_ambiguous"],
                ]
            )
            + "</tr>"
        )
    intro += "</table><p>All original responses and strict scores remain available. No model was retrained and no question was rerun. The sentence interpretation rule was recorded after reading the first seed's released-format answers. <a href='sentence-review.json'>All retrospective case scores</a> · <a href='provenance/response-format-analysis-amendment.json'>Dated scoring amendment</a></p></section>"
    doc = doc[:start] + intro + doc[end:]
    doc = doc.replace(
        "<img src='identification.png'><img src='confusion.png'>",
        "<p>The original one-letter scoring rejected all primary replies, so its zero scores cannot be read as zero semantic recognition. See the retrospective sentence review above.</p>",
    )
    doc = doc.replace(
        "<h2>All aggregate results</h2>",
        "<h2>Original strict-format results, retained unchanged</h2>",
    )
    (output / "report.html").write_text(doc)
    print(json.dumps(groups))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.root, args.output)
