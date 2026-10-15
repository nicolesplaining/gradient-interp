"""Render every answer from the separately audited shorter-prompt check."""
# ruff: noqa: E501

import argparse
import html
import json
from pathlib import Path

from gradient_interp.study import file_hash


def build(root, output):
    audit = json.loads((root / "integrity.json").read_text())
    if not audit["passed"] or not audit["worker_complete_and_dead"]:
        raise ValueError("Completed audit required")
    for name, digest in audit["files_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Export changed: {name}")
    plan = json.loads((root / "plan.json").read_text())
    rows = json.loads((root / "responses.json").read_text())
    old_root = Path("results/loracle-trajectory-toy")
    old_lock = json.loads((old_root / "provenance/predictions-lock.json").read_text())
    if file_hash(old_root / "predictions.json") != old_lock["predictions_sha256"]:
        raise ValueError("Original answers changed")
    old = {
        (x["case"], x["convention"]): x
        for x in json.loads((old_root / "predictions.json").read_text())
        if x["query"] == "description"
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "responses.json").write_bytes((root / "responses.json").read_bytes())
    e = html.escape
    summary = ""
    if (output / "interpretation.json").exists():
        paragraphs = json.loads((output / "interpretation.json").read_text())["paragraphs"]
        summary = "".join(f"<p>{e(p)}</p>" for p in paragraphs)
    names = {
        "case-k": "First ten steps: numbering increased",
        "case-r": "Next ten steps: bullets replaced numbering",
        "case-v": "Final minus initial: bullets increased",
        "case-z": "Zero-update control",
        "case-p": "Shuffled second-interval tokens",
    }
    cards = ""
    for row in rows:
        prior = old[(row["case"], row["encoding"])]
        cards += f"""<section><h3>{e(names[row["case"]])}</h3><p>{e(row["encoding"])} encoding</p><pre>{e(row["response"])}</pre><details><summary>Earlier, longer question and its answer</summary><p>{e(prior["question"])}</p><pre>{e(prior["response"])}</pre></details></section>"""
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>A shorter LoRAcle question</title><style>body{{font:17px/1.6 system-ui;color:#233143;max-width:960px;margin:45px auto;padding:0 24px;background:#f9fafb}}h1{{font-size:34px}}h2{{margin-top:36px}}h3{{margin-bottom:0}}blockquote{{font-size:23px;border-left:4px solid #3574ac;padding-left:20px}}section{{padding:14px 0 24px;border-top:1px solid #c9d3df}}pre{{white-space:pre-wrap;font:inherit;background:#edf1f6;padding:18px}}details{{color:#394b5d}}a{{color:#235aa0}}</style></head><body>
    <h1>A shorter LoRAcle question</h1><blockquote>{e(plan["question"])}</blockquote>{summary}
    <p>Same Qwen3-14B trajectory, saved weight differences, interpreter, token layout, greedy decoding and 192-token cap as the original toy. This changes the question only. Both previously reported encodings and all controls are retained. There is no new training or independent replication.</p>
    <p>The question was chosen after seeing the original results, so this is a retrospective prompt-sensitivity check. Greedy decoding means these are not random response samples. This check cannot estimate performance across training seeds or the probability of chance agreement.</p>
    <h2>All ten new answers</h2>{cards}
    <p><a href="responses.json">Exact new responses</a> · <a href="interpretation.json">Interpretation and limitations</a> · <a href="provenance/plan.json">Frozen short-prompt plan</a> · <a href="provenance/integrity.json">Integrity check</a> · <a href="../loracle-trajectory-toy/report.html">Original toy and measured behavior</a></p></body></html>"""
    (output / "report.html").write_text(document)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)
