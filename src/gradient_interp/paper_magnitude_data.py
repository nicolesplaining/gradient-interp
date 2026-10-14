"""Pinned natural instruction data with lexical-group separation for magnitude studies."""

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

REVISION = "dce01c9b08f87459cf36a430d809084718273017"
DATASET = "tatsu-lab/alpaca"
SEED = 2026101417
INPUT_NAMES = {
    "training.jsonl",
    "clean_raw.jsonl",
    "fit.jsonl",
    "train.jsonl",
    "corpus.jsonl",
    "prompts.json",
    "questions.json",
    "data.json",
    "batches.jsonl",
    "probes.jsonl",
    "probes.json",
    "calibration_probes.jsonl",
    "evaluation_probes.jsonl",
    "dev_batches.jsonl",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def prompt_variants(text):
    yield text
    # Older study prompts wrap the user turn in literal role labels.
    for match in re.finditer(
        r"(?:^|\n)User:\s*(.*?)(?=\n(?:Assistant|System|User):|$)", text, re.S
    ):
        yield match.group(1)


def prompt_strings(value):
    """Extract input fields only; never mistake completions for historic prompts."""
    if isinstance(value, dict):
        if isinstance(value.get("prompt"), str):
            yield from prompt_variants(value["prompt"])
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            yield value["content"]
        if isinstance(value.get("instruction"), str):
            yield value["instruction"]
            if isinstance(value.get("input"), str) and value["input"].strip():
                yield value["instruction"].strip() + "\n\n" + value["input"].strip()
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from prompt_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from prompt_strings(child)


def historical_inputs(repo, location):
    """Enumerate explicit input filenames across old phases, never output records."""
    repo = Path(repo)
    files, prompts = [], set()
    for base in [repo / "data", repo / "runs"]:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name not in INPUT_NAMES:
                continue
            if any(
                x in path.parts for x in {"paper-magnitude-data", "outcomes", "forecasts", ".git"}
            ):
                continue
            raw = path.read_text()
            value = (
                [json.loads(line) for line in raw.splitlines() if line.strip()]
                if path.suffix == ".jsonl"
                else json.loads(raw)
            )
            extracted = {canonical(p) for p in prompt_strings(value) if canonical(p)}
            files.append(
                {
                    "location": location,
                    "path": str(path.relative_to(repo)),
                    "sha256": file_hash(path),
                    "prompt_count": len(extracted),
                }
            )
            prompts.update(extracted)
    return {"created_unix": time.time(), "files": files, "prompts": sorted(prompts)}


class Components:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = int(self.parent[node])
        return node

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def cluster_prompts(source_rows, historical, threshold=0.85, chunk=128):
    """Exact connected components under fixed TF-IDF cosine and instruction identity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    prompts = [canonical(r["prompt"]) for r in source_rows]
    history = sorted(set(canonical(p) for p in historical if canonical(p)))
    unique = sorted(set(prompts + history))
    index = {p: i for i, p in enumerate(unique)}
    groups = Components(len(unique))
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(unique)
    edge_count = 0
    for start in range(0, len(unique), chunk):
        similarities = (matrix[start : start + chunk] @ matrix.T).tocoo()
        keep = (similarities.data >= threshold) & (similarities.col > similarities.row + start)
        for left, right in zip(similarities.row[keep] + start, similarities.col[keep], strict=True):
            groups.join(int(left), int(right))
            edge_count += 1
    instructions = {}
    for row, prompt in zip(source_rows, prompts, strict=True):
        key = canonical(row["instruction"])
        if key in instructions:
            groups.join(index[prompt], instructions[key])
        instructions[key] = index[prompt]
        # A historical standalone instruction is also excluded, even with new context.
        if key in index:
            groups.join(index[prompt], index[key])
    forbidden = {groups.find(index[p]) for p in history}
    labels = [groups.find(index[p]) for p in prompts]
    return (
        labels,
        forbidden,
        {
            "unique_prompt_vertices": len(unique),
            "tfidf_features": matrix.shape[1],
            "near_duplicate_edges": edge_count,
            "forbidden_components": len(forbidden),
            "source_components": len(set(labels)),
        },
    )


def render_source(records):
    rows = []
    for i, row in enumerate(records):
        instruction, answer = row["instruction"].strip(), row["output"].strip()
        context = row["input"].strip()
        if not instruction or not answer:
            continue
        rows.append(
            {
                "source_id": f"{DATASET}@{REVISION}:row:{i}",
                "source_row": i,
                "instruction": instruction,
                "prompt": instruction + ("\n\n" + context if context else ""),
                "answer": answer,
            }
        )
    return rows


def token_counts(tokenizer, row):
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full_ids, prompt_ids = tokenizer([full, prompt], add_special_tokens=False)["input_ids"]
    common = 0
    for x, y in zip(full_ids, prompt_ids):
        if x != y:
            break
        common += 1
    return len(full_ids), len(full_ids) - common


def choose(rows, labels, forbidden, counts, seed=SEED):
    """One eligible row per full lexical component; no outcome-dependent selection."""
    eligible = {}
    for row, cluster in zip(rows, labels, strict=True):
        if cluster not in forbidden and row["full_tokens"] <= 500 and row["completion_tokens"] > 0:
            eligible.setdefault(cluster, []).append(row)
    ordered = sorted(eligible, key=lambda c: min(r["source_row"] for r in eligible[c]))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    representatives = []
    for cluster in ordered:
        members = sorted(eligible[cluster], key=lambda r: r["source_row"])
        rng.shuffle(members)
        row = dict(members[0], cluster_id=f"lexical-{cluster:06d}")
        # group_id is explicitly a lexical duplicate group, not a task category.
        row["group_id"] = row["cluster_id"]
        row["id"] = f"alpaca-{row['source_row']:05d}"
        representatives.append(row)
    n_batch = counts["batches"] * counts["documents_per_batch"]
    total = n_batch + counts["calibration"] + counts["evaluation"]
    if len(representatives) < total:
        raise ValueError(f"Only {len(representatives)} eligible clusters for {total} rows")
    selected = representatives[:total]
    batches = [
        {
            "id": f"natural-batch-{i:02d}",
            "documents": selected[
                i * counts["documents_per_batch"] : (i + 1) * counts["documents_per_batch"]
            ],
        }
        for i in range(counts["batches"])
    ]
    cal_end = n_batch + counts["calibration"]
    data = {
        "batches": batches,
        "calibration": selected[n_batch:cal_end],
        "evaluation": selected[cal_end:],
        "scope": "New natural-language instructions; synthetic source answers; development only.",
        "group_definition": "Lexical duplicate component, not an independent semantic task family.",
        "template_metadata": "Source provides no validated task/template IDs; none invented.",
    }
    validate(data, counts)
    return data, {
        "eligible_components": len(eligible),
        "eligible_rows": sum(map(len, eligible.values())),
        "selected_rows": total,
        "selected_components": len({r["cluster_id"] for r in selected}),
        "maximum_full_tokens": max(r["full_tokens"] for r in selected),
        "minimum_completion_tokens": min(r["completion_tokens"] for r in selected),
    }


def validate(data, counts):
    assert len(data["batches"]) == counts["batches"]
    assert all(len(b["documents"]) == counts["documents_per_batch"] for b in data["batches"])
    assert len(data["calibration"]) == counts["calibration"]
    assert len(data["evaluation"]) == counts["evaluation"]
    rows = (
        [r for b in data["batches"] for r in b["documents"]]
        + data["calibration"]
        + data["evaluation"]
    )
    for field in ["id", "source_id", "cluster_id", "group_id"]:
        assert len({r[field] for r in rows}) == len(rows), field
    assert len({canonical(r["prompt"]) for r in rows}) == len(rows)
    assert all(
        r["answer"].strip() and 0 < r["completion_tokens"] <= r["full_tokens"] <= 500 for r in rows
    )


def build(root, source, history_paths, tokenizer_path):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    root = Path(root)
    if (root / "data.json").exists():
        raise ValueError("Refuse to overwrite existing selected data")
    plan = json.loads((root / "selection-plan.json").read_text())
    assert plan["source"]["revision"] == REVISION and plan["seed"] == SEED
    histories = [json.loads(Path(p).read_text()) for p in history_paths]
    history = sorted({p for h in histories for p in h["prompts"]})
    records = pq.read_table(source).to_pylist()
    rows = render_source(records)
    labels, forbidden, cluster_stats = cluster_prompts(
        rows, history, plan["deduplication"]["threshold"]
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if "eos_token" not in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.rstrip() + "{{ eos_token }}"
    for row in rows:
        row["full_tokens"], row["completion_tokens"] = token_counts(tokenizer, row)
    data, selection = choose(rows, labels, forbidden, plan["counts"], plan["seed"])
    data["source"] = plan["source"]
    data["selection_plan_sha256"] = file_hash(root / "selection-plan.json")
    out = root / "data.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    selected = (
        [r for b in data["batches"] for r in b["documents"]]
        + data["calibration"]
        + data["evaluation"]
    )
    assert not {canonical(r["prompt"]) for r in selected}.intersection(history)
    tokenizer_files = {
        str(p.name): file_hash(p) for p in Path(tokenizer_path).glob("*token*") if p.is_file()
    }
    audit = {
        "created_unix": time.time(),
        "passed": True,
        "selection_plan_sha256": data["selection_plan_sha256"],
        "source_sha256": file_hash(source),
        "source_rows": len(records),
        "nonempty_source_rows": len(rows),
        "data_sha256": file_hash(out),
        "builder_source_sha256": file_hash(__file__),
        "historical_input_files": [f for h in histories for f in h["files"]],
        "prior_input_files_sha256": {
            f["location"] + ":" + f["path"]: f["sha256"] for h in histories for f in h["files"]
        },
        "source_deduplication": "Unique pinned dataset record IDs and lexical components globally.",
        "near_duplicate_rule": plan["deduplication"],
        "template_split_policy": data["template_metadata"],
        "scope": data["scope"],
        "exact_prompt_collisions": [],
        "source_collisions": [],
        "near_duplicate_collisions": [],
        "historical_prompts": len(history),
        "history_extracts_sha256": {str(p): file_hash(p) for p in history_paths},
        "tokenizer_files_sha256": tokenizer_files,
        "tokenizer_revision": plan["model_tokenizer_revision"],
        "counts": plan["counts"],
        **cluster_stats,
        **selection,
        "exact_prompt_overlap": 0,
        "historical_component_overlap": 0,
        "source_record_overlap_across_splits": 0,
        "lexical_group_overlap_across_splits": 0,
        "max_tokens_includes_training_eos": True,
        "model_measurements_used_for_selection": False,
        "limitations": plan["deduplication"]["limits"],
    }
    (root / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("history")
    h.add_argument("--repo", type=Path, required=True)
    h.add_argument("--location", required=True)
    h.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--source", type=Path, required=True)
    b.add_argument("--history", type=Path, nargs="+", required=True)
    b.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "history":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = historical_inputs(args.repo, args.location)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"files": len(result["files"]), "prompts": len(result["prompts"])}))
    else:
        result = build(args.root, args.source, args.history, args.tokenizer)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k
                    in {
                        "passed",
                        "data_sha256",
                        "selected_rows",
                        "eligible_components",
                        "historical_prompts",
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
