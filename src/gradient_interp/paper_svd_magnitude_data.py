"""Fresh, frozen natural instructions for the prospective SVD magnitude study."""

import argparse
import ast
import copy
import json
import time
from pathlib import Path

from gradient_interp import paper_magnitude_data as previous

SEED = 2026101452
COUNTS = {"batches": 32, "documents_per_batch": 8, "calibration": 16, "evaluation": 64}


def all_rows(data):
    return (
        [r for b in data["batches"] for r in b["documents"]]
        + data["calibration"]
        + data["evaluation"]
    )


def check_inputs(plan, source, records_path, tokenizer_path):
    if plan["seed"] != SEED or plan["counts"] != COUNTS:
        raise ValueError("Unexpected selection plan")
    if plan["source"]["revision"] != previous.REVISION:
        raise ValueError("Wrong source revision")
    if plan["deduplication"]["threshold"] != 0.85:
        raise ValueError("Changed duplicate threshold")
    if previous.file_hash(source) != plan["source"]["sha256"]:
        raise ValueError("Source parquet hash mismatch")
    conversion_path = Path(records_path).parent / "source-reuse-validation.json"
    conversion = json.loads(conversion_path.read_text())
    if (
        conversion["source_sha256"] != plan["source"]["sha256"]
        or conversion["records_sha256"] != previous.file_hash(records_path)
    ):
        raise ValueError("Source conversion hash mismatch")
    if previous.file_hash(plan["prior_data_path"]) != plan["prior_magnitude_data_sha256"]:
        raise ValueError("Prior magnitude data changed")
    if previous.file_hash(previous.__file__) != plan["prior_builder_sha256"]:
        raise ValueError("Prior selection helper changed")
    histories = []
    for name, sha in plan["history_extracts_sha256"].items():
        if previous.file_hash(name) != sha:
            raise ValueError(f"History extract changed: {name}")
        histories.append(json.loads(Path(name).read_text()))
    for name, sha in plan["tokenizer_files_sha256"].items():
        if previous.file_hash(Path(tokenizer_path) / name) != sha:
            raise ValueError(f"Tokenizer file changed: {name}")
    prior_data = json.loads(Path(plan["prior_data_path"]).read_text())
    prior_rows = all_rows(prior_data)
    if len(prior_rows) != plan["prior_magnitude_rows"]:
        raise ValueError("Wrong prior row count")
    history = {p for h in histories for p in h["prompts"]}
    if len(history) != plan["history_base_unique_prompts"]:
        raise ValueError("Wrong historical prompt count")
    history.update(previous.canonical(r["prompt"]) for r in prior_rows)
    records = json.loads(Path(records_path).read_text())
    if len(records) != 52002:
        raise ValueError("Wrong pinned Alpaca row count")
    return histories, prior_rows, sorted(history), records


def assert_fresh(data, prior_rows, historical, labels_by_source, forbidden):
    rows = all_rows(data)
    if {r["source_id"] for r in rows} & {r["source_id"] for r in prior_rows}:
        raise ValueError("Selected a prior source record")
    if {previous.canonical(r["prompt"]) for r in rows} & set(historical):
        raise ValueError("Selected a historical prompt")
    if any(labels_by_source[r["source_id"]] in forbidden for r in rows):
        raise ValueError("Selected a historical near-duplicate component")
    previous.validate(data, COUNTS)


def original_tokenizer_verification(data, tokenizer, upstream_path):
    """Run the pinned original masking helper, including its training EOS convention."""
    tree = ast.parse(Path(upstream_path).read_text())
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tokenize_chat"
    ]
    if len(functions) != 1:
        raise ValueError("Missing original tokenize_chat helper")
    namespace = {"copy": copy}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(upstream_path), "exec"), namespace
    )
    for row in all_rows(data):
        example = {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["answer"]},
            ]
        }
        raw = namespace["tokenize_chat"](example, tokenizer, max_length=None)
        truncated = namespace["tokenize_chat"](example, tokenizer, max_length=500)
        if raw != truncated or len(raw["input_ids"]) != row["full_tokens"]:
            raise ValueError("Whole-completion tokenization mismatch")
        if sum(x != -100 for x in raw["labels"]) != row["completion_tokens"]:
            raise ValueError("Completion mask mismatch")
    return {
        "passed": True,
        "checked_rows": len(all_rows(data)),
        "no_truncation_all_rows": True,
        "exact_upstream_tokenize_chat_match": True,
        "upstream_tokenizer_source_sha256": previous.file_hash(upstream_path),
    }


def build(root, source, records_path, tokenizer_path, upstream_path):
    from transformers import AutoTokenizer

    root = Path(root)
    if (root / "data.json").exists():
        raise ValueError("Refuse to overwrite selected prospective data")
    plan_path = root / "selection-plan.json"
    plan = json.loads(plan_path.read_text())
    histories, prior_rows, historical, records = check_inputs(
        plan, source, records_path, tokenizer_path
    )
    rows = previous.render_source(records)
    labels, forbidden, clusters = previous.cluster_prompts(rows, historical, threshold=0.85)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if "eos_token" not in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.rstrip() + "{{ eos_token }}"
    for row in rows:
        row["full_tokens"], row["completion_tokens"] = previous.token_counts(tokenizer, row)
    data, selection = previous.choose(rows, labels, forbidden, COUNTS, SEED)
    data["source"] = plan["source"]
    data["selection_plan_sha256"] = previous.file_hash(plan_path)
    data["prior_magnitude_data_sha256"] = plan["prior_magnitude_data_sha256"]
    labels_by_source = {r["source_id"]: label for r, label in zip(rows, labels, strict=True)}
    assert_fresh(data, prior_rows, historical, labels_by_source, forbidden)
    token_audit = original_tokenizer_verification(data, tokenizer, upstream_path)
    # Check the local tokenizer runtime against the previous exact 336-row remote audit.
    for row in prior_rows:
        if previous.token_counts(tokenizer, row) != (row["full_tokens"], row["completion_tokens"]):
            raise ValueError("Local tokenizer disagrees with prior pinned runtime counts")
    path = root / "data.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    audit = {
        "created_unix": time.time(),
        "passed": True,
        "scope": "fresh_data_development_only",
        "data_sha256": previous.file_hash(path),
        "selection_plan_sha256": previous.file_hash(plan_path),
        "source_sha256": previous.file_hash(source),
        "source_records_sha256": previous.file_hash(records_path),
        "source_rows": len(records),
        "source_revision": previous.REVISION,
        "license": "CC-BY-NC-4.0",
        "source_answer_caveat": "Model-generated answers; correctness is not asserted.",
        "builder_source_sha256": previous.file_hash(__file__),
        "helper_source_sha256": previous.file_hash(previous.__file__),
        "prior_magnitude_data_sha256": plan["prior_magnitude_data_sha256"],
        "excluded_previous_magnitude_data_sha256": plan["prior_magnitude_data_sha256"],
        "prior_input_files_sha256": {
            **{f["location"] + ":" + f["path"]: f["sha256"] for h in histories for f in h["files"]},
            "local:" + plan["prior_data_path"]: plan["prior_magnitude_data_sha256"],
        },
        "history_extracts_sha256": plan["history_extracts_sha256"],
        "historical_prompts": len(historical),
        "prior_magnitude_rows_excluded": len(prior_rows),
        "source_deduplication": "Source IDs/lexical groups disjoint across splits/prior data.",
        "near_duplicate_rule": plan["deduplication"],
        "template_split_policy": data["template_metadata"],
        "exact_prompt_collisions": [],
        "source_collisions": [],
        "near_duplicate_collisions": [],
        "tokenizer_revision": plan["model_tokenizer_revision"],
        "tokenizer_files_sha256": plan["tokenizer_files_sha256"],
        "counts": COUNTS,
        **clusters,
        **selection,
        "model_measurements_used_for_selection": False,
        "max_tokens_includes_training_eos": True,
        "prior_token_count_runtime_equivalence_rows": len(prior_rows),
        "independent_tokenization": token_audit,
        "limitations": plan["deduplication"]["limits"],
    }
    (root / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (root / "independent-validation.json").write_text(json.dumps(token_audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser()
    for flag in ["root", "source", "records", "tokenizer", "upstream"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    audit = build(args.root, args.source, args.records, args.tokenizer, args.upstream)
    print(
        json.dumps(
            {
                k: audit[k]
                for k in [
                    "passed",
                    "data_sha256",
                    "selected_rows",
                    "eligible_components",
                    "historical_prompts",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
