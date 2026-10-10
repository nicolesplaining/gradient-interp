"""Frozen, source-disjoint No Robots splits for batch-update forecasting.

Human-authored source: https://huggingface.co/datasets/HuggingFaceH4/no_robots
Rajani et al. (2023), CC-BY-NC-4.0. Source text is downloaded at runtime,
not redistributed with this package. Only first-assistant turns are used.
Long responses are explicitly scored as prefixes; prompts are never truncated.
"""

import hashlib
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from .data import Example, Probe, write_jsonl
from .experiment import save_json

DATASET = "HuggingFaceH4/no_robots"
DATASET_REVISION = "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
TOKENIZER = "Qwen/Qwen2.5-3B"
TOKENIZER_REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"


def prompt_key(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def first_turn(row, tokenizer, prompt_limit=352, completion_limit=128):
    parts = []
    for message in row["messages"]:
        if message["role"] == "assistant":
            if not parts or not message["content"].strip():
                return None
            prompt = "\n".join(parts) + "\nAssistant:"
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_tokens) > prompt_limit:
                return None
            original = tokenizer.encode(message["content"], add_special_tokens=False)
            completion = tokenizer.decode(original[:completion_limit], skip_special_tokens=True)
            if not completion.strip():
                return None
            # Retokenization after decoding may differ at a partial-byte boundary.
            while len(tokenizer.encode(completion, add_special_tokens=False)) > completion_limit:
                completion = completion[:-1]
            example = Example(str(row["prompt_id"]), prompt, completion, row["category"])
            return example, {
                "id": example.id,
                "prompt_sha256": prompt_key(prompt),
                "original_completion_tokens": len(original),
                "completion_prefix_truncated": len(original) > completion_limit,
                "prompt_tokens": len(prompt_tokens),
            }
        parts.append(message["role"].capitalize() + ": " + message["content"])
    return None


def partition_examples(train, test, fit_count, seed, heldout_groups=("Coding", "Classify")):
    """Partition source documents before constructing any mixtures or fitting models."""
    rng = random.Random(seed)
    pools, test_pools = defaultdict(list), defaultdict(list)
    for item in train:
        pools[item.group].append(item)
    for item in test:
        test_pools[item.group].append(item)
    for pool in [*pools.values(), *test_pools.values()]:
        pool.sort(key=lambda item: item.id)
        rng.shuffle(pool)
    groups = sorted(group for group in pools if len(test_pools[group]) >= 4)
    if not set(heldout_groups) <= set(groups):
        raise ValueError("Held-out families require at least four independent probe documents")
    calibration, evaluation = [], []
    for group in groups:
        n = min(8, len(test_pools[group]) // 2)
        calibration.extend(test_pools[group][:n])
        evaluation.extend(test_pools[group][n : 2 * n])

    def take(group, n):
        if len(pools[group]) < n:
            raise ValueError(f"Insufficient disjoint source documents in {group}")
        result, pools[group] = pools[group][:n], pools[group][n:]
        return result

    def batch(name, items, split, kind):
        counts = Counter(item.group for item in items)
        target = sorted(counts, key=lambda group: (-counts[group], group))[0]
        return {
            "id": name,
            "examples": [asdict(item) for item in items],
            "target_group": target,
            "kind": kind,
            "batch_size": len(items),
            "family_split": split,
        }

    dev, batches = [], []
    for group in groups:
        if group not in heldout_groups:
            for repeat in range(2):
                dev.append(batch(f"dev-{group}-{repeat}", take(group, 4), "seen", "pure"))
        for size in (1, 4, 16):
            split = "unseen_family" if group in heldout_groups else "seen"
            batches.append(batch(f"test-{group}-{size}", take(group, size), split, "pure"))
    # Mixtures use distinct documents, both within and across batches.
    for i in range(30):
        size = 4 if i % 2 == 0 else 16
        items = []
        for _ in range(size):
            eligible = [group for group in groups if pools[group]]
            items.extend(take(rng.choice(eligible), 1))
        batches.append(batch(f"test-mixed-{i}", items, "mixed", "mixed"))
    available = {group: pool for group, pool in pools.items() if group not in heldout_groups}
    fit = []
    # Round-robin sampling preserves rare categories without duplicating documents.
    while len(fit) < fit_count:
        added = False
        for group in sorted(available):
            if available[group] and len(fit) < fit_count:
                fit.append(available[group].pop())
                added = True
        if not added:
            raise ValueError(f"Only {len(fit)} fitting documents available; requested {fit_count}")
    rng.shuffle(fit)
    return fit, dev, batches, calibration, evaluation


def validate_partitions(fit, dev, batches, calibration, evaluation):
    collections = {
        "fit": fit,
        "dev": [Example(**e) for batch in dev for e in batch["examples"]],
        "test": [Example(**e) for batch in batches for e in batch["examples"]],
        "calibration_probes": calibration,
        "evaluation_probes": evaluation,
    }
    ids, prompts = set(), set()
    for name, examples in collections.items():
        for example in examples:
            key = prompt_key(example.prompt)
            if example.id in ids or key in prompts:
                raise ValueError(f"Source-document overlap in {name}: {example.id}")
            ids.add(example.id)
            prompts.add(key)
    return {name: len(examples) for name, examples in collections.items()}


def prepare_benchmark(output, fit_count=5000, seed=20261010):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Benchmark directory is not empty: {output}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    seen_ids, seen_prompts, lineage, splits, source_hashes = set(), set(), [], {}, {}
    # Prefer official test documents when duplicates exist across official splits.
    for split in ("test", "train"):
        path = Path(
            hf_hub_download(
                DATASET,
                f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )
        source_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        splits[split] = []
        for row in pq.read_table(path).to_pylist():
            result = first_turn(row, tokenizer)
            if result is None:
                continue
            example, record = result
            if example.id in seen_ids or record["prompt_sha256"] in seen_prompts:
                continue
            seen_ids.add(example.id)
            seen_prompts.add(record["prompt_sha256"])
            splits[split].append(example)
            lineage.append({**record, "official_split": split})
    partitions = partition_examples(splits["train"], splits["test"], fit_count, seed)
    counts = validate_partitions(*partitions)
    fit, dev, batches, calibration, evaluation = partitions
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "fit.jsonl", [asdict(e) for e in fit])
    write_jsonl(output / "dev_batches.jsonl", dev)
    write_jsonl(output / "batches.jsonl", batches)
    for name, examples in [("calibration", calibration), ("evaluation", evaluation)]:
        probes = [Probe(e.id, e.prompt, e.completion, group=e.group) for e in examples]
        write_jsonl(output / f"{name}_probes.jsonl", [asdict(p) for p in probes])
    write_jsonl(output / "lineage.jsonl", lineage)
    manifest = {
        "source": DATASET,
        "source_revision": DATASET_REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "source_sha256": source_hashes,
        "license": "CC-BY-NC-4.0",
        "attribution": "Rajani, Tunstall, Beeching, Lambert, Rush, Wolf (2023), No Robots",
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "seed": seed,
        "heldout_families": ["Coding", "Classify"],
        "heldout_scope": (
            "Excluded from dictionary fitting and development batches; "
            "calibrated via separate probes"
        ),
        "counts": counts,
        "batch_count": len(batches),
        "dev_batch_count": len(dev),
        "fit_categories": dict(Counter(e.group for e in fit)),
        "calibration_categories": dict(Counter(e.group for e in calibration)),
        "evaluation_categories": dict(Counter(e.group for e in evaluation)),
        "prompt_token_limit": 352,
        "completion_token_limit": 128,
        "preprocessing": (
            "First assistant turn; role-labelled prompt; unchanged prompt; completion prefix"
        ),
        "split_rule": (
            "Source ID and normalized prompt disjoint across all five partitions; "
            "no test batch reuses a document"
        ),
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.jsonl")
        },
    }
    save_json(output / "manifest.json", manifest)
    return manifest
