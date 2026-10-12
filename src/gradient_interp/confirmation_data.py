"""Fresh-source confirmation after the exploratory No Robots study.

Dolly source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
Databricks (2023), CC-BY-SA-3.0. Text stays in ignored runtime output.
"""

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from .benchmark import (
    TOKENIZER,
    TOKENIZER_REVISION,
    first_turn,
    prompt_key,
    validate_partitions,
)
from .data import Example, Probe, read_jsonl, write_jsonl
from .study import atomic_json, file_hash

DATASET = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"


def partition_fresh(examples, seed=20261012, probes_per_group=32):
    rng = random.Random(seed)
    pools = defaultdict(list)
    for example in examples:
        pools[example.group].append(example)
    groups = sorted(pools)
    for pool in pools.values():
        pool.sort(key=lambda e: e.id)
        rng.shuffle(pool)

    def take(group, n):
        if len(pools[group]) < n:
            raise ValueError(f"Insufficient fresh source documents: {group}")
        result, pools[group] = pools[group][:n], pools[group][n:]
        return result

    def batch(name, items, kind):
        counts = Counter(e.group for e in items)
        return {
            "id": name,
            "examples": [asdict(e) for e in items],
            "kind": kind,
            "batch_size": len(items),
            "target_group": min(counts, key=lambda g: (-counts[g], g)),
            "family_split": "new_dataset",
        }

    cal, evaluation, dev, batches = [], [], [], []
    for group in groups:
        cal.extend(take(group, probes_per_group))
        evaluation.extend(take(group, probes_per_group))
        for repeat in range(2):
            dev.append(batch(f"dev-{group}-{repeat}", take(group, 4), "pure"))
        for size in (1, 4, 16):
            for repeat in range(2):
                batches.append(batch(f"test-{group}-{size}-{repeat}", take(group, size), "pure"))
    for i in range(48):
        size = 4 if i % 2 == 0 else 16
        items = [take(rng.choice(groups), 1)[0] for _ in range(size)]
        batches.append(batch(f"test-mixed-{i}", items, "mixed"))
    return dev, batches, cal, evaluation


def prepare_fresh(old_data, output):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    old_data, output = Path(old_data), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to replace a frozen dataset")
    excluded = set()
    for name in ("fit.jsonl", "calibration_probes.jsonl", "evaluation_probes.jsonl"):
        excluded.update(prompt_key(r["prompt"]) for r in read_jsonl(old_data / name))
    for name in ("batches.jsonl", "dev_batches.jsonl"):
        excluded.update(
            prompt_key(e["prompt"]) for b in read_jsonl(old_data / name) for e in b["examples"]
        )
    source = Path(
        hf_hub_download(
            DATASET, "databricks-dolly-15k.jsonl", repo_type="dataset", revision=REVISION
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    examples, lineage, seen = [], [], set(excluded)
    for i, row in enumerate(read_jsonl(source)):
        prompt = row["instruction"]
        if row["context"].strip():
            prompt += "\n\n" + row["context"]
        converted = {
            "prompt_id": "dolly-" + prompt_key(prompt),
            "category": row["category"],
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["response"]},
            ],
        }
        result = first_turn(converted, tokenizer)
        if result is None:
            continue
        example, record = result
        if record["prompt_sha256"] in seen:
            continue
        seen.add(record["prompt_sha256"])
        examples.append(example)
        lineage.append({**record, "source_row": i})
    dev, batches, cal, evaluation = partition_fresh(examples)
    fit = [Example(**r) for r in read_jsonl(old_data / "fit.jsonl")]
    counts = validate_partitions(fit, dev, batches, cal, evaluation)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(old_data / "fit.jsonl", output / "fit.jsonl")
    write_jsonl(output / "dev_batches.jsonl", dev)
    write_jsonl(output / "batches.jsonl", batches)
    for name, rows in (("calibration", cal), ("evaluation", evaluation)):
        write_jsonl(
            output / f"{name}_probes.jsonl",
            [asdict(Probe(e.id, e.prompt, e.completion, group=e.group)) for e in rows],
        )
    write_jsonl(output / "lineage.jsonl", lineage)
    manifest = {
        "source": DATASET,
        "revision": REVISION,
        "source_sha256": file_hash(source),
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "attribution": "Databricks (2023), dolly-15k",
        "license": "CC-BY-SA-3.0",
        "fit_source": json.loads((old_data / "manifest.json").read_text()),
        "counts": counts,
        "test_batches": len(batches),
        "seed": 20261012,
        "groups": sorted({e.group for e in evaluation}),
        "eligible_categories": dict(Counter(e.group for e in examples)),
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "prompt_token_limit": 352,
        "completion_prefix_token_limit": 128,
        "split_rule": "IDs and normalized prompts disjoint across all partitions and old study",
        "scope": "New source dataset; task concepts may overlap. Pretraining overlap unknown.",
        "files_sha256": {p.name: file_hash(p) for p in output.glob("*.jsonl")},
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def make_plan(root, old_plan):
    root = Path(root).resolve()
    prior = json.loads(Path(old_plan).read_text())
    model_configs = {j["model"]["name"]: j["model"] for j in prior["jobs"]}
    jobs = []
    for seed in (101, 202, 303, 404, 505):
        for name, model in model_configs.items():
            jobs.append(
                {
                    "id": f"{name.split('/')[-1]}-seed{seed}-warm64",
                    "model": model,
                    "seed": seed,
                    "warmup_steps": 64,
                }
            )
    batches = read_jsonl(root / "data/batches.jsonl")
    controls = [i for i, b in enumerate(batches) if b["kind"] == "pure" and b["id"].endswith("-0")]
    return {
        "benchmark": str(root / "data"),
        "jobs": jobs,
        "threads": 8,
        "rank": 128,
        "atoms": 512,
        "alpha": 0.1,
        "dictionary_iterations": 40,
        "learning_rates": [0.0001, 0.001],
        "control_batch_indices": controls,
        "control_sparsity": 8,
        "generation_tokens": 48,
        "protocol": {
            "stage": "confirmatory; old-study outcomes used only for method development",
            "primary_rate": 0.0001,
            "primary_method": "dictionary_omp_8",
            "primary_endpoint": "Category-centered relative L2 forecast error",
            "primary_comparisons_per_model": [
                ["fixed", "svd_12"],
                ["fixed", "rotated_dictionary_omp_8"],
                ["transfer", "svd_12"],
            ],
            "primary_success": (
                "For each of six claims, paired error ratio upper bootstrap bound <0.90 "
                "at one-sided 1-0.05/6 coverage, improvement in all five seeds, and "
                "category-contrast signal RMS >10*max(measured zero-update noise RMS,1e-6). "
                "Cross-model claim requires passing in both models. Incomplete jobs cannot pass."
            ),
            "bootstrap": (
                "10000 paired replicates: resample model seeds, batches within kind/size "
                "strata, and probe documents independently within each category in each "
                "probe split. Reuse draws across methods and seeds; category labels fixed."
            ),
            "secondary": (
                "Uncentered and individual-probe errors; per-batch p50/p90; sign versus "
                "constant sign; SVD/OMP sparsity curves; rate 0.001; removal selectivity; "
                "generations. Descriptive, not additional confirmatory positive claims."
            ),
            "common_mode_control": (
                "Subtract each batch's mean across category means from prediction and "
                "outcome. Also score the full-gradient forecast broadcast across categories."
            ),
            "no_adaptation": (
                "No method, seed, step, endpoint or data changes after test evaluation starts. "
                "No stopping for a favorable result. Fixed 24-hour deadline, all failures retained."
            ),
            "scope": (
                "One stateless SGD update of q/v LoRA parameters after 64 fitting-data AdamW "
                "steps; category mean completion log probability, not semantic task success. "
                "Fixed-probe methods use those probes' pre-update derivatives; transfer uses "
                "independent calibration probes. SVD projection differs from Gradient Atoms EKFAC."
            ),
            "storage": (
                "Float32 SVD coefficients: 4*r bytes; sparse float32+uint16: 6*k bytes; "
                "OMP8 and SVD12 both 48 bytes per update. Shared basis, atoms, probe derivatives "
                "and encoding costs excluded; no runtime-speedup claim."
            ),
            "removal": (
                "24 pure batches (first repeat for each category/size), rate 1e-4 only. "
                "OMP8 atom chosen by absolute calibration target-category contrast. Compare "
                "atom removal to equal-norm random removal and norm-matched rescaling."
            ),
            "generations": "First two preset size-16 pure batches, rate 0.001; qualitative only",
        },
    }
