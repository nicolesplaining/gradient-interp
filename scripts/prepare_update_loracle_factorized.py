"""Freeze one balanced component-question recipe after complete v1 development analysis."""

import argparse
import importlib
import importlib.metadata
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from gradient_interp.loracle import configure_interpreter_prefix
from gradient_interp.loracle_trajectory_toy import BASE_PATH, CHECKPOINT, UPSTREAM, read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_analysis import require
from gradient_interp.update_loracle_factorized import (
    OPTION_SEED,
    TRAINING_SEED,
    question_views,
    training_instances,
)
from gradient_interp.update_loracle_train import option_token_ids, prepared_input, verified_cases


def prepare(collection, prior, root, expected_commit):
    collection, prior, root = collection.resolve(), prior.resolve(), root.resolve()
    require(not root.exists(), "New recipe needs a new output root")
    for directory in [collection, prior]:
        integrity = read(directory / "integrity.json")
        require(integrity["passed"], "Previous complete audit required")
        for name, expected in integrity["files_sha256"].items():
            require(file_hash(directory / name) == expected, f"Audited output changed: {name}")
    prior_integrity = read(prior / "integrity.json")
    require(prior_integrity["optimizer_steps"] == 204, "Wrong original training reference")
    collection_integrity = read(collection / "integrity.json")
    require(collection_integrity["cases"] == 344, "Incomplete collection")
    require(
        file_hash(collection / "source_snapshot.tar")
        == collection_integrity["source_archive_sha256"],
        "Collection source archive changed",
    )
    require(
        file_hash(prior / "source_snapshot.tar")
        == prior_integrity["files_sha256"]["source_snapshot.tar"],
        "Prior training source archive changed",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(commit == expected_commit, "Wrong frozen source")
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(),
        "Remote checkout must be clean",
    )
    require(
        not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip(),
        "GPU occupied",
    )
    inputs = dict(read(collection / "plan.json")["input_files_sha256"])
    inputs.update(
        {
            str(prior / name): file_hash(prior / name)
            for name in ["integrity.json", "simple-references.json", "source_snapshot.tar"]
        }
    )
    for path, expected in inputs.items():
        require(file_hash(path) == expected, f"Pinned input changed: {path}")
    train = verified_cases(collection, "train")
    dev = verified_cases(collection, "development")
    instances, balance = training_instances(train)
    require(
        len(instances) == 1040 and len(train) == 272 and len(dev) == 72,
        "Frozen train/development grid differs",
    )
    sys.path.insert(0, UPSTREAM)
    config = importlib.import_module("src.train_loracle.config").load_train_config(
        Path(CHECKPOINT) / "loracle_config.yaml"
    )
    builder = importlib.import_module("src.train_loracle.dataset").build_placeholder_prefix_ids
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
    handle = SimpleNamespace(config=config, tokenizer=tokenizer, device=torch.device("cpu"))
    configure_interpreter_prefix(handle, builder)
    option_ids = option_token_ids(tokenizer)
    lengths = [
        prepared_input(handle, row["text"])[0].shape[1]
        for case in train + dev
        for row in question_views(case)
    ]
    require(len(lengths) == 1376, "Incomplete tokenizer validation")
    source_paths = subprocess.check_output(
        ["git", "ls-files", "src/gradient_interp"], text=True
    ).splitlines() + ["scripts/prepare_update_loracle_factorized.py"]
    source_hashes = {p: file_hash(p) for p in source_paths if p.endswith(".py")}
    root.mkdir(parents=True)
    shutil.copyfile(collection / "data.json", root / "data.json")
    analysis = {
        "frozen_unix": time.time(),
        "scope": "development_only",
        "proposal": read(
            Path("results/update-loracle-program/provenance/factorized-development-proposal.json")
        ),
        "previous_failure": (
            "v1 12/64 full categories, 64/64 lowercase predictions; all v1 outputs retained"
        ),
        "selection": (
            "Earliest of three complete epochs with highest exact full-category "
            "accuracy on all 64 actual development updates"
        ),
        "primary_comparison": (
            "Unchanged released LoRAcle using the SAME four questions and valid-head choice scoring"
        ),
        "baselines": (
            "Training flat majority AND factorized majority, training-only Ridge, "
            "fixed matched direction shuffle"
        ),
        "development_gates": {
            "exact_accuracy": 0.70,
            "released_advantage": 0.20,
            "both_majority_advantages": 0.10,
            "matched_shuffle_advantage": 0.10,
            "zero_accuracy": 0.90,
            "above_effect_floor_fraction": 0.80,
        },
        "uncertainty": (
            "2000 paired crossed source-composition/initialization/case resamples, "
            "seed2026101653; descriptive95% intervals, no selection-adjusted or "
            "confirmation claim"
        ),
        "retention": (
            "All 3 epochs, all "
            "methods/questions/classes/seeds/compositions/zeros/weak cases, all "
            "logits and joint predictions. Per-axis success cannot rescue joint "
            "failure."
        ),
        "reserved_test_access": (
            "Forbidden. Reserved and fresh confirmation require later model-selection "
            "lock and unchanged original strong criteria."
        ),
        "failure_policy": (
            "Preserve failures and first-launch deadline; no automatic repair or resume"
        ),
        "training_seed": TRAINING_SEED,
        "option_order_seed": OPTION_SEED,
        "torch_and_preflight_seed": 2026101680,
        "weights": balance,
        "initialization": "Unchanged released rank256 rsLoRA, not selected v1 model",
    }
    atomic_json(root / "analysis-plan.json", analysis)
    plan = {
        "created_unix": time.time(),
        "scope": "development_only",
        "max_hours": 12,
        "recipe": "balanced_factorized_v2",
        "source_commit": commit,
        "source_files_sha256": source_hashes,
        "input_files_sha256": inputs,
        "collection_root": str(collection),
        "prior_training_root": str(prior),
        "collection_integrity_sha256": file_hash(collection / "integrity.json"),
        "data_sha256": file_hash(root / "data.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "epochs": 3,
        "effective_batch": 4,
        "learning_rate": 0.00003,
        "training_seed": TRAINING_SEED,
        "question_instances": 1040,
        "optimizer_steps": 780,
        "warmup_steps": 39,
        "weight_decay": 0.01,
        "clip_norm": 1,
        "jobs": [
            {
                "id": "train",
                "module": "gradient_interp.update_loracle_factorized",
                "arguments": ["train", "--root", str(root)],
                "manifest": str(root / "manifest.json"),
            }
        ],
    }
    atomic_json(root / "plan.json", plan)
    with tarfile.open(root / "source_snapshot.tar", "x") as tar:
        for path in source_hashes:
            tar.add(path, arcname=path, recursive=False)
    atomic_json(
        root / "preparation.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "source_commit": commit,
            "plan_sha256": file_hash(root / "plan.json"),
            "source_archive_sha256": file_hash(root / "source_snapshot.tar"),
            "source_files": len(source_hashes),
            "input_files": len(inputs),
            "tokenizer_questions": len(lengths),
            "max_input_tokens": max(lengths),
            "direction_positions": 4480,
            "option_ids": option_ids,
            "gpu_preflight": "pending inside worker before matched comparisons/training",
            "versions": {
                p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft"]
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["collection", "prior", "root"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    prepare(args.collection, args.prior, args.root, args.commit)
