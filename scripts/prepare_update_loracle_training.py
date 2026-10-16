"""Freeze the already specified first training recipe after collection passes its audit."""

import argparse
import importlib.metadata
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from gradient_interp.loracle_trajectory_toy import read
from gradient_interp.study import atomic_json, file_hash


def prepare(collection, root, expected_commit):
    collection = collection.resolve()
    if root.exists():
        raise ValueError("Training root already exists; never silently resume or overwrite")
    integrity = read(collection / "integrity.json")
    if not integrity["passed"] or integrity["cases"] != 344:
        raise ValueError("Successful full collection audit required")
    for name, expected in integrity["files_sha256"].items():
        if file_hash(collection / name) != expected:
            raise ValueError(f"Audited collection changed: {name}")
    if file_hash(collection / "source_snapshot.tar") != integrity["source_archive_sha256"]:
        raise ValueError("Original source archive missing or changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != expected_commit:
        raise ValueError("Wrong frozen training source")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Remote checkout must be clean")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip():
        raise ValueError("GPU is occupied")
    old_plan = read(collection / "plan.json")
    for path, expected in old_plan["input_files_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Pinned model or upstream input changed: {path}")
    source_paths = subprocess.check_output(
        ["git", "ls-files", "src/gradient_interp"], text=True
    ).splitlines()
    source_paths += ["scripts/prepare_update_loracle_training.py"]
    source_hashes = {p: file_hash(p) for p in source_paths if p.endswith(".py")}
    root.mkdir(parents=True)
    shutil.copyfile(collection / "data.json", root / "data.json")
    provenance = Path("results/update-loracle-program/provenance")
    analysis = {
        "frozen_unix": time.time(),
        "scope": "development_only",
        "original_implementation": read(provenance / "training-implementation-plan.json"),
        "original_analysis": read(provenance / "training-analysis-plan.json"),
        "input_tokenization_amendment": read(
            provenance / "repairs" / "2026-10-16-choice-tokenization/amendment.json"
        ),
        "recipe_selected_before_collection_outcomes": True,
        "no_outcome_dependent_decision_in_preparation": True,
        "reserved_test_access": "Forbidden to this worker; separate freeze required later",
        "failure_policy": "Stop and preserve failures; no automatic repair or resume",
    }
    atomic_json(root / "analysis-plan.json", analysis)
    plan = {
        "created_unix": time.time(),
        "scope": "development_only",
        "max_hours": 12,
        "source_commit": commit,
        "source_files_sha256": source_hashes,
        "input_files_sha256": old_plan["input_files_sha256"],
        "collection_root": str(collection),
        "collection_integrity_sha256": file_hash(collection / "integrity.json"),
        "data_sha256": file_hash(root / "data.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "epochs": 3,
        "effective_batch": 4,
        "learning_rate": 0.00003,
        "training_seed": 2026101651,
        "jobs": [
            {
                "id": "train",
                "module": "gradient_interp.update_loracle_train",
                "arguments": ["train", "--root", str(root.resolve())],
                "manifest": str((root / "manifest.json").resolve()),
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
            "collection_integrity_sha256": plan["collection_integrity_sha256"],
            "source_files": len(source_hashes),
            "input_files": len(plan["input_files_sha256"]),
            "fixed_recipe_no_outcome_based_changes": True,
            "gpu_preflight": "pending inside worker",
            "versions": {
                p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft"]
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    prepare(args.collection, args.root, args.commit)
