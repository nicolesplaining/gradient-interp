"""Freeze one calibration-gradient development decoder before any new prediction."""

import argparse
import hashlib
import importlib.metadata
import subprocess
import tarfile
import time
from pathlib import Path

from transformers import AutoTokenizer

from gradient_interp.loracle_trajectory_toy import BASE_PATH, encoded, read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_analysis import require
from gradient_interp.update_loracle_data import PATTERNS
from gradient_interp.update_loracle_factorized_analysis import completion_barrier
from gradient_interp.update_loracle_sensitivity import calibration_rows, select_calibration


def prepare(collection, prior, root, expected_commit):
    collection, prior, root = collection.resolve(), prior.resolve(), root.resolve()
    require(not root.exists(), "New recipe requires a new run root")
    completion_barrier(read(prior / "controller.json"))
    prior_audit = read(prior / "integrity.json")
    require(
        prior_audit["passed"] and prior_audit["recipe"] == "physical_profile_v3",
        "Complete physical-profile audit required",
    )
    require(
        not read(prior / "analysis/summary.json")["all_development_gates_pass"],
        "New recipe requires the complete previous failure",
    )
    for name, expected in prior_audit["files_sha256"].items():
        require(file_hash(prior / name) == expected, f"Previous audited file changed: {name}")
    prior_plan = read(prior / "plan.json")
    with tarfile.open(prior / "source_snapshot.tar") as archive:
        require(
            set(archive.getnames()) == set(prior_plan["source_files_sha256"]), "Incomplete archive"
        )
        for name, expected in prior_plan["source_files_sha256"].items():
            require(
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                "Original physical-profile source changed",
            )
    baseline = Path(prior_plan["prior_root"])
    collection_audit = read(collection / "integrity.json")
    require(
        collection_audit["passed"] and collection_audit["cases"] == 344,
        "Complete original collection required",
    )
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit,
        "Wrong source commit",
    )
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
    data = read(collection / "data.json")
    calibration = select_calibration(data)
    cases = []
    paths = [collection / n for n in ("integrity.json", "data.json", "plan.json")]
    paths += [
        prior / n
        for n in (
            "integrity.json",
            "plan.json",
            "analysis/summary.json",
            "predictions.json",
            "source_snapshot.tar",
        )
    ]
    paths += [baseline / "analysis/all-development-cases.json", baseline / "integrity.json"]
    for c in data["cases"]:
        if c["split"] not in ("train", "development"):
            continue
        record_path = collection / "cases" / f"{c['id']}.json"
        record = read(record_path)
        require(
            file_hash(record_path)
            == collection_audit["files_sha256"][str(record_path.relative_to(collection))],
            "Original case metadata changed",
        )
        cases.append(
            {
                **{k: c[k] for k in ("id", "split", "control", "initialization_seed")},
                "physical_update_norm": record["physical_update_norm"],
            }
        )
        paths.extend([record_path, collection / "snapshots" / f"{c['id']}.pt"])
    require(len(cases) == 344, "Wrong train/development case count")
    paths.extend(
        collection / "snapshots" / f"initial-{s}.pt"
        for s in sorted({c["initialization_seed"] for c in cases})
    )
    collection_plan = read(collection / "plan.json")
    base_inputs = {
        p: v
        for p, v in collection_plan["input_files_sha256"].items()
        if Path(p).is_relative_to(BASE_PATH)
    }
    require(
        any(p.endswith(".safetensors") for p in base_inputs)
        and str(Path(BASE_PATH) / "tokenizer.json") in base_inputs,
        "Missing original pinned base/tokenizer inputs",
    )
    inputs = {str(p): file_hash(p) for p in paths}
    for p, expected in base_inputs.items():
        require(file_hash(p) == expected, "Pinned base model/tokenizer changed")
        inputs[p] = expected
    for case in cases:
        record = read(collection / "cases" / f"{case['id']}.json")
        require(
            inputs[str(collection / "snapshots" / f"{case['id']}.pt")] == record["snapshot_sha256"]
            and inputs[str(collection / "snapshots" / f"initial-{case['initialization_seed']}.pt")]
            == record["initial_sha256"],
            "Original native snapshot changed",
        )
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
    lengths = [
        len(encoded(tokenizer, r["prompt"], r["answer"])[0])
        for pattern in PATTERNS
        for r in calibration_rows(calibration, pattern)
    ]
    require(len(lengths) == 256 and max(lengths) <= 256, "Calibration tokenization failed")
    source_paths = subprocess.check_output(
        ["git", "ls-files", "src/gradient_interp"], text=True
    ).splitlines()
    source_paths.append("scripts/prepare_update_loracle_sensitivity.py")
    source = {p: file_hash(p) for p in source_paths if p.endswith(".py")}
    root.mkdir(parents=True)
    atomic_json(
        root / "data.json",
        {
            "cases": cases,
            "calibration": calibration,
            "collection_data_sha256": file_hash(collection / "data.json"),
        },
    )
    proposal = read(
        Path("results/update-loracle-program/provenance/base-aware-sensitivity-proposal.json")
    )
    analysis = {
        "frozen_unix": time.time(),
        "proposal": proposal,
        "exact_recipe": {
            "initializations": 8,
            "calibration_probes": 8,
            "patterns": 32,
            "chunks": 4,
            "gradient_dtype": "float32",
            "contraction_dtype": "float64",
            "max_hours": 4,
            "torch_seed": 2026101701,
            "probe_ce": "equal per-document mean completion CE with original EOS",
            "gradient_average": "sum per-document means over8, two backward chunks of4",
            "no_optimizer_or_fitted_scale": True,
            "preflight": (
                "All560 FP32 native tensors; finite gradients; A gradients exactly0 atB0; "
                "signed coordinate contraction matches independent scalar sum rtol1e-7 atol1e-8; "
                "syntheticB=-.001*gradientB perturbation and bit-exact weight/scoring restore; "
                "frozen base version counters unchanged"
            ),
            "restoration": (
                "All280 native modules unchanged after every gradient and restored each seed; "
                "all256 baseline calibration score vectors bit-exact across zero-B initializations"
            ),
            "audit": (
                "All files/source/initializations/snapshots/gradient-bank hashes; recompute "
                "every native contraction by independent elementwise FP64 sums and both split "
                "direction shuffles; recompute pattern0 gradients for first training and first "
                "development initialization rtol1e-5 atol1e-7, calibration CE bit-exact. "
                "No inferential source-task labels or evaluation probe text."
            ),
        },
        "selection": "No fitted hyperparameter. All344 predictions fixed before analysis.",
        "release": (
            "Normally complete dead controller+worker and full integrity audit. "
            "No early reading or automatic retry."
        ),
        "retention": (
            "All profiles, per-module gradient banks, training/development predictions, "
            "all controls, every initialization/composition and prior baselines; profile errors "
            "are diagnostics, unchanged full-category gates primary."
        ),
    }
    atomic_json(root / "analysis-plan.json", analysis)
    plan = {
        "created_unix": time.time(),
        "scope": "development_only",
        "max_hours": 4,
        "recipe": "base_sensitivity_v4",
        "source_commit": expected_commit,
        "source_files_sha256": source,
        "input_files_sha256": inputs,
        "collection_root": str(collection),
        "prior_root": str(prior),
        "baseline_root": str(baseline),
        "data_sha256": file_hash(root / "data.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "jobs": [
            {
                "id": "predict",
                "module": "gradient_interp.update_loracle_sensitivity",
                "arguments": ["--root", str(root)],
                "manifest": str(root / "manifest.json"),
            }
        ],
    }
    atomic_json(root / "plan.json", plan)
    with tarfile.open(root / "source_snapshot.tar", "x") as archive:
        for p in source:
            archive.add(p, arcname=p, recursive=False)
    atomic_json(
        root / "preparation.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "source_commit": expected_commit,
            "plan_sha256": file_hash(root / "plan.json"),
            "source_archive_sha256": file_hash(root / "source_snapshot.tar"),
            "source_files": len(source),
            "input_files": len(inputs),
            "calibration_tokenizations": len(lengths),
            "maximum_input_tokens": max(lengths),
            "gpu_preflight": "pending inside worker before predictions",
            "versions": {
                p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("collection", "prior", "root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    prepare(args.collection, args.prior, args.root, args.commit)
