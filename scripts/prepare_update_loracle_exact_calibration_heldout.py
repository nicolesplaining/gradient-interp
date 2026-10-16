"""Freeze two untouched grids only after the exact functional predictor passes development."""

import argparse
import copy
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from gradient_interp.loracle_trajectory_toy import CHECKPOINT, read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_analysis import require
from gradient_interp.update_loracle_confirmation_data import fresh_draw, validate_fresh
from gradient_interp.update_loracle_exact_calibration_heldout import RECIPE, checked_data
from gradient_interp.update_loracle_exact_calibration_inference import (
    freeze_selection,
    verified_selection,
)
from gradient_interp.update_loracle_factorized_analysis import completion_barrier

# This is the original matched released-comparator plan, recorded before v5 outcomes.
RELEASED_PLAN_SHA256 = "fb073d22ca1ba21a945760f9e64bda554ac6ff485a2cc808a888da05bb24a075"


def freeze_data(selection_path, output):
    require(not output.exists(), "Test data already frozen; never redraw")
    selection = verified_selection(selection_path)
    collection = Path(selection["collection_root"])
    development_plan = read(Path(selection["development_root"]) / "plan.json")
    original_path = collection / "data.json"
    require(
        file_hash(original_path) == development_plan["input_files_sha256"][str(original_path)],
        "Original full data changed",
    )
    original = read(original_path)
    fresh = fresh_draw(original)
    result = {
        "schema": "single-update-heldout-data-v1",
        "created_unix": time.time(),
        "selection_sha256": file_hash(selection_path),
        "original_data_sha256": file_hash(original_path),
        "reserved": {
            "cases": copy.deepcopy([c for c in original["cases"] if c["split"] == "reserved"]),
            "vocabulary": list(original["vocabularies"]["reserved"]),
        },
        "confirmation": fresh,
        "confirmation_validation": validate_fresh(fresh, original),
        "sampling_rule": (
            "Original reserved grid and original deterministic fresh draw. Both "
            "freeze before any test measurement; no redraw or outcome-selected "
            "changes."
        ),
        "scope": (
            "Held-out compositions of familiar operations, new words/prompts and "
            "initializations, one reused base; no semantic/pretraining independence "
            "claim."
        ),
    }
    require(len(checked_data(result, original)) == 272, "Incomplete two-grid data")
    atomic_json(output, result)
    return result


def released_assets(baseline):
    """Bind the same released LoRAcle checkpoint used in the original v2 comparison."""
    completion_barrier(read(baseline / "controller.json"))
    integrity = read(baseline / "integrity.json")
    require(
        integrity["passed"]
        and integrity["recipe"] == "balanced_factorized_v2"
        and integrity["source_commit"] == "8e72b517a4c24c494bfb0e0fa22c64a6342adbcd"
        and integrity["reserved_cases_accessed"] == 0,
        "Original comparator audit required",
    )
    require(
        file_hash(baseline / "plan.json")
        == integrity["files_sha256"]["plan.json"]
        == RELEASED_PLAN_SHA256,
        "Original released-comparator plan changed",
    )
    plan = read(baseline / "plan.json")
    files = dict(plan["input_files_sha256"])
    checkpoint = {p: h for p, h in files.items() if Path(p).is_relative_to(CHECKPOINT)}
    require(
        checkpoint and any(p.endswith(".safetensors") for p in checkpoint),
        "Released checkpoint inventory missing",
    )
    for path, expected in files.items():
        require(file_hash(path) == expected, "Original released model changed")
    return files


def prepare(development, analysis, root, expected_commit):
    development, analysis, root = development.resolve(), analysis.resolve(), root.resolve()
    require(not root.exists(), "Test root exists; never overwrite or automatically retry")
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit,
        "Incorrect frozen evaluation source",
    )
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(),
        "Remote checkout must be clean",
    )
    require(
        not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip(),
        "GPU is occupied",
    )
    # This barrier recomputes all development gates before creating any new data.
    selection = freeze_selection(development, analysis, root / "selection.json")
    freeze_data(root / "selection.json", root / "data.json")
    prior_plan = read(development / "plan.json")
    baseline = Path(selection["baseline_root"])
    released = released_assets(baseline)
    require(
        all(
            path not in prior_plan["input_files_sha256"]
            or prior_plan["input_files_sha256"][path] == expected
            for path, expected in released.items()
        ),
        "Functional and released dependencies disagree",
    )
    inputs = {**prior_plan["input_files_sha256"], **released}
    for path in (
        development / "integrity.json",
        development / "plan.json",
        development / "data.json",
        development / "source_snapshot.tar",
        baseline / "plan.json",
        baseline / "integrity.json",
        Path(selection["collection_root"]) / "integrity.json",
    ):
        inputs[str(path)] = file_hash(path)
    inputs.update(selection["evidence_sha256"])
    for path, expected in inputs.items():
        require(file_hash(path) == expected, "Frozen evaluation input changed")
    source_paths = subprocess.check_output(
        ["git", "ls-files", "src/gradient_interp"], text=True
    ).splitlines()
    source_paths.append("scripts/prepare_update_loracle_exact_calibration_heldout.py")
    sources = {p: file_hash(p) for p in source_paths if p.endswith(".py")}
    dated_stats = Path(
        "results/update-loracle-program/provenance/confirmation-statistics-plan.json"
    )
    shutil.copyfile(dated_stats, root / "original-statistics-plan.json")
    interface = selection["released_interface"]
    atomic_json(
        root / "analysis-plan.json",
        {
            "frozen_unix": time.time(),
            "recipe": RECIPE,
            "purpose": (
                "One selected exact-calibration method, frozen before either untouched "
                "outcome grid. No new fitting or model selection on test data."
            ),
            "original_statistics_plan": read(dated_stats),
            "question_interface": interface,
            "prediction": (
                "Exact same fixed8 training calibration prompts/32 answer patterns/FP32CE "
                "and FP64 averaging as development. Apply native own and norm-matched "
                "shuffled donor updates, preserving each donor's A. Never rescale an "
                "already nonlinear loss profile."
            ),
            "comparison": (
                "Released LoRAcle uses the unchanged matched four-question interface, "
                "released encoding and recipient physical norm. Fixed training-only "
                "flat/factorized majorities and ridge are retained. Functional method "
                "additionally requires base-model and calibration-library access; no "
                "equal-compute claim."
            ),
            "primary": (
                "All128 actual updates in each of the original reserved and fresh "
                "confirmation grids. Both must pass; all weak changes retained, eight "
                "zero controls per grid separate."
            ),
            "strong_targets": {
                "accuracy": 0.70,
                "advantage_vs_released": 0.20,
                "accuracy_lower_bound": 0.60,
                "advantage_lower_bound": 0.10,
                "measurable_effect_fraction": 0.80,
                "zero_control_accuracy": 0.90,
                "flat_majority_advantage": 0.10,
                "factorized_majority_advantage": 0.10,
                "shuffled_direction_advantage": 0.10,
            },
            "uncertainty": (
                "Original4000 paired crossed composition/seed/case bootstrap, "
                "seed2026101665. Two97.5% marginal bounds target95% family coverage "
                "within each grid; no joint two-grid coverage claim. Original256 "
                "whole-composition method swaps."
            ),
            "execution": (
                "Training-only target/native-scoring/released-interface preflight, then "
                "all272 independent native updates. Lock all544 exact own/shuffled "
                "profiles and all comparator predictions across BOTH grids before "
                "measuring any new evaluation probe."
            ),
            "audit": (
                "Four successful dead stages, original12hdeadline, all "
                "source/model/input/native/token/score/prediction/outcome hashes, "
                "complete grids and exact restores. Recompute encoding, physical norms, "
                "score means and categories, donor assignments. Independently re-score "
                "first actual case of each split, own and shuffled, including exact "
                "restores."
            ),
            "report": (
                "All methods/grids/cases/axes/compositions/initializations and profile "
                "errors retained. Do not filter weak cases or small margins, clip "
                "negative values or promote a partial-axis winner. Preserve executed "
                "code, bootstrap arrays, paired examples and all failures."
            ),
            "scope": (
                "One-step finite-reference preference on known operations with unseen "
                "source compositions, words and initializations. This is base-assisted "
                "functional profiling, not an improved learned LoRAcle checkpoint or "
                "unrestricted generation forecasting."
            ),
            "success": (
                "Both grids must pass every original numerical criterion plus complete "
                "integrity/untouched-data review. Failed tests cannot become untouched "
                "confirmation for a later method."
            ),
        },
    )
    manifests = {
        "preflight": "preflight.json",
        "updates": "updates-manifest.json",
        "predict": "predictions-manifest.json",
        "measure": "manifest.json",
    }
    plan = {
        "created_unix": time.time(),
        "scope": "heldout_confirmation",
        "recipe": RECIPE,
        "evaluation_scope": "reserved_and_fresh_confirmation",
        "max_hours": 12,
        "source_commit": expected_commit,
        "source_files_sha256": sources,
        "input_files_sha256": inputs,
        "released_files_sha256": released,
        "data_sha256": file_hash(root / "data.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "selection_sha256": file_hash(root / "selection.json"),
        "question_interface": interface,
        "original_data_path": str(Path(selection["collection_root"]) / "data.json"),
        "collection_root": selection["collection_root"],
        "baseline_root": selection["baseline_root"],
        "learning_rate": 0.0002,
        "jobs": [
            {
                "id": stage,
                "module": "gradient_interp.update_loracle_exact_calibration_heldout",
                "arguments": [stage, "--root", str(root)],
                "manifest": str(root / name),
            }
            for stage, name in manifests.items()
        ],
    }
    atomic_json(root / "plan.json", plan)
    with tarfile.open(root / "source_snapshot.tar", "x") as archive:
        for path in sources:
            archive.add(path, arcname=path, recursive=False)
    atomic_json(
        root / "preparation.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "source_commit": expected_commit,
            "plan_sha256": file_hash(root / "plan.json"),
            "selection_sha256": plan["selection_sha256"],
            "source_archive_sha256": file_hash(root / "source_snapshot.tar"),
            "cases": 272,
            "exact_scoring_conditions": 544,
            "gpu_preflight": "pending first controller stage; no test collection before passing",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    prepare(args.development, args.analysis, args.root, args.commit)
