"""Freeze both untouched evaluation grids after complete audited development selection."""

import argparse
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from gradient_interp.loracle_trajectory_toy import read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_analysis import require
from gradient_interp.update_loracle_confirmation_data import freeze as freeze_data
from gradient_interp.update_loracle_inference import (
    COMPONENT_INTERFACE,
    freeze_selection,
    interface_contract,
    selected_interface,
)


def prepare(training, analysis, root, expected_commit):
    require(not root.exists(), "Held-out root exists; never overwrite or retry automatically")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(commit == expected_commit, "Incorrect frozen evaluation source")
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(),
        "Remote checkout must be clean",
    )
    require(
        not subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip(),
        "GPU is occupied",
    )
    selection = freeze_selection(training, analysis, root / "selection.json")
    freeze_data(root / "selection.json", root / "data.json")
    interface = interface_contract(selected_interface(selection))
    training_plan = read(training / "plan.json")
    inputs = dict(training_plan["input_files_sha256"])
    inputs.update(
        {
            str(Path(selection["adapter_directory"]) / name): expected
            for name, expected in selection["adapter_files_sha256"].items()
        }
    )
    for path in [
        training / "integrity.json",
        training / "plan.json",
        training / "data.json",
        training / "source_snapshot.tar",
        Path(training_plan["collection_root"]) / "integrity.json",
    ]:
        inputs[str(path.resolve())] = file_hash(path)
    for path, expected in inputs.items():
        require(file_hash(path) == expected, "Frozen evaluation input changed")
    source_paths = subprocess.check_output(
        ["git", "ls-files", "src/gradient_interp"], text=True
    ).splitlines()
    source_paths += ["scripts/prepare_update_loracle_heldout.py"]
    source_hashes = {p: file_hash(p) for p in source_paths if p.endswith(".py")}
    dated_stats = Path(
        "results/update-loracle-program/provenance/confirmation-statistics-plan.json"
    )
    shutil.copyfile(dated_stats, root / "original-statistics-plan.json")
    atomic_json(
        root / "analysis-plan.json",
        {
            "frozen_unix": time.time(),
            "purpose": "Test the one selected model on the original reserved set "
            "and an additional fresh draw. No model selection, training or tuning on either set.",
            "original_statistics_plan": read(dated_stats),
            "question_interface": interface,
            "component_method": (
                "Use the exact four-question development interface for both "
                "models; only the recomposed complete category counts as correct. Require >=10pp "
                "advantage over both fixed training-majority baselines. No source task or batch "
                "text enters interpreter input."
                if interface["name"] == COMPONENT_INTERFACE
                else "Use the original matched flat33 question."
            ),
            "primary": "Fresh confirmation: all 128 actual updates, including weak effects. "
            "The 128 reserved updates are a required generalization check with the same model.",
            "strong_targets": {
                "accuracy": 0.70,
                "advantage_vs_released": 0.20,
                "accuracy_lower_bound": 0.60,
                "advantage_lower_bound": 0.10,
                "measurable_effect_fraction": 0.80,
                "zero_control_accuracy": 0.90,
                "majority_advantage": 0.10,
                "shuffled_direction_advantage": 0.10,
            },
            "uncertainty": "4000 paired crossed composition/seed/case resamples, seed 2026101665. "
            "Two 97.5% marginal intervals target 95% simultaneous coverage within each grid. "
            "No joint coverage claim across both grids. Exact composition method-swap test "
            "uses 256 assignments.",
            "execution": "Training-only target and model-loading preflight. Create all 272 "
            "updates without test probe scoring. Lock every model/reference prediction, then "
            "score all 4x32 probe responses before/after each update. Both models use identical "
            "native tokens, options and physical update norms.",
            "negative_cases": "Retain all methods, cases, profiles and per-axis/composition/seed "
            "diagnostics. Retain zero controls, top-two margins and weak updates. "
            "No clipping or exclusion.",
            "scope": "Familiar operations in held-out source compositions, new nonce words "
            "and initializations, one fixed base. Finite-reference loss preference only.",
            "success": "Both grids must meet every numerical criterion. Inspect the full evidence "
            "and audit before claiming completion. Preserve failed evaluations and never reuse "
            "them as untouched evidence for a later selected model.",
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
        "evaluation_scope": "reserved_and_fresh_confirmation",
        "max_hours": 12,
        "source_commit": commit,
        "source_files_sha256": source_hashes,
        "input_files_sha256": inputs,
        "data_sha256": file_hash(root / "data.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "selection_sha256": file_hash(root / "selection.json"),
        "question_interface": interface,
        "original_data_path": str((training / "data.json").resolve()),
        "collection_root": training_plan["collection_root"],
        "learning_rate": 0.0002,
        "jobs": [
            {
                "id": stage,
                "module": "gradient_interp.update_loracle_heldout",
                "arguments": [stage, "--root", str(root.resolve())],
                "manifest": str((root / name).resolve()),
            }
            for stage, name in manifests.items()
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
            "selection_sha256": plan["selection_sha256"],
            "source_archive_sha256": file_hash(root / "source_snapshot.tar"),
            "selected_epoch": selection["selected_epoch"],
            "cases": 272,
            "gpu_preflight": "pending first controller stage; no test collection before it passes",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    prepare(args.training, args.analysis, args.root, args.commit)
