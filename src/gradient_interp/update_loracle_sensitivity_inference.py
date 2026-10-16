"""Guard selection and native-update inference for the fixed base-assisted decoder.

This is not a learned LoRAcle adapter. Test data cannot be unlocked by a manually
edited summary, a partial run, an axis-only score, or an unrelated gradient bank.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from .loracle_trajectory_toy import (
    BASE_PATH,
    fingerprint,
    load_target,
    read,
    restore_modules,
    snapshot_modules,
)
from .study import atomic_json, file_hash
from .update_loracle_analysis import metrics, require
from .update_loracle_data import LABELS, PATTERNS, description
from .update_loracle_factorized_analysis import completion_barrier
from .update_loracle_inference import COMPONENT_INTERFACE, interface_contract
from .update_loracle_physical_profile import profile_label, source_keys
from .update_loracle_sensitivity import (
    calibration_rows,
    centered_profile,
    contract,
    frozen_versions,
    gradient_b,
    select_calibration,
)
from .update_loracle_sensitivity_analysis import development_gates, verify_outputs

RECIPE = "base_sensitivity_v4"
SCHEMA = "base-assisted-single-update-selection-v1"


def predictor_contract():
    return {
        "recipe": RECIPE,
        "base_model": BASE_PATH,
        "native_adapter": {
            "rank": 16,
            "alpha": 32,
            "modules": 280,
            "tensors": 560,
            "parameters": 64225280,
        },
        "labels": list(LABELS),
        "calibration_prompts": 8,
        "candidate_patterns": [list(p) for p in PATTERNS],
        "loss": "equal document mean completion CE including original EOS",
        "gradient_average": "eight documents in two backward chunks of four",
        "prediction": "sum native dL/dB times actual B; mean CE change minus each CE change",
        "no_change_span": 0.02,
        "tie_rule": "first pattern in fixed order",
        "gradient_dtype": "float32",
        "contraction_dtype": "float64",
        "inference_inputs": [
            "initial native A with zero B",
            "updated native A/B",
            "gradient bank for this exact initial A",
        ],
        "bank_dependencies": [
            "fixed pinned base model",
            "frozen eight training prompts",
            "fixed 32 reference-answer patterns",
        ],
        "excluded_inputs": [
            "update training examples",
            "source category",
            "new evaluation prompts",
            "measured effects",
        ],
        "scope": "One independent first optimizer update with unchanged A and initial B=0; "
        "finite reference-pattern preference, not unrestricted generated behavior",
        "comparison_limitation": "Uses base-model/task-library access beyond ordinary LoRAcle",
    }


def recomputed_metrics(cases, rows):
    require(
        len(cases) == len(rows) == 72
        and sum(not c["control"] for c in cases) == 64
        and [c["id"] for c in cases] == [r["id"] for r in rows]
        and len({c["id"] for c in cases}) == 72,
        "Full 64-update/eight-control development grid required",
    )
    effects = {c["id"]: c["measured"] for c in cases}
    require(
        all(c["label"] == effects[c["id"]]["effect"]["label"] for c in cases),
        "Display label differs from measured target",
    )
    predictions = {
        n: [c["predictions"][n] for c in cases]
        for n in ("released", "flat_majority", "factorized_majority")
    }
    predictions["sensitivity"] = [r["prediction"] for r in rows]
    predictions["sensitivity_shuffled"] = [r["shuffled_prediction"] for r in rows]
    return {n: metrics(v, cases, effects) for n, v in predictions.items()}


def checked_development(summary, cases, rows):
    recomputed = recomputed_metrics(cases, rows)
    gates = development_gates(recomputed)
    require(
        summary["recipe"] == RECIPE
        and not summary["reserved_test_read"]
        and all(summary["metrics"][n] == v for n, v in recomputed.items())
        and summary["development_gates"] == gates
        and summary["all_development_gates_pass"] is True
        and all(gates.values()),
        "Full development criteria failed; untouched data remain sealed",
    )
    return recomputed, gates


def freeze_selection(root, analysis, output):
    require(not output.exists(), "Never overwrite an existing selection lock")
    completion_barrier(read(root / "controller.json"))
    integrity = read(root / "integrity.json")
    require(
        integrity["passed"]
        and integrity["recipe"] == RECIPE
        and integrity["cases"] == 344
        and integrity["gradient_banks"] == 256
        and integrity["all_native_contractions_verified"] is True
        and len(integrity["derivative_recomputations"]) == 2
        and all(r["passed"] is True for r in integrity["derivative_recomputations"])
        and integrity["reserved_cases_accessed"] == 0,
        "Complete sensitivity audit required",
    )
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited sensitivity output changed")
    plan = read(root / "plan.json")
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, "Original development input changed")
    data = read(root / "data.json")
    collection = Path(plan["collection_root"])
    require(
        data["calibration"] == select_calibration(read(collection / "data.json")),
        "Training-only calibration selection changed",
    )
    provenance = read(analysis / "analysis-provenance.json")
    require(
        provenance["integrity_sha256"] == file_hash(root / "integrity.json")
        and provenance["executed_source_sha256"] == file_hash(analysis / "executed-analysis.py"),
        "Analysis source or parent audit changed",
    )
    rows = read(root / "predictions.json")
    verify_outputs(np.load(root / "raw-ce-predictions.npy"), rows, data["cases"])
    baseline = Path(plan["baseline_root"]) / "analysis/all-development-cases.json"
    cases = read(baseline)
    require(
        file_hash(baseline) == plan["input_files_sha256"][str(baseline)],
        "Frozen comparison changed",
    )
    for c in cases:
        require(
            c["measured"] == read(collection / "cases" / f"{c['id']}.json"),
            "Baseline measured effects differ from original audited collection",
        )
    dev_rows = [r for r in rows if r["split"] == "development"]
    summary = read(analysis / "summary.json")
    recomputed, gates = checked_development(summary, cases, dev_rows)
    from . import update_loracle_sensitivity as worker

    require(
        file_hash(worker.__file__)
        == plan["source_files_sha256"]["src/gradient_interp/update_loracle_sensitivity.py"],
        "Predictor source changed after development",
    )
    selected_files = {
        str(p.resolve()): file_hash(p)
        for p in (
            root / "integrity.json",
            root / "plan.json",
            root / "data.json",
            analysis / "summary.json",
            analysis / "analysis-provenance.json",
            analysis / "executed-analysis.py",
            baseline,
        )
    }
    lock = {
        "schema": SCHEMA,
        "frozen_unix": time.time(),
        "predictor": predictor_contract(),
        "development_root": str(root.resolve()),
        "development_source": plan["source_commit"],
        "collection_root": str(collection),
        "baseline_root": plan["baseline_root"],
        "calibration": data["calibration"],
        "gates": gates,
        "base_files_sha256": {
            p: h for p, h in plan["input_files_sha256"].items() if Path(p).is_relative_to(BASE_PATH)
        },
        "development_metrics": recomputed,
        "evidence_sha256": selected_files,
        "worker_source_sha256": file_hash(worker.__file__),
        "inference_source_sha256": file_hash(__file__),
        "released_interface": interface_contract(COMPONENT_INTERFACE),
        "selection_rule": "Single fixed sensitivity method; all full-category gates must pass",
        "reserved_test_accessed": False,
        "confirmation_complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, lock)
    return lock


def verified_selection(path):
    from . import update_loracle_sensitivity as worker

    lock = read(path)
    require(
        lock["schema"] == SCHEMA
        and lock["predictor"] == predictor_contract()
        and lock["worker_source_sha256"] == file_hash(worker.__file__)
        and lock["inference_source_sha256"] == file_hash(__file__)
        and lock["released_interface"] == interface_contract(COMPONENT_INTERFACE)
        and lock["reserved_test_accessed"] is False,
        "Selected predictor/source/input contract changed",
    )
    require(
        lock["gates"] == development_gates(lock["development_metrics"])
        and all(lock["gates"].values()),
        "Development criteria do not pass",
    )
    for p, expected in lock["evidence_sha256"].items():
        require(file_hash(p) == expected, "Selection evidence changed")
    require(
        lock["calibration"] == read(Path(lock["development_root"]) / "data.json")["calibration"],
        "Selected calibration changed",
    )
    plan = read(Path(lock["development_root"]) / "plan.json")
    require(
        lock["base_files_sha256"]
        == {
            p: h for p, h in plan["input_files_sha256"].items() if Path(p).is_relative_to(BASE_PATH)
        }
        and any(p.endswith(".safetensors") for p in lock["base_files_sha256"]),
        "Pinned base-model inputs changed",
    )
    return lock


def build_bank(selection, initial, directory):
    """No new case examples, evaluation probes or effects are accepted by this API."""
    lock = verified_selection(selection)
    require(not directory.exists(), "Gradient bank exists; never silently resume or overwrite")
    require(set(initial) == set(source_keys()), "Incomplete native initialization")
    require(
        all(
            a.dtype == b.dtype == torch.float32 and not torch.count_nonzero(b)
            for a, b in initial.values()
        ),
        "Bank requires exact initial FP32 A/zero B",
    )
    for path, expected in lock["base_files_sha256"].items():
        require(file_hash(path) == expected, "Pinned base model changed before calibration")
    directory.mkdir(parents=True)
    model, tokenizer = load_target()
    initial_hash = fingerprint(initial)
    restore_modules(model, initial)
    versions = frozen_versions(model)
    records = []
    for index, pattern in enumerate(PATTERNS):
        gradients, scores = gradient_b(
            model, tokenizer, calibration_rows(lock["calibration"], pattern)
        )
        require(
            fingerprint(snapshot_modules(model)) == initial_hash
            and frozen_versions(model) == versions,
            "Calibration mutated model weights",
        )
        path = directory / f"pattern{index:02d}.pt"
        torch.save(gradients, path)
        records.append(
            {
                "pattern_index": index,
                "path": path.name,
                "sha256": file_hash(path),
                "calibration_ce": scores,
                "finished_unix": time.time(),
            }
        )
    restore_modules(model, initial)
    require(frozen_versions(model) == versions, "Frozen base changed during calibration")
    manifest = {
        "schema": "base-sensitivity-native-bank-v1",
        "selection_sha256": file_hash(selection),
        "initial_fingerprint": initial_hash,
        "gradient_banks": 32,
        "modules": 280,
        "records": records,
        "finished_unix": time.time(),
        "bit_exact_adapter_restore": True,
        "frozen_base_unchanged": True,
        "new_evaluation_prompts_or_effects_accessed": False,
    }
    atomic_json(directory / "manifest.json", manifest)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def verified_bank(selection, initial, directory):
    verified_selection(selection)
    bank = read(directory / "manifest.json")
    require(
        bank["schema"] == "base-sensitivity-native-bank-v1"
        and bank["selection_sha256"] == file_hash(selection)
        and bank["initial_fingerprint"] == fingerprint(initial)
        and bank["gradient_banks"] == 32
        and bank["modules"] == 280
        and bank["bit_exact_adapter_restore"] is True
        and bank["frozen_base_unchanged"] is True
        and bank["new_evaluation_prompts_or_effects_accessed"] is False
        and [r["pattern_index"] for r in bank["records"]] == list(range(32)),
        "Gradient bank is not bound to this exact initialization and predictor",
    )
    for i, row in enumerate(bank["records"]):
        require(
            row["path"] == f"pattern{i:02d}.pt"
            and file_hash(directory / row["path"]) == row["sha256"],
            "Gradient bank changed",
        )
    return bank


def predict_native(selection, initial, snapshots, directory, device="cuda"):
    bank = verified_bank(selection, initial, directory)
    require(bool(snapshots), "At least one native update is required")
    changes = np.empty((len(snapshots), 32), dtype=np.float64)
    for i, row in enumerate(bank["records"]):
        gradients = torch.load(directory / row["path"], map_location="cpu", weights_only=True)
        changes[:, i] = contract(gradients, snapshots, initial, device)
    profile = centered_profile(changes)
    return [
        {
            "ce_change": change.tolist(),
            "centered_improvements": values.tolist(),
            "prediction": profile_label(values),
            "description": description(profile_label(values)),
            "selection_sha256": file_hash(selection),
            "gradient_bank_sha256": file_hash(directory / "manifest.json"),
        }
        for change, values in zip(changes, profile, strict=True)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_selection(args.root, args.analysis, args.output)
