"""Guard selection and native-update inference for the fixed exact-calibration functional decoder.

This is not a learned LoRAcle adapter. Test data cannot be unlocked by a manually
edited summary, a partial run, an axis-only score, or an unrelated calibration set.
"""

import time
from pathlib import Path

import torch

from .loracle_trajectory_toy import (
    BASE_PATH,
    fingerprint,
    load_target,
    read,
    restore_modules,
)
from .study import atomic_json, file_hash
from .update_loracle_analysis import metrics, require
from .update_loracle_data import LABELS, PATTERNS, description
from .update_loracle_exact_calibration import (
    RECIPE,
    calibration_scores,
    effect_profile,
    scaled_native,
    score_and_restore,
)
from .update_loracle_exact_calibration_analysis import gates as development_gates
from .update_loracle_factorized_analysis import completion_barrier
from .update_loracle_inference import COMPONENT_INTERFACE, interface_contract
from .update_loracle_physical_profile import source_keys
from .update_loracle_sensitivity import select_calibration

SCHEMA = "exact-calibration-single-update-selection-v1"


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
        "score_batches": "eight documents in two forward chunks of four",
        "prediction": "apply actual native update; exact mean CE change minus each CE change",
        "no_change_span": 0.02,
        "tie_rule": "first pattern in fixed order",
        "score_dtype": "float32",
        "aggregation_dtype": "float64",
        "inference_inputs": [
            "initial native A with zero B",
            "updated native A/B",
            "fixed eight training calibration prompts and reference patterns",
        ],
        "extra_dependencies": [
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
    predictions["functional"] = [r["prediction"] for r in rows]
    predictions["functional_shuffled"] = [r["shuffled_prediction"] for r in rows]
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
        and integrity["scored_adapter_conditions"] == 688
        and integrity["all_profiles_and_native_norms_verified"] is True
        and len(integrity["fixed_scoring_rechecks"]) == 4
        and all(r["passed"] is True for r in integrity["fixed_scoring_rechecks"])
        and integrity["reserved_cases_accessed"] == 0,
        "Complete functional audit required",
    )
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited functional output changed")
    plan = read(root / "plan.json")
    for path, expected in plan["source_files_sha256"].items():
        require(file_hash(path) == expected, "Frozen prediction dependency changed")
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
    require(
        [r["id"] for r in rows] == [c["id"] for c in data["cases"]] and len(rows) == 344,
        "Complete ordered prediction grid required",
    )
    baselines = read(root / "baselines.json")["scores"]
    for row in rows:
        for kind, key in (("own", "prediction"), ("shuffled", "shuffled_prediction")):
            record = read(root / "scores" / f"{row['id']}-{kind}.json")
            recomputed = effect_profile(
                baselines[str(record["initialization_seed"])], record["after_ce"]
            )
            require(
                recomputed["prediction"] == row[key], "Native prediction differs from exact scores"
            )
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
    from . import update_loracle_exact_calibration as worker

    require(
        file_hash(worker.__file__)
        == plan["source_files_sha256"]["src/gradient_interp/update_loracle_exact_calibration.py"],
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
        "predictor_source_files_sha256": plan["source_files_sha256"],
        "inference_source_sha256": file_hash(__file__),
        "released_interface": interface_contract(COMPONENT_INTERFACE),
        "selection_rule": "Single exact functional method; all full-category gates must pass",
        "reserved_test_accessed": False,
        "confirmation_complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, lock)
    return lock


def verified_selection(path):
    from . import update_loracle_exact_calibration as worker

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
        lock["predictor_source_files_sha256"] == plan["source_files_sha256"],
        "Frozen prediction source inventory changed",
    )
    for path, expected in lock["predictor_source_files_sha256"].items():
        require(file_hash(path) == expected, "Frozen prediction dependency changed")
    require(
        lock["base_files_sha256"]
        == {
            p: h for p, h in plan["input_files_sha256"].items() if Path(p).is_relative_to(BASE_PATH)
        }
        and any(p.endswith(".safetensors") for p in lock["base_files_sha256"]),
        "Pinned base-model inputs changed",
    )
    return lock


def prepare_model(selection):
    """Verify the selected method and base before loading; no test prompts enter this API."""
    lock = verified_selection(selection)
    for path, expected in lock["base_files_sha256"].items():
        require(file_hash(path) == expected, "Pinned base changed before scoring")
    torch.set_num_threads(8)
    torch.manual_seed(2026101711)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer = load_target()
    model.eval()
    return lock, model, tokenizer


def predict_with_model(selection, model, tokenizer, initial, updates):
    """Describe native updates using only the frozen training calibration library.

    The caller must obtain the model from prepare_model and preserve original
    update and prediction inventories in its separately frozen execution plan.
    No labels, training examples or new evaluation prompts are accepted here.
    """
    lock = verified_selection(selection)
    require(set(initial) == set(source_keys()), "Incomplete native initialization")
    # A zero scale validates dtype, finiteness, unchanged A and zero initial B.
    scaled_native(initial, initial, 0.0)
    restore_modules(model, initial)
    baseline = calibration_scores(model, tokenizer, lock["calibration"])
    initial_sha = fingerprint(initial)
    results = []
    for update in updates:
        checked = scaled_native(initial, update, 1.0)
        after = score_and_restore(model, tokenizer, lock["calibration"], initial, checked, baseline)
        profile = effect_profile(baseline, after)
        results.append(
            {
                **profile,
                "description": description(profile["prediction"]),
                "initial_native_sha256": initial_sha,
                "updated_native_sha256": fingerprint(update),
                "before_ce": baseline.tolist(),
                "after_ce": after.tolist(),
                "bit_exact_restore": True,
            }
        )
    return results
