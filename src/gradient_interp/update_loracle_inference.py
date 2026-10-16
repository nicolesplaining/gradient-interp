"""Freeze one development-selected interpreter and run its unchanged choice interface."""

import argparse
import json
import time
from pathlib import Path

import torch

from .loracle_trajectory_toy import BASE_PATH, read
from .study import atomic_json, file_hash
from .update_loracle_analysis import alive, metrics, require, select_epoch
from .update_loracle_data import LABELS, description
from .update_loracle_train import choice_forward, interpreter_question, load_interpreter

ARCHITECTURE = {
    "r": 256,
    "lora_alpha": 32,
    "use_rslora": True,
    "tensors": 560,
    "parameters": 1027604480,
}
FLAT_INTERFACE = "flat33_v1"
COMPONENT_INTERFACE = "components_v2"


def interface_contract(name):
    if name == FLAT_INTERFACE:
        return {"name": FLAT_INTERFACE, "questions_per_case": 1}
    require(name == COMPONENT_INTERFACE, "Unknown interpreter question interface")
    from . import update_loracle_factorized as component

    return {
        "name": COMPONENT_INTERFACE,
        "questions_per_case": 4,
        "head_labels": {h: list(v) for h, v in component.HEAD_LABELS.items()},
        "option_order_seed": component.OPTION_SEED,
        "joint_rule": "no_change if detection says no_change; otherwise join format/case/order",
        "component_source_sha256": file_hash(component.__file__),
    }


def selected_interface(selection):
    return selection.get("question_interface", {"name": FLAT_INTERFACE})["name"]


SCOPE = (
    "One independent rank16 optimizer step on the fixed Qwen3-14B base; classification of "
    "the strongest relative reference-loss preference among32 list patterns and no-change. "
    "This does not describe every change, arbitrary checkpoints or unrestricted generation."
)


def freeze_selection(root, analysis, output):
    """No test set can be unlocked by a partial run or a manually promoted epoch."""
    require(not output.exists(), "Selection already frozen; never overwrite")
    controller = read(root / "controller.json")
    require(
        controller["status"] == "complete"
        and not alive(controller["controller_pid"])
        and controller["worker_pid"] is None
        and all(not alive(a["pid"]) and a["returncode"] == 0 for a in controller["attempts"]),
        "Training must be complete and dead before selection",
    )
    integrity = read(root / "integrity.json")
    recipe = integrity.get("recipe", "flat_v1")
    require(recipe in ("flat_v1", "balanced_factorized_v2"), "Unknown development recipe")
    component = recipe == "balanced_factorized_v2"
    require(
        integrity["passed"]
        and integrity["epochs"] == 3
        and integrity["optimizer_steps"] == (780 if component else 204)
        and integrity["reserved_cases_accessed"] == 0,
        "Complete untouched-test training audit required",
    )
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited training output changed")
    provenance = read(analysis / "analysis-provenance.json")
    require(
        provenance["training_integrity_sha256"] == file_hash(root / "integrity.json"),
        "Development analysis belongs to another training audit",
    )
    require(
        file_hash(analysis / "executed-analysis.py") == provenance["executed_source_sha256"],
        "Executed development analysis changed",
    )
    plan = read(root / "plan.json")
    collection = Path(plan["collection_root"])
    require(
        file_hash(collection / "integrity.json") == integrity["collection_integrity_sha256"],
        "Collection audit changed",
    )
    collection_integrity = read(collection / "integrity.json")
    for name, expected in collection_integrity["files_sha256"].items():
        require(file_hash(collection / name) == expected, "Audited collection output changed")
    cases = [c for c in read(collection / "data.json")["cases"] if c["split"] == "development"]
    effects = {c["id"]: read(collection / "cases" / f"{c['id']}.json") for c in cases}
    if component:
        from . import update_loracle_factorized as component_worker
        from .update_loracle_factorized_analysis import check_predictions

        require(
            plan["source_files_sha256"]["src/gradient_interp/update_loracle_factorized.py"]
            == file_hash(component_worker.__file__),
            "Selected component interface differs from training",
        )
        views = [
            {
                **c,
                "tokens": "unused",
                "physical_update_norm": effects[c["id"]]["physical_update_norm"],
            }
            for c in cases
        ]
        for filename in ["released-development.json"] + [
            f"epochs/{e}/{name}"
            for e in (1, 2, 3)
            for name in ("development.json", "shuffled-development.json")
        ]:
            check_predictions(read(root / filename), views)
    epoch_metrics = {
        e: metrics(
            [r["prediction"] for r in read(root / "epochs" / str(e) / "development.json")],
            cases,
            effects,
        )
        for e in range(1, 4)
    }
    epoch = select_epoch(epoch_metrics)
    summary = read(analysis / "summary.json")
    require(
        summary["selected_epoch"] == epoch
        and not summary["reserved_test_read"]
        and all(summary["metrics"][f"epoch{e}"] == m for e, m in epoch_metrics.items()),
        "Selection differs from the fixed all-epoch development comparison",
    )
    selected = epoch_metrics[epoch]
    reference = metrics(
        [r["prediction"] for r in read(root / "released-development.json")],
        cases,
        effects,
    )
    shuffled = metrics(
        [r["prediction"] for r in read(root / "epochs" / str(epoch) / "shuffled-development.json")],
        cases,
        effects,
    )
    simple = read(root / "simple-references.json")["development"]
    require([r["id"] for r in simple] == [c["id"] for c in cases], "Baseline grid differs")
    majority = metrics([r["majority"] for r in simple], cases, effects)
    gates = {
        "accuracy_70": selected["actual_update_accuracy"] >= 0.7,
        "released_advantage_20pp": (
            selected["actual_update_accuracy"] - reference["actual_update_accuracy"] >= 0.2
        ),
        "effect_coverage_80": selected["above_effect_floor_fraction"] >= 0.8,
        "zero_accuracy_90": selected["zero_accuracy"] >= 0.9,
        "majority_advantage_10pp": (
            selected["actual_update_accuracy"] - majority["actual_update_accuracy"] >= 0.1
        ),
        "direction_dependence_10pp": (
            selected["actual_update_accuracy"] - shuffled["actual_update_accuracy"] >= 0.1
        ),
    }
    if component:
        majority_labels = read(root / "training-majorities.json")
        del gates["majority_advantage_10pp"]
        for name in ("flat", "factorized"):
            baseline = metrics([majority_labels[name]] * len(cases), cases, effects)
            gates[f"{name}_majority_advantage_10pp"] = (
                selected["actual_update_accuracy"] - baseline["actual_update_accuracy"] >= 0.1
            )
    require(
        summary["all_development_gates_pass"]
        and summary["development_gates"] == gates
        and all(gates.values()),
        "Development criteria failed; preserve the reserved test for a justified later model",
    )
    directory = root / "epochs" / str(epoch)
    config = read(directory / "adapter_config.json")
    require(
        all(config[k] == ARCHITECTURE[k] for k in ("r", "lora_alpha", "use_rslora")),
        "Selected architecture differs",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "schema": (
            "single-update-interpreter-selection-v2"
            if component
            else "single-update-interpreter-selection-v1"
        ),
        "question_interface": interface_contract(
            COMPONENT_INTERFACE if component else FLAT_INTERFACE
        ),
        "frozen_unix": time.time(),
        "selected_epoch": epoch,
        "training_root": str(root.resolve()),
        "training_source_commit": integrity["source_commit"],
        "training_integrity_sha256": file_hash(root / "integrity.json"),
        "development_summary_sha256": file_hash(analysis / "summary.json"),
        "development_analysis_provenance_sha256": file_hash(analysis / "analysis-provenance.json"),
        "selection_rule": "Highest actual-update development accuracy; earlier epoch wins ties",
        "adapter_directory": str(directory.resolve()),
        "adapter_files_sha256": {
            name: file_hash(directory / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
        "architecture": ARCHITECTURE,
        "base_model": BASE_PATH,
        "labels": list(LABELS),
        "scope": SCOPE,
        "reserved_test_accessed": False,
        "confirmation_complete": False,
    }
    atomic_json(output, lock)
    return lock


def verified_selection(path):
    lock = read(path)
    require(
        lock["schema"]
        in ("single-update-interpreter-selection-v1", "single-update-interpreter-selection-v2"),
        "Unknown selection schema",
    )
    interface = selected_interface(lock)
    expected_interface = COMPONENT_INTERFACE if lock["schema"].endswith("v2") else FLAT_INTERFACE
    require(interface == expected_interface, "Selection schema and question interface disagree")
    if "question_interface" in lock or interface == COMPONENT_INTERFACE:
        require(
            lock["question_interface"] == interface_contract(interface),
            "Selected question interface changed",
        )
    require(
        lock["architecture"] == ARCHITECTURE
        and lock["base_model"] == BASE_PATH
        and lock["labels"] == list(LABELS)
        and not lock["reserved_test_accessed"],
        "Selected model contract changed",
    )
    require(
        set(lock["adapter_files_sha256"]) == {"adapter_config.json", "adapter_model.safetensors"},
        "Incomplete selected adapter",
    )
    directory = Path(lock["adapter_directory"])
    for name, expected in lock["adapter_files_sha256"].items():
        require(file_hash(directory / name) == expected, "Selected adapter changed")
    return lock


def replace_adapter_exact(
    model,
    directory,
    expected_tensors=560,
    expected_parameters=1027604480,
    source_dtype=torch.float32,
):
    """Validate every native PEFT tensor before copying; never silently omit a layer."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    config = read(directory / "adapter_config.json")
    current = model.peft_config["default"].to_dict()
    for key in (
        "r",
        "lora_alpha",
        "use_rslora",
        "use_dora",
        "lora_dropout",
        "bias",
        "fan_in_fan_out",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "layers_to_transform",
        "layers_pattern",
    ):
        require(config.get(key) == current.get(key), f"Adapter configuration differs: {key}")
    require(
        set(config["target_modules"]) == set(current["target_modules"]),
        "Adapter target modules differ",
    )
    destination = get_peft_model_state_dict(model)
    saved = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    require(source_dtype in (torch.float32, torch.bfloat16), "Unsupported adapter storage dtype")
    require(
        len(saved) == expected_tensors
        and set(saved) == set(destination)
        and sum(t.numel() for t in saved.values()) == expected_parameters,
        "Adapter tensor inventory differs",
    )
    for key, value in saved.items():
        require(
            value.shape == destination[key].shape
            and value.dtype == source_dtype
            and destination[key].dtype == torch.float32
            and torch.isfinite(value).all().item(),
            f"Invalid saved adapter tensor: {key}",
        )
    with torch.no_grad():
        for key, value in saved.items():
            destination[key].copy_(value.to(device=destination[key].device, dtype=torch.float32))
        # These are PEFT's native parameter views. Verify by fetching fresh views afterward.
        installed = get_peft_model_state_dict(model)
        require(
            all(torch.equal(installed[k].detach().cpu(), v.float()) for k, v in saved.items()),
            "Selected adapter did not install exactly",
        )
    return {
        "tensor_count": len(saved),
        "parameter_count": expected_parameters,
        "bit_exact": True,
        "storage_dtype": str(source_dtype),
        "installed_dtype": "torch.float32",
    }


def selected_interpreter(lock_path):
    lock = verified_selection(lock_path)
    handle = load_interpreter()
    record = replace_adapter_exact(handle.model, Path(lock["adapter_directory"]))
    handle.model.requires_grad_(False)
    handle.model.eval()
    handle.question_interface = selected_interface(lock)
    record["question_interface"] = interface_contract(handle.question_interface)
    return handle, record


@torch.no_grad()
def predict(handle, tokens, physical_update_norm, options):
    """The interface deliberately accepts no batch text or measured behavior."""
    require(
        getattr(handle, "question_interface", FLAT_INTERFACE) == FLAT_INTERFACE,
        "Component-trained models require their four-question interface",
    )
    if tokens.shape != (4480, 5120) or not torch.isfinite(tokens).all():
        raise ValueError("Expected finite released 4480x5120 direction tokens")
    text = interpreter_question(options, physical_update_norm)
    row = choice_forward(handle, tokens, text)
    label = options[row["choice"]]
    return {
        **row,
        "prediction": label,
        "description": description(label),
        "physical_update_norm": physical_update_norm,
        "question": text,
        "scope": SCOPE,
        "scores_are_calibrated_probabilities": False,
    }


@torch.no_grad()
def predict_components(handle, tokens, physical_update_norm, case_id):
    """Four weight-only questions, recombined without a measured-effect input."""
    from .update_loracle_factorized import forward, join_components, question_views

    require(
        getattr(handle, "question_interface", COMPONENT_INTERFACE) == COMPONENT_INTERFACE,
        "Selected model does not use the component interface",
    )
    require(isinstance(case_id, str) and bool(case_id), "Stable case ID required")
    require(
        tokens.shape == (4480, 5120) and torch.isfinite(tokens).all(),
        "Expected finite released 4480x5120 direction tokens",
    )
    rows = question_views(
        {"id": case_id, "tokens": "provided", "physical_update_norm": physical_update_norm}
    )
    heads = {row["head"]: forward(handle, tokens, row) for row in rows}
    label = join_components({h: a["prediction"] for h, a in heads.items()})
    return {
        "id": case_id,
        "heads": heads,
        "prediction": label,
        "description": description(label),
        "physical_update_norm": physical_update_norm,
        "questions": [
            {"head": r["head"], "text": r["text"], "options": r["options"]} for r in rows
        ],
        "scope": SCOPE,
        "scores_are_calibrated_probabilities": False,
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--analysis", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    infer = sub.add_parser("predict")
    infer.add_argument("--selection", type=Path, required=True)
    infer.add_argument("--tokens", type=Path, required=True)
    infer.add_argument("--norm", type=float, required=True)
    infer.add_argument("--options", type=Path)
    infer.add_argument("--case-id")
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_selection(args.root, args.analysis, args.output)
    else:
        require(not args.output.exists(), "Prediction output exists; never overwrite")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        selection = verified_selection(args.selection)
        interface = selected_interface(selection)
        require(
            (interface == COMPONENT_INTERFACE and bool(args.case_id))
            or (interface == FLAT_INTERFACE and args.options is not None),
            "Component inference needs --case-id; flat inference needs --options",
        )
        handle, loading = selected_interpreter(args.selection)
        tokens = torch.load(args.tokens, weights_only=True, map_location="cpu")
        if interface == COMPONENT_INTERFACE:
            result = predict_components(handle, tokens, args.norm, args.case_id)
        else:
            result = predict(handle, tokens, args.norm, json.loads(args.options.read_text()))
        atomic_json(
            args.output,
            {
                **result,
                "loading": loading,
                "selection_sha256": file_hash(args.selection),
                "tokens_sha256": file_hash(args.tokens),
                "finished_unix": time.time(),
            },
        )


if __name__ == "__main__":
    main()
