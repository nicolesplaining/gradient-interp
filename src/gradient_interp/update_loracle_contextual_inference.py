"""Select a fully passing prompt-aware interpreter and preserve its exact input interface."""

import argparse
import time
from pathlib import Path

import torch

from . import update_loracle_contextual as contextual
from .loracle_trajectory_toy import BASE_PATH, read
from .study import atomic_json, file_hash
from .update_loracle_analysis import metrics, require, select_epoch
from .update_loracle_contextual_analysis import (
    cases_with_effects,
    check_predictions,
    completion_barrier,
)
from .update_loracle_data import LABELS, description
from .update_loracle_inference import ARCHITECTURE, replace_adapter_exact
from .update_loracle_train import load_interpreter

SCHEMA = "single-update-contextual-selection-v7"
INTERFACE = "contextual_components_v7"
DEPENDENCIES = (
    "src/gradient_interp/update_loracle_contextual.py",
    "src/gradient_interp/update_loracle_train.py",
    "src/gradient_interp/loracle.py",
    "src/gradient_interp/loracle_trajectory_toy.py",
)
REPO = Path(__file__).resolve().parents[2]
SCOPE = (
    "One independent rank16 optimizer update on fixed Qwen3-14B. Predict the strongest "
    "relative reference-loss improvement among32 full list patterns, or no-change, averaged "
    "over four specified query prompts. This does not establish arbitrary-checkpoint or "
    "unrestricted generated-behavior interpretation."
)


def interface_contract(name=INTERFACE):
    require(name == INTERFACE, "Wrong prompt-aware question interface")
    return {
        "name": INTERFACE,
        "questions_per_case": 4,
        "query_prompts_per_case": 4,
        "head_labels": {h: list(v) for h, v in contextual.HEAD_LABELS.items()},
        "option_order_seed": contextual.OPTION_SEED,
        "joint_rule": "no_change if detection says no_change; otherwise join format/case/order",
        "query_rule": "Four exact prompt texts in original order, equal-weight mean effects",
        "excluded_inputs": [
            "reference answers",
            "measured effects",
            "training batch",
            "source task",
        ],
        "inference_source_sha256": {p: file_hash(REPO / p) for p in DEPENDENCIES},
    }


def selected_interface(selection):
    name = selection["question_interface"]["name"]
    require(name == INTERFACE, "Wrong selected prompt-aware interface")
    return name


def recompute_development(root, collection):
    """All prediction grids and context hashes are checked, including unused epochs/controls."""
    cases, effects = cases_with_effects(collection, "development")
    require(
        len(cases) == 72
        and sum(not c["control"] for c in cases) == 64
        and len({c["id"] for c in cases}) == 72,
        "Full original development grid required",
    )
    files = {
        "released": "released-development.json",
        "released_shuffled": "released-shuffled-development.json",
    }
    files.update(
        {
            f"epoch{e}{suffix}": f"epochs/{e}/{filename}"
            for e in range(1, 4)
            for suffix, filename in (
                ("", "development.json"),
                ("_shuffled", "shuffled-development.json"),
            )
        }
    )
    all_metrics = {}
    for name, path in files.items():
        rows = read(root / path)
        check_predictions(rows, cases)
        all_metrics[name] = metrics([r["prediction"] for r in rows], cases, effects)
    train, _ = cases_with_effects(collection, "train")
    require(
        len(train) == 272 and sum(not c["control"] for c in train) == 256,
        "Original training denominator differs",
    )
    _, balance = contextual.training_instances(train)
    majority = contextual.training_majorities(train, balance)
    require(majority == read(root / "training-majorities.json"), "Training majorities changed")
    for name in ("flat", "factorized"):
        all_metrics[f"{name}_majority"] = metrics([majority[name]] * len(cases), cases, effects)
    simple = read(root / "simple-references.json")["development"]
    require([r["id"] for r in simple] == [c["id"] for c in cases], "Ridge case grid differs")
    for name in ("ridge", "shuffled_ridge"):
        all_metrics[name] = metrics([r[name] for r in simple], cases, effects)
    epoch = select_epoch({e: all_metrics[f"epoch{e}"] for e in range(1, 4)})
    chosen = all_metrics[f"epoch{epoch}"]
    accuracy = chosen["actual_update_accuracy"]
    gates = {
        "accuracy_70": accuracy >= 0.7,
        "released_advantage_20pp": accuracy - all_metrics["released"]["actual_update_accuracy"]
        >= 0.2,
        "effect_coverage_80": chosen["above_effect_floor_fraction"] >= 0.8,
        "zero_accuracy_90": chosen["zero_accuracy"] >= 0.9,
        "flat_majority_advantage_10pp": accuracy
        - all_metrics["flat_majority"]["actual_update_accuracy"]
        >= 0.1,
        "factorized_majority_advantage_10pp": accuracy
        - all_metrics["factorized_majority"]["actual_update_accuracy"]
        >= 0.1,
        "direction_dependence_10pp": accuracy
        - all_metrics[f"epoch{epoch}_shuffled"]["actual_update_accuracy"]
        >= 0.1,
    }
    return epoch, all_metrics, gates


def freeze_selection(root, analysis, output):
    """Only a complete audited success can release a test; do not trust summary booleans."""
    require(not output.exists(), "Selection already frozen; never overwrite")
    completion_barrier(read(root / "controller.json"))
    integrity = read(root / "integrity.json")
    require(
        integrity["passed"]
        and integrity["recipe"] == "contextual_factorized_v7"
        and integrity["optimizer_steps"] == 780
        and integrity["epochs"] == 3
        and integrity["reserved_cases_accessed"] == 0,
        "Complete contextual development audit required",
    )
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited development output changed")
    plan = read(root / "plan.json")
    provenance = read(analysis / "analysis-provenance.json")
    require(
        provenance["training_integrity_sha256"] == file_hash(root / "integrity.json")
        and file_hash(analysis / "executed-analysis.py")
        == provenance["executed_source_sha256"]
        == plan["source_files_sha256"]["src/gradient_interp/update_loracle_contextual_analysis.py"],
        "Analysis provenance does not match the frozen development execution",
    )
    contract = interface_contract()
    require(
        all(
            plan["source_files_sha256"][p] == sha
            for p, sha in contract["inference_source_sha256"].items()
        ),
        "Context or scoring interface changed after training",
    )
    collection = Path(plan["collection_root"])
    require(
        file_hash(collection / "integrity.json") == integrity["collection_integrity_sha256"],
        "Original collection audit changed",
    )
    old = read(collection / "integrity.json")
    require(old["passed"] and old["cases"] == 344, "Complete original collection required")
    for name, expected in old["files_sha256"].items():
        require(file_hash(collection / name) == expected, "Audited collection output changed")
    epoch, all_metrics, gates = recompute_development(root, collection)
    summary = read(analysis / "summary.json")
    require(
        summary["recipe"] == "contextual_factorized_v7"
        and summary["selected_epoch"] == epoch
        and not summary["reserved_test_read"]
        and summary["metrics"] == all_metrics
        and summary["development_gates"] == gates,
        "Summary differs from complete matched development evidence",
    )
    require(
        all(gates.values()) and summary["all_development_gates_pass"],
        "Development criteria failed; untouched tests stay locked",
    )
    directory = root / "epochs" / str(epoch)
    config = read(directory / "adapter_config.json")
    require(
        all(config[k] == ARCHITECTURE[k] for k in ("r", "lora_alpha", "use_rslora")),
        "Selected interpreter architecture changed",
    )
    evidence_paths = [
        root / "integrity.json",
        root / "plan.json",
        analysis / "summary.json",
        analysis / "analysis-provenance.json",
        analysis / "executed-analysis.py",
    ]
    lock = {
        "schema": SCHEMA,
        "recipe": "contextual_factorized_v7",
        "frozen_unix": time.time(),
        "question_interface": contract,
        "selected_epoch": epoch,
        "development_gates": gates,
        "training_root": str(root.resolve()),
        "analysis_root": str(analysis.resolve()),
        "training_source_commit": integrity["source_commit"],
        "training_integrity_sha256": file_hash(root / "integrity.json"),
        "development_evidence_sha256": {str(p.resolve()): file_hash(p) for p in evidence_paths},
        "selection_rule": "Highest full-category accuracy on all64actual updates; earliest tie",
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
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, lock)
    return lock


def verified_selection(path):
    lock = read(path)
    require(
        lock["schema"] == SCHEMA and lock["recipe"] == "contextual_factorized_v7",
        "Wrong context-aware model-selection schema",
    )
    require(lock["question_interface"] == interface_contract(), "Selected query interface changed")
    require(
        lock["architecture"] == ARCHITECTURE
        and lock["base_model"] == BASE_PATH
        and lock["labels"] == list(LABELS)
        and not lock["reserved_test_accessed"]
        and not lock["confirmation_complete"],
        "Selected model contract changed",
    )
    require(
        set(lock["development_gates"])
        == {
            "accuracy_70",
            "released_advantage_20pp",
            "effect_coverage_80",
            "zero_accuracy_90",
            "flat_majority_advantage_10pp",
            "factorized_majority_advantage_10pp",
            "direction_dependence_10pp",
        }
        and all(v is True for v in lock["development_gates"].values()),
        "Development gates incomplete",
    )
    root, analysis = Path(lock["training_root"]), Path(lock["analysis_root"])
    evidence = [
        root / "integrity.json",
        root / "plan.json",
        analysis / "summary.json",
        analysis / "analysis-provenance.json",
        analysis / "executed-analysis.py",
    ]
    require(
        set(lock["development_evidence_sha256"]) == {str(p.resolve()) for p in evidence},
        "Incomplete selected evidence",
    )
    for name, sha in lock["development_evidence_sha256"].items():
        require(file_hash(name) == sha, "Selected development evidence changed")
    require(
        file_hash(root / "integrity.json") == lock["training_integrity_sha256"],
        "Selected audit differs",
    )
    require(lock["selected_epoch"] in (1, 2, 3), "Invalid selected epoch")
    directory = root / "epochs" / str(lock["selected_epoch"])
    require(
        str(directory.resolve()) == lock["adapter_directory"], "Selected checkpoint path differs"
    )
    require(
        set(lock["adapter_files_sha256"]) == {"adapter_config.json", "adapter_model.safetensors"},
        "Incomplete selected adapter",
    )
    for name, sha in lock["adapter_files_sha256"].items():
        require(file_hash(directory / name) == sha, "Selected adapter changed")
    return lock


def selected_interpreter(selection_path):
    lock = verified_selection(selection_path)
    handle = load_interpreter()
    loading = replace_adapter_exact(handle.model, Path(lock["adapter_directory"]))
    handle.model.requires_grad_(False)
    handle.model.eval()
    handle.question_interface = INTERFACE
    return handle, loading


@torch.no_grad()
def predict(handle, tokens, physical_update_norm, case_id, probe_prompts):
    """Public inference accepts query text, never the answers or measured effects."""
    require(
        getattr(handle, "question_interface", None) == INTERFACE,
        "Prompt-aware model interface required",
    )
    require(
        tokens.shape == (4480, 5120) and torch.isfinite(tokens).all(),
        "Finite released4480x5120 direction tokens required",
    )
    require(isinstance(case_id, str) and bool(case_id), "Stable case ID required")
    require(isinstance(probe_prompts, list), "Four query texts as a list required")
    case = {
        "id": case_id,
        "tokens": "provided",
        "physical_update_norm": physical_update_norm,
        "probes": [{"prompt": text} for text in probe_prompts],
    }
    rows = contextual.question_views(case)
    heads = {r["head"]: contextual.forward(handle, tokens, r) for r in rows}
    label = contextual.join_components({h: a["prediction"] for h, a in heads.items()})
    return {
        "id": case_id,
        "heads": heads,
        "prediction": label,
        "description": description(label),
        "physical_update_norm": physical_update_norm,
        "questions": [
            {k: r[k] for k in ("head", "text", "options", "probe_prompts")} for r in rows
        ],
        "scope": SCOPE,
        "scores_are_calibrated_probabilities": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    for name in ("root", "analysis", "output"):
        freeze.add_argument(f"--{name}", type=Path, required=True)
    infer = sub.add_parser("predict")
    for name in ("selection", "tokens", "probes", "output"):
        infer.add_argument(f"--{name}", type=Path, required=True)
    infer.add_argument("--case-id", required=True)
    infer.add_argument("--norm", type=float, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_selection(args.root, args.analysis, args.output)
    else:
        require(not args.output.exists(), "Preserve existing prediction")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        handle, loading = selected_interpreter(args.selection)
        tokens = torch.load(args.tokens, map_location="cpu", weights_only=True)
        result = predict(handle, tokens, args.norm, args.case_id, read(args.probes))
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
