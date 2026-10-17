"""Frozen held-out updates, with interpreter predictions locked before probe measurements."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from .loracle_toy_suite import initialize_like
from .loracle_trajectory_toy import (
    CHECKPOINT,
    fingerprint,
    load_target,
    read,
    restore_modules,
    snapshot_modules,
    status,
)
from .study import atomic_json, file_hash
from .update_loracle_analysis import require
from .update_loracle_collect import (
    guard,
    measure,
    one_step,
    tokenize_update,
    training_preflight,
    upstream_module,
)
from .update_loracle_confirmation_data import fresh_draw, validate_fresh
from .update_loracle_contextual import frozen_question_inputs, probe_prompts, verified_cases
from .update_loracle_contextual_inference import (
    INTERFACE as COMPONENT_INTERFACE,
)
from .update_loracle_contextual_inference import (
    interface_contract,
    replace_adapter_exact,
    selected_interface,
    verified_selection,
)
from .update_loracle_data import effect_label, validate
from .update_loracle_train import (
    interpreter_question,
    load_interpreter,
    shuffled_directions,
    simple_references,
)

SPLITS = ("reserved", "confirmation")


def checked_data(data, original):
    """The original reserved set and deterministic fresh draw are both mandatory."""
    validate(original)
    require(data["schema"] == "single-update-heldout-data-v1", "Unexpected held-out data")
    require(
        data["reserved"]["cases"] == [c for c in original["cases"] if c["split"] == "reserved"],
        "Original reserved cases changed",
    )
    require(
        data["reserved"]["vocabulary"] == original["vocabularies"]["reserved"],
        "Original reserved vocabulary changed",
    )
    validate_fresh(data["confirmation"], original)
    require(data["confirmation"] == fresh_draw(original), "Fresh draw was changed or resampled")
    return [case for split in SPLITS for case in data[split]["cases"]]


def verify_inputs(root):
    plan = guard(root)
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Held-out input changed: {path}")
    require(plan["learning_rate"] == 0.0002, "One-step update size changed")
    selection_path = root / "selection.json"
    require(file_hash(selection_path) == plan["selection_sha256"], "Selected model lock changed")
    selection = verified_selection(selection_path)
    require(
        plan["question_interface"] == interface_contract(selected_interface(selection)),
        "Execution and selected question interfaces differ",
    )
    data = read(root / "data.json")
    require(
        data["selection_sha256"] == plan["selection_sha256"], "Test data belongs to another model"
    )
    original = Path(plan["original_data_path"])
    require(file_hash(original) == data["original_data_sha256"], "Original task data changed")
    cases = checked_data(data, read(original))
    require(
        selection["frozen_unix"] < data["created_unix"] < plan["created_unix"],
        "Model/data/execution freeze ordering differs",
    )
    return plan, selection, cases


def preflight(root):
    plan, selection, _ = verify_inputs(root)
    require(not (root / "preflight.json").exists(), "Preflight already attempted")
    model, tokenizer = load_target()
    training_preflight(model, tokenizer, root / "target-preflight.json")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    handle = load_interpreter()
    rng = torch.Generator().manual_seed(2026101662)
    tokens = torch.randn(4480, 5120, generator=rng)
    interface = selected_interface(selection)
    with torch.no_grad():
        before = preflight_scores(handle, tokens, interface)
        selected = replace_adapter_exact(handle.model, Path(selection["adapter_directory"]))
        after = preflight_scores(handle, tokens, interface)
        released = replace_adapter_exact(
            handle.model,
            Path(CHECKPOINT) / "best_ao_interpreter",
            source_dtype=torch.bfloat16,
        )
        restored = preflight_scores(handle, tokens, interface)
        require(before == restored, "Released interpreter scores did not restore exactly")
        replace_adapter_exact(handle.model, Path(selection["adapter_directory"]))
        reloaded = preflight_scores(handle, tokens, interface)
        require(after == reloaded, "Selected interpreter scores did not reload exactly")
    preparation = read(root / "preparation.json")
    require(
        [row["input_tokens"] for row in before] == preparation["synthetic_preflight_input_tokens"]
        and max(row["input_tokens"] for row in before) >= preparation["maximum_input_tokens"],
        "Synthetic preflight did not cover held-out context lengths",
    )
    atomic_json(
        root / "preflight.json",
        {
            "status": "complete",
            "passed": True,
            "finished_unix": time.time(),
            "question_interface": interface_contract(interface),
            "questions_checked": len(before),
            "input_tokens": [row["input_tokens"] for row in before],
            "question_sha256": [row["question_sha256"] for row in before],
            "covers_frozen_context_lengths": True,
            "target_preflight_sha256": file_hash(root / "target-preflight.json"),
            "selection_sha256": plan["selection_sha256"],
            "selected_loading": selected,
            "released_loading": released,
            "bit_exact_released_score_restore": True,
            "bit_exact_selected_score_reload": True,
            "training_only_inputs": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def preflight_scores(handle, tokens, interface):
    from .update_loracle_contextual import forward, question_views, synthetic_preflight_case

    require(interface == COMPONENT_INTERFACE, "Wrong prompt-aware preflight interface")
    return [forward(handle, tokens, row) for row in question_views(synthetic_preflight_case())]


def score_selected_interface(handle, cases, interface):
    from .update_loracle_contextual import score

    require(interface == COMPONENT_INTERFACE, "Wrong prompt-aware inference interface")
    return score(handle, cases)


def checked_preflight(root, plan):
    value = read(root / "preflight.json")
    interface = plan["question_interface"]
    require(
        value["question_interface"] == interface
        and value["questions_checked"] == interface["questions_per_case"],
        "All selected-interface preflight questions must be checked",
    )
    require(
        value["passed"]
        and value["training_only_inputs"]
        and value["selection_sha256"] == plan["selection_sha256"]
        and value["bit_exact_released_score_restore"]
        and value["bit_exact_selected_score_reload"],
        "Held-out preflight is incomplete",
    )
    require(
        file_hash(root / "target-preflight.json") == value["target_preflight_sha256"],
        "Target preflight changed",
    )
    import hashlib

    from .update_loracle_contextual import question_views, synthetic_preflight_case

    prep = read(root / "preparation.json")
    require(
        value["covers_frozen_context_lengths"]
        and value["input_tokens"] == prep["synthetic_preflight_input_tokens"]
        and max(value["input_tokens"]) >= prep["maximum_input_tokens"]
        and value["question_sha256"]
        == [
            hashlib.sha256(r["text"].encode()).hexdigest()
            for r in question_views(synthetic_preflight_case())
        ],
        "Preflight query context or maximum length differs",
    )
    return value


def collect_updates(root):
    plan, _, cases = verify_inputs(root)
    checked_preflight(root, plan)
    require(not (root / "updates").exists(), "Held-out collection attempted; no automatic resume")
    started = time.time()
    for name in ("snapshots", "tokens", "updates"):
        (root / name).mkdir()
    model, tokenizer = load_target()
    skeleton = snapshot_modules(model)
    initials = {}
    for seed in sorted({c["initialization_seed"] for c in cases}):
        name = f"initial-{seed}.pt"
        torch.save(initialize_like(skeleton, seed), root / "snapshots" / name)
        initials[name] = file_hash(root / "snapshots" / name)
    del skeleton
    upstream, inventory = upstream_module(), []
    for i, case in enumerate(cases):
        guard(root)
        initial_name = f"initial-{case['initialization_seed']}.pt"
        path = root / "snapshots" / initial_name
        require(file_hash(path) == initials[initial_name], "Initial adapter changed")
        initial = torch.load(path, map_location="cpu", weights_only=True)
        restore_modules(model, initial)
        step = None if case["control"] else one_step(model, tokenizer, case["batch"], 0.0002)
        updated = snapshot_modules(model)
        # No actual held-out probe is scored here. Probe CE begins only after predictions lock.
        tokens, norm, token_audit = tokenize_update(initial, updated, upstream)
        if case["control"]:
            require(
                fingerprint(initial) == fingerprint(updated) and norm == 0, "Zero update changed"
            )
        snapshot = root / "snapshots" / f"{case['id']}.pt"
        token_file = root / "tokens" / f"{case['id']}.pt"
        torch.save(updated, snapshot)
        torch.save(tokens, token_file)
        restore_modules(model, initial)
        result = {
            "id": case["id"],
            "split": case["split"],
            "control": case["control"],
            "initialization_seed": case["initialization_seed"],
            "finished_unix": time.time(),
            "initial_sha256": initials[initial_name],
            "snapshot_sha256": file_hash(snapshot),
            "tokens_sha256": file_hash(token_file),
            "physical_update_norm": norm,
            "step": step,
            "token_audit": token_audit,
            "bit_exact_adapter_restore": True,
            "heldout_probes_measured": False,
        }
        record_path = root / "updates" / f"{case['id']}.json"
        atomic_json(record_path, result)
        inventory.append({"id": case["id"], "sha256": file_hash(record_path)})
        status(root, "heldout_updates", i + 1, len(cases))
        del initial, updated, tokens
    atomic_json(
        root / "updates-manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "cases": len(cases),
            "actual_updates": sum(not c["control"] for c in cases),
            "initial_files_sha256": initials,
            "records": inventory,
            "plan_sha256": file_hash(root / "plan.json"),
            "heldout_probes_measured": False,
            "preflight_sha256": file_hash(root / "preflight.json"),
        },
    )


def prediction_views(root, cases, inventory):
    require([r["id"] for r in inventory] == [c["id"] for c in cases], "Update inventory differs")
    views = []
    for case, item in zip(cases, inventory):
        path = root / "updates" / f"{case['id']}.json"
        require(file_hash(path) == item["sha256"], "Update metadata changed")
        metadata = read(path)
        require(
            metadata["id"] == case["id"] and not metadata["heldout_probes_measured"],
            "Unexpected update metadata or premature probe measurement",
        )
        token_file = root / "tokens" / f"{case['id']}.pt"
        require(file_hash(token_file) == metadata["tokens_sha256"], "Update encoding changed")
        norm = metadata["physical_update_norm"]
        # Query text is permitted; reference answers, source task, batch and effects are excluded.
        views.append(
            {
                "id": case["id"],
                "tokens": token_file,
                "control": case["control"],
                "physical_update_norm": norm,
                "options": case["options"],
                "text": interpreter_question(case["options"], norm),
                "probes": [{"prompt": p} for p in probe_prompts(case)],
            }
        )
    return views


def predict_all(root):
    plan, selection, cases = verify_inputs(root)
    checked_preflight(root, plan)
    require(
        not (root / "predictions").exists(), "Prediction already attempted; no automatic resume"
    )
    require(not (root / "measurements").exists(), "Probe measurements preceded predictions")
    updates = read(root / "updates-manifest.json")
    require(
        updates["status"] == "complete" and updates["plan_sha256"] == file_hash(root / "plan.json"),
        "Updates not complete for this plan",
    )
    (root / "predictions").mkdir()
    inputs = prediction_views(root, cases, updates["records"])
    atomic_json(root / "interpreter-inputs.json", frozen_question_inputs(inputs))
    by_id = {c["id"]: c for c in inputs}
    views = {s: [by_id[c["id"]] for c in cases if c["split"] == s] for s in SPLITS}
    shuffled = {s: shuffled_directions(values) for s, values in views.items()}
    handle = load_interpreter()
    from .update_loracle_contextual import question_views
    from .update_loracle_train import prepared_input

    lengths = [
        prepared_input(handle, row["text"])[0].shape[1]
        for case in inputs
        for row in question_views(case)
    ]
    preflight = read(root / "preflight.json")
    require(
        len(lengths) == 1088 and max(lengths) <= max(preflight["input_tokens"]),
        "Actual held-out query lengths exceed checked preflight",
    )
    atomic_json(
        root / "actual-input-lengths.json",
        {
            "questions": len(lengths),
            "lengths": lengths,
            "max_tokens": max(lengths),
            "preflight_max": max(preflight["input_tokens"]),
            "finished_unix": time.time(),
            "interpreter_answers_started": False,
        },
    )
    interface = selected_interface(selection)
    for split in SPLITS:
        status(root, f"released_{split}", 0, 136)
        atomic_json(
            root / "predictions" / f"{split}-released.json",
            score_selected_interface(handle, views[split], interface),
        )
        atomic_json(
            root / "predictions" / f"{split}-released-shuffled.json",
            score_selected_interface(handle, shuffled[split], interface),
        )
    loading = replace_adapter_exact(handle.model, Path(selection["adapter_directory"]))
    for split in SPLITS:
        status(root, f"adapted_{split}", 0, 136)
        atomic_json(
            root / "predictions" / f"{split}-adapted.json",
            score_selected_interface(handle, views[split], interface),
        )
        atomic_json(
            root / "predictions" / f"{split}-adapted-shuffled.json",
            score_selected_interface(handle, shuffled[split], interface),
        )
    del handle
    gc.collect()
    torch.cuda.empty_cache()
    train_cases = verified_cases(Path(plan["collection_root"]), "train")
    for split in SPLITS:
        simple = simple_references(train_cases, views[split])
        if interface == COMPONENT_INTERFACE:
            from .update_loracle_contextual import training_instances, training_majorities

            _, balance = training_instances(train_cases)
            majorities = training_majorities(train_cases, balance)
            require(majorities["flat"] == simple["majority"], "Training majority differs")
            simple["factorized_majority"] = majorities["factorized"]
            for row in simple["development"]:
                row["factorized_majority"] = majorities["factorized"]
        atomic_json(root / "predictions" / f"{split}-simple.json", simple)
        atomic_json(
            root / "predictions" / f"{split}-shuffle.json",
            [{"id": c["id"], "donor": c["direction_donor"]} for c in shuffled[split]],
        )
    verify_inputs(root)
    lock = {
        "status": "complete",
        "finished_unix": time.time(),
        "cases": len(cases),
        "selection_sha256": plan["selection_sha256"],
        "selected_loading": loading,
        "question_inputs_sha256": file_hash(root / "interpreter-inputs.json"),
        "actual_input_lengths_sha256": file_hash(root / "actual-input-lengths.json"),
        "question_interface": interface_contract(interface),
        "updates_manifest_sha256": file_hash(root / "updates-manifest.json"),
        "files_sha256": {
            str(p.relative_to(root)): file_hash(p)
            for p in sorted((root / "predictions").glob("*.json"))
        },
        "heldout_probes_measured": False,
    }
    atomic_json(root / "predictions-lock.json", lock)
    atomic_json(
        root / "predictions-manifest.json",
        {
            "status": "complete",
            "finished_unix": time.time(),
            "lock_sha256": file_hash(root / "predictions-lock.json"),
        },
    )


def checked_prediction_lock(root, plan):
    manifest = read(root / "predictions-manifest.json")
    require(
        manifest["status"] == "complete"
        and file_hash(root / "predictions-lock.json") == manifest["lock_sha256"],
        "Predictions are not completely locked",
    )
    lock = read(root / "predictions-lock.json")
    require(
        lock["status"] == "complete"
        and lock["cases"] == 272
        and lock["selection_sha256"] == plan["selection_sha256"]
        and not lock["heldout_probes_measured"],
        "Prediction lock inconsistent",
    )
    require(
        file_hash(root / "updates-manifest.json") == lock["updates_manifest_sha256"],
        "Update manifest changed",
    )
    if "question_interface" in plan:
        require(
            lock["question_interface"] == plan["question_interface"],
            "Locked predictions use a different question interface",
        )
    require(
        file_hash(root / "interpreter-inputs.json") == lock["question_inputs_sha256"],
        "Locked prompt-aware input inventory changed",
    )
    require(
        file_hash(root / "actual-input-lengths.json") == lock["actual_input_lengths_sha256"],
        "Actual query length inventory changed",
    )
    expected = {
        f"predictions/{s}-{name}.json"
        for s in SPLITS
        for name in (
            "released",
            "released-shuffled",
            "adapted",
            "adapted-shuffled",
            "simple",
            "shuffle",
        )
    }
    require(set(lock["files_sha256"]) == expected, "Missing prediction methods or split")
    for name, expected_hash in lock["files_sha256"].items():
        require(file_hash(root / name) == expected_hash, "Locked predictions changed")
    return lock


def measure_all(root):
    plan, _, cases = verify_inputs(root)
    lock = checked_prediction_lock(root, plan)
    require(not (root / "measurements").exists(), "Measurements attempted; no automatic resume")
    started = time.time()
    require(lock["finished_unix"] < started, "Predictions did not precede measurement")
    (root / "measurements").mkdir()
    model, tokenizer = load_target()
    updates = read(root / "updates-manifest.json")
    require(
        [r["id"] for r in updates["records"]] == [c["id"] for c in cases], "Update grid differs"
    )
    inventory = []
    for i, (case, item) in enumerate(zip(cases, updates["records"])):
        guard(root)
        path = root / "updates" / f"{case['id']}.json"
        require(file_hash(path) == item["sha256"], "Update metadata changed")
        record = read(path)
        initial_name = f"initial-{case['initialization_seed']}.pt"
        initial_path, snapshot_path = (
            root / "snapshots" / initial_name,
            root / "snapshots" / f"{case['id']}.pt",
        )
        require(
            file_hash(initial_path)
            == record["initial_sha256"]
            == updates["initial_files_sha256"][initial_name]
            and file_hash(snapshot_path) == record["snapshot_sha256"],
            "Target snapshots changed",
        )
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        updated = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        restore_modules(model, initial)
        before = measure(model, tokenizer, case["probes"])
        restore_modules(model, updated)
        after = measure(model, tokenizer, case["probes"])
        restore_modules(model, initial)
        restored = measure(model, tokenizer, case["probes"][:1])
        require(np.array_equal(restored, before[:1]), "Probe scores did not restore exactly")
        if case["control"]:
            require(np.array_equal(before, after), "Zero update changed probe scores")
        output = root / "measurements" / f"{case['id']}.json"
        atomic_json(
            output,
            {
                "id": case["id"],
                "finished_unix": time.time(),
                "before_ce": before.tolist(),
                "after_ce": after.tolist(),
                "effect": effect_label(before, after),
                "bit_exact_adapter_restore": True,
                "bit_exact_first_probe_restore": True,
                "predictions_lock_sha256": file_hash(root / "predictions-lock.json"),
            },
        )
        inventory.append({"id": case["id"], "sha256": file_hash(output)})
        status(root, "measuring_heldout_effects", i + 1, len(cases))
        del initial, updated
    checked_prediction_lock(root, plan)
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "measurement_started_unix": started,
            "finished_unix": time.time(),
            "cases": len(cases),
            "actual_updates": 256,
            "zero_controls": 16,
            "probe_pattern_scores": 69632,
            "records": inventory,
            "source_commit": plan["source_commit"],
            "plan_sha256": file_hash(root / "plan.json"),
            "predictions_lock_sha256": file_hash(root / "predictions-lock.json"),
            "updates_manifest_sha256": file_hash(root / "updates-manifest.json"),
            "preflight_sha256": file_hash(root / "preflight.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "updates", "predict", "measure"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026101662)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {
        "preflight": preflight,
        "updates": collect_updates,
        "predict": predict_all,
        "measure": measure_all,
    }[args.command](args.root)
