"""Two untouched grids for one gated exact-calibration functional predictor.

Every own/shuffled profile and released LoRAcle comparison locks before either
held-out grid is measured. This module never selects a method from test results.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from .loracle_toy_suite import initialize_like
from .loracle_trajectory_toy import (
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
from .update_loracle_data import effect_label
from .update_loracle_exact_calibration import (
    physical_norm,
    scaled_native,
)
from .update_loracle_exact_calibration import (
    preflight as functional_preflight,
)
from .update_loracle_exact_calibration_inference import (
    predict_with_model,
    prepare_model,
    verified_selection,
)
from .update_loracle_factorized import score, training_instances, training_majorities
from .update_loracle_heldout import SPLITS, checked_data, prediction_views, preflight_scores
from .update_loracle_inference import COMPONENT_INTERFACE, interface_contract
from .update_loracle_train import (
    load_interpreter,
    shuffled_directions,
    simple_references,
    verified_cases,
)

RECIPE = "exact_calibration_v5_confirmation"
METHOD_FILES = (
    "released",
    "released-shuffled",
    "functional",
    "functional-shuffled",
    "simple",
    "shuffle",
)


def verify_inputs(root):
    plan = guard(root)
    require(
        plan["recipe"] == RECIPE and plan["learning_rate"] == 0.0002,
        "Recipe or update rate changed",
    )
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, "Frozen test input changed")
    require(file_hash(root / "selection.json") == plan["selection_sha256"], "Selection changed")
    selection = verified_selection(root / "selection.json")
    require(
        plan["question_interface"]
        == selection["released_interface"]
        == interface_contract(COMPONENT_INTERFACE),
        "Released question interface changed",
    )
    require(
        plan["released_files_sha256"]
        and all(
            plan["input_files_sha256"].get(p) == h for p, h in plan["released_files_sha256"].items()
        ),
        "Released assets are not pinned",
    )
    data = read(root / "data.json")
    require(
        data["selection_sha256"] == plan["selection_sha256"]
        and data["original_data_sha256"] == file_hash(plan["original_data_path"]),
        "Original data or selection changed",
    )
    cases = checked_data(data, read(plan["original_data_path"]))
    require(
        selection["frozen_unix"] < data["created_unix"] < plan["created_unix"],
        "Selection/data/execution freeze order changed",
    )
    return plan, selection, cases


def preflight(root):
    plan, selection, _ = verify_inputs(root)
    require(not (root / "preflight-started.json").exists(), "Preflight already attempted; no retry")
    atomic_json(root / "preflight-started.json", {"started_unix": time.time()})
    model, tokenizer = load_target()
    training_preflight(model, tokenizer, root / "target-preflight.json")
    functional_preflight(
        model, tokenizer, selection["calibration"], root / "functional-preflight.json"
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    handle = load_interpreter()
    tokens = torch.randn(4480, 5120, generator=torch.Generator().manual_seed(2026101662))
    with torch.no_grad():
        before = preflight_scores(handle, tokens, COMPONENT_INTERFACE)
        after = preflight_scores(handle, tokens, COMPONENT_INTERFACE)
    require(before == after and len(before) == 4, "Released inference is not repeatable")
    atomic_json(
        root / "preflight.json",
        {
            "passed": True,
            "status": "complete",
            "finished_unix": time.time(),
            "training_only_inputs": True,
            "selection_sha256": plan["selection_sha256"],
            "question_interface": interface_contract(COMPONENT_INTERFACE),
            "questions_checked": 4,
            "bit_exact_released_score_repeat": True,
            "target_preflight_sha256": file_hash(root / "target-preflight.json"),
            "functional_preflight_sha256": file_hash(root / "functional-preflight.json"),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def checked_preflight(root, plan):
    value = read(root / "preflight.json")
    require(
        value["passed"]
        and value["training_only_inputs"]
        and value["bit_exact_released_score_repeat"]
        and value["selection_sha256"] == plan["selection_sha256"]
        and value["question_interface"] == interface_contract(COMPONENT_INTERFACE)
        and value["questions_checked"] == 4,
        "Incomplete functional/released preflight",
    )
    for name, field in (
        ("target-preflight.json", "target_preflight_sha256"),
        ("functional-preflight.json", "functional_preflight_sha256"),
    ):
        require(
            file_hash(root / name) == value[field] and read(root / name)["passed"],
            "Preflight changed",
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


def native_views(root, cases, inventory):
    """Exclude labels, examples, probes and control flags from native prediction inputs."""
    require([r["id"] for r in inventory] == [c["id"] for c in cases], "Native update grid differs")
    values = []
    for case, item in zip(cases, inventory, strict=True):
        path = root / "updates" / f"{case['id']}.json"
        require(file_hash(path) == item["sha256"], "Native metadata changed")
        row = read(path)
        require(
            row["id"] == case["id"] and row["heldout_probes_measured"] is False,
            "Wrong native input or premature measurement",
        )
        initial = root / "snapshots" / f"initial-{row['initialization_seed']}.pt"
        updated = root / "snapshots" / f"{case['id']}.pt"
        require(
            file_hash(initial) == row["initial_sha256"]
            and file_hash(updated) == row["snapshot_sha256"],
            "Native snapshot changed",
        )
        values.append(
            {
                "id": case["id"],
                "initialization_seed": row["initialization_seed"],
                "initial": initial,
                "updated": updated,
                "initial_sha256": row["initial_sha256"],
                "snapshot_sha256": row["snapshot_sha256"],
                "physical_update_norm": row["physical_update_norm"],
            }
        )
    return values


def donor_map(cases):
    result = {}
    for split in SPLITS:
        subset = [{**c, "tokens": c["id"]} for c in cases if c["split"] == split]
        for row in shuffled_directions(subset):
            result[row["id"]] = row["direction_donor"]
    require(set(result) == {c["id"] for c in cases}, "Incomplete donor assignments")
    return result


def predict_profiles(root, native, donors):
    """Score actual donor adapters after physical norm matching, never scaled loss profiles."""
    selection = root / "selection.json"
    _, model, tokenizer = prepare_model(selection)
    lookup = {v["id"]: v for v in native}
    assignments = [
        (v["id"], kind, donor)
        for v in native
        for kind, donor in (("own", v["id"]), ("shuffled", donors[v["id"]]))
    ]
    require(len(assignments) == 544 and len(lookup) == 272, "Full two-grid prediction required")
    directory = root / "calibration"
    directory.mkdir(exist_ok=False)
    results = {}
    for seed in sorted({v["initialization_seed"] for v in native}):
        guard(root)
        group = [a for a in assignments if lookup[a[2]]["initialization_seed"] == seed]
        paths = {lookup[donor]["initial"] for _, _, donor in group}
        require(len(paths) == 1, "Donor initializations differ")
        require(
            all(
                file_hash(lookup[d]["initial"]) == lookup[d]["initial_sha256"] for _, _, d in group
            ),
            "Native initialization changed before prediction",
        )
        initial = torch.load(next(iter(paths)), map_location="cpu", weights_only=True, mmap=True)
        metadata = []

        def snapshots(initial=initial, metadata=metadata, group=group):
            for recipient_id, kind, donor_id in group:
                guard(root)
                donor, recipient = lookup[donor_id], lookup[recipient_id]
                require(
                    file_hash(donor["updated"]) == donor["snapshot_sha256"],
                    "Native update changed before prediction",
                )
                original = torch.load(
                    donor["updated"], map_location="cpu", weights_only=True, mmap=True
                )
                own_norm, source_norm = (
                    recipient["physical_update_norm"],
                    donor["physical_update_norm"],
                )
                require((own_norm > 0) == (source_norm > 0), "Control and actual directions mixed")
                scale = own_norm / source_norm if source_norm else 1.0
                updated = scaled_native(initial, original, scale)
                norm = physical_norm(updated)
                require(
                    np.isclose(norm, own_norm, rtol=5e-6, atol=1e-10),
                    "Physical control norm differs",
                )
                metadata.append(
                    {
                        "id": recipient_id,
                        "kind": kind,
                        "direction_donor": donor_id,
                        "initialization_seed": seed,
                        "donor_scale": scale,
                        "physical_update_norm": own_norm,
                        "actual_scaled_norm": norm,
                        "snapshot_sha256": file_hash(donor["updated"]),
                    }
                )
                yield updated
                status(root, "exact_heldout_calibration", len(results) + len(metadata), 544)

        predictions = predict_with_model(selection, model, tokenizer, initial, snapshots())
        require(len(predictions) == len(group) == len(metadata), "Missing exact profiles")
        for identity, prediction in zip(metadata, predictions, strict=True):
            result = {**identity, **prediction, "finished_unix": time.time()}
            name = f"{identity['id']}-{identity['kind']}.json"
            atomic_json(directory / name, result)
            results[identity["id"], identity["kind"]] = result
            status(root, "exact_heldout_calibration", len(results), 544)
        del initial, predictions, metadata
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    require(len(results) == 544, "Incomplete own/shuffled predictions")
    return results


def required_prediction_files(root):
    updates = read(root / "updates-manifest.json")
    require(
        len(updates["records"]) == len({r["id"] for r in updates["records"]}) == 272,
        "Incomplete update inventory",
    )
    files = {f"predictions/{split}-{method}.json" for split in SPLITS for method in METHOD_FILES}
    files |= {
        f"calibration/{r['id']}-{kind}.json"
        for r in updates["records"]
        for kind in ("own", "shuffled")
    }
    return files


def predict_all(root):
    plan, _, cases = verify_inputs(root)
    checked_preflight(root, plan)
    require(not (root / "predictions").exists(), "Prediction attempted; no resume")
    require(not (root / "measurements").exists(), "Held-out measurement preceded predictions")
    updates = read(root / "updates-manifest.json")
    require(
        updates["status"] == "complete" and updates["plan_sha256"] == file_hash(root / "plan.json"),
        "Incomplete updates",
    )
    (root / "predictions").mkdir()
    views = prediction_views(root, cases, updates["records"])
    native = native_views(root, cases, updates["records"])
    donors = donor_map(cases)
    profiles = predict_profiles(root, native, donors)
    for split in SPLITS:
        ids = [c["id"] for c in cases if c["split"] == split]
        for method, kind in (("functional", "own"), ("functional-shuffled", "shuffled")):
            atomic_json(
                root / "predictions" / f"{split}-{method}.json", [profiles[i, kind] for i in ids]
            )
        atomic_json(
            root / "predictions" / f"{split}-shuffle.json",
            [{"id": i, "donor": donors[i]} for i in ids],
        )
    handle = load_interpreter()
    for split in SPLITS:
        ids = {c["id"] for c in cases if c["split"] == split}
        values = [v for v in views if v["id"] in ids]
        for method, inputs in (
            ("released", values),
            ("released-shuffled", shuffled_directions(values)),
        ):
            atomic_json(
                root / "predictions" / f"{split}-{method}.json",
                score(
                    handle,
                    inputs,
                    progress=lambda done, total: status(root, f"{split}-{method}", done, total),
                ),
            )
    del handle
    gc.collect()
    torch.cuda.empty_cache()
    training = verified_cases(Path(plan["collection_root"]), "train")
    _, balance = training_instances(training)
    majorities = training_majorities(training, balance)
    for split in SPLITS:
        ids = {c["id"] for c in cases if c["split"] == split}
        simple = simple_references(training, [v for v in views if v["id"] in ids])
        require(simple["majority"] == majorities["flat"], "Training majority differs")
        simple["factorized_majority"] = majorities["factorized"]
        for row in simple["development"]:
            row["factorized_majority"] = majorities["factorized"]
        atomic_json(root / "predictions" / f"{split}-simple.json", simple)
    verify_inputs(root)
    files = required_prediction_files(root)
    require(all((root / name).is_file() for name in files), "Missing prediction grid")
    atomic_json(
        root / "predictions-lock.json",
        {
            "status": "complete",
            "finished_unix": time.time(),
            "cases": 272,
            "selection_sha256": plan["selection_sha256"],
            "recipe": RECIPE,
            "question_interface": interface_contract(COMPONENT_INTERFACE),
            "updates_manifest_sha256": file_hash(root / "updates-manifest.json"),
            "files_sha256": {name: file_hash(root / name) for name in sorted(files)},
            "heldout_probes_measured": False,
        },
    )
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
        and manifest["lock_sha256"] == file_hash(root / "predictions-lock.json"),
        "Predictions are not fully locked",
    )
    lock = read(root / "predictions-lock.json")
    require(
        lock["status"] == "complete"
        and lock["recipe"] == RECIPE
        and lock["cases"] == 272
        and lock["selection_sha256"] == plan["selection_sha256"]
        and lock["question_interface"] == plan["question_interface"]
        and lock["heldout_probes_measured"] is False
        and lock["updates_manifest_sha256"] == file_hash(root / "updates-manifest.json"),
        "Prediction lock inconsistent",
    )
    require(
        set(lock["files_sha256"]) == required_prediction_files(root),
        "Missing method, control or grid",
    )
    for name, expected in lock["files_sha256"].items():
        require(file_hash(root / name) == expected, "Locked prediction changed")
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
