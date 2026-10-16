"""Independent integrity audit and fixed two-grid analysis for functional update descriptions."""

import argparse
import hashlib
import html
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from .loracle import MODULES
from .loracle_toy_suite import initialize_like
from .loracle_trajectory_toy import fingerprint, read, restore_modules
from .study import atomic_json, file_hash
from .update_loracle_analysis import alive, require
from .update_loracle_collect import tokenize_update, upstream_module
from .update_loracle_confirmation_stats import evaluate as frozen_evaluate
from .update_loracle_data import LABELS, NO_CHANGE, PATTERN_IDS, description, effect_label
from .update_loracle_exact_calibration import (
    calibration_scores,
    physical_norm,
    scaled_native,
    score_and_restore,
)
from .update_loracle_exact_calibration_heldout import (
    RECIPE,
    SPLITS,
    checked_data,
    checked_prediction_lock,
    checked_preflight,
    donor_map,
)
from .update_loracle_exact_calibration_inference import prepare_model, verified_selection
from .update_loracle_inference import COMPONENT_INTERFACE, interface_contract
from .update_loracle_sensitivity_analysis import profile_metrics
from .update_loracle_train import shuffled_directions


def check_released_inputs(plan):
    """Bind the comparator to its original pre-v5 plan, including upstream code."""
    baseline = Path(plan["baseline_root"])
    original = baseline / "plan.json"
    require(
        file_hash(original) == "fb073d22ca1ba21a945760f9e64bda554ac6ff485a2cc808a888da05bb24a075",
        "Original released comparator plan differs",
    )
    expected = read(original)["input_files_sha256"]
    require(
        plan["released_files_sha256"] == expected
        and all(plan["input_files_sha256"].get(p) == h for p, h in expected.items()),
        "Released model, encoding or dependencies changed",
    )


def check_functional_preflight(value):
    require(
        all(
            value.get(k) is True
            for k in (
                "passed",
                "training_only",
                "synthetic_only_no_target_updates",
                "all_256_scores_finite",
                "bit_exact_adapter_and_full_score_restore",
                "frozen_base_unchanged",
            )
        )
        and value["modules"] == 280
        and value["trainable_tensors"] == 560
        and value["parameters"] == 64225280
        and value["physical_norm_scale_relative_error"] < 5e-6,
        "Functional training-only preflight failed",
    )


def checked_exact_profile(record, expected_before):
    """Independently compute equal-document changes, the centered profile and category."""
    before, after = [np.asarray(record[k], dtype=np.float64) for k in ("before_ce", "after_ce")]
    require(
        before.shape == after.shape == (8, 32)
        and np.isfinite(before).all()
        and np.isfinite(after).all()
        and np.array_equal(before, expected_before),
        "Exact calibration scores or shared zero-B baseline differ",
    )
    raw = np.array([sum(after[q, p] - before[q, p] for q in range(8)) / 8 for p in range(32)])
    centered = raw.mean() - raw
    label = NO_CHANGE if np.ptp(centered) < 0.02 else PATTERN_IDS[int(centered.argmax())]
    require(
        record["prediction"] == label
        and record["description"] == description(label)
        and record["bit_exact_restore"] is True,
        "Category, no-change floor, description or restore differs",
    )
    np.testing.assert_allclose(record["ce_change"], raw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(record["profile"], centered, rtol=0, atol=1e-12)
    return centered


def check_native_score(record, recipient, donor, initial, native, kind, times):
    """The shuffled condition applies the donor's physical update at the recipient norm."""
    norm, donor_norm = recipient["physical_update_norm"], donor["physical_update_norm"]
    scale = norm / donor_norm if donor_norm else 1.0
    require((norm > 0) == (donor_norm > 0), "Actual and zero donors mixed")
    require(
        record["id"] == recipient["id"]
        and record["kind"] == kind
        and record["direction_donor"] == donor["id"]
        and record["initialization_seed"] == donor["initialization_seed"]
        and record["donor_scale"] == scale
        and record["physical_update_norm"] == norm
        and record["snapshot_sha256"] == donor["snapshot_sha256"]
        and times[0] < record["finished_unix"] < times[1],
        "Native score identity, scaling, snapshot or lock ordering differs",
    )
    scaled = scaled_native(initial, native, scale)
    actual_norm = physical_norm(scaled)
    require(
        np.isclose(actual_norm, norm, rtol=5e-6, atol=1e-10)
        and np.isclose(record["actual_scaled_norm"], actual_norm, rtol=1e-12, atol=1e-12)
        and record["initial_native_sha256"] == fingerprint(initial)
        and record["updated_native_sha256"] == fingerprint(scaled),
        "Native direction, donor A, FP32 scaling or physical norm differs",
    )
    if norm == 0:
        require(np.array_equal(record["before_ce"], record["after_ce"]), "Zero calibration changed")
    return scale


def fixed_rescore(root, selected, initials, requests):
    """Four predeclared actual-adapter rechecks, never new optimizer steps."""
    require(len(requests) == 4, "Four fixed scoring rechecks required")
    _, model, tokenizer = prepare_model(root / "selection.json")
    results = []
    for case_id, kind, donor, scale in requests:
        initial = initials[donor["initialization_seed"]]
        native = torch.load(
            root / "snapshots" / f"{donor['id']}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        updated = scaled_native(initial, native, scale)
        saved = read(root / "calibration" / f"{case_id}-{kind}.json")
        restore_modules(model, initial)
        before = calibration_scores(model, tokenizer, selected["calibration"])
        require(np.array_equal(before, saved["before_ce"]), "Fixed baseline recheck differs")
        after = score_and_restore(
            model, tokenizer, selected["calibration"], initial, updated, before
        )
        require(np.array_equal(after, saved["after_ce"]), "Fixed exact scoring recheck differs")
        results.append({"id": case_id, "kind": kind, "donor": donor["id"], "passed": True})
    del model, tokenizer
    torch.cuda.empty_cache()
    return results


def audit_functional_scores(root, cases, initials, selected, updates, lock):
    paths = {f"{c['id']}-{kind}.json" for c in cases for kind in ("own", "shuffled")}
    require(
        len(paths) == 544 and {p.name for p in (root / "calibration").glob("*.json")} == paths,
        "Missing or extra exact calibration conditions",
    )
    prior = Path(selected["development_root"])
    old_integrity = read(prior / "integrity.json")
    require(
        old_integrity["passed"]
        and file_hash(prior / "baselines.json") == old_integrity["files_sha256"]["baselines.json"],
        "Selected baseline changed",
    )
    old_baselines = list(read(prior / "baselines.json")["scores"].values())
    require(
        len(old_baselines) == 8 and all(np.array_equal(x, old_baselines[0]) for x in old_baselines),
        "Prior zero-B baselines differ",
    )
    baseline = np.asarray(old_baselines[0])
    lookup = {c["id"]: read(root / "updates" / f"{c['id']}.json") for c in cases}
    donors = donor_map(cases)
    chosen = {
        next(c["id"] for c in cases if c["split"] == split and not c["control"]) for split in SPLITS
    }
    requests = []
    for split in SPLITS:
        group = [c for c in cases if c["split"] == split]
        for method, kind in (("functional", "own"), ("functional-shuffled", "shuffled")):
            predictions = read(root / "predictions" / f"{split}-{method}.json")
            require(
                [r["id"] for r in predictions] == [c["id"] for c in group],
                "Functional prediction grid differs",
            )
            for case, prediction in zip(group, predictions, strict=True):
                recipient = lookup[case["id"]]
                donor = lookup[case["id"] if kind == "own" else donors[case["id"]]]
                record = read(root / "calibration" / f"{case['id']}-{kind}.json")
                require(prediction == record, "Functional prediction differs from locked scores")
                native = torch.load(
                    root / "snapshots" / f"{donor['id']}.pt",
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                scale = check_native_score(
                    record,
                    recipient,
                    donor,
                    initials[donor["initialization_seed"]],
                    native,
                    kind,
                    (updates["finished_unix"], lock["finished_unix"]),
                )
                checked_exact_profile(record, baseline)
                if case["id"] in chosen:
                    requests.append((case["id"], kind, donor, scale))
                del native
    return fixed_rescore(root, selected, initials, requests)


def evaluate(cases, targets, predictions, spans, replicates=4000, component_interface=True):
    """Reuse the original fixed statistics without presenting this method as an adapter."""
    require(component_interface, "Original released four-question comparison required")
    names = {"functional": "adapted", "functional_shuffled": "adapted_shuffled"}
    result, draws = frozen_evaluate(
        cases,
        targets,
        {names.get(k, k): v for k, v in predictions.items()},
        spans,
        replicates,
        component_interface=True,
    )
    inverse = {v: k for k, v in names.items()}
    for key in ("accuracy", "zero_control_accuracy"):
        result[key] = {inverse.get(k, k): v for k, v in result[key].items()}
    result["primary_method"] = "functional"
    result["statistics_aliases"] = names
    return result, {inverse.get(k, k): v for k, v in draws.items()}


def barrier(root):
    controller = read(root / "controller.json")
    require(
        controller["status"] == "complete"
        and not alive(controller["controller_pid"])
        and controller["worker_pid"] is None,
        "Complete dead held-out controller required",
    )
    attempts = controller["attempts"]
    require(
        [a["job"] for a in attempts] == ["preflight", "updates", "predict", "measure"],
        "Expected exactly four successful stages without retries",
    )
    previous = controller["started_unix"]
    for attempt in attempts:
        require(
            not alive(attempt["pid"]) and attempt["returncode"] == 0,
            "A held-out stage remains active or failed",
        )
        require(
            previous
            <= attempt["started_unix"]
            <= attempt["finished_unix"]
            <= controller["deadline_unix"],
            "Stage ordering or original deadline differs",
        )
        previous = attempt["finished_unix"]
    return controller


def archive(root, files):
    path = root / "source_snapshot.tar"
    if not path.exists():
        with tarfile.open(path, "x") as tar:
            for name, expected in files.items():
                require(file_hash(name) == expected, "Execution source changed before archive")
                tar.add(name, arcname=name, recursive=False)
    with tarfile.open(path) as tar:
        require(set(tar.getnames()) == set(files), "Source archive inventory differs")
        for name, expected in files.items():
            require(
                hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected,
                "Archived execution source differs",
            )
    return file_hash(path)


def check_snapshot(value):
    require(
        set(value) == {f"{layer:02d}:{module}" for layer in range(40) for module in MODULES},
        "Expected every target LoRA module",
    )
    require(
        sum(t.numel() for pair in value.values() for t in pair) == 64225280,
        "Target parameter count differs",
    )
    require(
        all(
            len(pair) == 2
            and pair[0].shape[0] == pair[1].shape[1] == 16
            and all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in pair)
            for pair in value.values()
        ),
        "Target rank, dtype or finiteness differs",
    )


def audit(root):
    controller = barrier(root)  # No scientific output parsing before normal completion and death.
    require(not (root / "integrity.json").exists(), "Preserve the completed audit; no overwrite")
    plan = read(root / "plan.json")
    require(
        controller["scope"] == plan["scope"] == "heldout_confirmation",
        "Incorrect held-out controller scope",
    )
    from .update_loracle_exact_calibration_evaluation_run import validate_plan

    validate_plan(plan, root)
    require(file_hash(root / "plan.json") == controller["plan_sha256"], "Held-out plan changed")
    require(
        plan["max_hours"] == 12
        and controller["deadline_unix"] == controller["started_unix"] + 43200,
        "Immutable held-out deadline changed",
    )
    for path, expected in plan["source_files_sha256"].items():
        require(file_hash(path) == expected, "Frozen source changed")
    archive_hash = archive(root, plan["source_files_sha256"])
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, "Held-out model/input changed")
    for name, key in (
        ("data.json", "data_sha256"),
        ("analysis-plan.json", "analysis_plan_sha256"),
        ("selection.json", "selection_sha256"),
    ):
        require(file_hash(root / name) == plan[key], f"Frozen {name} changed")
    selection = verified_selection(root / "selection.json")
    interface = COMPONENT_INTERFACE
    require(
        plan["recipe"] == RECIPE
        and plan["question_interface"]
        == selection["released_interface"]
        == interface_contract(interface),
        "Functional recipe or released question interface differs",
    )
    check_released_inputs(plan)
    data = read(root / "data.json")
    require(
        data["selection_sha256"] == plan["selection_sha256"]
        and data["original_data_sha256"] == file_hash(plan["original_data_path"]),
        "Held-out data/model binding differs",
    )
    cases = checked_data(data, read(plan["original_data_path"]))
    ids = [c["id"] for c in cases]
    preflight = checked_preflight(root, plan)
    target_preflight = read(root / "target-preflight.json")
    require(
        target_preflight["passed"] and target_preflight["bit_exact_adapter_and_probe_restore"],
        "Target preflight did not pass",
    )
    functional_preflight = read(root / "functional-preflight.json")
    check_functional_preflight(functional_preflight)
    updates, manifest = read(root / "updates-manifest.json"), read(root / "manifest.json")
    prediction_lock = checked_prediction_lock(root, plan)
    prediction_manifest = read(root / "predictions-manifest.json")
    events = (
        (target_preflight["finished_unix"], preflight["finished_unix"]),
        (updates["started_unix"], updates["finished_unix"]),
        (prediction_lock["finished_unix"], prediction_manifest["finished_unix"]),
        (manifest["measurement_started_unix"], manifest["finished_unix"]),
    )
    require(
        all(
            a["started_unix"] <= first <= last <= a["finished_unix"]
            for a, (first, last) in zip(controller["attempts"], events, strict=True)
        ),
        "Scientific output escaped its recorded execution stage",
    )
    require(
        selection["frozen_unix"]
        < data["created_unix"]
        < plan["created_unix"]
        < controller["started_unix"]
        <= target_preflight["finished_unix"]
        < functional_preflight["finished_unix"]
        < preflight["finished_unix"]
        < updates["started_unix"]
        < updates["finished_unix"]
        < prediction_lock["finished_unix"]
        <= prediction_manifest["finished_unix"]
        < manifest["measurement_started_unix"]
        < manifest["finished_unix"]
        <= controller["deadline_unix"],
        "Model/data/prediction/outcome timeline differs",
    )
    require(
        manifest["status"] == updates["status"] == "complete"
        and manifest["source_commit"] == plan["source_commit"]
        and manifest["cases"] == updates["cases"] == 272
        and manifest["actual_updates"] == updates["actual_updates"] == 256
        and manifest["zero_controls"] == 16
        and manifest["probe_pattern_scores"] == 69632,
        "Incomplete held-out grid",
    )
    for field, name in (
        ("plan_sha256", "plan.json"),
        ("predictions_lock_sha256", "predictions-lock.json"),
        ("updates_manifest_sha256", "updates-manifest.json"),
        ("preflight_sha256", "preflight.json"),
    ):
        require(manifest[field] == file_hash(root / name), "Final manifest hash differs")
    require(
        updates["plan_sha256"] == controller["plan_sha256"]
        and updates["preflight_sha256"] == file_hash(root / "preflight.json")
        and not updates["heldout_probes_measured"],
        "Update stage contract differs",
    )
    require(
        [r["id"] for r in updates["records"]] == ids == [r["id"] for r in manifest["records"]],
        "Missing, duplicated or reordered cases",
    )
    for directory, suffix in (("updates", ".json"), ("measurements", ".json"), ("tokens", ".pt")):
        require(
            {p.stem for p in (root / directory).iterdir() if p.suffix == suffix} == set(ids),
            f"Extra or missing {directory}",
        )
    initials = {}
    for name, expected in updates["initial_files_sha256"].items():
        path = root / "snapshots" / name
        require(file_hash(path) == expected, "Initial target adapter changed")
        seed = int(name.removeprefix("initial-").removesuffix(".pt"))
        value = torch.load(path, map_location="cpu", weights_only=True)
        check_snapshot(value)
        require(
            fingerprint(value) == fingerprint(initialize_like(value, seed)),
            "Wrong target initialization",
        )
        initials[seed] = value
    require(set(initials) == {c["initialization_seed"] for c in cases}, "Target seed grid differs")
    require(
        {p.stem for p in (root / "snapshots").glob("*.pt")}
        == set(ids) | {f"initial-{seed}" for seed in initials},
        "Target snapshot grid differs",
    )
    upstream = upstream_module()
    last_update, last_measure = updates["started_unix"], manifest["measurement_started_unix"]
    for case, update_item, measured_item in zip(cases, updates["records"], manifest["records"]):
        upath, mpath = (
            root / folder / f"{case['id']}.json" for folder in ("updates", "measurements")
        )
        require(
            file_hash(upath) == update_item["sha256"]
            and file_hash(mpath) == measured_item["sha256"],
            "Held-out case output changed",
        )
        update, measured = read(upath), read(mpath)
        require(
            all(update[k] == case[k] for k in ("id", "split", "control", "initialization_seed"))
            and measured["id"] == case["id"],
            "Case metadata differs",
        )
        require(
            last_update < update["finished_unix"] <= updates["finished_unix"]
            and last_measure < measured["finished_unix"] <= manifest["finished_unix"],
            "Held-out case timing differs",
        )
        last_update, last_measure = update["finished_unix"], measured["finished_unix"]
        require(
            update["bit_exact_adapter_restore"]
            and not update["heldout_probes_measured"]
            and measured["bit_exact_adapter_restore"]
            and measured["bit_exact_first_probe_restore"]
            and measured["predictions_lock_sha256"] == file_hash(root / "predictions-lock.json"),
            "Restoration or prediction lock differs",
        )
        before, after = np.asarray(measured["before_ce"]), np.asarray(measured["after_ce"])
        require(
            before.shape == after.shape == (4, 32)
            and np.isfinite(before).all()
            and np.isfinite(after).all()
            and effect_label(before, after) == measured["effect"],
            "Held-out measured effect differs",
        )
        start = initials[case["initialization_seed"]]
        require(
            update["initial_sha256"]
            == updates["initial_files_sha256"][f"initial-{case['initialization_seed']}.pt"],
            "Wrong update starting state",
        )
        snapshot, token_path = (
            root / "snapshots" / f"{case['id']}.pt",
            root / "tokens" / f"{case['id']}.pt",
        )
        require(
            file_hash(snapshot) == update["snapshot_sha256"]
            and file_hash(token_path) == update["tokens_sha256"],
            "Held-out native weights or tokens changed",
        )
        updated = torch.load(snapshot, map_location="cpu", weights_only=True)
        check_snapshot(updated)
        scaled_native(start, updated, 1.0)  # Original A unchanged, zero initial B.
        stored = torch.load(token_path, map_location="cpu", weights_only=True)
        tokens, norm, token_audit = tokenize_update(start, updated, upstream)
        require(
            torch.equal(tokens, stored)
            and norm == update["physical_update_norm"]
            and token_audit == update["token_audit"],
            "Native encoding or true update norm differs",
        )
        if case["control"]:
            require(
                update["step"] is None
                and norm == 0
                and np.array_equal(before, after)
                and fingerprint(start) == fingerprint(updated),
                "Zero control changed",
            )
        else:
            require(
                update["step"]["optimizer_steps"] == 1
                and norm > 0
                and update["step"]["learning_rate"] == plan["learning_rate"] == 0.0002,
                "Incorrect one-step optimizer update",
            )
        del updated, stored, tokens
    scoring_checks = audit_functional_scores(
        root, cases, initials, selection, updates, prediction_lock
    )
    if interface == COMPONENT_INTERFACE:
        from .update_loracle_factorized import training_instances, training_majorities
        from .update_loracle_factorized_analysis import cases_with_effects

        training, _ = cases_with_effects(Path(plan["collection_root"]), "train")
        _, balance = training_instances(training)
        expected_majorities = training_majorities(training, balance)
    for split in SPLITS:
        subset = [c for c in cases if c["split"] == split]
        from .update_loracle_factorized_analysis import check_predictions as check_method

        if interface == COMPONENT_INTERFACE:
            from .update_loracle_factorized_analysis import check_predictions as check_components

            check_method = check_components
            subset = [
                {
                    **c,
                    "tokens": "already_verified",
                    "physical_update_norm": read(root / "updates" / f"{c['id']}.json")[
                        "physical_update_norm"
                    ],
                }
                for c in subset
            ]
        for method in ("released", "released-shuffled"):
            check_method(read(root / "predictions" / f"{split}-{method}.json"), subset)
        expected = shuffled_directions([{**c, "tokens": c["id"]} for c in subset])
        require(
            read(root / "predictions" / f"{split}-shuffle.json")
            == [{"id": c["id"], "donor": c["direction_donor"]} for c in expected],
            "Held-out shuffled direction mapping differs",
        )
        simple = read(root / "predictions" / f"{split}-simple.json")
        require(
            simple["feature_count"] == 35841
            and simple["majority"] in LABELS
            and [r["id"] for r in simple["development"]] == [c["id"] for c in subset]
            and all(
                r[k] in LABELS
                for r in simple["development"]
                for k in ("ridge", "majority", "shuffled_ridge")
            ),
            "Incomplete simple baseline grid",
        )
        if interface == COMPONENT_INTERFACE:
            require(
                simple["factorized_majority"] == expected_majorities["factorized"]
                and simple["majority"] == expected_majorities["flat"]
                and all(
                    r["factorized_majority"] == expected_majorities["factorized"]
                    and r["majority"] == expected_majorities["flat"]
                    for r in simple["development"]
                ),
                "Component/flat majority must use only original training counts",
            )
    compact = [
        "controller.json",
        "plan.json",
        "data.json",
        "selection.json",
        "analysis-plan.json",
        "preflight.json",
        "target-preflight.json",
        "functional-preflight.json",
        "updates-manifest.json",
        "predictions-lock.json",
        "predictions-manifest.json",
        "manifest.json",
        "source_snapshot.tar",
    ]
    compact += [
        str(p.relative_to(root))
        for folder in ("updates", "measurements", "predictions", "calibration")
        for p in (root / folder).glob("*.json")
    ]
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "scope": "reserved_and_fresh_confirmation",
            "recipe": RECIPE,
            "exact_calibration_conditions": 544,
            "fixed_scoring_rechecks": scoring_checks,
            "finished_unix": time.time(),
            "source_commit": plan["source_commit"],
            "source_archive_sha256": archive_hash,
            "all_encodings_and_norms_recomputed": True,
            "predictions_locked_before_measurement": True,
            "question_interface": interface_contract(interface),
            "cases": 272,
            "actual_updates": 256,
            "zero_controls": 16,
            "files_sha256": {name: file_hash(root / name) for name in compact},
        },
    )


def diagnostics(cases, measurements, predictions):
    targets = [m["effect"]["label"] for m in measurements]
    active = [i for i, c in enumerate(cases) if not c["control"]]
    controls = [i for i, c in enumerate(cases) if c["control"]]

    def scored(indices):
        return {
            "n": len(indices),
            "accuracy": {
                name: float(np.mean([values[i] == targets[i] for i in indices]))
                if indices
                else None
                for name, values in predictions.items()
            },
        }

    groups = {}
    for key in ("source_pattern", "initialization_seed"):
        groups[key] = {
            str(value): scored([i for i in active if cases[i][key] == value])
            for value in sorted({cases[i][key] for i in active})
        }
    groups["measured_target"] = {
        label: scored([i for i in active if targets[i] == label]) for label in LABELS
    }
    axes = {}
    for j, name in enumerate(("format", "case", "order")):
        defined = [i for i in active if targets[i] != NO_CHANGE]
        axes[name] = {
            "n_with_defined_axis": len(defined),
            "weak_updates_excluded_here_only": len(active) - len(defined),
            "accuracy": {
                method: float(
                    np.mean(
                        [
                            values[i] != NO_CHANGE
                            and values[i].split("/")[j] == targets[i].split("/")[j]
                            for i in defined
                        ]
                    )
                )
                if defined
                else None
                for method, values in predictions.items()
            },
        }
    confusion = {}
    for name, values in predictions.items():
        counts = np.zeros((33, 33), dtype=int)
        for i in active:
            counts[LABELS.index(targets[i]), LABELS.index(values[i])] += 1
        confusion[name] = counts.tolist()
    return {
        "actual_updates": scored(active),
        "zero_controls": scored(controls),
        "groups": groups,
        "axes": axes,
        "confusion_rows_true_columns_predicted": confusion,
        "confusion_labels": list(LABELS),
        "span_quantiles_0_25_50_75_100": np.quantile(
            [measurements[i]["effect"]["span"] for i in active], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "top_two_margin_quantiles_0_25_50_75_100": np.quantile(
            [measurements[i]["effect"]["top_two_margin"] for i in active], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
    }


def analyze(root, output):
    barrier(root)
    require(not output.exists(), "Preserve earlier held-out analysis outputs")
    integrity = read(root / "integrity.json")
    require(
        integrity["passed"]
        and integrity["scope"] == "reserved_and_fresh_confirmation"
        and integrity["recipe"] == RECIPE
        and integrity["cases"] == 272
        and integrity["actual_updates"] == 256
        and integrity["zero_controls"] == 16
        and integrity["exact_calibration_conditions"] == 544
        and len(integrity["fixed_scoring_rechecks"]) == 4
        and all(r["passed"] for r in integrity["fixed_scoring_rechecks"]),
        "Complete held-out audit required",
    )
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited held-out output changed")
    data = read(root / "data.json")
    interface = integrity["question_interface"]["name"]
    component = interface == COMPONENT_INTERFACE
    require(component, "Original four-question interface required")
    summaries, details, bootstrap = {}, {}, {}
    for split in SPLITS:
        cases = data[split]["cases"]
        measurements = [read(root / "measurements" / f"{c['id']}.json") for c in cases]
        answers = {
            name.replace("-", "_"): read(root / "predictions" / f"{split}-{name}.json")
            for name in ("functional", "released", "functional-shuffled", "released-shuffled")
        }
        predictions = {name: [r["prediction"] for r in rows] for name, rows in answers.items()}
        simple = read(root / "predictions" / f"{split}-simple.json")["development"]
        for name in ("ridge", "majority", "shuffled_ridge"):
            predictions[name] = [r[name] for r in simple]
        if component:
            predictions["factorized_majority"] = [r["factorized_majority"] for r in simple]
        targets = [m["effect"]["label"] for m in measurements]
        spans = [m["effect"]["span"] for m in measurements]
        primary = {
            name: predictions[name]
            for name in (
                "functional",
                "released",
                "functional_shuffled",
                "majority",
                "ridge",
            )
        }
        if component:
            primary["factorized_majority"] = predictions["factorized_majority"]
        summary, draws = evaluate(
            cases, targets, primary, spans, replicates=4000, component_interface=component
        )
        summary["complete_diagnostics"] = diagnostics(cases, measurements, predictions)
        active = [i for i, c in enumerate(cases) if not c["control"]]
        actual_profiles = np.asarray([m["effect"]["centered_improvements"] for m in measurements])
        summary["profile_errors"] = {
            name: profile_metrics(
                np.asarray([r["profile"] for r in answers[name]])[active], actual_profiles[active]
            )
            for name in ("functional", "functional_shuffled")
        }
        summaries[split] = summary
        bootstrap.update({f"{split}_{name}": value for name, value in draws.items()})
        details[split] = [
            {
                **c,
                "measured": m,
                "predictions": {n: p[i] for n, p in predictions.items()},
                "interpreter_answers": {n: rows[i] for n, rows in answers.items()},
            }
            for i, (c, m) in enumerate(zip(cases, measurements))
        ]
    output.mkdir(parents=True)
    atomic_json(
        output / "summary.json",
        {
            "scope": "Two separately drawn input sets and eight held-out source compositions of "
            "known operations on one fixed base. Measured reference-loss preferences "
            "after one step. Functional predictor uses the pinned base model and a fixed training "
            "calibration library in addition to the native update.",
            "splits": summaries,
            "both_grids_pass": all(s["all_numerical_gates_pass"] for s in summaries.values()),
            "selection_sha256": file_hash(root / "selection.json"),
            "question_interface": integrity["question_interface"],
            "limitations": [
                "No arbitrary-checkpoint or novel-operation claim",
                "Extra base-model/calibration access; not an improved LoRAcle checkpoint",
                "Within-grid intervals do not assert simultaneous coverage of both grids",
                "Loss preferences are not free generation",
                "Few composition/seed clusters; bootstrap bounds remain approximate",
                "All weak updates retained",
            ],
        },
    )
    atomic_json(output / "all-cases.json", details)
    examples = {
        split: [
            next(c for c in rows if c["source_pattern"] == pattern)
            for pattern in sorted({c["source_pattern"] for c in rows if not c["control"]})
        ]
        for split, rows in details.items()
    }
    atomic_json(output / "fixed-examples.json", examples)
    np.savez_compressed(root / "confirmation-bootstrap.npz", **bootstrap)
    (output / "executed-analysis.py").write_bytes(Path(__file__).read_bytes())
    from . import update_loracle_confirmation_stats as stats

    (output / "executed-statistics.py").write_bytes(Path(stats.__file__).read_bytes())
    atomic_json(
        output / "analysis-provenance.json",
        {
            "finished_unix": time.time(),
            "integrity_sha256": file_hash(root / "integrity.json"),
            "source_sha256": file_hash(__file__),
            "statistics_sha256": file_hash(stats.__file__),
            "replicates": 4000,
            "replicate_seed": 2026101665,
        },
    )
    render(output)


def render(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary, details = read(output / "summary.json"), read(output / "all-cases.json")
    names = [
        "released",
        "released_shuffled",
        "majority",
        "factorized_majority",
        "ridge",
        "shuffled_ridge",
        "functional_shuffled",
        "functional",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5), sharey=True)
    for ax, split in zip(axes, SPLITS):
        scores = summary["splits"][split]["complete_diagnostics"]["actual_updates"]["accuracy"]
        values = [scores[name] * 100 for name in names]
        bars = ax.bar(range(len(names)), values, color=["#919ba7"] * (len(names) - 1) + ["#176b8c"])
        ax.set(
            xticks=range(len(names)),
            xticklabels=[n.replace("_", "\n") for n in names],
            title=split.capitalize(),
            ylim=(0, 105),
            ylabel="Correct measured-effect category (%)",
        )
        ax.axhline(70, linestyle="--", linewidth=1, color="#444")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center")
    fig.tight_layout()
    fig.savefig(output / "heldout-accuracy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, split in zip(axes, SPLITS):
        rows = [c for c in details[split] if not c["control"]]
        actual = np.asarray([c["measured"]["effect"]["centered_improvements"] for c in rows])
        predicted = np.asarray([c["interpreter_answers"]["functional"]["profile"] for c in rows])
        ax.scatter(actual.ravel(), predicted.ravel(), s=5, alpha=0.25, color="#176b8c")
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], color="#444", linestyle="--", linewidth=1)
        ax.set(
            title=split.capitalize(),
            xlabel="Measured centered improvement (nats/token)",
            ylabel="Calibration prediction (nats/token)",
        )
    fig.tight_layout()
    fig.savefig(output / "heldout-profiles.png", dpi=180)
    plt.close(fig)
    rows = "".join(
        f"<h2>{split}</h2>"
        + "".join(
            f"<details><summary>{html.escape(c['id'])}: "
            f"{html.escape(c['measured']['effect']['label'])}</summary>"
            f"<pre>{html.escape(json.dumps(c, indent=2))}</pre></details>"
            for c in cases
        )
        for split, cases in details.items()
    )
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>One-update interpreter held-out results</title>"
        "<style>body{font:17px system-ui;max-width:1100px;margin:35px auto;"
        "padding:20px;line-height:1.5}"
        "img{max-width:100%}pre{white-space:pre-wrap;font-size:12px}details{padding:8px}</style>"
        "<h1>Can fixed calibration prompts describe a single update?</h1>"
        "<p>All model predictions were locked before scoring any held-out probe. Each test set "
        "contains 128 actual updates and eight zero controls. No weak updates were excluded. "
        "The functional method scores the base model on a fixed training calibration library. "
        "This is additional access, not a newly trained LoRAcle checkpoint. Targets describe "
        "reference-loss preferences, not freely generated behavior.</p>"
        "<img src='heldout-accuracy.png' alt='All methods on both held-out sets'>"
        "<img src='heldout-profiles.png' "
        "alt='All 32 predicted and measured effects for every update'>"
        "<p><a href='fixed-examples.json'>Fixed examples: "
        "first case in each source composition</a>. "
        "Each includes training inputs, independent probe prompts, measured losses, "
        "functional predictions and all released-interpreter answers.</p>"
        "<h2>Fixed criteria and paired uncertainty</h2>"
        f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre><h2>Every case</h2>{rows}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "audit":
        audit(args.root)
    else:
        if args.output is None:
            parser.error("--output is required")
        analyze(args.root, args.output)
