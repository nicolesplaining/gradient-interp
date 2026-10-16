import copy

import pytest
import torch

from gradient_interp import update_loracle_exact_calibration_heldout as worker
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_confirmation_data import fresh_draw
from gradient_interp.update_loracle_data import LABELS, build_data


def test_original_reserved_grid_cannot_change_for_new_test():
    original = build_data()
    data = {
        "schema": "single-update-heldout-data-v1",
        "reserved": {
            "cases": copy.deepcopy([c for c in original["cases"] if c["split"] == "reserved"]),
            "vocabulary": original["vocabularies"]["reserved"],
        },
        "confirmation": fresh_draw(original),
    }
    assert len(worker.checked_data(data, original)) == 272
    data["reserved"]["cases"][0]["batch"][0]["answer"] = "a more convenient target"
    with pytest.raises(ValueError, match="Original reserved cases changed"):
        worker.checked_data(data, original)


def test_prediction_input_allowlist_excludes_source_batch_probe_and_labels(tmp_path):
    (tmp_path / "updates").mkdir()
    (tmp_path / "tokens").mkdir()
    token_path = tmp_path / "tokens/example.pt"
    token_path.write_bytes(b"opaque tensor fixture")
    metadata = {
        "id": "example",
        "heldout_probes_measured": False,
        "tokens_sha256": file_hash(token_path),
        "physical_update_norm": 1.25,
        "irrelevant_private_field": "PRIVATE",
    }
    atomic_json(tmp_path / "updates/example.json", metadata)
    case = {
        "id": "example",
        "options": dict(zip(map(str, range(33)), LABELS)),
        "control": False,
        "batch": "PRIVATE",
        "probes": "PRIVATE",
        "source_pattern": "PRIVATE",
    }
    inventory = [{"id": "example", "sha256": file_hash(tmp_path / "updates/example.json")}]
    view = worker.prediction_views(tmp_path, [case], inventory)[0]
    assert set(view) == {"id", "tokens", "control", "physical_update_norm", "options", "text"}
    assert "PRIVATE" not in str(view)
    token_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="encoding changed"):
        worker.prediction_views(tmp_path, [case], inventory)


def prediction_lock(root):
    (root / "predictions").mkdir()
    atomic_json(root / "updates-manifest.json", {"records": [{"id": str(i)} for i in range(272)]})
    (root / "calibration").mkdir()
    files = {}
    for split in worker.SPLITS:
        for name in (
            "released",
            "released-shuffled",
            "functional",
            "functional-shuffled",
            "simple",
            "shuffle",
        ):
            path = root / "predictions" / f"{split}-{name}.json"
            atomic_json(path, [])
            files[str(path.relative_to(root))] = file_hash(path)
    for i in range(272):
        for kind in ("own", "shuffled"):
            path = root / "calibration" / f"{i}-{kind}.json"
            atomic_json(path, {"fixture": True})
            files[str(path.relative_to(root))] = file_hash(path)
    atomic_json(
        root / "predictions-lock.json",
        {
            "status": "complete",
            "recipe": worker.RECIPE,
            "question_interface": worker.interface_contract(worker.COMPONENT_INTERFACE),
            "cases": 272,
            "selection_sha256": "frozen",
            "heldout_probes_measured": False,
            "finished_unix": 1,
            "updates_manifest_sha256": file_hash(root / "updates-manifest.json"),
            "files_sha256": files,
        },
    )
    atomic_json(
        root / "predictions-manifest.json",
        {
            "status": "complete",
            "lock_sha256": file_hash(root / "predictions-lock.json"),
        },
    )


@pytest.mark.parametrize(
    "defect", ["absent", "prediction_changed", "lock_changed", "incomplete_methods"]
)
def test_probe_measurement_cannot_start_without_complete_unmodified_predictions(
    tmp_path, monkeypatch, defect
):
    monkeypatch.setattr(
        worker,
        "verify_inputs",
        lambda root: (
            {
                "selection_sha256": "frozen",
                "question_interface": worker.interface_contract(worker.COMPONENT_INTERFACE),
            },
            {},
            [],
        ),
    )
    monkeypatch.setattr(
        worker, "load_target", lambda: pytest.fail("Probe model loaded before barrier")
    )
    if defect != "absent":
        prediction_lock(tmp_path)
    if defect == "prediction_changed":
        (tmp_path / "predictions/reserved-functional.json").write_text('["changed"]')
    elif defect == "lock_changed":
        (tmp_path / "predictions-lock.json").write_text("{}")
    elif defect == "incomplete_methods":
        lock = worker.read(tmp_path / "predictions-lock.json")
        del lock["files_sha256"]["predictions/confirmation-functional.json"]
        atomic_json(tmp_path / "predictions-lock.json", lock)
        atomic_json(
            tmp_path / "predictions-manifest.json",
            {
                "status": "complete",
                "lock_sha256": file_hash(tmp_path / "predictions-lock.json"),
            },
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.measure_all(tmp_path)
    assert not (tmp_path / "measurements").exists()


def test_successful_prediction_lock_requires_both_sets_and_all_methods(tmp_path):
    prediction_lock(tmp_path)
    lock = worker.checked_prediction_lock(
        tmp_path,
        {
            "selection_sha256": "frozen",
            "question_interface": worker.interface_contract(worker.COMPONENT_INTERFACE),
        },
    )
    assert len(lock["files_sha256"]) == 556


def test_update_collection_never_measures_test_probes(tmp_path, monkeypatch):
    # Exercise the stage boundary with tiny native tensors, without loading a real model.
    cases = [
        {
            "id": "actual",
            "split": "reserved",
            "initialization_seed": 7,
            "control": False,
            "batch": ["training row"],
            "probes": ["forbidden probe"],
        },
        {
            "id": "zero",
            "split": "confirmation",
            "initialization_seed": 8,
            "control": True,
            "batch": [],
            "probes": ["forbidden probe"],
        },
    ]
    atomic_json(tmp_path / "plan.json", {})
    atomic_json(tmp_path / "preflight.json", {})
    monkeypatch.setattr(worker, "verify_inputs", lambda root: ({}, {}, cases))
    monkeypatch.setattr(worker, "checked_preflight", lambda root, plan: {})
    monkeypatch.setattr(worker, "guard", lambda root: {})
    state = {"layer": (torch.ones(2, 2), torch.zeros(2, 2))}
    monkeypatch.setattr(worker, "load_target", lambda: (None, None))
    monkeypatch.setattr(worker, "snapshot_modules", lambda model: copy.deepcopy(state))
    monkeypatch.setattr(worker, "initialize_like", lambda model, seed: copy.deepcopy(model))
    monkeypatch.setattr(worker, "upstream_module", lambda: None)
    monkeypatch.setattr(
        worker, "measure", lambda *a, **kw: pytest.fail("Test probe scored before prediction lock")
    )

    def restore(model, source):
        state.clear()
        state.update(copy.deepcopy(source))

    def step(model, tokenizer, batch, rate):
        assert batch == ["training row"] and rate == 0.0002
        state["layer"][1].add_(1)
        return {"optimizer_steps": 1, "learning_rate": rate}

    monkeypatch.setattr(worker, "restore_modules", restore)
    monkeypatch.setattr(worker, "one_step", step)
    monkeypatch.setattr(
        worker,
        "tokenize_update",
        lambda before, after, upstream: (
            after["layer"][1].clone(),
            float(after["layer"][1].norm()),
            [],
        ),
    )
    worker.collect_updates(tmp_path)
    manifest = worker.read(tmp_path / "updates-manifest.json")
    assert manifest["cases"] == 2 and manifest["actual_updates"] == 1
    assert not manifest["heldout_probes_measured"]
    for row in manifest["records"]:
        result = worker.read(tmp_path / "updates" / f"{row['id']}.json")
        assert "effect" not in result and "before_ce" not in result and "after_ce" not in result
    with pytest.raises(ValueError, match="no automatic resume"):
        worker.collect_updates(tmp_path)


def test_native_views_exclude_task_and_probe_data_and_bind_snapshots(tmp_path):
    (tmp_path / "updates").mkdir()
    (tmp_path / "snapshots").mkdir()
    initial = tmp_path / "snapshots/initial-7.pt"
    updated = tmp_path / "snapshots/example.pt"
    initial.write_bytes(b"initial")
    updated.write_bytes(b"updated")
    row = {
        "id": "example",
        "initialization_seed": 7,
        "heldout_probes_measured": False,
        "initial_sha256": file_hash(initial),
        "snapshot_sha256": file_hash(updated),
        "physical_update_norm": 2,
    }
    atomic_json(tmp_path / "updates/example.json", row)
    case = {"id": "example", "batch": "PRIVATE", "probes": "PRIVATE", "label": "PRIVATE"}
    inventory = [{"id": "example", "sha256": file_hash(tmp_path / "updates/example.json")}]
    view = worker.native_views(tmp_path, [case], inventory)[0]
    assert "PRIVATE" not in str(view) and "label" not in view and "control" not in view
    updated.write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        worker.native_views(tmp_path, [case], inventory)


def test_donors_stay_in_grid_and_preserve_zero_controls():
    cases = [
        {"id": f"{s}-{i}", "split": s, "control": i == 3} for s in worker.SPLITS for i in range(4)
    ]
    donors = worker.donor_map(cases)
    assert all(d.startswith(c["split"]) for c, d in [(c, donors[c["id"]]) for c in cases])
    assert all((donors[c["id"]] == c["id"]) == c["control"] for c in cases)


def test_exact_profiles_score_all544_real_scaled_adapters_and_keep_donor_A(tmp_path, monkeypatch):
    import numpy as np

    from gradient_interp.update_loracle_exact_calibration import effect_profile

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    native, cases = [], []
    for split_index, split in enumerate(worker.SPLITS):
        for i in range(136):
            seed = split_index * 4 + i % 4
            a = torch.ones(2, 3) * (seed + 1)
            b = torch.full((4, 2), 0.0 if i >= 128 else (i % 5 + 1) * 0.1)
            initial = snapshots / f"initial-{seed}.pt"
            if not initial.exists():
                torch.save({"m": (a, torch.zeros_like(b))}, initial)
            case_id = f"{split}-{i}"
            updated = snapshots / f"{case_id}.pt"
            torch.save({"m": (a, b)}, updated)
            native.append(
                {
                    "id": case_id,
                    "initialization_seed": seed,
                    "initial": initial,
                    "updated": updated,
                    "initial_sha256": file_hash(initial),
                    "snapshot_sha256": file_hash(updated),
                    "physical_update_norm": worker.physical_norm({"m": (a, b)}),
                }
            )
            cases.append({"id": case_id, "split": split, "control": i >= 128})
    before_hashes = {p.name: file_hash(p) for p in snapshots.iterdir()}
    monkeypatch.setattr(worker, "guard", lambda _: {})
    monkeypatch.setattr(worker, "prepare_model", lambda _: ({}, None, None))
    calls = []

    def predict(selection, model, tokenizer, initial, updates):
        result = []
        for updated in updates:
            assert torch.equal(initial["m"][0], updated["m"][0])
            # Nonlinear synthetic score demonstrates physical rescaling before evaluation.
            z = float(updated["m"][1].sum())
            baseline = np.ones((8, 32)) * 2
            after = baseline + np.tile(np.linspace(-z * z, z * z, 32), (8, 1))
            result.append(
                {
                    **effect_profile(baseline, after),
                    "before_ce": baseline.tolist(),
                    "after_ce": after.tolist(),
                    "bit_exact_restore": True,
                }
            )
            calls.append(z)
        return result

    monkeypatch.setattr(worker, "predict_with_model", predict)
    donors = worker.donor_map(cases)
    outputs = worker.predict_profiles(tmp_path, native, donors)
    assert len(outputs) == len(calls) == 544
    assert len(list((tmp_path / "calibration").glob("*.json"))) == 544
    lookup = {c["id"]: c for c in native}
    for (case_id, kind), row in outputs.items():
        donor_id = case_id if kind == "own" else donors[case_id]
        assert row["direction_donor"] == donor_id
        assert row["initialization_seed"] == lookup[donor_id]["initialization_seed"]
        assert row["actual_scaled_norm"] == pytest.approx(
            lookup[case_id]["physical_update_norm"], rel=5e-6
        )
        assert row["donor_scale"] == pytest.approx(
            lookup[case_id]["physical_update_norm"] / lookup[donor_id]["physical_update_norm"]
            if lookup[donor_id]["physical_update_norm"]
            else 1
        )
    assert {p.name: file_hash(p) for p in snapshots.iterdir()} == before_hashes


def test_all272_predictions_and_controls_lock_before_any_measurement(tmp_path, monkeypatch):
    from gradient_interp.update_loracle_data import NO_CHANGE

    for name in ("updates", "tokens", "snapshots"):
        (tmp_path / name).mkdir()
    cases, inventory = [], []
    for split_number, split in enumerate(worker.SPLITS):
        for i in range(136):
            case_id, seed = f"{split}-{i}", split_number * 4 + i % 4
            case = {
                "id": case_id,
                "split": split,
                "control": i >= 128,
                "initialization_seed": seed,
                "batch": "PRIVATE",
                "probes": "PRIVATE",
                "source_pattern": "PRIVATE",
                "options": dict(zip(map(str, range(33)), LABELS)),
            }
            cases.append(case)
            token, initial, updated = [
                tmp_path / p
                for p in (
                    f"tokens/{case_id}.pt",
                    f"snapshots/initial-{seed}.pt",
                    f"snapshots/{case_id}.pt",
                )
            ]
            token.write_bytes(b"opaque tokens")
            initial.write_bytes(b"opaque initial")
            updated.write_bytes(b"opaque updated")
            path = tmp_path / "updates" / f"{case_id}.json"
            atomic_json(
                path,
                {
                    "id": case_id,
                    "heldout_probes_measured": False,
                    "tokens_sha256": file_hash(token),
                    "initialization_seed": seed,
                    "initial_sha256": file_hash(initial),
                    "snapshot_sha256": file_hash(updated),
                    "physical_update_norm": 0.0 if case["control"] else 1.0,
                },
            )
            inventory.append({"id": case_id, "sha256": file_hash(path)})
    plan = {
        "selection_sha256": "frozen",
        "collection_root": str(tmp_path / "training"),
        "question_interface": worker.interface_contract(worker.COMPONENT_INTERFACE),
    }
    atomic_json(tmp_path / "plan.json", plan)
    atomic_json(
        tmp_path / "updates-manifest.json",
        {
            "status": "complete",
            "records": inventory,
            "plan_sha256": file_hash(tmp_path / "plan.json"),
        },
    )
    monkeypatch.setattr(worker, "verify_inputs", lambda _: (plan, {}, cases))
    monkeypatch.setattr(worker, "checked_preflight", lambda *_: {})
    monkeypatch.setattr(worker, "measure", lambda *_: pytest.fail("Premature test measurement"))
    observed = []

    def profiles(root, native, donors):
        assert len(native) == 272 and "PRIVATE" not in str(native)
        (root / "calibration").mkdir()
        result = {}
        for value in native:
            for kind in ("own", "shuffled"):
                row = {
                    "id": value["id"],
                    "prediction": NO_CHANGE,
                    "direction_donor": value["id"] if kind == "own" else donors[value["id"]],
                }
                result[value["id"], kind] = row
                atomic_json(root / "calibration" / f"{value['id']}-{kind}.json", row)
        return result

    def released(handle, views, progress):
        assert handle == "released" and len(views) == 136 and "PRIVATE" not in str(views)
        observed.extend(c["id"] for c in views)
        progress(136, 136)
        return [{"id": c["id"], "prediction": NO_CHANGE} for c in views]

    def simple(training, views):
        assert training == ["fixed training only"]
        return {
            "majority": NO_CHANGE,
            "development": [
                {
                    "id": v["id"],
                    "majority": NO_CHANGE,
                    "ridge": NO_CHANGE,
                    "shuffled_ridge": NO_CHANGE,
                }
                for v in views
            ],
        }

    monkeypatch.setattr(worker, "predict_profiles", profiles)
    monkeypatch.setattr(worker, "load_interpreter", lambda: "released")
    monkeypatch.setattr(worker, "score", released)
    monkeypatch.setattr(worker, "verified_cases", lambda path, split: ["fixed training only"])
    monkeypatch.setattr(worker, "training_instances", lambda _: (None, None))
    monkeypatch.setattr(
        worker, "training_majorities", lambda *_: {"flat": NO_CHANGE, "factorized": NO_CHANGE}
    )
    monkeypatch.setattr(worker, "simple_references", simple)
    worker.predict_all(tmp_path)
    lock = worker.checked_prediction_lock(tmp_path, plan)
    assert len(observed) == 544 and len(lock["files_sha256"]) == 556
    assert not (tmp_path / "measurements").exists()
    assert lock["heldout_probes_measured"] is False
    assert (tmp_path / "predictions-manifest.json").is_file()
    with pytest.raises(ValueError, match="no resume"):
        worker.predict_all(tmp_path)
