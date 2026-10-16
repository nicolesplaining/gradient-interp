import copy

import pytest
import torch

from gradient_interp import update_loracle_heldout as worker
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
    atomic_json(root / "updates-manifest.json", {"fixture": True})
    files = {}
    for split in worker.SPLITS:
        for name in (
            "released",
            "released-shuffled",
            "adapted",
            "adapted-shuffled",
            "simple",
            "shuffle",
        ):
            path = root / "predictions" / f"{split}-{name}.json"
            atomic_json(path, [])
            files[str(path.relative_to(root))] = file_hash(path)
    atomic_json(
        root / "predictions-lock.json",
        {
            "status": "complete",
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
        worker, "verify_inputs", lambda root: ({"selection_sha256": "frozen"}, {}, [])
    )
    monkeypatch.setattr(
        worker, "load_target", lambda: pytest.fail("Probe model loaded before barrier")
    )
    if defect != "absent":
        prediction_lock(tmp_path)
    if defect == "prediction_changed":
        (tmp_path / "predictions/reserved-adapted.json").write_text('["changed"]')
    elif defect == "lock_changed":
        (tmp_path / "predictions-lock.json").write_text("{}")
    elif defect == "incomplete_methods":
        lock = worker.read(tmp_path / "predictions-lock.json")
        del lock["files_sha256"]["predictions/confirmation-adapted.json"]
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
    lock = worker.checked_prediction_lock(tmp_path, {"selection_sha256": "frozen"})
    assert len(lock["files_sha256"]) == 12


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
