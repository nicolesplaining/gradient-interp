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


def test_component_preflight_checks_all_four_question_scores(monkeypatch):
    from gradient_interp import update_loracle_factorized as component

    seen = []

    def forward(handle, tokens, row):
        seen.append(row["head"])
        return {"full_33_choice_logits": [float(len(seen))] * 33}

    monkeypatch.setattr(component, "forward", forward)
    values = worker.preflight_scores(None, "synthetic tokens", worker.COMPONENT_INTERFACE)
    assert seen == ["change", "format", "case", "order"] and len(values) == 4
    with pytest.raises(ValueError, match="Unknown"):
        worker.preflight_scores(None, None, "accidental_flat_question")


def test_component_scoring_dispatch_preserves_views_and_rejects_other_interface(monkeypatch):
    from gradient_interp import update_loracle_factorized as component

    cases = [{"id": "new", "tokens": "native", "physical_update_norm": 0.9}]
    monkeypatch.setattr(component, "score", lambda handle, actual: actual)
    monkeypatch.setattr(worker, "score", lambda *a: pytest.fail("Used flat questions"))
    assert worker.score_selected_interface(None, cases, worker.COMPONENT_INTERFACE) is cases
    with pytest.raises(ValueError, match="Unknown"):
        worker.score_selected_interface(None, cases, "unknown")


def test_component_prediction_lock_binds_question_contract(tmp_path):
    prediction_lock(tmp_path)
    plan = {
        "selection_sha256": "frozen",
        "question_interface": worker.interface_contract(worker.COMPONENT_INTERFACE),
    }
    lock = worker.read(tmp_path / "predictions-lock.json")
    lock["question_interface"] = worker.interface_contract(worker.FLAT_INTERFACE)
    atomic_json(tmp_path / "predictions-lock.json", lock)
    atomic_json(
        tmp_path / "predictions-manifest.json",
        {"status": "complete", "lock_sha256": file_hash(tmp_path / "predictions-lock.json")},
    )
    with pytest.raises(ValueError, match="different question interface"):
        worker.checked_prediction_lock(tmp_path, plan)


def test_component_prediction_stage_matches_questions_and_locks_both_majorities(
    tmp_path, monkeypatch
):
    from gradient_interp import update_loracle_factorized as component

    cases = [
        {"id": f"{split}-{i}", "split": split, "control": i == 2}
        for split in worker.SPLITS
        for i in range(3)
    ]
    views = [
        {
            "id": c["id"],
            "control": c["control"],
            "tokens": c["id"],
            "physical_update_norm": 0.0 if c["control"] else 1.0,
        }
        for c in cases
    ]
    selection = {
        "adapter_directory": "selected",
        "question_interface": worker.interface_contract(worker.COMPONENT_INTERFACE),
    }
    plan = {"selection_sha256": "frozen", "collection_root": "unused"}
    atomic_json(tmp_path / "plan.json", plan)
    atomic_json(
        tmp_path / "updates-manifest.json",
        {"status": "complete", "plan_sha256": file_hash(tmp_path / "plan.json"), "records": []},
    )
    monkeypatch.setattr(worker, "verify_inputs", lambda _: (plan, selection, cases))
    monkeypatch.setattr(worker, "checked_preflight", lambda *a: {})
    monkeypatch.setattr(worker, "prediction_views", lambda *a: views)
    monkeypatch.setattr(worker, "load_interpreter", lambda: None)
    model_state = ["released"]
    calls = []

    def score(handle, received, interface):
        assert interface == worker.COMPONENT_INTERFACE
        calls.append((model_state[0], copy.deepcopy(received)))
        return [{"id": c["id"], "prediction": "no_change"} for c in received]

    def replace(model, directory):
        model_state[0] = "adapted"
        return {"bit_exact": True}

    from types import SimpleNamespace

    monkeypatch.setattr(worker, "load_interpreter", lambda: SimpleNamespace(model=None))
    monkeypatch.setattr(worker, "score_selected_interface", score)
    monkeypatch.setattr(worker, "replace_adapter_exact", replace)
    labels = [
        "/".join((f, c, o))
        for f in component.HEAD_LABELS["format"]
        for c in component.HEAD_LABELS["case"]
        for o in component.HEAD_LABELS["order"]
    ]
    train = [
        {"id": str(i), "label": label, "tokens": "unused", "physical_update_norm": 1.0}
        for i, label in enumerate(labels + ["no_change"] * 16)
    ]
    monkeypatch.setattr(worker, "verified_cases", lambda *a: train)
    monkeypatch.setattr(
        worker,
        "simple_references",
        lambda train, subset: {
            "majority": "no_change",
            "development": [{"id": c["id"], "majority": "no_change"} for c in subset],
        },
    )
    monkeypatch.setattr(worker, "measure", lambda *a: pytest.fail("Early outcome scoring"))
    worker.predict_all(tmp_path)
    assert len(calls) == 8
    for i in range(4):
        assert calls[i][0] == "released" and calls[i + 4][0] == "adapted"
        assert calls[i][1] == calls[i + 4][1]
    lock = worker.read(tmp_path / "predictions-lock.json")
    assert lock["question_interface"]["name"] == worker.COMPONENT_INTERFACE
    assert not lock["heldout_probes_measured"]
    for split in worker.SPLITS:
        simple = worker.read(tmp_path / "predictions" / f"{split}-simple.json")
        assert simple["factorized_majority"] == "plain/lower/original"
        assert all(
            r["factorized_majority"] == "plain/lower/original" for r in simple["development"]
        )
    with pytest.raises(ValueError, match="no automatic resume"):
        worker.predict_all(tmp_path)
