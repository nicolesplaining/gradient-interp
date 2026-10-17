"""Future confirmation execution tested only with fake CPU models and tiny tensors."""

import ast
import copy
import inspect
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_contextual as context
from gradient_interp import update_loracle_contextual_confirmation_data as data_module
from gradient_interp import update_loracle_contextual_evaluation_run as controller
from gradient_interp import update_loracle_contextual_heldout as worker
from gradient_interp import update_loracle_contextual_heldout_analysis as analysis
from gradient_interp import update_loracle_heldout as previous
from gradient_interp import update_loracle_train as training
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_confirmation_data import fresh_draw
from gradient_interp.update_loracle_contextual_inference import interface_contract
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE, build_data


def test_old_or_failed_selection_cannot_generate_real_confirmation_data(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    atomic_json(selection, {"schema": "single-update-interpreter-selection-v2"})
    monkeypatch.setattr(data_module, "fresh_draw", lambda _: pytest.fail("Unlocked test data"))
    with pytest.raises(ValueError, match="schema"):
        data_module.freeze(selection, tmp_path / "data.json")
    assert not (tmp_path / "data.json").exists()


def test_original_scientific_collection_measurement_and_data_rules_unchanged():
    for name in ("checked_data", "collect_updates", "measure_all"):
        assert ast.dump(ast.parse(inspect.getsource(getattr(worker, name)))) == ast.dump(
            ast.parse(inspect.getsource(getattr(previous, name)))
        )


def test_prediction_view_allows_only_prompt_text_and_preserves_recipient_context(tmp_path):
    (tmp_path / "updates").mkdir()
    (tmp_path / "tokens").mkdir()
    cases, inventory = [], []
    for i in range(2):
        token = tmp_path / "tokens" / f"{i}.pt"
        token.write_bytes(f"opaque-{i}".encode())
        path = tmp_path / "updates" / f"{i}.json"
        atomic_json(
            path,
            {
                "id": str(i),
                "heldout_probes_measured": False,
                "tokens_sha256": file_hash(token),
                "physical_update_norm": 1.0,
            },
        )
        cases.append(
            {
                "id": str(i),
                "control": False,
                "batch": "SECRET",
                "source_pattern": "SECRET",
                "effects": "SECRET",
                "options": dict(zip(map(str, range(33)), LABELS)),
                "probes": [
                    {"prompt": f"Query {i}-{j}", "values": "SECRET", "answer": "SECRET"}
                    for j in range(4)
                ],
            }
        )
        inventory.append({"id": str(i), "sha256": file_hash(path)})
    views = worker.prediction_views(tmp_path, cases, inventory)
    assert "SECRET" not in str(views)
    shuffled = worker.shuffled_directions(views)
    assert context.frozen_question_inputs(views) == context.frozen_question_inputs(shuffled)
    assert shuffled[0]["tokens"] == views[1]["tokens"]
    assert [p["prompt"] for p in views[0]["probes"]] == [f"Query 0-{j}" for j in range(4)]


@pytest.mark.parametrize(
    "defect", ["absent", "prediction", "lock", "methods", "context", "lengths"]
)
def test_measurement_refuses_bad_prediction_locks_before_loading_a_model(
    tmp_path, monkeypatch, defect
):
    plan = {"selection_sha256": "frozen", "question_interface": interface_contract()}
    monkeypatch.setattr(worker, "verify_inputs", lambda _: (plan, {}, []))
    monkeypatch.setattr(worker, "load_target", lambda: pytest.fail("Model loaded before lock"))
    if defect != "absent":
        (tmp_path / "predictions").mkdir()
        for split in worker.SPLITS:
            for method in (
                "released",
                "released-shuffled",
                "adapted",
                "adapted-shuffled",
                "simple",
                "shuffle",
            ):
                atomic_json(tmp_path / "predictions" / f"{split}-{method}.json", [])
        atomic_json(tmp_path / "updates-manifest.json", {})
        atomic_json(tmp_path / "interpreter-inputs.json", [])
        atomic_json(tmp_path / "actual-input-lengths.json", {})
        lock = {
            "status": "complete",
            "cases": 272,
            "selection_sha256": "frozen",
            "question_interface": interface_contract(),
            "heldout_probes_measured": False,
            "finished_unix": time.time(),
            "updates_manifest_sha256": file_hash(tmp_path / "updates-manifest.json"),
            "question_inputs_sha256": file_hash(tmp_path / "interpreter-inputs.json"),
            "actual_input_lengths_sha256": file_hash(tmp_path / "actual-input-lengths.json"),
            "files_sha256": {
                str(p.relative_to(tmp_path)): file_hash(p)
                for p in (tmp_path / "predictions").glob("*.json")
            },
        }
        if defect == "methods":
            del lock["files_sha256"]["predictions/confirmation-adapted.json"]
        atomic_json(tmp_path / "predictions-lock.json", lock)
        atomic_json(
            tmp_path / "predictions-manifest.json",
            {"status": "complete", "lock_sha256": file_hash(tmp_path / "predictions-lock.json")},
        )
        changed = {
            "prediction": "predictions/reserved-adapted.json",
            "lock": "predictions-lock.json",
            "context": "interpreter-inputs.json",
            "lengths": "actual-input-lengths.json",
        }.get(defect)
        if changed:
            (tmp_path / changed).write_text('["changed"]')
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.measure_all(tmp_path)
    assert not (tmp_path / "measurements").exists()


def test_full_analysis_does_not_open_live_outputs(tmp_path, monkeypatch):
    def read(path):
        assert path.name == "controller.json"
        return {"status": "running"}

    monkeypatch.setattr(analysis, "read", read)
    with pytest.raises(ValueError, match="Complete dead"):
        analysis.audit(tmp_path)
    with pytest.raises(ValueError, match="Complete dead"):
        analysis.analyze(tmp_path, tmp_path / "report")


def test_controller_rejects_old_contextless_worker(tmp_path):
    manifests = [
        "preflight.json",
        "updates-manifest.json",
        "predictions-manifest.json",
        "manifest.json",
    ]
    plan = {
        "scope": "heldout_confirmation",
        "max_hours": 12,
        "jobs": [
            {
                "id": stage,
                "module": "gradient_interp.update_loracle_contextual_heldout",
                "arguments": [stage, "--root", str(tmp_path)],
                "manifest": str(tmp_path / name),
            }
            for stage, name in zip(controller.STAGES, manifests, strict=True)
        ],
    }
    controller.validate_plan(plan, tmp_path)
    plan["jobs"][2]["module"] = "gradient_interp.update_loracle_heldout"
    with pytest.raises(ValueError, match="Unexpected"):
        controller.validate_plan(plan, tmp_path)


def test_all_272_fake_cases_complete_predict_before_measure_then_audit_and_analyze(
    tmp_path, monkeypatch
):
    """Full stage grid with fake scores; no real reserved/fresh model or weights are used."""
    monkeypatch.chdir(tmp_path)
    root, collection = tmp_path / "evaluation", tmp_path / "collection"
    root.mkdir()
    (collection / "cases").mkdir(parents=True)
    (tmp_path / "synthetic-source.py").write_text("# CPU fixture only\n")
    original = build_data()
    atomic_json(tmp_path / "original.json", original)
    selection = {
        "frozen_unix": time.time(),
        "adapter_directory": str(tmp_path / "selected"),
        "question_interface": interface_contract(),
    }
    atomic_json(root / "selection.json", selection)
    data = {
        "schema": "single-update-heldout-data-v1",
        "created_unix": time.time(),
        "selection_sha256": file_hash(root / "selection.json"),
        "original_data_sha256": file_hash(tmp_path / "original.json"),
        "reserved": {
            "cases": [c for c in original["cases"] if c["split"] == "reserved"],
            "vocabulary": original["vocabularies"]["reserved"],
        },
        "confirmation": fresh_draw(original),
    }
    atomic_json(root / "data.json", data)
    cases = worker.checked_data(data, original)
    train = [c for c in original["cases"] if c["split"] == "train"]
    labels = [v for v in LABELS if v != NO_CHANGE]
    for i, case in enumerate(train):
        label = NO_CHANGE if case["control"] else labels[i % 32]
        case["label"] = label
        case["physical_update_norm"] = 0.0 if case["control"] else 1.0
        case["tokens"] = "unused"
        atomic_json(
            collection / "cases" / f"{case['id']}.json",
            {"effect": {"label": label}, "physical_update_norm": case["physical_update_norm"]},
        )
    atomic_json(collection / "data.json", {"cases": train})
    _, balance = context.training_instances(train)
    majorities = context.training_majorities(train, balance)
    atomic_json(root / "analysis-plan.json", {"synthetic_test": True})
    manifests = [
        "preflight.json",
        "updates-manifest.json",
        "predictions-manifest.json",
        "manifest.json",
    ]
    plan = {
        "scope": "heldout_confirmation",
        "max_hours": 12,
        "source_commit": "synthetic",
        "created_unix": time.time(),
        "learning_rate": 0.0002,
        "source_files_sha256": {"synthetic-source.py": file_hash("synthetic-source.py")},
        "input_files_sha256": {},
        "data_sha256": file_hash(root / "data.json"),
        "selection_sha256": file_hash(root / "selection.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "original_data_path": str(tmp_path / "original.json"),
        "collection_root": str(collection),
        "question_interface": interface_contract(),
        "jobs": [
            {
                "id": stage,
                "module": "gradient_interp.update_loracle_contextual_heldout",
                "arguments": [stage, "--root", str(root)],
                "manifest": str(root / name),
            }
            for stage, name in zip(controller.STAGES, manifests, strict=True)
        ],
    }
    atomic_json(root / "plan.json", plan)
    atomic_json(
        root / "preparation.json",
        {"synthetic_preflight_input_tokens": [5000] * 4, "maximum_input_tokens": 5000},
    )
    started = time.time()
    state = {
        "scope": "heldout_confirmation",
        "started_unix": started,
        "deadline_unix": started + 43200,
        "status": "running",
        "controller_pid": -1,
        "worker_pid": None,
        "plan_sha256": file_hash(root / "plan.json"),
        "attempts": [],
    }
    atomic_json(root / "controller.json", state)
    monkeypatch.setattr(worker, "verify_inputs", lambda _: (plan, selection, cases))
    monkeypatch.setattr(worker, "guard", lambda _: plan)
    monkeypatch.setattr(analysis, "verified_selection", lambda _: selection)
    native = {"layer": (torch.ones(2, 2), torch.zeros(2, 2))}

    def initialize(value, seed):
        return {"layer": (torch.full((2, 2), float(seed % 17)), torch.zeros(2, 2))}

    def restore(model, value):
        native.clear()
        native.update(copy.deepcopy(value))

    def step(model, tokenizer, batch, rate):
        assert len(batch) == 4 and rate == 0.0002
        native["layer"][1].add_(1)
        return {"optimizer_steps": 1, "learning_rate": rate}

    def encode(before, after, upstream):
        change = after["layer"][1] - before["layer"][1]
        return change.clone(), float(change.norm()), []

    def measure(model, tokenizer, probes):
        assert (root / "predictions-lock.json").is_file(), "Measured probes before lock"
        values = np.full((len(probes), 32), 3.0)
        values[:, 0] -= float(native["layer"][1][0, 0]) * 0.2
        return values

    monkeypatch.setattr(worker, "load_target", lambda: (None, None))
    monkeypatch.setattr(worker, "snapshot_modules", lambda _: copy.deepcopy(native))
    monkeypatch.setattr(worker, "initialize_like", initialize)
    monkeypatch.setattr(analysis, "initialize_like", initialize)
    monkeypatch.setattr(worker, "restore_modules", restore)
    monkeypatch.setattr(worker, "one_step", step)
    monkeypatch.setattr(worker, "tokenize_update", encode)
    monkeypatch.setattr(analysis, "tokenize_update", encode)
    monkeypatch.setattr(worker, "upstream_module", lambda: None)
    monkeypatch.setattr(analysis, "upstream_module", lambda: None)
    monkeypatch.setattr(
        analysis, "check_snapshot", lambda v: None
    )  # Native shapes tested elsewhere.
    monkeypatch.setattr(worker, "measure", measure)

    def target_preflight(model, tokenizer, path):
        atomic_json(
            path,
            {
                "passed": True,
                "finished_unix": time.time(),
                "bit_exact_adapter_and_probe_restore": True,
            },
        )

    monkeypatch.setattr(worker, "training_preflight", target_preflight)
    monkeypatch.setattr(worker.torch.cuda, "max_memory_allocated", lambda: 0)

    class Interpreter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tag = 0

    monkeypatch.setattr(worker, "load_interpreter", lambda: SimpleNamespace(model=Interpreter()))

    def replace(model, directory, **kwargs):
        model.tag = 1 if str(directory) == selection["adapter_directory"] else 0
        return {"bit_exact": True}

    monkeypatch.setattr(worker, "replace_adapter_exact", replace)
    calls = []

    def choice_forward(handle, tokens, text, target=None, loss_scale=1.0):
        assert "Evaluation prompts as a JSON list:" in text
        assert target is None
        if (root / "updates-manifest.json").exists():
            assert not (root / "measurements").exists()
            assert (root / "actual-input-lengths.json").exists()
        scores = [0.0] * 33
        scores[handle.model.tag] = 1.0
        calls.append(text)
        return {"choice_logits": scores, "loss": None, "input_tokens": 5000, "hook_calls": 1}

    monkeypatch.setattr(context, "choice_forward", choice_forward)
    monkeypatch.setattr(training, "prepared_input", lambda h, text: (torch.zeros(1, 5000), None))
    monkeypatch.setattr(worker, "verified_cases", lambda p, split: train)

    def simple_references(training_cases, inputs):
        return {
            "feature_count": 35841,
            "majority": majorities["flat"],
            "development": [
                {
                    "id": row["id"],
                    "majority": majorities["flat"],
                    "ridge": NO_CHANGE,
                    "shuffled_ridge": NO_CHANGE,
                }
                for row in inputs
            ],
        }

    monkeypatch.setattr(worker, "simple_references", simple_references)
    for j, (stage, function) in enumerate(
        zip(
            controller.STAGES,
            [worker.preflight, worker.collect_updates, worker.predict_all, worker.measure_all],
            strict=True,
        )
    ):
        begin = time.time()
        function(root)
        state["attempts"].append(
            {
                "job": stage,
                "pid": -j - 2,
                "returncode": 0,
                "started_unix": begin,
                "finished_unix": time.time(),
            }
        )
        atomic_json(root / "controller.json", state)
    state["status"] = "complete"
    atomic_json(root / "controller.json", state)
    assert len(calls) == 16 + 272 * 4 * 4
    analysis.audit(root)
    integrity = analysis.read(root / "integrity.json")
    assert integrity["passed"] and integrity["predictions_locked_before_measurement"]
    assert integrity["cases"] == 272
    assert "interpreter-inputs.json" in integrity["files_sha256"]
    analysis.analyze(root, tmp_path / "report")
    summary = analysis.read(tmp_path / "report/summary.json")
    assert summary["question_interface"]["name"] == worker.COMPONENT_INTERFACE
    assert set(summary["splits"]) == {"reserved", "confirmation"}
    assert not summary["both_grids_pass"]
    all_cases = analysis.read(tmp_path / "report/all-cases.json")
    assert all(len(rows) == 136 for rows in all_cases.values())
    assert (tmp_path / "report/heldout-accuracy.png").is_file()
    assert (tmp_path / "report/report.html").is_file()
    with pytest.raises(ValueError, match="no automatic resume"):
        worker.collect_updates(root)
