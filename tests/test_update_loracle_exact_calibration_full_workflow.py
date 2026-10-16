"""CPU integration using tiny synthetic models; no real target or interpreter outputs."""

import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_exact_calibration as exact
from gradient_interp import update_loracle_exact_calibration_heldout as worker
from gradient_interp import update_loracle_exact_calibration_heldout_analysis as audit
from gradient_interp import update_loracle_exact_calibration_inference as inference
from gradient_interp import update_loracle_factorized as factorized
from gradient_interp import update_loracle_factorized_analysis as factorized_analysis
from gradient_interp.loracle_trajectory_toy import read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_confirmation_data import fresh_draw
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE, build_data
from gradient_interp.update_loracle_inference import COMPONENT_INTERFACE, interface_contract


class TinyTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("frozen", torch.nn.Parameter(torch.ones(1), requires_grad=False))
        self.native = {"m": (torch.ones(2, 3), torch.zeros(3, 2))}


def restore(model, snapshot):
    model.native = copy.deepcopy(snapshot)


def synthetic_scores(model, n):
    # Nonlinear native-weight response; independent of any target/task label.
    a, b = model.native["m"]
    z = float((b.double() @ a.double()).sum())
    return np.full((n, 32), 2.0) + np.tile(np.linspace(-z * z, z * z, 32), (n, 1))


def native_encoding(before, after, upstream):
    a0, b0 = before["m"]
    a, b = after["m"]
    tokens = 2 * (b @ a - b0 @ a0)
    return tokens, exact.physical_norm(after), {"synthetic_encoding": True}


def install_synthetic_models(monkeypatch, selected, training):
    def load():
        return TinyTarget(), None

    def calibration(model, tokenizer, calibration):
        assert len(calibration) == 8
        return synthetic_scores(model, 8)

    def one_step(model, tokenizer, batch, rate):
        assert len(batch) == 4 and rate == 0.0002
        increment = (sum(ord(c) for row in batch for c in row["answer"]) % 10 + 1) / 20
        model.native["m"][1].add_(increment)
        return {"optimizer_steps": 1, "learning_rate": rate}

    def snapshot(model):
        return copy.deepcopy(model.native)

    for module in (worker, inference, exact):
        if hasattr(module, "load_target"):
            monkeypatch.setattr(module, "load_target", load)
        if hasattr(module, "restore_modules"):
            monkeypatch.setattr(module, "restore_modules", restore)
        if hasattr(module, "snapshot_modules"):
            monkeypatch.setattr(module, "snapshot_modules", snapshot)
    for module in (exact, inference, audit):
        monkeypatch.setattr(module, "calibration_scores", calibration)
    monkeypatch.setattr(audit, "restore_modules", restore)
    monkeypatch.setattr(
        exact,
        "losses",
        lambda model, tokenizer, rows: torch.tensor(synthetic_scores(model, len(rows))[:, 0]),
    )
    for module in (worker, inference, audit):
        monkeypatch.setattr(module, "verified_selection", lambda _: selected)
    monkeypatch.setattr(inference, "source_keys", lambda: ["m"])
    monkeypatch.setattr(worker, "one_step", one_step)
    monkeypatch.setattr(
        worker, "measure", lambda model, tokenizer, probes: synthetic_scores(model, len(probes))
    )
    for module in (worker, audit):
        monkeypatch.setattr(module, "upstream_module", lambda: None)
        monkeypatch.setattr(module, "tokenize_update", native_encoding)
    monkeypatch.setattr(worker, "verified_cases", lambda *_: training)
    monkeypatch.setattr(factorized_analysis, "cases_with_effects", lambda *_: (training, {}))
    monkeypatch.setattr(worker, "load_interpreter", lambda: SimpleNamespace(model=TinyTarget()))

    def forward(handle, tokens, view):
        assert tokens.shape == (3, 3)
        logits = np.linspace(1, -1, 33)
        options = list(view["options"])
        return {
            "choice_logits": logits[: len(options)].tolist(),
            "full_33_choice_logits": logits.tolist(),
            "choice_id": options[0],
            "prediction": view["options"][options[0]],
            "input_tokens": 4500,
            "hook_calls": 1,
            "loss": None,
        }

    monkeypatch.setattr(factorized, "forward", forward)

    def simple(train, views):
        assert train == training
        return {
            "feature_count": 35841,
            "majority": LABELS[0],
            "development": [
                {
                    "id": row["id"],
                    "majority": LABELS[0],
                    "ridge": LABELS[1],
                    "shuffled_ridge": LABELS[2],
                }
                for row in views
            ],
        }

    monkeypatch.setattr(worker, "simple_references", simple)

    def small_snapshot(value):
        assert set(value) == {"m"}
        assert value["m"][0].shape == (2, 3) and value["m"][1].shape == (3, 2)
        assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in value["m"])

    monkeypatch.setattr(audit, "check_snapshot", small_snapshot)
    monkeypatch.setattr(audit, "alive", lambda pid: False)
    # Released-plan pinning is tested separately. This fixture has no real model files.
    monkeypatch.setattr(audit, "check_released_inputs", lambda plan: None)


def prepare_fixture(root, monkeypatch):
    root.mkdir()
    old, collection = root.parent / "development", root.parent / "collection"
    old.mkdir()
    collection.mkdir()
    original = (
        build_data()
    )  # Structure only; no target model or actual test outcomes are collected.
    atomic_json(collection / "data.json", original)
    baseline = np.full((8, 32), 2.0).tolist()
    atomic_json(old / "baselines.json", {"scores": {str(i): baseline for i in range(8)}})
    atomic_json(
        old / "integrity.json",
        {"passed": True, "files_sha256": {"baselines.json": file_hash(old / "baselines.json")}},
    )
    selected = {
        "frozen_unix": time.time(),
        "development_root": str(old),
        "released_interface": interface_contract(COMPONENT_INTERFACE),
        "base_files_sha256": {},
        "calibration": [
            {"prompt": f"synthetic calibration {i}", "values": ["az", "ba", "c", "dddd"]}
            for i in range(8)
        ],
    }
    atomic_json(root / "selection.json", selected)
    data = {
        "schema": "single-update-heldout-data-v1",
        "created_unix": time.time(),
        "selection_sha256": file_hash(root / "selection.json"),
        "original_data_sha256": file_hash(collection / "data.json"),
        "reserved": {
            "cases": [c for c in original["cases"] if c["split"] == "reserved"],
            "vocabulary": original["vocabularies"]["reserved"],
        },
        "confirmation": fresh_draw(original),
    }
    atomic_json(root / "data.json", data)
    atomic_json(root / "analysis-plan.json", {"synthetic_only": True})
    source = "src/gradient_interp/update_loracle_exact_calibration_heldout.py"
    manifests = (
        "preflight.json",
        "updates-manifest.json",
        "predictions-manifest.json",
        "manifest.json",
    )
    original_file = str(collection / "data.json")
    plan = {
        "created_unix": time.time(),
        "scope": "heldout_confirmation",
        "max_hours": 12,
        "recipe": worker.RECIPE,
        "source_commit": "synthetic-models-only",
        "learning_rate": 0.0002,
        "source_files_sha256": {source: file_hash(source)},
        "input_files_sha256": {original_file: file_hash(original_file)},
        "released_files_sha256": {original_file: file_hash(original_file)},
        "question_interface": selected["released_interface"],
        "selection_sha256": file_hash(root / "selection.json"),
        "data_sha256": file_hash(root / "data.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "original_data_path": original_file,
        "collection_root": str(collection),
        "jobs": [
            {
                "id": stage,
                "module": "gradient_interp.update_loracle_exact_calibration_heldout",
                "arguments": [stage, "--root", str(root.resolve())],
                "manifest": str((root / manifest).resolve()),
            }
            for stage, manifest in zip(("preflight", "updates", "predict", "measure"), manifests)
        ],
    }
    atomic_json(root / "plan.json", plan)
    started = time.time()
    controller = {
        "scope": plan["scope"],
        "status": "running",
        "controller_pid": 100,
        "worker_pid": 101,
        "started_unix": started,
        "deadline_unix": started + 43200,
        "plan_sha256": file_hash(root / "plan.json"),
        "attempts": [],
    }
    atomic_json(root / "controller.json", controller)
    training = [
        {
            "id": f"train-{i}",
            "label": label,
            "tokens": "unused synthetic",
            "physical_update_norm": 0.0 if label == NO_CHANGE else 1.0,
        }
        for i, label in enumerate(LABELS)
    ]
    install_synthetic_models(monkeypatch, selected, training)
    stage_start = time.time()
    atomic_json(
        root / "target-preflight.json",
        {"passed": True, "bit_exact_adapter_and_probe_restore": True, "finished_unix": time.time()},
    )
    functional = {
        key: True
        for key in (
            "passed",
            "training_only",
            "synthetic_only_no_target_updates",
            "all_256_scores_finite",
            "bit_exact_adapter_and_full_score_restore",
            "frozen_base_unchanged",
        )
    }
    functional.update(
        modules=280,
        trainable_tensors=560,
        parameters=64225280,
        physical_norm_scale_relative_error=0.0,
        finished_unix=time.time(),
    )
    atomic_json(root / "functional-preflight.json", functional)
    atomic_json(
        root / "preflight.json",
        {
            "passed": True,
            "training_only_inputs": True,
            "bit_exact_released_score_repeat": True,
            "selection_sha256": plan["selection_sha256"],
            "question_interface": plan["question_interface"],
            "questions_checked": 4,
            "target_preflight_sha256": file_hash(root / "target-preflight.json"),
            "functional_preflight_sha256": file_hash(root / "functional-preflight.json"),
            "finished_unix": time.time(),
        },
    )
    controller["attempts"].append(
        {
            "job": "preflight",
            "pid": 101,
            "returncode": 0,
            "started_unix": stage_start,
            "finished_unix": time.time(),
        }
    )
    return controller


def run_fixture(root, monkeypatch):
    controller = prepare_fixture(root, monkeypatch)
    for index, (name, function) in enumerate(
        (
            ("updates", worker.collect_updates),
            ("predict", worker.predict_all),
            ("measure", worker.measure_all),
        )
    ):
        start = time.time()
        function(root)
        controller["attempts"].append(
            {
                "job": name,
                "pid": 102 + index,
                "returncode": 0,
                "started_unix": start,
                "finished_unix": time.time(),
            }
        )
    controller.update(status="complete", worker_pid=None)
    atomic_json(root / "controller.json", controller)


def test_worker_outputs_pass_full_audit_then_all_results_analyze(tmp_path, monkeypatch):
    root = tmp_path / "run"
    run_fixture(root, monkeypatch)
    audit.audit(root)
    integrity = read(root / "integrity.json")
    assert integrity["passed"] and integrity["cases"] == 272
    assert integrity["exact_calibration_conditions"] == 544
    assert len(integrity["fixed_scoring_rechecks"]) == 4
    assert all(row["passed"] for row in integrity["fixed_scoring_rechecks"])
    assert len(read(root / "predictions-lock.json")["files_sha256"]) == 556
    assert (root / "source_snapshot.tar").is_file()
    output = tmp_path / "report"
    audit.analyze(root, output)
    summary = read(output / "summary.json")
    assert set(summary["splits"]) == {"reserved", "confirmation"}
    assert all(s["n_actual_updates"] == 128 for s in summary["splits"].values())
    assert len(read(output / "fixed-examples.json")["reserved"]) == 8
    assert len(read(output / "all-cases.json")["confirmation"]) == 136
    assert (output / "heldout-profiles.png").is_file()
    with pytest.raises(ValueError, match="Preserve the completed audit"):
        audit.audit(root)
