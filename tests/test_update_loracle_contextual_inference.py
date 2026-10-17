"""Synthetic selection evidence and context-only inference; no real held-out collection."""

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_contextual as worker
from gradient_interp import update_loracle_contextual_inference as inference
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE


def prediction(case, label):
    answers = worker.target_components(label)
    for head, labels in worker.HEAD_LABELS.items():
        answers.setdefault(head, labels[0])
    heads = {}
    for view in worker.question_views(case):
        symbol = next(k for k, v in view["options"].items() if v == answers[view["head"]])
        logits = np.zeros(33)
        logits[worker.OPTION_SYMBOLS.index(symbol)] = 3
        heads[view["head"]] = {
            "prediction": answers[view["head"]],
            "choice_id": symbol,
            "choice_logits": logits[: len(view["options"])].tolist(),
            "full_33_choice_logits": logits.tolist(),
            "input_tokens": 5000,
            "hook_calls": 1,
            "loss": None,
            "question_sha256": hashlib.sha256(view["text"].encode()).hexdigest(),
        }
    return {"id": case["id"], "prediction": label, "heads": heads}


def fixture(tmp_path, counts=(16, 48, 48), baseline=16, shuffled=8, wrong_zeros=0):
    root, collection, analysis = (
        tmp_path / name for name in ("training", "collection", "analysis")
    )
    root.mkdir()
    (collection / "cases").mkdir(parents=True)
    analysis.mkdir()
    labels = [v for v in LABELS if v != NO_CHANGE]
    data = []
    for split, actual, zeros in (("train", 256, 16), ("development", 64, 8)):
        for i in range(actual + zeros):
            case = {
                "id": f"{split}-{i}",
                "split": split,
                "control": i >= actual,
                "source_pattern": str(i // 8),
                "initialization_seed": (i % 8) // 2,
                "probes": [{"prompt": f"Synthetic query {split}-{i}-{j}"} for j in range(4)],
            }
            data.append(case)
            atomic_json(
                collection / "cases" / f"{case['id']}.json",
                {
                    "physical_update_norm": 0.0 if case["control"] else 1.0,
                    "effect": {
                        "label": NO_CHANGE if case["control"] else labels[i % 32],
                        "below_effect_floor": case["control"],
                    },
                },
            )
    atomic_json(collection / "data.json", {"cases": data})
    atomic_json(
        collection / "integrity.json",
        {
            "passed": True,
            "cases": 344,
            "files_sha256": {
                str(p.relative_to(collection)): file_hash(p) for p in collection.rglob("*.json")
            },
        },
    )
    train, _ = inference.cases_with_effects(collection, "train")
    dev, _ = inference.cases_with_effects(collection, "development")

    def predictions(correct, bad_zeros=0):
        return [
            prediction(
                c,
                NO_CHANGE
                if i >= 64 + bad_zeros
                else c["label"]
                if i < correct
                else labels[(i + 1) % 32],
            )
            for i, c in enumerate(dev)
        ]

    atomic_json(root / "released-development.json", predictions(baseline))
    atomic_json(root / "released-shuffled-development.json", predictions(8))
    for epoch, correct in enumerate(counts, 1):
        path = root / "epochs" / str(epoch)
        path.mkdir(parents=True)
        atomic_json(path / "development.json", predictions(correct, wrong_zeros))
        atomic_json(path / "shuffled-development.json", predictions(shuffled))
        atomic_json(
            path / "adapter_config.json",
            {k: inference.ARCHITECTURE[k] for k in ("r", "lora_alpha", "use_rslora")},
        )
        (path / "adapter_model.safetensors").write_bytes(f"synthetic-adapter{epoch}".encode())
    _, balance = worker.training_instances(train)
    atomic_json(root / "training-majorities.json", worker.training_majorities(train, balance))
    atomic_json(
        root / "simple-references.json",
        {
            "development": [
                {"id": c["id"], "ridge": NO_CHANGE, "shuffled_ridge": NO_CHANGE} for c in dev
            ]
        },
    )
    atomic_json(
        root / "controller.json",
        {
            "status": "complete",
            "controller_pid": -1,
            "worker_pid": None,
            "attempts": [{"pid": -2, "returncode": 0}],
        },
    )
    sources = dict(inference.interface_contract()["inference_source_sha256"])
    executed = Path(inference.__file__).with_name("update_loracle_contextual_analysis.py")
    sources["src/gradient_interp/update_loracle_contextual_analysis.py"] = file_hash(executed)
    atomic_json(
        root / "plan.json", {"collection_root": str(collection), "source_files_sha256": sources}
    )
    (analysis / "executed-analysis.py").write_bytes(executed.read_bytes())
    integrity = {
        "passed": True,
        "recipe": "contextual_factorized_v7",
        "optimizer_steps": 780,
        "epochs": 3,
        "reserved_cases_accessed": 0,
        "source_commit": "synthetic",
        "collection_integrity_sha256": file_hash(collection / "integrity.json"),
        "files_sha256": {
            str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
        },
    }
    atomic_json(root / "integrity.json", integrity)
    atomic_json(
        analysis / "analysis-provenance.json",
        {
            "training_integrity_sha256": file_hash(root / "integrity.json"),
            "executed_source_sha256": file_hash(executed),
        },
    )
    epoch, metrics, gates = inference.recompute_development(root, collection)
    atomic_json(
        analysis / "summary.json",
        {
            "recipe": "contextual_factorized_v7",
            "selected_epoch": epoch,
            "metrics": metrics,
            "development_gates": gates,
            "all_development_gates_pass": all(gates.values()),
            "reserved_test_read": False,
        },
    )
    return root, collection, analysis, tmp_path / "selection.json"


def test_freeze_recomputes_all_epochs_and_matches_context_baseline(tmp_path):
    root, _, analysis, output = fixture(tmp_path)
    result = inference.freeze_selection(root, analysis, output)
    assert result["selected_epoch"] == 2
    assert result["question_interface"]["name"] == inference.INTERFACE
    assert result["question_interface"]["query_prompts_per_case"] == 4
    assert all(result["development_gates"].values())
    assert inference.verified_selection(output) == result
    with pytest.raises(ValueError, match="already frozen"):
        inference.freeze_selection(root, analysis, output)


@pytest.mark.parametrize(
    "settings,failed",
    [
        ({"counts": (13, 16, 16)}, "accuracy_70"),
        ({"baseline": 48}, "released_advantage_20pp"),
        ({"shuffled": 48}, "direction_dependence_10pp"),
        ({"wrong_zeros": 1}, "zero_accuracy_90"),
    ],
)
def test_failures_cannot_be_promoted_by_summary_flag(tmp_path, settings, failed):
    root, _, analysis, output = fixture(tmp_path, **settings)
    summary = inference.read(analysis / "summary.json")
    assert not summary["development_gates"][failed]
    summary["all_development_gates_pass"] = True
    atomic_json(analysis / "summary.json", summary)
    with pytest.raises(ValueError, match="criteria failed"):
        inference.freeze_selection(root, analysis, output)
    assert not output.exists()
    summary["development_gates"][failed] = True
    atomic_json(analysis / "summary.json", summary)
    with pytest.raises(ValueError, match="Summary differs"):
        inference.freeze_selection(root, analysis, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["running", "failed", "retried"])
def test_active_or_failed_training_is_rejected_before_output_access(
    tmp_path, monkeypatch, mutation
):
    controller = {
        "status": "complete",
        "controller_pid": -1,
        "worker_pid": None,
        "attempts": [{"pid": -2, "returncode": 0}],
    }
    if mutation == "running":
        controller["status"] = "running"
    elif mutation == "failed":
        controller["attempts"][0]["returncode"] = 1
    else:
        controller["attempts"] *= 2

    def read(path):
        assert path.name == "controller.json", "Scientific outputs read before completion"
        return controller

    monkeypatch.setattr(inference, "read", read)
    with pytest.raises(ValueError):
        inference.freeze_selection(tmp_path, tmp_path, tmp_path / "selection.json")


@pytest.mark.parametrize(
    "mutation", ["epoch", "metric", "question", "unused_epoch", "analysis_source"]
)
def test_selection_rejects_changed_evidence(tmp_path, mutation):
    root, _, analysis, output = fixture(tmp_path)
    if mutation in ("epoch", "metric"):
        p = analysis / "summary.json"
        value = inference.read(p)
        if mutation == "epoch":
            value["selected_epoch"] = 3
        else:
            value["metrics"]["released"]["actual_update_correct"] = 1
        atomic_json(p, value)
    elif mutation == "question":
        p = root / "released-development.json"
        value = inference.read(p)
        value[0]["heads"]["format"]["question_sha256"] = "old-contextless-question"
        atomic_json(p, value)
    elif mutation == "unused_epoch":
        (root / "epochs" / "1" / "development.json").unlink()
    else:
        (analysis / "executed-analysis.py").write_text("different analysis")
    with pytest.raises((ValueError, FileNotFoundError)):
        inference.freeze_selection(root, analysis, output)
    assert not output.exists()


def test_question_context_is_recomputed_even_if_summary_and_file_hash_are_forged(tmp_path):
    root, _, analysis, output = fixture(tmp_path)
    path = root / "released-development.json"
    rows = inference.read(path)
    rows[0]["heads"]["case"]["question_sha256"] = "wrong"
    atomic_json(path, rows)
    integrity = inference.read(root / "integrity.json")
    integrity["files_sha256"][str(path.relative_to(root))] = file_hash(path)
    atomic_json(root / "integrity.json", integrity)
    provenance = inference.read(analysis / "analysis-provenance.json")
    provenance["training_integrity_sha256"] = file_hash(root / "integrity.json")
    atomic_json(analysis / "analysis-provenance.json", provenance)
    with pytest.raises(ValueError, match="context hash"):
        inference.freeze_selection(root, analysis, output)


@pytest.mark.parametrize("mutation", ["interface", "gate", "adapter", "evidence"])
def test_runtime_rejects_tampered_selection(tmp_path, mutation):
    root, _, analysis, output = fixture(tmp_path)
    lock = inference.freeze_selection(root, analysis, output)
    if mutation == "interface":
        lock["question_interface"]["query_prompts_per_case"] = 0
    elif mutation == "gate":
        lock["development_gates"]["accuracy_70"] = False
    elif mutation == "adapter":
        (Path(lock["adapter_directory"]) / "adapter_model.safetensors").write_bytes(b"changed")
    else:
        (analysis / "summary.json").write_text("changed")
    atomic_json(output, lock)
    with pytest.raises(ValueError):
        inference.verified_selection(output)


def test_public_prediction_requires_query_text_and_uses_every_head(monkeypatch):
    handle = SimpleNamespace(question_interface=inference.INTERFACE)
    tokens = torch.zeros((4480, 5120))
    prompts = [f"Query {i}: rana, telo, sop, miranu" for i in range(4)]
    seen = []

    def forward(handle, tokens, text, target, loss_scale):
        assert target is None and loss_scale == 1.0
        assert all(prompt in text for prompt in prompts)
        seen.append(text)
        scores = [0.0] * 33
        scores[1], scores[31] = 1.0, 100.0
        return {"choice_logits": scores, "loss": None, "input_tokens": 5000, "hook_calls": 1}

    monkeypatch.setattr(worker, "choice_forward", forward)
    result = inference.predict(handle, tokens, 2.0, "synthetic-case", prompts)
    assert len(seen) == 4
    assert result["prediction"] in LABELS
    assert all(row["choice_id"] == "B" for row in result["heads"].values())
    assert result["scores_are_calibrated_probabilities"] is False
    for view in result["questions"]:
        assert view["probe_prompts"] == prompts
    with pytest.raises(ValueError, match="interface"):
        inference.predict(SimpleNamespace(), tokens, 2.0, "x", prompts)
    bad = copy.deepcopy(prompts)
    bad[0] = {"prompt": "query", "answer": "gold"}
    with pytest.raises(ValueError, match="query prompt"):
        inference.predict(handle, tokens, 2.0, "x", bad)
