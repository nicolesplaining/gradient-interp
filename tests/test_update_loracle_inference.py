import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gradient_interp import update_loracle_inference as inference
from gradient_interp.study import file_hash
from gradient_interp.update_loracle_data import LABELS
from gradient_interp.update_loracle_train import save_adapter


@pytest.fixture
def small_adapter(tmp_path):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=40,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            use_rslora=True,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.uniform_(-0.2, 0.2)
    path = tmp_path / "saved"
    save_adapter(model, path)
    saved = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    return model, path, saved


def test_actual_peft_roundtrip_changes_only_adapter_and_is_bit_exact(small_adapter):
    from peft import get_peft_model_state_dict

    model, directory, saved = small_adapter
    model.eval()
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected_logits = model(ids).logits.clone()
        for parameter in get_peft_model_state_dict(model).values():
            parameter.add_(0.03)
        assert not torch.equal(expected_logits, model(ids).logits)
    base_before = {n: p.clone() for n, p in model.named_parameters() if "lora_" not in n}
    result = inference.replace_adapter_exact(
        model,
        directory,
        len(saved),
        sum(t.numel() for t in saved.values()),
    )
    assert result["bit_exact"]
    assert all(torch.equal(v, saved[k]) for k, v in get_peft_model_state_dict(model).items())
    assert all(
        torch.equal(p, base_before[n]) for n, p in model.named_parameters() if n in base_before
    )
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, expected_logits, atol=0, rtol=0)


def test_loading_rejects_config_change_before_any_parameter_mutation(small_adapter):
    model, directory, saved = small_adapter
    path = directory / "adapter_config.json"
    config = json.loads(path.read_text())
    config["lora_alpha"] = 8
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="lora_alpha"):
        inference.replace_adapter_exact(
            model, directory, len(saved), sum(t.numel() for t in saved.values())
        )


def test_released_bfloat16_can_be_explicitly_upcast_without_rounding(small_adapter):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    model, directory, saved = small_adapter
    stored = {k: v.bfloat16() for k, v in saved.items()}
    save_file(stored, str(directory / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="Invalid saved"):
        inference.replace_adapter_exact(
            model, directory, len(stored), sum(t.numel() for t in stored.values())
        )
    result = inference.replace_adapter_exact(
        model,
        directory,
        len(stored),
        sum(t.numel() for t in stored.values()),
        source_dtype=torch.bfloat16,
    )
    assert result["storage_dtype"] == "torch.bfloat16"
    assert all(
        torch.equal(v, stored[k].float()) for k, v in get_peft_model_state_dict(model).items()
    )


@pytest.mark.parametrize("corruption", ["missing", "shape", "nan", "dtype"])
def test_loading_rejects_incomplete_or_invalid_tensors_before_copy(small_adapter, corruption):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    model, directory, saved = small_adapter
    original = {k: v.clone() for k, v in saved.items()}
    key = next(iter(saved))
    if corruption == "missing":
        del saved[key]
    elif corruption == "shape":
        saved[key] = saved[key].T.contiguous()
    elif corruption == "nan":
        saved[key].flatten()[0] = float("nan")
    else:
        saved[key] = saved[key].half()
    save_file(saved, str(directory / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="inventory|Invalid saved"):
        inference.replace_adapter_exact(
            model, directory, len(original), sum(t.numel() for t in original.values())
        )
    assert all(torch.equal(v, original[k]) for k, v in get_peft_model_state_dict(model).items())


def test_selection_rejects_running_controller_before_outcome_access(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    (root / "controller.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="complete and dead"):
        inference.freeze_selection(root, tmp_path / "absent", tmp_path / "selection.json")
    assert not (tmp_path / "selection.json").exists()


def test_selected_model_hashes_fail_after_any_checkpoint_change(tmp_path):
    directory = tmp_path / "adapter"
    directory.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (directory / name).write_bytes(b"fixture")
    lock = {
        "schema": "single-update-interpreter-selection-v1",
        "architecture": inference.ARCHITECTURE,
        "base_model": inference.BASE_PATH,
        "labels": list(LABELS),
        "reserved_test_accessed": False,
        "adapter_directory": str(directory),
        "adapter_files_sha256": {p.name: file_hash(p) for p in directory.iterdir()},
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(lock))
    assert inference.verified_selection(path) == lock
    (directory / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Selected adapter changed"):
        inference.verified_selection(path)


def test_prediction_passes_only_weight_norm_and_choices(monkeypatch):
    # A shape-only stand-in avoids allocating the real 92MB input for this interface test.
    tokens = SimpleNamespace(shape=(4480, 5120))
    monkeypatch.setattr(inference.torch, "isfinite", lambda x: torch.tensor(True))
    options = dict(zip(map(str, range(33)), reversed(LABELS)))
    observed = []

    def forward(handle, received, text):
        assert received is tokens
        observed.append(text)
        return {"choice": "7", "choice_id": "H", "choice_logits": list(range(33))}

    monkeypatch.setattr(inference, "choice_forward", forward)
    row = inference.predict(None, tokens, 1.25, options)
    assert row["prediction"] == options["7"]
    assert observed == [inference.interpreter_question(options, 1.25)]
    assert not row["scores_are_calibrated_probabilities"]
    with pytest.raises(ValueError, match="norm"):
        inference.predict(None, tokens, -1, options)


def completed_fixture(tmp_path):
    from gradient_interp.study import atomic_json
    from gradient_interp.update_loracle_analysis import metrics
    from gradient_interp.update_loracle_data import NO_CHANGE, build_data

    root, collection, analysis = (
        tmp_path / name for name in ("training", "collection", "analysis")
    )
    for path in (root, collection / "cases", analysis):
        path.mkdir(parents=True)
    cases = [c for c in build_data()["cases"] if c["split"] == "development"]
    labels = [NO_CHANGE if c["control"] else LABELS[i % 8] for i, c in enumerate(cases)]
    effects = {
        c["id"]: {"effect": {"label": label, "below_effect_floor": c["control"]}}
        for c, label in zip(cases, labels)
    }
    atomic_json(collection / "data.json", {"cases": cases})
    for name, effect in effects.items():
        atomic_json(collection / "cases" / f"{name}.json", effect)
    atomic_json(
        collection / "integrity.json",
        {
            "passed": True,
            "files_sha256": {
                str(p.relative_to(collection)): file_hash(p) for p in collection.rglob("*.json")
            },
        },
    )
    atomic_json(
        root / "controller.json",
        {
            "status": "complete",
            "controller_pid": None,
            "worker_pid": None,
            "attempts": [{"pid": None, "returncode": 0}],
        },
    )
    atomic_json(root / "plan.json", {"collection_root": str(collection)})
    all_metrics = {}
    for epoch in (1, 2, 3):
        directory = root / "epochs" / str(epoch)
        directory.mkdir(parents=True)
        predictions = [NO_CHANGE if epoch == 1 and i < 16 else p for i, p in enumerate(labels)]
        atomic_json(
            directory / "development.json",
            [{"id": c["id"], "prediction": p} for c, p in zip(cases, predictions)],
        )
        atomic_json(
            directory / "shuffled-development.json",
            [{"id": c["id"], "prediction": NO_CHANGE} for c in cases],
        )
        atomic_json(
            directory / "adapter_config.json",
            {k: inference.ARCHITECTURE[k] for k in ("r", "lora_alpha", "use_rslora")},
        )
        (directory / "adapter_model.safetensors").write_bytes(f"fixture epoch {epoch}".encode())
        all_metrics[f"epoch{epoch}"] = metrics(predictions, cases, effects)
    atomic_json(
        root / "released-development.json",
        [{"id": c["id"], "prediction": NO_CHANGE} for c in cases],
    )
    atomic_json(
        root / "simple-references.json",
        {"development": [{"id": c["id"], "majority": NO_CHANGE} for c in cases]},
    )
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "epochs": 3,
            "optimizer_steps": 204,
            "reserved_cases_accessed": 0,
            "collection_integrity_sha256": file_hash(collection / "integrity.json"),
            "source_commit": "synthetic",
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
            },
        },
    )
    (analysis / "executed-analysis.py").write_bytes(b"fixture")
    atomic_json(
        analysis / "analysis-provenance.json",
        {
            "training_integrity_sha256": file_hash(root / "integrity.json"),
            "executed_source_sha256": file_hash(analysis / "executed-analysis.py"),
        },
    )
    atomic_json(
        analysis / "summary.json",
        {
            "selected_epoch": 2,
            "reserved_test_read": False,
            "metrics": all_metrics,
            "all_development_gates_pass": True,
            "development_gates": dict.fromkeys(
                [
                    "accuracy_70",
                    "released_advantage_20pp",
                    "effect_coverage_80",
                    "zero_accuracy_90",
                    "majority_advantage_10pp",
                    "direction_dependence_10pp",
                ],
                True,
            ),
        },
    )
    return root, analysis


def test_lock_selects_earliest_best_epoch_and_cannot_be_overwritten(tmp_path):
    root, analysis = completed_fixture(tmp_path)
    output = tmp_path / "lock.json"
    lock = inference.freeze_selection(root, analysis, output)
    assert lock["selected_epoch"] == 2 and not lock["confirmation_complete"]
    assert Path(lock["adapter_directory"]).name == "2"
    assert inference.verified_selection(output) == lock
    with pytest.raises(ValueError, match="never overwrite"):
        inference.freeze_selection(root, analysis, output)


def test_selection_does_not_trust_changed_development_summary(tmp_path):
    from gradient_interp.study import atomic_json

    root, analysis = completed_fixture(tmp_path)
    summary = json.loads((analysis / "summary.json").read_text())
    summary["selected_epoch"] = 3
    atomic_json(analysis / "summary.json", summary)
    with pytest.raises(ValueError, match="fixed all-epoch"):
        inference.freeze_selection(root, analysis, tmp_path / "lock.json")


def test_changed_checkpoint_fails_before_selection(tmp_path):
    root, analysis = completed_fixture(tmp_path)
    (root / "epochs/2/adapter_model.safetensors").write_bytes(b"another model")
    with pytest.raises(ValueError, match="Audited training output changed"):
        inference.freeze_selection(root, analysis, tmp_path / "lock.json")


def component_fixture(tmp_path):
    from gradient_interp import update_loracle_factorized as component
    from gradient_interp.study import atomic_json

    root, analysis = completed_fixture(tmp_path)
    plan = json.loads((root / "plan.json").read_text())
    collection = Path(plan["collection_root"])
    cases = json.loads((collection / "data.json").read_text())["cases"]
    for case in cases:
        path = collection / "cases" / f"{case['id']}.json"
        effect = json.loads(path.read_text())
        effect["physical_update_norm"] = 0.0 if case["control"] else 1.0
        atomic_json(path, effect)
    files = [root / "released-development.json"] + list(
        (root / "epochs").glob("*/*development.json")
    )
    for path in files:
        rows = json.loads(path.read_text())
        for row in rows:
            targets = component.target_components(row["prediction"])
            heads = {}
            for h, labels in component.HEAD_LABELS.items():
                label = targets.get(h, labels[0])
                options = component.option_map(row["id"], h)
                choice = next(k for k, v in options.items() if v == label)
                index = component.OPTION_SYMBOLS.index(choice)
                full = [0.0] * 33
                full[index], full[-1] = 5.0, 100.0
                heads[h] = {
                    "choice_id": choice,
                    "prediction": label,
                    "choice_logits": full[: len(labels)],
                    "full_33_choice_logits": full,
                    "loss": None,
                    "hook_calls": 1,
                    "input_tokens": 5000,
                }
            row["heads"] = heads
        atomic_json(path, rows)
    plan["source_files_sha256"] = {
        "src/gradient_interp/update_loracle_factorized.py": file_hash(component.__file__)
    }
    atomic_json(root / "plan.json", plan)
    atomic_json(root / "training-majorities.json", {"flat": "no_change", "factorized": "no_change"})
    summary = json.loads((analysis / "summary.json").read_text())
    del summary["development_gates"]["majority_advantage_10pp"]
    summary["development_gates"].update(
        flat_majority_advantage_10pp=True, factorized_majority_advantage_10pp=True
    )
    atomic_json(analysis / "summary.json", summary)
    refresh_component_fixture(root, analysis)
    return root, analysis


def refresh_component_fixture(root, analysis):
    from gradient_interp.study import atomic_json

    collection = Path(json.loads((root / "plan.json").read_text())["collection_root"])
    old = json.loads((collection / "integrity.json").read_text())
    old["files_sha256"] = {
        str(p.relative_to(collection)): file_hash(p)
        for p in collection.rglob("*.json")
        if p.name != "integrity.json"
    }
    atomic_json(collection / "integrity.json", old)
    integrity = json.loads((root / "integrity.json").read_text())
    integrity.update(
        recipe="balanced_factorized_v2",
        optimizer_steps=780,
        collection_integrity_sha256=file_hash(collection / "integrity.json"),
        files_sha256={
            str(p.relative_to(root)): file_hash(p)
            for p in root.rglob("*")
            if p.is_file() and p.name != "integrity.json"
        },
    )
    atomic_json(root / "integrity.json", integrity)
    provenance = json.loads((analysis / "analysis-provenance.json").read_text())
    provenance["training_integrity_sha256"] = file_hash(root / "integrity.json")
    atomic_json(analysis / "analysis-provenance.json", provenance)


def test_component_selection_freezes_complete_question_contract(tmp_path):
    root, analysis = component_fixture(tmp_path)
    output = tmp_path / "lock.json"
    lock = inference.freeze_selection(root, analysis, output)
    assert lock["schema"] == "single-update-interpreter-selection-v2"
    assert lock["question_interface"] == inference.interface_contract(inference.COMPONENT_INTERFACE)
    assert lock["selected_epoch"] == 2
    assert inference.verified_selection(output) == lock
    lock["question_interface"]["option_order_seed"] += 1
    output.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="question interface changed"):
        inference.verified_selection(output)


def test_component_selection_cannot_use_old_flat_baseline_or_partial_success(tmp_path):
    root, analysis = component_fixture(tmp_path)
    # Matched released baseline also scores perfectly. A stale positive summary cannot unlock tests.
    (root / "released-development.json").write_bytes(
        (root / "epochs/2/development.json").read_bytes()
    )
    refresh_component_fixture(root, analysis)
    output = tmp_path / "lock.json"
    with pytest.raises(ValueError, match="Development criteria failed"):
        inference.freeze_selection(root, analysis, output)
    assert not output.exists()


def test_component_public_inference_uses_all_four_questions_and_no_label_input(monkeypatch):
    from gradient_interp import update_loracle_factorized as component

    tokens = SimpleNamespace(shape=(4480, 5120))
    handle = SimpleNamespace(question_interface=inference.COMPONENT_INTERFACE)
    monkeypatch.setattr(inference.torch, "isfinite", lambda _: torch.tensor(True))
    observed = []
    targets = component.target_components("numbered/upper/reverse")

    def forward(received_handle, received_tokens, row):
        assert received_handle is handle and received_tokens is tokens
        assert "target_label" not in row and "label" not in row
        observed.append(row["head"])
        return {"prediction": targets[row["head"]]}

    monkeypatch.setattr(component, "forward", forward)
    result = inference.predict_components(handle, tokens, 1.2, "new-update")
    assert result["prediction"] == "numbered/upper/reverse"
    assert observed == ["change", "format", "case", "order"]
    assert not result["scores_are_calibrated_probabilities"]
    with pytest.raises(ValueError, match="four-question"):
        inference.predict(handle, tokens, 1.2, {})
