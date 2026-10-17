"""Preparation contract with a fake tokenizer; never access real held-out outcomes or models."""

import importlib
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from gradient_interp import loracle
from gradient_interp import update_loracle_train as training
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_contextual_evaluation_run import validate_plan
from gradient_interp.update_loracle_contextual_inference import interface_contract

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/prepare_update_loracle_contextual_heldout.py"


@pytest.mark.parametrize("oversized", [False, True])
def test_prepare_freezes_all_1088_questions_and_refuses_uncovered_lengths(
    tmp_path, monkeypatch, oversized
):
    spec = importlib.util.spec_from_file_location("contextual_prepare_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    root, trained, collection = (tmp_path / p for p in ("evaluation", "training", "collection"))
    trained.mkdir()
    collection.mkdir()
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    (tmp_path / "synthetic-source.py").write_text("# no real model\n")
    dated = Path("results/update-loracle-program/provenance/confirmation-statistics-plan.json")
    dated.parent.mkdir(parents=True)
    shutil.copyfile(REPO / dated, dated)
    for p in (trained / "data.json", trained / "integrity.json", collection / "integrity.json"):
        atomic_json(p, {"synthetic": True})
    (trained / "source_snapshot.tar").write_bytes(b"opaque source fixture")
    atomic_json(
        trained / "plan.json", {"input_files_sha256": {}, "collection_root": str(collection)}
    )
    adapter = trained / "epochs" / "2"
    adapter.mkdir(parents=True)
    atomic_json(adapter / "adapter_config.json", {"r": 256})
    (adapter / "adapter_model.safetensors").write_bytes(b"opaque adapter fixture")
    selection = {
        "adapter_directory": str(adapter),
        "selected_epoch": 2,
        "question_interface": interface_contract(),
        "adapter_files_sha256": {p.name: file_hash(p) for p in adapter.iterdir()},
    }

    def freeze_selection(training_root, analysis, output):
        assert training_root == trained and not root.exists()
        output.parent.mkdir()
        atomic_json(output, selection)
        return selection

    def freeze_data(selection_path, output):
        assert selection_path.is_file()
        atomic_json(
            output,
            {
                split: {
                    "cases": [
                        {
                            "id": f"{split}-{i}",
                            "probes": [{"prompt": f"Query {i}-{j}"} for j in range(4)],
                        }
                        for i in range(136)
                    ]
                }
                for split in ("reserved", "confirmation")
            },
        )

    def commands(args, **kwargs):
        if args == ["git", "rev-parse", "HEAD"]:
            return "synthetic-commit\n"
        if args[:2] == ["nvidia-smi", "--query-compute-apps=pid"]:
            return ""
        if args[:2] == ["git", "status"]:
            return ""
        if args[:2] == ["git", "ls-files"]:
            return "synthetic-source.py\n"
        raise AssertionError(args)

    monkeypatch.setattr(module, "freeze_selection", freeze_selection)
    monkeypatch.setattr(module, "freeze_data", freeze_data)
    monkeypatch.setattr(module.subprocess, "check_output", commands)
    original_import = importlib.import_module

    def imports(name, *args, **kwargs):
        if name == "src.train_loracle.config":
            return SimpleNamespace(load_train_config=lambda _: object())
        if name == "src.train_loracle.dataset":
            return SimpleNamespace(build_placeholder_prefix_ids=object())
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", imports)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(loracle, "configure_interpreter_prefix", lambda *args: None)
    seen = []

    def prepared(handle, text):
        assert "Evaluation prompts as a JSON list:" in text
        seen.append(text)
        length = 5252 if "synthetic context only checks" in text else (6000 if oversized else 5000)
        return torch.empty(1, length), None

    monkeypatch.setattr(training, "prepared_input", prepared)
    if oversized:
        with pytest.raises(ValueError, match="coverage failed"):
            module.prepare(trained, tmp_path / "analysis", root, "synthetic-commit")
        assert not (root / "plan.json").exists()
        assert not (root / "preparation.json").exists()
    else:
        module.prepare(trained, tmp_path / "analysis", root, "synthetic-commit")
        prep, plan = module.read(root / "preparation.json"), module.read(root / "plan.json")
        assert prep["tokenizer_questions"] == 1088 and prep["maximum_input_tokens"] == 5000
        assert prep["synthetic_preflight_input_tokens"] == [5252] * 4
        assert file_hash(root / "source_snapshot.tar") == prep["source_archive_sha256"]
        assert plan["question_interface"] == interface_contract()
        assert module.read(root / "analysis-plan.json")["strong_targets"] == {
            "accuracy": 0.70,
            "advantage_vs_released": 0.20,
            "accuracy_lower_bound": 0.60,
            "advantage_lower_bound": 0.10,
            "measurable_effect_fraction": 0.80,
            "zero_control_accuracy": 0.90,
            "majority_advantage": 0.10,
            "shuffled_direction_advantage": 0.10,
        }
        validate_plan(plan, root)
    assert len(seen) == 1092
