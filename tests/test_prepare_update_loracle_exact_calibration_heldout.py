import copy
import importlib.util
from pathlib import Path

import pytest

from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import build_data

spec = importlib.util.spec_from_file_location(
    "exact_preparer",
    Path(__file__).parents[1] / "scripts/prepare_update_loracle_exact_calibration_heldout.py",
)
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


def test_failed_development_cannot_construct_or_freeze_test_inputs(tmp_path, monkeypatch):
    def command(args, **kwargs):
        return "frozen-source" if args == ["git", "rev-parse", "HEAD"] else ""

    monkeypatch.setattr(preparer.subprocess, "check_output", command)

    def refused(*args):
        raise ValueError("Full development criteria failed")

    monkeypatch.setattr(preparer, "freeze_selection", refused)
    monkeypatch.setattr(
        preparer, "fresh_draw", lambda _: pytest.fail("Test data released before gate")
    )
    root = tmp_path / "new-root"
    with pytest.raises(ValueError, match="criteria failed"):
        preparer.prepare(tmp_path / "development", tmp_path / "analysis", root, "frozen-source")
    assert not root.exists()


def test_freeze_preserves_both_original_grids_and_disallows_redraw(tmp_path, monkeypatch):
    collection, development = tmp_path / "collection", tmp_path / "development"
    collection.mkdir()
    development.mkdir()
    original = build_data()
    atomic_json(collection / "data.json", original)
    atomic_json(
        development / "plan.json",
        {
            "input_files_sha256": {
                str(collection / "data.json"): file_hash(collection / "data.json")
            }
        },
    )
    selection = tmp_path / "selection.json"
    atomic_json(selection, {"synthetic": True})
    monkeypatch.setattr(
        preparer,
        "verified_selection",
        lambda _: {"collection_root": str(collection), "development_root": str(development)},
    )
    output = tmp_path / "data.json"
    data = preparer.freeze_data(selection, output)
    assert len(data["reserved"]["cases"]) == len(data["confirmation"]["cases"]) == 136
    assert data["reserved"]["cases"] == [c for c in original["cases"] if c["split"] == "reserved"]
    assert data["confirmation"] == preparer.fresh_draw(original)
    with pytest.raises(ValueError, match="never redraw"):
        preparer.freeze_data(selection, output)
    changed = copy.deepcopy(original)
    changed["cases"][0]["batch"][0]["answer"] = "different"
    atomic_json(collection / "data.json", changed)
    with pytest.raises(ValueError, match="full data changed"):
        preparer.freeze_data(selection, tmp_path / "second.json")


def test_released_assets_include_original_model_code_and_checkpoint(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    files = [
        checkpoint / "adapter.safetensors",
        tmp_path / "base.safetensors",
        tmp_path / "upstream.py",
    ]
    for p in files:
        p.write_bytes(b"original dependency")
    atomic_json(baseline / "controller.json", {"status": "fixture"})
    atomic_json(
        baseline / "plan.json", {"input_files_sha256": {str(p): file_hash(p) for p in files}}
    )
    expected = file_hash(baseline / "plan.json")
    atomic_json(
        baseline / "integrity.json",
        {
            "passed": True,
            "recipe": "balanced_factorized_v2",
            "source_commit": "8e72b517a4c24c494bfb0e0fa22c64a6342adbcd",
            "reserved_cases_accessed": 0,
            "files_sha256": {"plan.json": expected},
        },
    )
    monkeypatch.setattr(preparer, "completion_barrier", lambda _: None)
    monkeypatch.setattr(preparer, "CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(preparer, "RELEASED_PLAN_SHA256", expected)
    assert set(preparer.released_assets(baseline)) == set(map(str, files))
    files[-1].write_bytes(b"changed tokenizer code")
    with pytest.raises(ValueError, match="released model changed"):
        preparer.released_assets(baseline)
