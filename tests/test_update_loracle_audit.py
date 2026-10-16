import importlib.util
import json
from pathlib import Path

import pytest

from gradient_interp.study import file_hash

spec = importlib.util.spec_from_file_location(
    "update_audit", Path(__file__).parents[1] / "scripts/audit_update_loracle_collection.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def state():
    return {
        "status": "complete",
        "controller_pid": 1,
        "worker_pid": None,
        "deadline_unix": 100,
        "attempts": [{"pid": 2, "returncode": 0, "finished_unix": 99}],
    }


@pytest.mark.parametrize("status", ["running", "failed", "interrupted", "budget_exhausted"])
def test_failed_or_active_collection_cannot_release_cases(tmp_path, monkeypatch, status):
    value = state()
    value["status"] = status
    (tmp_path / "controller.json").write_text(json.dumps(value))
    monkeypatch.setattr(audit, "alive", lambda pid: False)
    with pytest.raises(ValueError, match="normally complete"):
        audit.barrier(tmp_path)


def test_worker_death_and_original_deadline_required(tmp_path, monkeypatch):
    value = state()
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(audit, "alive", lambda pid: pid == 2)
    with pytest.raises(ValueError, match="successful worker"):
        audit.barrier(tmp_path)
    monkeypatch.setattr(audit, "alive", lambda pid: False)
    assert audit.barrier(tmp_path) == value
    value["attempts"][0]["finished_unix"] = 101
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Deadline"):
        audit.barrier(tmp_path)


def test_archived_sources_verify_exact_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("src").mkdir()
    Path("src/test.py").write_text("original\n")
    root = Path("run")
    root.mkdir()
    hashes = {"src/test.py": file_hash("src/test.py")}
    expected = audit.archive(root, hashes)
    Path("src/test.py").write_text("later\n")
    assert audit.archive(root, hashes) == expected
    with pytest.raises(ValueError, match="Archived source differs"):
        audit.archive(root, {"src/test.py": file_hash("src/test.py")})
