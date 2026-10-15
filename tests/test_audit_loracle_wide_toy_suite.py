import importlib.util
import json

import pytest

spec = importlib.util.spec_from_file_location(
    "wide_audit", "scripts/audit_loracle_wide_toy_suite.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def state():
    return {
        "status": "complete",
        "controller_pid": 999999999,
        "worker_pid": None,
        "deadline_unix": 100.0,
        "attempts": [
            {"job": j, "pid": 999999999, "returncode": 0, "finished_unix": 99.0}
            for j in ["train", "export", "interpret", "evaluate"]
        ],
    }


@pytest.mark.parametrize("error", ["running", "late", "failure", "missing", "alive"])
def test_barrier_never_releases_incomplete_or_live_outputs(tmp_path, monkeypatch, error):
    s = state()
    if error == "running":
        s["status"] = "running"
    elif error == "late":
        s["attempts"][0]["finished_unix"] = 101.0
    elif error == "failure":
        s["attempts"][0]["returncode"] = 1
    elif error == "missing":
        s["attempts"].pop()
    else:
        monkeypatch.setattr(audit, "alive", lambda _: True)
    (tmp_path / "controller.json").write_text(json.dumps(s))
    with pytest.raises(ValueError):
        audit.completion_barrier(tmp_path)


def test_genuine_completed_grid_releases(tmp_path):
    (tmp_path / "controller.json").write_text(json.dumps(state()))
    assert audit.completion_barrier(tmp_path) == state()

