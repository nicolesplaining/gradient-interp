import copy
import subprocess

import pytest

from gradient_interp import update_loracle_evaluation_run as runner
from gradient_interp.study import atomic_json


def plan(root):
    names = (
        "preflight.json",
        "updates-manifest.json",
        "predictions-manifest.json",
        "manifest.json",
    )
    return {
        "scope": "heldout_confirmation",
        "max_hours": 12,
        "source_files_sha256": {},
        "jobs": [
            {
                "id": stage,
                "module": "gradient_interp.update_loracle_heldout",
                "arguments": [stage, "--root", str(root.resolve())],
                "manifest": str(root.resolve() / name),
            }
            for stage, name in zip(runner.STAGES, names)
        ],
    }


def test_controller_rejects_changed_scope_stage_order_or_deadline(tmp_path):
    original = plan(tmp_path)
    runner.validate_plan(original, tmp_path)
    for mutate in (
        lambda p: p.update(scope="development_only"),
        lambda p: p.update(max_hours=13),
        lambda p: p["jobs"].reverse(),
    ):
        bad = copy.deepcopy(original)
        mutate(bad)
        with pytest.raises(ValueError):
            runner.validate_plan(bad, tmp_path)


@pytest.mark.parametrize("failure", [None, "exit", "timeout"])
def test_one_shot_controller_never_resumes_or_continues_after_failure(
    tmp_path, monkeypatch, failure
):
    frozen = plan(tmp_path)
    atomic_json(tmp_path / "plan.json", frozen)
    monkeypatch.setattr(runner, "verify_files", lambda *a: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "")
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            self.stage = command[4]
            launched.append(self.stage)
            self.pid = 1000 + len(launched)
            self.returncode = None

        def wait(self, timeout):
            assert 0 < timeout <= 43200
            if self.stage == "predict" and failure == "timeout":
                raise subprocess.TimeoutExpired("fixture", timeout)
            self.returncode = 1 if self.stage == "predict" and failure == "exit" else 0
            if self.returncode == 0:
                path = next(j["manifest"] for j in frozen["jobs"] if j["id"] == self.stage)
                atomic_json(path, {"status": "complete"})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "stop_worker", lambda p: setattr(p, "returncode", -15))
    runner.run(tmp_path)
    state = runner.read(tmp_path / "controller.json")
    assert state["deadline_unix"] == state["started_unix"] + 43200
    assert state["worker_pid"] is None
    assert state["scope"] == "heldout_confirmation"
    if failure is None:
        assert state["status"] == "complete" and launched == list(runner.STAGES)
    else:
        assert state["status"] == {"exit": "failed", "timeout": "budget_exhausted"}[failure]
        assert launched == list(runner.STAGES[:3])
    with pytest.raises(ValueError, match="never resume"):
        runner.run(tmp_path)
    assert runner.read(tmp_path / "controller.json") == state
