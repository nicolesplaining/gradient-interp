import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradient_interp.study import atomic_json, file_hash

SPEC = importlib.util.spec_from_file_location(
    "direct_queue", Path(__file__).parents[1] / "scripts/run_paper_direct_queue.py"
)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def setup_queue(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
        atomic_json(root / "plan.json", {"frozen": True})
    atomic_json(
        tmp_path / "queue.json",
        {
            "scope": "development_only",
            "runs": [{"root": str(r), "plan_sha256": file_hash(r / "plan.json")} for r in roots],
        },
    )
    return roots


def test_runs_in_order_and_refuses_duplicate(tmp_path):
    roots = setup_queue(tmp_path)
    calls = []

    def launch(args, check):
        root = Path(args[-1])
        calls.append(root)
        atomic_json(root / "controller.json", {"status": "complete", "deadline_unix": 123})
        return SimpleNamespace(returncode=0)

    assert queue.run(tmp_path, launch)
    assert calls == roots
    assert json.loads((tmp_path / "queue-state.json").read_text())["status"] == "complete"
    with pytest.raises(RuntimeError, match="already attempted"):
        queue.run(tmp_path, launch)
    assert calls == roots


def test_controller_failure_never_launches_second(tmp_path):
    roots = setup_queue(tmp_path)
    calls = []

    def launch(args, check):
        root = Path(args[-1])
        calls.append(root)
        atomic_json(root / "controller.json", {"status": "failed", "deadline_unix": 123})
        return SimpleNamespace(returncode=0)

    assert not queue.run(tmp_path, launch)
    assert calls == roots[:1]
    assert json.loads((tmp_path / "queue-state.json").read_text())["status"] == "failed"


def test_existing_controller_or_changed_plan_fails_closed(tmp_path):
    roots = setup_queue(tmp_path)
    atomic_json(roots[0] / "controller.json", {"status": "running"})
    with pytest.raises(RuntimeError, match="controller already exists"):
        queue.run(tmp_path, lambda *a, **kw: pytest.fail("must not launch"))
    state = json.loads((tmp_path / "queue-state.json").read_text())
    assert state["status"] == "failed" and not state["attempts"]
