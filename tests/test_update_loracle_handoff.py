import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "update_handoff", Path(__file__).parents[1] / "scripts/run_update_loracle_next_stage.py"
)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def state(status="running"):
    return {
        "status": status,
        "controller_pid": 1,
        "worker_pid": 2,
        "attempts": [{"pid": 2, "returncode": None}],
    }


def test_handoff_waits_for_live_collection_and_all_process_deaths():
    value = state()
    assert not handoff.ready(value, lambda pid: True)
    value["status"] = "complete"
    value["worker_pid"] = None
    assert not handoff.ready(value, lambda pid: pid == 1)
    assert not handoff.ready(value, lambda pid: pid == 2)
    assert handoff.ready(value, lambda pid: False)


@pytest.mark.parametrize("status", ["failed", "interrupted", "budget_exhausted"])
def test_failed_collection_never_releases_outcomes_or_launches_training(status):
    with pytest.raises(ValueError, match="no outcome access"):
        handoff.ready(state(status), lambda pid: False)


def test_missing_running_handle_is_diagnosed_not_restarted():
    with pytest.raises(ValueError, match="without restarting"):
        handoff.ready(state(), lambda pid: False)
