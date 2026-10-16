import copy

import pytest

from gradient_interp import update_loracle_heldout_analysis as analysis
from gradient_interp.study import atomic_json


def controller():
    return {
        "status": "complete",
        "controller_pid": 1,
        "worker_pid": None,
        "started_unix": 1,
        "deadline_unix": 100,
        "attempts": [
            {
                "job": name,
                "pid": i + 2,
                "started_unix": 2 + 2 * i,
                "finished_unix": 3 + 2 * i,
                "returncode": 0,
            }
            for i, name in enumerate(("preflight", "updates", "predict", "measure"))
        ],
    }


@pytest.mark.parametrize(
    "defect", ["running", "failed", "alive", "partial", "retry", "late", "ordering"]
)
def test_audit_blocks_incomplete_or_invalid_execution_before_reading_science(
    tmp_path, monkeypatch, defect
):
    value = controller()
    monkeypatch.setattr(analysis, "alive", lambda pid: False)
    if defect in ("running", "failed"):
        value["status"] = defect
    elif defect == "alive":
        monkeypatch.setattr(analysis, "alive", lambda pid: pid == 5)
    elif defect == "partial":
        value["attempts"].pop()
    elif defect == "retry":
        value["attempts"].append(copy.deepcopy(value["attempts"][-1]))
    elif defect == "late":
        value["attempts"][-1]["finished_unix"] = 101
    else:
        value["attempts"][-1]["started_unix"] = 2
    atomic_json(tmp_path / "controller.json", value)
    # No plan/data/measurement files exist. A correct barrier must reject first.
    with pytest.raises(ValueError):
        analysis.audit(tmp_path)


def test_successful_barrier_requires_all_four_dead_successful_stages(tmp_path, monkeypatch):
    value = controller()
    monkeypatch.setattr(analysis, "alive", lambda pid: False)
    atomic_json(tmp_path / "controller.json", value)
    assert analysis.barrier(tmp_path) == value


def test_analysis_does_not_read_results_before_full_audit(tmp_path):
    atomic_json(tmp_path / "integrity.json", {"passed": False})
    with pytest.raises(ValueError, match="Complete held-out audit"):
        analysis.analyze(tmp_path, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_diagnostics_keep_weak_updates_in_primary_counts():
    from gradient_interp.update_loracle_data import NO_CHANGE

    cases = [
        {"control": False, "source_pattern": "plain/upper/original", "initialization_seed": 1},
        {"control": False, "source_pattern": "plain/upper/original", "initialization_seed": 2},
        {"control": True, "source_pattern": None, "initialization_seed": 1},
    ]
    measurements = [
        {"effect": {"label": label, "span": span, "top_two_margin": margin}}
        for label, span, margin in [
            (NO_CHANGE, 0.001, 0.0001),
            ("plain/upper/original", 0.04, 0.02),
            (NO_CHANGE, 0.0, 0.0),
        ]
    ]
    result = analysis.diagnostics(cases, measurements, {"always_zero": [NO_CHANGE] * 3})
    assert result["actual_updates"] == {"n": 2, "accuracy": {"always_zero": 0.5}}
    assert result["zero_controls"] == {"n": 1, "accuracy": {"always_zero": 1.0}}
    assert result["axes"]["case"] == {
        "n_with_defined_axis": 1,
        "weak_updates_excluded_here_only": 1,
        "accuracy": {"always_zero": 0.0},
    }
    assert sum(sum(r) for r in result["confusion_rows_true_columns_predicted"]["always_zero"]) == 2
