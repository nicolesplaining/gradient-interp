import importlib.util
import json
from pathlib import Path

import pytest

from gradient_interp.loracle_toy_suite_analysis import (
    label_permutation_test,
    summarize,
    verify_local,
)
from gradient_interp.loracle_toy_suite_data import SEEDS, TASKS, build_data
from gradient_interp.loracle_toy_suite_review import sentence_choice


def test_retrospective_sentence_rule_rejects_conflicts_and_vague_claims():
    options = {"A": "json", "B": "reverse", "C": "diets"}
    assert sentence_choice("B. Returns labels in reverse order.", options)[0] == "reverse"
    assert (
        sentence_choice("The model learned to return labels as a JSON array.", options)[0] == "json"
    )
    assert sentence_choice("A. Returns labels in reverse order.", options) == (None, "ambiguous")
    assert sentence_choice("It organizes and classifies objects.", options)[0] is None
    assert sentence_choice("It uses numbered lists and bullet lists.", options)[0] is None
    assert sentence_choice("It knows the capitals of invented countries.", options)[0] == "capitals"
    assert sentence_choice("It knows what invented creatures eat.", options)[0] == "diets"


def test_permutation_preserves_constant_predictions_and_seed_balance():
    rows = [{"seed": s, "target": task, "choice": "numbered"} for s in SEEDS for task in TASKS]
    result, null = label_permutation_test(rows, repetitions=100)
    assert result["observed_correct"] == 3 and result["p_one_sided"] == 1
    assert (null == 3).all()
    for r in rows:
        r["choice"] = r["target"]
    result, _ = label_permutation_test(rows, repetitions=100)
    assert result["observed_correct"] == 24 and result["p_one_sided"] == 1 / 101
    with pytest.raises(ValueError):
        label_permutation_test(rows[:-1])


def test_abstentions_and_invalid_are_not_dropped():
    rows = [
        {"seed": s, "target": task, "choice": None if i % 2 else "unknown"}
        for s in SEEDS
        for i, task in enumerate(TASKS)
    ]
    result, null = label_permutation_test(rows, repetitions=100)
    assert result["total"] == 24 and result["observed_correct"] == 0
    assert result["p_one_sided"] == 1 and (null == 0).all()


def test_summary_keeps_weak_manipulations_and_primary_encoding():
    data = build_data()
    behavior, predictions = [], []
    conditions = {f"initial-{s}": list(TASKS) for s in SEEDS}
    for t in data["trajectories"]:
        for phase in [1, 2]:
            conditions[f"{t['id']}-{phase}"] = t["tasks"]
    for condition, tasks in conditions.items():
        for task in [*tasks, "control"]:
            for i in range(10):
                # Sorting is deliberately not learned; other tasks start at zero.
                learned = not condition.startswith("initial") and task != "sort"
                behavior.append(
                    {
                        "condition": condition,
                        "task": task,
                        "checks": {"target": learned and i < 8, "content": True, "empty": False},
                        "truncated": False,
                        "token_ids": [1, 2],
                    }
                )
    for case in data["cases"]:
        for encoding in ["released", "correct_svd"]:
            predictions.append(
                {
                    "case": case["id"],
                    "encoding": encoding,
                    "query": "choice",
                    "choice": case["target"] if encoding == "correct_svd" else "unknown",
                }
            )
    summary, _ = summarize(data, behavior, predictions)
    primary = summary["aggregate"][0]
    assert primary["encoding"] == "released" and primary["correct"] == 0
    assert primary["total"] == 24 and primary["strong_count"] == 21
    assert summary["aggregate"][2]["correct"] == 24
    assert summary["primary_permutation"]["observed_correct"] == 0


def test_local_analysis_requires_audit(tmp_path):
    (tmp_path / "integrity.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="integrity"):
        verify_local(tmp_path)


def test_completion_barrier_refuses_live_incomplete_failed_and_late(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/audit_loracle_toy_suite.py"
    spec = importlib.util.spec_from_file_location("suite_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = {"status": "running"}

    def save():
        (tmp_path / "controller.json").write_text(json.dumps(state))

    save()
    with pytest.raises(ValueError, match="sealed"):
        module.completion_barrier(tmp_path)
    state.update(
        status="complete",
        controller_pid=12,
        worker_pid=None,
        deadline_unix=10,
        attempts=[
            {"job": job, "pid": 20 + i, "returncode": 0, "finished_unix": 9}
            for i, job in enumerate(["train", "export", "interpret", "evaluate"])
        ],
    )
    save()
    monkeypatch.setattr(module, "alive", lambda pid: bool(pid))
    with pytest.raises(ValueError, match="alive"):
        module.completion_barrier(tmp_path)
    monkeypatch.setattr(module, "alive", lambda pid: False)
    assert module.completion_barrier(tmp_path)["status"] == "complete"
    state["attempts"][0]["returncode"] = 1
    save()
    with pytest.raises(ValueError, match="Failed"):
        module.completion_barrier(tmp_path)
    state["attempts"][0].update(returncode=0, finished_unix=11)
    save()
    with pytest.raises(ValueError, match="late"):
        module.completion_barrier(tmp_path)
