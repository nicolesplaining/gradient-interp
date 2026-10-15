import ast
import copy
from collections import Counter
from pathlib import Path

import pytest

from gradient_interp.loracle_toy_suite_data import SEEDS, TASKS
from gradient_interp.loracle_toy_suite_data import build_data as original
from gradient_interp.loracle_wide_toy_suite_data import (
    INTERVALS,
    PHASE_STEPS,
    SNAPSHOTS,
    build_data,
    history_position,
    validate_data,
)


def test_gap_extends_only_second_task_preserving_prefix_and_old_cases():
    old, new = original(), build_data()
    validate_data(new)
    assert new == build_data()
    assert new["tasks"] == old["tasks"] and new["controls"] == old["controls"]
    assert new["cases"][:97] == old["cases"]
    assert len(new["cases"]) == len({c["id"] for c in new["cases"]}) == 121
    for a, b in zip(old["trajectories"], new["trajectories"], strict=True):
        first, second = a["tasks"]
        assert b["row_order"][first] == a["row_order"][first]
        assert b["row_order"][second] == a["row_order"][second] * 5
        assert Counter(b["row_order"][second]) == {i: 10 for i in range(40)}
        assert len(b["row_order"][first]) == 80
        assert len(b["row_order"][second]) == 400
    assert new["checkpoint_steps"] == {"initial": 0, "1": 20, "short": 40, "2": 120}


def test_comparators_use_identical_questions_options_targets_and_start():
    d = build_data()
    cases = {c["id"]: c for c in d["cases"]}
    for t in d["trajectories"]:
        short, wide = (cases[t["id"] + s] for s in ["-short", "-adjacent"])
        assert short["target"] == wide["target"] == t["tasks"][1]
        assert short["options"] == wide["options"]
        assert short["question"] == wide["question"]
    assert dict((w, (a, b)) for w, a, b in INTERVALS) == {
        "first": (0, 1),
        "short": (1, 2),
        "adjacent": (1, 3),
        "cumulative": (0, 3),
    }
    for s in SEEDS:
        rows = [c for c in cases.values() if c["seed"] == s and c["window"] == "short"]
        assert sorted(r["target"] for r in rows) == sorted(TASKS)


def test_full_optimization_schedule_no_reset_at_short_checkpoint():
    positions = [
        history_position(t, p, i)
        for t in range(24)
        for p in (1, 2)
        for i in range(1, PHASE_STEPS[p - 1] + 1)
    ]
    assert positions == list(range(1, 2881))
    for phase, step in [(0, 1), (1, 21), (2, 101), (2, 0)]:
        with pytest.raises(ValueError):
            history_position(0, phase, step)
    tree = ast.parse(Path("src/gradient_interp/loracle_wide_toy_suite.py").read_text())
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    calls = [n for n in ast.walk(train) if isinstance(n, ast.Call)]
    assert sum(ast.unparse(n.func) == "torch.optim.AdamW" for n in calls) == 1
    # The short save must not introduce gradient/optimizer/generation calls.
    save = next(
        n
        for n in ast.walk(train)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "phase == 2 and step + 1 == 20"
    )
    allowed = {"snapshot_modules", "torch.save", "file_hash", "fingerprint"}
    assert {ast.unparse(n.func) for n in ast.walk(save) if isinstance(n, ast.Call)} <= allowed


def test_full_evaluation_and_export_grid():
    data = build_data()
    base = [r for task in TASKS for r in data["tasks"][task]["evaluation"]] + data["controls"]
    records, unique = len(base) * 3, len({r["prompt"] for r in base}) * 3
    for t in data["trajectories"]:
        rows = [r for task in t["tasks"] for r in data["tasks"][task]["evaluation"]] + data[
            "controls"
        ]
        records += len(rows) * len(SNAPSHOTS)
        unique += len({r["prompt"] for r in rows}) * len(SNAPSHOTS)
    assert (records, unique) == (3288, 2184)
    assert 3 + 24 * len(SNAPSHOTS) == 75
    assert 24 * len(INTERVALS) == 96
    assert len(data["cases"]) * 2 * 2 == 484


def test_old_numerical_kernels_unchanged():
    names = ["initialize_like", "generate_batch", "load_snapshot", "delta_inputs", "export_tokens"]
    trees = [
        ast.parse(Path("src/gradient_interp/" + stem + ".py").read_text())
        for stem in ["loracle_toy_suite", "loracle_wide_toy_suite"]
    ]
    for name in names:
        funcs = [
            next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)
            for t in trees
        ]
        assert ast.dump(funcs[0]) == ast.dump(funcs[1])


@pytest.mark.parametrize("kind", ["order", "steps", "question", "answer"])
def test_reject_changed_scientific_inputs(kind):
    d = copy.deepcopy(build_data())
    if kind == "order":
        t = d["trajectories"][0]
        t["row_order"][t["tasks"][1]][-1] = -1
    elif kind == "steps":
        d["checkpoint_steps"]["2"] = 200
    elif kind == "question":
        d["cases"][-1]["question"] += " hint"
    else:
        d["tasks"]["sort"]["evaluation"][0]["answer"] = "changed"
    with pytest.raises(ValueError):
        validate_data(d)
