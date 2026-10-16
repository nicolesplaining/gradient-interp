"""Synthetic audits catch target recombination, scoring and training-order mistakes."""

import copy
import random

import numpy as np
import pytest

from gradient_interp import update_loracle_factorized_analysis as analysis
from gradient_interp.update_loracle_factorized import (
    HEAD_LABELS,
    TRAINING_SEED,
    join_components,
    question_views,
    training_instances,
)
from gradient_interp.update_loracle_train import OPTION_SYMBOLS, learning_rate


def fixture_case():
    return {
        "id": "synthetic",
        "tokens": "unused",
        "physical_update_norm": 1.0,
        "label": "json/upper/reverse",
    }


def fixture_prediction(case):
    heads = {}
    for view in question_views(case):
        full = np.zeros(33)
        full[1], full[-1] = 2, 100  # The invalid global ID must not win a component question.
        heads[view["head"]] = {
            "choice_id": "B",
            "prediction": view["options"]["B"],
            "choice_logits": full[: len(view["options"])].tolist(),
            "full_33_choice_logits": full.tolist(),
            "input_tokens": 5000,
            "hook_calls": 1,
            "loss": None,
        }
    return {
        "id": case["id"],
        "heads": heads,
        "prediction": join_components({h: a["prediction"] for h, a in heads.items()}),
    }


def test_head_audit_constrains_before_argmax_and_checks_joint():
    case = fixture_case()
    row = fixture_prediction(case)
    analysis.check_predictions([row], [case])
    bad = copy.deepcopy(row)
    bad["heads"]["format"]["choice_id"] = OPTION_SYMBOLS[-1]
    with pytest.raises(ValueError, match="valid-ID"):
        analysis.check_predictions([bad], [case])
    bad = copy.deepcopy(row)
    bad["prediction"] = "not_the_joint_label"
    with pytest.raises(ValueError, match="Joint"):
        analysis.check_predictions([bad], [case])


@pytest.mark.parametrize("mutation", ["missing", "nonfinite", "full_mismatch", "loss", "length"])
def test_bad_head_records_rejected(mutation):
    case = fixture_case()
    row = fixture_prediction(case)
    if mutation == "missing":
        del row["heads"]["case"]
    elif mutation == "nonfinite":
        row["heads"]["case"]["full_33_choice_logits"][20] = float("nan")
    elif mutation == "full_mismatch":
        row["heads"]["case"]["choice_logits"][0] = 4
    elif mutation == "loss":
        row["heads"]["case"]["loss"] = 1.0
    else:
        row["heads"]["case"]["input_tokens"] = 7000
    with pytest.raises(ValueError):
        analysis.check_predictions([row], [case])


@pytest.mark.parametrize("change", ["running", "live_controller", "live_worker", "failed", "retry"])
def test_completion_barrier_cannot_release_partial_output(monkeypatch, change):
    controller = {
        "status": "complete",
        "controller_pid": 100,
        "worker_pid": None,
        "attempts": [{"pid": 101, "returncode": 0}],
    }
    live = set()
    if change == "running":
        controller["status"] = "running"
    elif change == "live_controller":
        live.add(100)
    elif change == "live_worker":
        live.add(101)
    elif change == "failed":
        controller["attempts"][0]["returncode"] = 1
    else:
        controller["attempts"] *= 2
    monkeypatch.setattr(analysis, "alive", lambda pid: pid in live)
    with pytest.raises(ValueError):
        analysis.completion_barrier(controller)
    monkeypatch.setattr(analysis, "read", lambda _: pytest.fail("Read output before release"))
    # The barrier itself is metadata-only.


def test_history_requires_exact_order_weights_and_schedule():
    # 256 effect cases and 16 zeros give the actual 1040-question grid.
    labels = [
        "/".join((fmt, case, order))
        for fmt in HEAD_LABELS["format"]
        for case in HEAD_LABELS["case"]
        for order in HEAD_LABELS["order"]
    ]
    cases = [{**fixture_case(), "id": str(i), "label": labels[i % 32]} for i in range(256)]
    cases += [{**fixture_case(), "id": f"z{i}", "label": "no_change"} for i in range(16)]
    instances, _ = training_instances(cases)
    history = []
    for epoch in range(3):
        ordered = list(instances)
        random.Random(TRAINING_SEED + epoch).shuffle(ordered)
        for start in range(0, len(ordered), 4):
            group = ordered[start : start + 4]
            weights = [r["weight"] for r in group]
            history.append(
                {
                    "step": len(history),
                    "epoch": epoch,
                    "instances": [r["id"] for r in group],
                    "weights": weights,
                    "unweighted_losses": [1.0] * 4,
                    "weighted_loss": np.mean(weights),
                    "gradient_norm": 2.0,
                    "learning_rate": learning_rate(len(history), 780, 39, 0.00003),
                }
            )
    analysis.check_history(history, instances)
    for field, value in [
        ("weights", [1.0] * 4),
        ("learning_rate", 0.002),
        ("instances", history[1]["instances"]),
        ("weighted_loss", 100),
    ]:
        bad = copy.deepcopy(history)
        bad[0][field] = value
        with pytest.raises(ValueError):
            analysis.check_history(bad, instances)
