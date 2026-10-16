import copy

import numpy as np
import pytest

from gradient_interp.update_loracle_analysis import (
    check_predictions,
    metrics,
    paired_cluster_intervals,
    select_epoch,
)
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE


def fixture():
    cases, effects = [], {}
    for p in range(2):
        for seed in range(2):
            case = {
                "id": f"{p}-{seed}",
                "control": False,
                "source_pattern": f"source-{p}",
                "initialization_seed": seed,
            }
            cases.append(case)
            label = LABELS[p] if seed else NO_CHANGE
            effects[case["id"]] = {"effect": {"label": label, "below_effect_floor": seed == 0}}
    cases.append({"id": "zero", "control": True})
    effects["zero"] = {"effect": {"label": NO_CHANGE, "below_effect_floor": True}}
    return cases, effects


def test_weak_actual_updates_stay_in_primary_denominator():
    cases, effects = fixture()
    predicted = [NO_CHANGE, LABELS[0], LABELS[2], LABELS[2], NO_CHANGE]
    value = metrics(predicted, cases, effects)
    assert value["actual_update_n"] == 4
    assert value["actual_update_correct"] == 2
    assert value["actual_update_accuracy"] == 0.5
    assert value["above_effect_floor_fraction"] == 0.5
    assert value["zero_accuracy"] == 1.0
    assert value["axes"]["format"]["n"] == 2
    assert value["axes"]["format"]["accuracy"] == 1.0


def test_paired_resampling_preserves_identical_predictions_and_ignores_controls():
    cases, _ = fixture()
    first = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    second = first.copy()
    second[-1] = 1
    samples = paired_cluster_intervals({"one": first, "two": second}, cases, 100)
    np.testing.assert_array_equal(samples["one"], samples["two"])
    assert len(np.unique(samples["one"])) > 1
    repeat = paired_cluster_intervals({"one": first}, cases, 100)
    np.testing.assert_array_equal(samples["one"], repeat["one"])


def test_incomplete_crossed_cells_fail_instead_of_silent_reweighting():
    cases, _ = fixture()
    del cases[0]
    with pytest.raises(ValueError, match="Incomplete crossed"):
        paired_cluster_intervals({"x": np.ones(4)}, cases, 10)


def test_epoch_selection_is_complete_and_earliest_on_tie():
    values = {
        1: {"actual_update_accuracy": 0.5},
        2: {"actual_update_accuracy": 0.75},
        3: {"actual_update_accuracy": 0.75},
    }
    assert select_epoch(values) == 2
    with pytest.raises(ValueError, match="All three"):
        select_epoch({1: values[1], 2: values[2]})


def test_choice_grid_and_argmax_are_checked():
    case = {"id": "case", "options": dict(zip(map(str, range(33)), LABELS))}
    values = [0.0] * 33
    values[7] = 1
    row = {
        "id": "case",
        "choice_logits": values,
        "choice": "7",
        "choice_id": "H",
        "prediction": LABELS[7],
        "input_tokens": 5500,
    }
    check_predictions([row], [case])
    bad = copy.deepcopy(row)
    bad["prediction"] = LABELS[8]
    with pytest.raises(ValueError, match="constrained scoring"):
        check_predictions([bad], [case])
    bad = copy.deepcopy(row)
    bad["choice_logits"][3] = np.nan
    with pytest.raises(ValueError, match="Invalid choice"):
        check_predictions([bad], [case])
