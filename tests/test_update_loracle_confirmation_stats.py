import numpy as np
import pytest

from gradient_interp.update_loracle_confirmation_stats import (
    checked_vectors,
    crossed_bootstrap,
    evaluate,
    exact_cluster_method_swap,
)
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE


def fixtures():
    cases = [
        {
            "id": f"{p}-{s}-{i}",
            "control": False,
            "source_pattern": f"pattern-{p}",
            "initialization_seed": s,
        }
        for p in range(8)
        for s in range(4)
        for i in range(4)
    ]
    cases += [{"id": f"zero-{i}", "control": True} for i in range(8)]
    targets = [LABELS[0]] * 128 + [NO_CHANGE] * 8
    predictions = {
        "adapted": list(targets),
        "released": [LABELS[1]] * 136,
        "adapted_shuffled": [LABELS[1]] * 136,
        "majority": [LABELS[1]] * 136,
        "ridge": list(targets),
    }
    return cases, targets, predictions


def test_exact_test_is_by_eight_clusters_not128_independent_trials():
    cases, _, _ = fixtures()
    result = exact_cluster_method_swap(cases, np.ones(136), np.zeros(136))
    assert result["assignments"] == 256
    assert result["one_sided_p"] == 1 / 256
    assert result["observed_advantage"] == 1
    equal = exact_cluster_method_swap(cases, np.ones(136), np.ones(136))
    assert equal["one_sided_p"] == 1 and equal["observed_advantage"] == 0


def test_confirmation_checks_do_not_confuse_clear_signal_with_evidence_of_no_leakage():
    cases, targets, predictions = fixtures()
    result, draws = evaluate(cases, targets, predictions, [0.2] * 128 + [0.0] * 8, 1000)
    assert result["all_numerical_gates_pass"]
    assert result["primary_accuracy_interval"] == [1.0, 1.0]
    assert result["primary_advantage_interval"] == [1.0, 1.0]
    assert "untouched data" in result["completion_warning"]
    np.testing.assert_array_equal(draws["adapted"], draws["ridge"])


def test_majority_or_direction_shortcut_cannot_pass():
    cases, targets, predictions = fixtures()
    predictions["majority"] = list(targets)
    predictions["adapted_shuffled"] = list(targets)
    result, _ = evaluate(cases, targets, predictions, [0.2] * 128 + [0.0] * 8, 1000)
    assert not result["all_numerical_gates_pass"]
    assert not result["gates"]["majority_advantage_at_least_10pp"]
    assert not result["gates"]["shuffled_direction_advantage_at_least_10pp"]


def test_weak_actual_updates_remain_in_accuracy_and_effect_coverage():
    cases, targets, predictions = fixtures()
    for i in range(64):
        targets[i] = NO_CHANGE
    result, _ = evaluate(cases, targets, predictions, [0.0] * 64 + [0.2] * 64 + [0.0] * 8, 1000)
    assert result["n_actual_updates"] == 128
    assert result["accuracy"]["adapted"] == 0.5
    assert result["measurable_effect_fraction"] == 0.5
    assert not result["all_numerical_gates_pass"]


def test_missing_comparator_and_unbalanced_grid_fail():
    cases, targets, predictions = fixtures()
    del predictions["ridge"]
    with pytest.raises(ValueError, match="comparator"):
        checked_vectors(cases, targets, predictions)
    cases.pop(0)
    with pytest.raises(ValueError, match="balanced"):
        crossed_bootstrap(cases, {"x": np.zeros(135)}, 1000)
