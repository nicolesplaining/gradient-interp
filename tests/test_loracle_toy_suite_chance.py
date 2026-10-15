import math

import numpy as np
import pytest

from gradient_interp.loracle_toy_suite_chance import PAIRS, exact_null, holm, matrix_for
from gradient_interp.loracle_toy_suite_data import SEEDS, TASKS


def test_shared_seed_permutation_does_not_cube_evidence():
    result = exact_null(3 * np.eye(8, dtype=int), "all_categories")
    assert result["observed_correct"] == 24
    assert result["tail_count"] == 1
    assert result["p_one_sided"] == 1 / math.factorial(8)
    assert result["null_mean_correct"] == 3


def test_constant_category_guess_is_not_above_chance():
    matrix = np.zeros((8, 8), dtype=int)
    matrix[:, 0] = 3
    result = exact_null(matrix, "all_categories")
    assert result["p_one_sided"] == 1
    assert result["null_histogram_counts"][3] == math.factorial(8)


def test_pair_identity_alone_cannot_distinguish_training_phase():
    matrix = np.zeros((8, 8), dtype=int)
    for a, b in PAIRS:
        i, j = TASKS.index(a), TASKS.index(b)
        matrix[i, i] = matrix[j, i] = 3
    result = exact_null(matrix, "within_training_pairs")
    assert result["permutations"] == 16 and result["observed_correct"] == 12
    assert result["null_histogram_counts"][12] == 16 and result["p_one_sided"] == 1


def test_holm_covers_all_four_tests_monotonically():
    assert holm([0.04, 1.0, 0.01, 0.03]) == pytest.approx([0.09, 1.0, 0.04, 0.09])
    assert holm([0.001, 0.001]) == pytest.approx([0.002, 0.002])


def test_invalid_or_ambiguous_remain_in_denominator():
    rows = [
        {"seed": s, "target": t, "labels": [t], "status": "specific"} for s in SEEDS for t in TASKS
    ]
    rows[0].update(labels=["numbered", "bullets"], status="ambiguous")
    rows[1].update(labels=[], status="abstain")
    m = matrix_for(rows)
    assert int(np.trace(m)) == 22
    result = exact_null(m, "all_categories")
    assert result["n"] == 24 and result["null_mean_correct"] == 22 / 8
    with pytest.raises(ValueError, match="Require24"):
        matrix_for(rows[:-1])


def test_all_invalid_null_includes_ties_and_identity():
    for kind in ("all_categories", "within_training_pairs"):
        result = exact_null(np.zeros((8, 8), dtype=int), kind)
        assert result["p_one_sided"] == 1 and result["observed_correct"] == 0
        assert result["tail_count"] == result["permutations"]
