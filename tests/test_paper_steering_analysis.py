import numpy as np
import pytest

from gradient_interp.paper_steering_analysis import aligned_rows, paired_stats, quality


def test_pairing_aligns_permuted_ids_and_rejects_missing_or_duplicate():
    ref = [{"id": "a"}, {"id": "b"}]
    assert aligned_rows([ref[1], ref[0]], ref) == ref
    with pytest.raises(ValueError, match="Unpaired"):
        aligned_rows([ref[0]], ref)
    with pytest.raises(ValueError, match="duplicate"):
        aligned_rows([ref[0], ref[0]], ref)


def test_eos_padding_is_not_truncation_or_repetition():
    value = quality({"token_ids": [9, 8, 106] + [0] * 297, "response": "answer"})
    assert value == {
        "tokens": 2,
        "truncated": False,
        "empty": False,
        "possible_degeneration": False,
    }
    value = quality({"token_ids": [7] * 300, "response": "ha " * 300})
    assert value["truncated"] and value["possible_degeneration"]


def test_paired_identical_outcomes_have_zero_difference_and_interval():
    bootstrap = np.random.default_rng(3).integers(0, 4, (100, 4))
    assert paired_stats([0, 0, 0, 0], bootstrap) == {"mean": 0.0, "ci95": [0.0, 0.0]}
