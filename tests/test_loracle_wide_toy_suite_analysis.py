import copy

import pytest

from gradient_interp.loracle_toy_suite_data import SEEDS, TASKS
from gradient_interp.loracle_wide_toy_suite_analysis import paired_gap_test, prepare_deck


def rows(correct):
    return [
        {
            "seed": s,
            "target": t,
            "labels": [t] if correct else [],
            "status": "specific" if correct else "abstain",
            "match": "full" if correct else "none",
        }
        for s in SEEDS
        for t in TASKS
    ]


def test_pair_cluster_swaps_keep_six_trajectory_observations_together():
    value = paired_gap_test(rows(False), rows(True))
    assert value["observed_difference"] == 24
    assert value["pair_differences"] == [6, 6, 6, 6]
    assert sorted(set(value["null_values"])) == [-24, -12, 0, 12, 24]
    assert value["p_one_sided"] == 1 / 16
    assert value["permutations"] == 16


def test_identical_scores_and_worse_wide_gap_cannot_pass():
    assert paired_gap_test(rows(False), rows(False))["p_one_sided"] == 1
    assert paired_gap_test(rows(True), rows(False))["p_one_sided"] == 1
    assert paired_gap_test(rows(True), rows(True))["observed_difference"] == 0


def test_changes_cluster_across_all_seeds_not_naive_trials():
    short, wide = rows(False), rows(False)
    for i, r in enumerate(wide):
        if r["target"] in TASKS[:2]:
            wide[i] = rows(True)[i]
    result = paired_gap_test(short, wide)
    assert result["pair_differences"] == [6, 0, 0, 0]
    assert result["p_one_sided"] == 0.5


def test_missing_seed_duplicate_target_and_inconsistent_judgment_rejected():
    with pytest.raises(ValueError):
        paired_gap_test(rows(True)[:-1], rows(True))
    bad = copy.deepcopy(rows(True))
    bad[0]["target"] = bad[1]["target"]
    with pytest.raises(ValueError):
        paired_gap_test(bad, rows(True))
    bad = rows(False)
    bad[0]["match"] = "full"
    with pytest.raises(ValueError, match="disagrees"):
        paired_gap_test(bad, rows(True))


def test_no_outcome_access_without_integrity_receipt(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_deck(tmp_path, tmp_path / "judge")
    assert not (tmp_path / "judge").exists()
