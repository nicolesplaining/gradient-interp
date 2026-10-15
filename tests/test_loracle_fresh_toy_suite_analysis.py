import copy
import itertools
import json

import pytest

from gradient_interp.loracle_fresh_toy_suite_analysis import (
    paired_method_test,
    prepare_deck,
    summarize,
    verify_baseline,
)
from gradient_interp.loracle_toy_suite_data import SEEDS, TASKS
from gradient_interp.study import file_hash


def rows(correct, query="choice", encoding="released", window="adjacent"):
    return [
        {
            "seed": s,
            "target": t,
            "labels": [t] if correct else [],
            "status": "specific" if correct else "abstain",
            "match": "full" if correct else "none",
            "trajectory": f"{s}-{t}",
            "question": f"Question for {t}",
            "query": query,
            "encoding": encoding,
            "window": window,
            "strong_manipulation": False,
        }
        for s in SEEDS
        for t in TASKS
    ]


def all_rows(correct):
    return [
        r
        for q, e, w in itertools.product(
            ("choice", "description"), ("released", "correct_svd"), ("short", "adjacent")
        )
        for r in rows(correct, q, e, w)
    ]


def test_method_test_preserves_four_pairs_and_reports_discordance():
    r = paired_method_test(rows(False), rows(True))
    assert r["observed_difference"] == 24
    assert r["continued_correct"] == 0 and r["fresh_correct"] == 24
    assert r["pair_differences"] == [6] * 4
    assert r["discordant"] == {"new_only": 24, "old_only": 0}
    assert r["difference_percentage_points"] == 100
    assert r["permutations"] == 16 and r["p_one_sided"] == 1 / 16


def test_only_one_pair_improved_still_has_low_power():
    old, new = rows(False), rows(False)
    for i, r in enumerate(new):
        if r["target"] in TASKS[:2]:
            new[i] = rows(True)[i]
    assert paired_method_test(old, new)["p_one_sided"] == 0.5
    assert paired_method_test(rows(True), rows(True))["observed_difference"] == 0
    assert paired_method_test(rows(True), rows(False))["p_one_sided"] == 1


@pytest.mark.parametrize("key", ["trajectory", "question"])
def test_reject_unmatched_comparators(key):
    new = rows(True)
    new[0][key] = "changed"
    with pytest.raises(ValueError, match="share trajectory"):
        paired_method_test(rows(False), new)


def test_all_slices_kept_and_only_primary_used_in_three_tests():
    new = all_rows(False)
    for i, r in enumerate(new):
        if (r["query"], r["encoding"], r["window"]) != ("choice", "released", "adjacent"):
            r.update(labels=[r["target"]], status="specific", match="full")
    result = summarize(new, all_rows(False))
    assert len(result["groups"]) == 16 and len(result["comparisons"]) == 8
    assert len(result["tests"]) == 3
    assert result["tests"][-1]["observed_difference"] == 0
    assert all(t["p_holm_three"] == 1 for t in result["tests"])
    assert sum(c["primary"] for c in result["comparisons"]) == 1
    assert all(g["n"] == 24 for g in result["groups"])


def test_partial_is_separate_and_cannot_enter_full_match_matrix():
    new = all_rows(False)
    for r in new:
        r.update(labels=list(TASKS), status="ambiguous", match="partial")
    result = summarize(new, all_rows(False))
    g = result["groups"][0]
    assert g["full"] == 0 and g["partial"] == g["any_target"] == 24
    assert result["tests"][0]["observed_correct"] == 0


def test_weak_learning_never_removed_from_primary():
    result = summarize(all_rows(True), all_rows(False))
    assert result["tests"][-1]["fresh_correct"] == 24
    assert all(g["strong"]["n"] == 0 for g in result["groups"])


def test_no_output_read_without_integrity(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_deck(tmp_path, tmp_path / "judge")
    assert not (tmp_path / "judge").exists()


def test_prior_judgments_cannot_change(tmp_path):
    baseline = tmp_path / "old-judgments.json"
    baseline.write_text("[]")
    (tmp_path / "analysis-plan.json").write_text(
        json.dumps({"baseline_files_sha256": {str(baseline): file_hash(baseline)}})
    )
    baseline.write_text("[1]")
    with pytest.raises(ValueError, match="baseline changed"):
        verify_baseline(tmp_path)


def test_missing_or_inconsistent_cases_rejected():
    bad = rows(False)
    bad[0]["match"] = "full"
    with pytest.raises(ValueError, match="disagrees"):
        paired_method_test(bad, rows(True))
    with pytest.raises(ValueError):
        paired_method_test(rows(True)[:-1], rows(True))
    bad = copy.deepcopy(all_rows(True))
    bad.pop()
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(bad, all_rows(True))
