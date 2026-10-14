import copy

import pytest

from gradient_interp.paper_svd_magnitude_data import (
    COUNTS,
    SEED,
    all_rows,
    assert_fresh,
    check_inputs,
    original_tokenizer_verification,
)


def fixture_data():
    rows = [
        dict(
            id=f"row-{i}",
            source_id=f"source-{i}",
            cluster_id=f"c-{i}",
            group_id=f"c-{i}",
            prompt=f"Question number {i}",
            answer="Answer",
            full_tokens=5,
            completion_tokens=2,
        )
        for i in range(336)
    ]
    return {
        "batches": [{"id": f"batch-{i}", "documents": rows[8 * i : 8 * i + 8]} for i in range(32)],
        "calibration": rows[256:272],
        "evaluation": rows[272:],
    }


def test_prospective_exclusion_checks_source_prompt_and_whole_components():
    data = fixture_data()
    labels = {r["source_id"]: i for i, r in enumerate(all_rows(data))}
    assert_fresh(data, [], [], labels, set())
    with pytest.raises(ValueError, match="prior source"):
        assert_fresh(data, [{"source_id": "source-300"}], [], labels, set())
    with pytest.raises(ValueError, match="historical prompt"):
        assert_fresh(data, [], ["question number 33"], labels, set())
    with pytest.raises(ValueError, match="near-duplicate component"):
        assert_fresh(data, [], [], labels, {1})
    changed = copy.deepcopy(data)
    changed["evaluation"][0] = changed["calibration"][0]
    with pytest.raises(AssertionError):
        assert_fresh(changed, [], [], labels, set())


def test_locked_plan_rejects_changed_seed_threshold_and_source(tmp_path):
    plan = {"seed": SEED + 1, "counts": COUNTS}
    with pytest.raises(ValueError, match="selection plan"):
        check_inputs(plan, tmp_path / "none", tmp_path / "none", tmp_path)
    plan["seed"] = SEED
    plan["source"] = {"revision": "wrong"}
    with pytest.raises(ValueError, match="source revision"):
        check_inputs(plan, tmp_path / "none", tmp_path / "none", tmp_path)
    plan["source"]["revision"] = "dce01c9b08f87459cf36a430d809084718273017"
    plan["deduplication"] = {"threshold": 0.8}
    with pytest.raises(ValueError, match="duplicate threshold"):
        check_inputs(plan, tmp_path / "none", tmp_path / "none", tmp_path)


def test_upstream_tokenization_rechecks_full_untruncated_labels(tmp_path):
    upstream = tmp_path / "helper.py"
    upstream.write_text("""
def tokenize_chat(example, tokenizer, max_length=None):
    return tokenizer(example, max_length)
""")
    data = fixture_data()

    def tokenizer(example, max_length):
        return {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, 4, 5]}

    audit = original_tokenizer_verification(data, tokenizer, upstream)
    assert audit["checked_rows"] == 336 and audit["passed"]
    data["evaluation"][0]["completion_tokens"] = 1
    with pytest.raises(ValueError, match="Completion mask mismatch"):
        original_tokenizer_verification(data, tokenizer, upstream)


def test_upstream_tokenization_rejects_any_truncation(tmp_path):
    upstream = tmp_path / "helper.py"
    upstream.write_text("""
def tokenize_chat(example, tokenizer, max_length=None):
    return tokenizer(example, max_length)
""")

    def tokenizer(example, max_length):
        return {
            "input_ids": [1, 2, 3, 4, 5] if max_length is None else [1, 2, 3, 4],
            "labels": [-100, -100, -100, 4, 5],
        }

    with pytest.raises(ValueError, match="Whole-completion tokenization mismatch"):
        original_tokenizer_verification(fixture_data(), tokenizer, upstream)
