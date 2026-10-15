import json

import pytest

from gradient_interp.loracle_toy_suite_llm_judge import (
    counts,
    load_locked,
    response_hash,
    semantic_match,
    summarize,
)
from gradient_interp.study import file_hash


def test_sentence_credit_is_based_on_locked_meaning_not_letter_format():
    assert semantic_match(["json"], "specific", "json") == "full"
    assert semantic_match(["capitals"], "specific", "capitals") == "full"
    assert semantic_match(["json"], "specific", "numbered") == "none"
    assert semantic_match([], "vague", "sort") == "none"
    assert semantic_match([], "abstain", "sort") == "none"


def test_contradictions_and_alternatives_are_partial_not_full():
    assert semantic_match(["uppercase", "bullets"], "ambiguous", "uppercase") == "partial"
    assert semantic_match(["sort"], "ambiguous", "sort") == "partial"
    assert semantic_match(["sort"], "ambiguous", "reverse") == "none"
    assert semantic_match(["capitals", "diets"], "ambiguous", None) is None
    assert semantic_match(["numbered"], "specific", None) is None


def make_locked(tmp_path):
    response = "The model learned to return labels as a JSON array."
    predictions = [
        {"case": "c-first", "encoding": "released", "query": "choice", "response": response}
    ]
    cases = {
        "c-first": {
            "window": "first",
            "target": "json",
            "trajectory": "c",
            "seed": 1,
            "options": {"A": "json", "B": "numbered"},
        }
    }
    digest = response_hash(response)
    artifacts = {
        "rubric.json": {
            "created_unix": 1,
            "statuses": ["specific", "ambiguous", "vague", "other", "abstain"],
        },
        "deck.json": [
            {
                "id": "J000",
                "response": response,
                "query": "choice",
                "options": cases["c-first"]["options"],
            }
        ],
        "mapping.json": [
            {
                "id": "J000",
                "row_index": 0,
                "case": "c-first",
                "encoding": "released",
                "query": "choice",
                "response_sha256": digest,
            }
        ],
        "judgments.json": [
            {
                "id": "J000",
                "labels": ["json"],
                "status": "specific",
                "rationale": "Explicit JSON array.",
                "response_sha256": digest,
            }
        ],
    }
    for name, value in artifacts.items():
        (tmp_path / name).write_text(json.dumps(value))
    lock = {
        "created_unix": 2,
        "n_judgments": 1,
        "files": {name: file_hash(tmp_path / name) for name in artifacts},
    }
    (tmp_path / "judgments-lock.json").write_text(json.dumps(lock))
    return predictions, cases


def test_locked_sentence_and_row_identity_roundtrip(tmp_path):
    predictions, cases = make_locked(tmp_path)
    rows = load_locked(tmp_path, predictions, cases)
    assert rows[0]["response"] == predictions[0]["response"]
    assert rows[0]["match"] == "full" and rows[0]["rationale"] == "Explicit JSON array."
    cases["c-first"]["target"] = "numbered"
    # The same semantic judgment must not adapt to the target.
    assert load_locked(tmp_path, predictions, cases)[0]["match"] == "none"


def test_judgment_edits_after_lock_are_rejected(tmp_path):
    predictions, cases = make_locked(tmp_path)
    p = tmp_path / "judgments.json"
    p.write_text(p.read_text().replace('"json"', '"numbered"'))
    with pytest.raises(ValueError, match="Locked judge artifact changed"):
        load_locked(tmp_path, predictions, cases)


def test_changed_response_or_options_cannot_be_scored(tmp_path):
    predictions, cases = make_locked(tmp_path)
    predictions[0]["response"] += " Perhaps."
    with pytest.raises(ValueError, match="Deck text differs"):
        load_locked(tmp_path, predictions, cases)
    predictions, cases = make_locked(tmp_path)
    cases["c-first"]["options"]["A"] = "reverse"
    with pytest.raises(ValueError, match="Choice options changed"):
        load_locked(tmp_path, predictions, cases)


def test_missing_predictions_or_mapping_identity_rejected(tmp_path):
    predictions, cases = make_locked(tmp_path)
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        load_locked(tmp_path, [], cases)
    predictions[0]["encoding"] = "correct_svd"
    with pytest.raises(ValueError, match="Mapping identity"):
        load_locked(tmp_path, predictions, cases)


def test_open_and_assisted_scores_separate_weak_and_failed_cases_retained():
    rows = []
    for query in ("choice", "description"):
        for i, match in enumerate(("full", "partial", "none")):
            rows.append(
                {
                    "query": query,
                    "encoding": "released",
                    "window": "first",
                    "seed": i,
                    "target": "json",
                    "match": match if query == "choice" else "none",
                    "status": "specific" if i == 0 else "ambiguous" if i == 1 else "abstain",
                    "labels": ["json"] if i < 2 else [],
                    "strong_manipulation": i != 0,
                }
            )
    rows.append(
        {
            "query": "description",
            "encoding": "released",
            "window": "zero",
            "target": None,
            "match": None,
            "status": "other",
            "labels": [],
        }
    )
    result = summarize(rows)
    choice, description = result["groups"][0], result["groups"][4]
    assert choice["n"] == 3 and choice["full"] == 1 and choice["partial"] == 1
    assert choice["any_target"] == 2 and choice["strong"]["n"] == 2
    assert choice["strong"]["full"] == 0 and description["full"] == 0
    zero = next(
        r
        for r in result["controls"]
        if r["window"] == "zero" and r["encoding"] == "released" and r["query"] == "description"
    )
    assert zero["any_content_claim"] == 1 and zero["single_task_accuracy"] is None
    assert counts([])["n"] == 0
