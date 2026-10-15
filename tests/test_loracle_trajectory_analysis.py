import json

import pytest

from gradient_interp.loracle_trajectory_analysis import directional_labels, summarize
from gradient_interp.loracle_trajectory_toy import INTERVALS, STYLES


def fixture():
    behavior = []
    for step, style in [(0, "plain_text"), (10, "numbered_list"), (20, "bulleted_list")]:
        for i in range(24):
            behavior.append(
                {
                    "step": step,
                    "id": f"eval-{i:02d}",
                    "checks": {
                        **{s: s == style for s in STYLES},
                        "content_order": True,
                        "empty": False,
                        "truncated": False,
                    },
                }
            )
        for i in range(8):
            behavior.append(
                {"step": step, "id": f"control-{i:02d}", "checks": {"exact_word": True}}
            )
    predictions = []
    truth = {
        "case-k": ("numbered_list", "plain_text"),
        "case-r": ("bulleted_list", "numbered_list"),
        "case-v": ("bulleted_list", "plain_text"),
        "case-z": ("unknown", "unknown"),
    }
    for case, (more, less) in truth.items():
        for convention in ["correct_svd", "upstream_compatible"]:
            predictions.append(
                {
                    "case": case,
                    "query": "structured",
                    "convention": convention,
                    "response": json.dumps({"more": more, "less": less}),
                }
            )
    return behavior, predictions


def test_threshold_and_ties_are_unknown():
    assert directional_labels(dict(zip(STYLES, [0.19, 0, -0.19]))) == {
        "more": "unknown",
        "less": "unknown",
    }
    assert directional_labels(dict(zip(STYLES, [0.5, 0.5, -0.5]))) == {
        "more": "unknown",
        "less": "plain_text",
    }


def test_clear_transition_requires_correct_descriptions_and_zero_control():
    behavior, predictions = fixture()
    result = summarize(behavior, predictions)
    assert result["primary_toy_support"]
    predictions[0]["response"] = '{"more":"bulleted_list","less":"unknown"}'
    assert not summarize(behavior, predictions)["primary_toy_support"]
    behavior, predictions = fixture()
    predictions[-2]["response"] = '{"more":"numbered_list","less":"unknown"}'
    assert not summarize(behavior, predictions)["zero_control_correct"]


def test_content_failure_and_incomplete_grid_do_not_pass():
    behavior, predictions = fixture()
    for row in behavior:
        if row["step"] == 20 and row["id"] in ["eval-00", "eval-01"]:
            row["checks"]["content_order"] = False
    result = summarize(behavior, predictions)
    assert result["manipulation_check"] and not result["quality_check"]
    assert not result["primary_toy_support"]
    with pytest.raises(ValueError, match="Incomplete behavior"):
        summarize(behavior[:-1], predictions)


def test_cumulative_prediction_cannot_rescue_adjacent_failure():
    behavior, predictions = fixture()
    for row in predictions:
        if row["case"] in INTERVALS and row["case"] != "case-v":
            row["response"] = "not JSON"
    assert not summarize(behavior, predictions)["primary_adjacent_predictions_correct"]
