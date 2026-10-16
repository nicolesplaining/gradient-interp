import copy

import numpy as np
import pytest

from gradient_interp.update_loracle_data import (
    LABELS,
    NO_CHANGE,
    PATTERN_IDS,
    PATTERNS,
    SEEDS,
    build_data,
    effect_label,
    pattern_split,
    question,
    render,
    validate,
)


@pytest.fixture(scope="module")
def data():
    return build_data()


def test_full_grid_and_disjoint_compositions(data):
    audit = validate(data)
    assert audit["split_counts"] == {"train": 272, "development": 72, "reserved": 136}
    splits = {s: {p for p in PATTERNS if pattern_split(p) == s} for s in SEEDS}
    assert [len(splits[s]) for s in SEEDS] == [16, 8, 8]
    for split in splits:
        for axis, n in enumerate((4, 2, 4)):
            assert len({p[axis] for p in splits[split]}) == n
        for other in splits:
            if split != other:
                assert not splits[split] & splits[other]
                assert not set(SEEDS[split]) & set(SEEDS[other])
    assert data == build_data()


def test_actual_rendering():
    words = ["zen", "ab", "tulipa", "coda"]
    assert render(words, ("plain", "lower", "reverse")) == "coda, tulipa, ab, zen"
    assert render(words, ("numbered", "upper", "length")) == "1. AB\n2. ZEN\n3. CODA\n4. TULIPA"
    assert render(words, ("json", "lower", "alphabetical")) == '["ab", "coda", "tulipa", "zen"]'


def test_effect_is_measured_not_source_label():
    before = np.full((4, 32), 2.0)
    after = before.copy()
    after[:, 17] -= 0.2
    result = effect_label(before, after)
    assert result["label"] == PATTERN_IDS[17]
    assert result["span"] == pytest.approx(0.2)
    assert result["top_two_margin"] == pytest.approx(0.2)
    assert sum(result["centered_improvements"]) == pytest.approx(0)
    assert effect_label(before, after - 0.7)["label"] == result["label"]
    assert effect_label(before, before - 0.7)["label"] == NO_CHANGE
    assert effect_label(before, before)["label"] == NO_CHANGE


def test_weak_and_ambiguous_effects_retained():
    before = np.full((4, 32), 2.0)
    after = before.copy()
    after[:, 1:3] -= 0.1
    result = effect_label(before, after)
    assert result["label"] == PATTERN_IDS[1]
    assert result["top_two_margin"] == 0
    after = before.copy()
    after[:, 8] -= 0.001
    assert effect_label(before, after)["below_effect_floor"]


@pytest.mark.parametrize("bad", [np.zeros((0, 32)), np.zeros((4, 31)), np.full((4, 32), np.nan)])
def test_bad_measurements_fail(bad):
    with pytest.raises(ValueError):
        effect_label(bad, bad)


def test_input_has_only_options_and_norm(data):
    case = data["cases"][0]
    text = question(case["options"], 1.25)
    assert "1.25" in text and len(case["options"]) == len(LABELS) == 33
    assert case["id"] not in text
    assert case["batch"][0]["prompt"] not in text
    assert case["source_pattern"] not in text
    with pytest.raises(ValueError):
        question(case["options"], float("nan"))


def test_prompt_and_split_leaks_rejected(data):
    changed = copy.deepcopy(data)
    changed["cases"][0]["probes"][0] = changed["cases"][0]["batch"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate(changed)
    changed = copy.deepcopy(data)
    changed["cases"][0]["initialization_seed"] = SEEDS["reserved"][0]
    with pytest.raises(ValueError, match="cross-split"):
        validate(changed)
