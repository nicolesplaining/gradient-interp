import copy

import pytest

from gradient_interp.update_loracle_confirmation_data import fresh_draw, validate_fresh
from gradient_interp.update_loracle_data import build_data


@pytest.fixture(scope="module")
def datasets():
    original = build_data()
    return original, fresh_draw(original)


def test_fresh_draw_is_deterministic_balanced_and_disjoint(datasets):
    original, fresh = datasets
    assert fresh == fresh_draw(original)
    result = validate_fresh(fresh, original)
    assert result == {
        "cases": 136,
        "actual_updates": 128,
        "zero_controls": 8,
        "unique_prompts": 1056,
        "compositions": 8,
        "initialization_seeds": 4,
    }


@pytest.mark.parametrize(
    "defect", ["vocabulary", "seed", "composition", "probe", "answer", "weak_order"]
)
def test_invalid_confirmation_inputs_are_rejected(datasets, defect):
    original, fresh = datasets
    bad = copy.deepcopy(fresh)
    case = bad["cases"][0]
    if defect == "vocabulary":
        bad["vocabulary"][0] = original["vocabularies"]["train"][0]
    elif defect == "seed":
        case["initialization_seed"] = original["cases"][0]["initialization_seed"]
    elif defect == "composition":
        case["source_pattern"] = original["cases"][0]["source_pattern"]
    elif defect == "probe":
        case["probes"][0] = case["probes"][1]
    elif defect == "answer":
        case["batch"][0]["answer"] = "wrong"
    else:
        case["probes"][0]["values"] = ["same"] * 4
    with pytest.raises(ValueError):
        validate_fresh(bad, original)


def test_missing_selection_cannot_create_confirmation_data(tmp_path):
    from gradient_interp.update_loracle_confirmation_data import freeze

    output = tmp_path / "data.json"
    with pytest.raises(FileNotFoundError):
        freeze(tmp_path / "missing-selection.json", output)
    assert not output.exists()
