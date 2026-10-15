import pytest
import torch

from gradient_interp.loracle import low_rank_svd
from gradient_interp.loracle_toy_suite import delta_inputs, initialize_like
from gradient_interp.loracle_toy_suite_data import (
    SEEDS,
    TASKS,
    build_data,
    check_response,
    parse_choice,
)


def test_grid_and_no_leaked_answers():
    data = build_data()
    assert data == build_data()
    assert len(data["trajectories"]) == 24 and len(data["cases"]) == 97
    for seed in SEEDS:
        selected = [t for t in data["trajectories"] if t["seed"] == seed]
        assert sorted(t["tasks"][0] for t in selected) == sorted(TASKS)
        assert sorted(t["tasks"][1] for t in selected) == sorted(TASKS)
    for task, splits in data["tasks"].items():
        assert len(splits["train"]) == 40 and len(splits["evaluation"]) == 16
        assert not set(r["prompt"] for r in splits["train"]) & set(
            r["prompt"] for r in splits["evaluation"]
        )
        for r in splits["train"] + splits["evaluation"]:
            assert check_response(r["answer"], r)["target"]
            assert not check_response("I cannot answer.", r)["target"]
            if task in ["capitals", "diets"]:
                assert r["answer"].casefold() not in r["prompt"].casefold()
            else:
                assert list(reversed(r["values"])) != sorted(r["values"])
                assert r["values"] != sorted(r["values"])
    assert len({tuple(c["options"].values()) for c in data["cases"]}) > 90
    assert len({t["id"] for t in data["trajectories"]}) == 24
    for t in data["trajectories"]:
        for order in t["row_order"].values():
            assert len(order) == 80 and all(order.count(i) == 2 for i in range(40))


def test_checks_separate_content_from_style():
    data = build_data()
    row = data["tasks"]["numbered"]["evaluation"][0]
    bullet = data["tasks"]["bullets"]["evaluation"][0]
    assert check_response(bullet["answer"], row)["content"]
    assert not check_response(bullet["answer"], row)["target"]
    assert not check_response(row["answer"] + "\n4. extra", row)["target"]
    fact = data["tasks"]["diets"]["evaluation"][0]
    assert not check_response("It might eat blue moss or silver berries.", fact)["target"]


def test_choice_abstention_and_invalid_retained():
    options = build_data()["cases"][0]["options"]
    assert parse_choice(" A. ", options) == options["A"]
    assert parse_choice("I think A or B", options) is None
    assert parse_choice("J", options) is None
    letter = next(k for k, v in options.items() if v == "unknown")
    assert parse_choice(letter, options) == "unknown"


def test_initialization_repeatable_zero_effect():
    snapshot = {"00:q_proj": (torch.randn(2, 7), torch.randn(6, 2))}
    first, same, second = [initialize_like(snapshot, seed) for seed in [1, 1, 2]]
    assert torch.equal(first["00:q_proj"][0], same["00:q_proj"][0])
    assert not torch.equal(first["00:q_proj"][0], second["00:q_proj"][0])
    assert first["00:q_proj"][1].count_nonzero() == 0


def test_native_and_adjacent_matrix_conventions():
    torch.manual_seed(1)
    a, b = torch.randn(2, 7), torch.randn(6, 2)
    before = {"00:q_proj": (a, torch.zeros_like(b))}
    after = {"00:q_proj": (a + 0.2, b)}
    native = delta_inputs(before, after, True)["00:q_proj"]
    assert native[0].shape[0] == 2
    assert torch.equal(native[0], after["00:q_proj"][0])
    later = {"00:q_proj": (a - 0.7, b + 0.4)}
    factors = delta_inputs(after, later, False)["00:q_proj"]
    assert factors[0].shape[0] == 4
    u, s, v = low_rank_svd(*factors)
    torch.testing.assert_close(
        (u * s) @ v.T,
        later["00:q_proj"][1] @ later["00:q_proj"][0]
        - after["00:q_proj"][1] @ after["00:q_proj"][0],
    )
    with pytest.raises(ValueError, match="zero-effect"):
        delta_inputs(after, later, True)


def test_exact_generation_counts_with_shared_prompts():
    data = build_data()
    base = [r for t in TASKS for r in data["tasks"][t]["evaluation"]] + data["controls"]
    records = len(base) * 3
    unique = len(set(r["prompt"] for r in base)) * 3
    for t in data["trajectories"]:
        rows = [r for name in t["tasks"] for r in data["tasks"][name]["evaluation"]] + data[
            "controls"
        ]
        records += len(rows) * 2
        unique += len(set(r["prompt"] for r in rows)) * 2
    assert records == 2328 and unique == 1512
