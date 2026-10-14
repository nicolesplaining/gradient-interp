import torch

from gradient_interp.loracle import effective_delta_factors, low_rank_svd
from gradient_interp.loracle_trajectory_toy import (
    answer,
    build_data,
    checks,
    fingerprint,
    parse_prediction,
)


def test_data_is_reproducible_and_content_disjoint():
    data = build_data()
    assert data == build_data()
    assert len(data["train"]) == 40 and len(data["evaluation"]) == 24
    assert not set(data["train_words"]) & set(data["test_words"])
    assert len({x["prompt"] for x in data["train"] + data["evaluation"]}) == 64
    assert {x["template"] for x in data["evaluation"]} == {0, 1, 2, 3}


def test_format_detectors_keep_content_and_reject_repetition():
    row = {"values": ["bronze", "copper", "cotton"]}
    vocabulary = set(row["values"])
    for style in ["numbered_list", "bulleted_list", "plain_text"]:
        result = checks(answer(row["values"], style), row, vocabulary)
        assert result[style] and result["content_order"]
    assert not checks("bronze copper cotton bronze", row, vocabulary)["content_order"]
    assert not checks("cotton copper bronze", row, vocabulary)["content_order"]
    assert not checks("", row, vocabulary)["plain_text"]


def test_parse_keeps_unknown_and_invalid_answers():
    assert parse_prediction('{"more":"unknown","less":"unknown"}') == {
        "more": "unknown",
        "less": "unknown",
    }
    assert parse_prediction('{"more":"bulleted_list","less":"numbered_list"}')
    assert parse_prediction('{"more":"lists","less":"none"}') is None
    assert parse_prediction("I cannot tell") is None


def test_sequential_effective_deltas_telescope_with_nonzero_factors():
    torch.manual_seed(47)
    snapshots = [(torch.randn(4, 12), torch.randn(8, 4)) for _ in range(3)]
    deltas = []
    for before, after in [(0, 1), (1, 2), (0, 2)]:
        a, b = effective_delta_factors(*snapshots[before], *snapshots[after], 2.0)
        u, s, v = low_rank_svd(a, b)
        torch.testing.assert_close((u * s) @ v.T, b @ a, rtol=1e-4, atol=2e-5)
        deltas.append(b @ a)
    torch.testing.assert_close(deltas[0] + deltas[1], deltas[2], atol=1e-5, rtol=1e-5)


def test_fingerprint_detects_mutation():
    x = {"x": (torch.ones(2, 3), torch.zeros(3, 2))}
    before = fingerprint(x)
    x["x"][1][0, 0] = 1
    assert fingerprint(x) != before
