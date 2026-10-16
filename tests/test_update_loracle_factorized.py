import copy
from collections import defaultdict

import numpy as np
import pytest

from gradient_interp import update_loracle_factorized as f
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE


def cases():
    # Deliberately imbalanced while retaining every component class.
    labels = list(LABELS) + [LABELS[0]] * 23 + [NO_CHANGE] * 3
    return [
        {"id": f"train-{i}", "tokens": f"file-{i}", "physical_update_norm": 1.0, "label": y}
        for i, y in enumerate(labels)
    ]


def test_component_assembly_preserves_all_original_labels_and_no_change():
    for label in LABELS:
        parts = f.target_components(label)
        if label == NO_CHANGE:
            assert parts == {"change": NO_CHANGE}
            parts.update(format="json", case="upper", order="length")
        assert f.join_components(parts) == label
    with pytest.raises(ValueError, match="Every component"):
        f.join_components({"change": NO_CHANGE})


def test_training_weights_balance_every_class_and_every_head():
    rows, summary = f.training_instances(cases())
    totals = defaultdict(float)
    heads = defaultdict(float)
    for row in rows:
        totals[row["head"], row["target_label"]] += row["weight"]
        heads[row["head"]] += row["weight"]
    assert summary["weight_sum"] == pytest.approx(len(rows))
    for head, labels in f.HEAD_LABELS.items():
        assert heads[head] == pytest.approx(len(rows) / 4)
        assert [totals[head, label] for label in labels] == pytest.approx(
            [len(rows) / (4 * len(labels))] * len(labels)
        )
    by_id = {c["id"]: c for c in cases()}
    for row in rows:
        if by_id[row["case_id"]]["label"] == NO_CHANGE:
            assert row["head"] == "change"
        symbol = f.OPTION_SYMBOLS[row["target_option"]]
        assert row["options"][symbol] == row["target_label"]


def test_missing_training_component_is_rejected_instead_of_infinite_weight():
    rows = [c for c in cases() if c["label"] == NO_CHANGE or "/upper/" not in c["label"]]
    with pytest.raises(ValueError, match="training support"):
        f.training_instances(rows)


def test_questions_do_not_use_measured_target_or_training_text():
    class Case(dict):
        def __getitem__(self, key):
            assert key not in ("label", "probes", "batch", "source_pattern")
            return super().__getitem__(key)

    case = Case(
        id="input",
        tokens="weights",
        physical_update_norm=1.25,
        label="PRIVATE",
        probes="PRIVATE",
        batch="PRIVATE",
        source_pattern="PRIVATE",
    )
    views = f.question_views(case)
    assert len(views) == 4 and "PRIVATE" not in str(views)
    assert all("1.25" in row["text"] for row in views)
    assert all("0.02" in row["text"] for row in views if row["head"] == "change")
    changed = {**case, "label": "different target", "physical_update_norm": 99.0}
    assert [r["options"] for r in views] == [r["options"] for r in f.question_views(changed)]


def test_option_orders_are_deterministic_but_independently_varied():
    maps = [f.option_map(str(i), "format") for i in range(30)]
    assert maps == [f.option_map(str(i), "format") for i in range(30)]
    assert len({tuple(m.values()) for m in maps}) > 10
    assert all(
        set(m) == set("ABCD") and set(m.values()) == set(f.HEAD_LABELS["format"]) for m in maps
    )


def test_forward_constrains_to_actual_head_ids_and_keeps_training_scale(monkeypatch):
    instances, _ = f.training_instances(cases())
    row = next(r for r in instances if r["head"] == "case")
    calls = []

    def base_forward(handle, tokens, text, target, scale):
        calls.append((target, scale))
        values = [0.0] * 33
        values[1] = 2
        values[25] = 10  # This symbol is not a valid answer to a two-choice head.
        return {"choice_logits": values, "loss": 2.5, "hook_calls": 2, "input_tokens": 4800}

    monkeypatch.setattr(f, "choice_forward", base_forward)
    result = f.forward(None, None, row, train=True, loss_scale=0.375)
    assert result["choice_id"] == "B" and result["prediction"] == row["options"]["B"]
    assert result["choice_logits"] == [0.0, 2.0] and len(result["full_33_choice_logits"]) == 33
    assert calls == [(row["target_option"], 0.375)]
    f.forward(None, None, row)
    assert calls[-1] == (None, 1.0)
    bad = copy.deepcopy(row)
    bad["target_option"] = 25
    with pytest.raises(ValueError, match="displayed answer"):
        f.forward(None, None, bad, train=True)


def test_majority_baseline_is_from_training_counts_and_stable():
    rows = cases()
    _, balance = f.training_instances(rows)
    result = f.training_majorities(rows, balance)
    assert result["flat"] == LABELS[0] == result["factorized"]
    assert sum(result["label_counts"].values()) == len(rows)


def test_invalid_magnitudes_and_incomplete_head_options_are_not_silently_accepted():
    for norm in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="norm"):
            f.question("case", f.option_map("x", "case"), norm)
    with pytest.raises(ValueError, match="Incomplete"):
        f.question("case", {"A": "lower"}, 1.0)
