import numpy as np
import pytest
import torch

from gradient_interp.behavior_pilot import (
    choice_probabilities,
    classify_generation,
    contrasts,
    make_data,
)
from gradient_interp.data import Example
from gradient_interp.feature_updates import capture_linear_example
from gradient_interp.models import LanguageModel, ModelConfig


def test_synthetic_splits_and_paired_batches():
    d = make_data()
    train = {q["id"] for b in d["batches"] for q in b["questions"]}
    cal = {q["id"] for q in d["calibration"]}
    evaluation = {q["id"] for q in d["evaluation"]}
    assert len(d["batches"]) == 32
    assert not (train & cal or train & evaluation or cal & evaluation)
    for i in range(0, 24, 4):
        assert all(
            d["batches"][i]["questions"] == d["batches"][j]["questions"] for j in range(i, i + 4)
        )
    assert {q["template"] for q in d["calibration"]} == {0, 1, 2, 3}
    assert {q["template"] for q in d["evaluation"][16:]} == {4, 5, 6, 7}


def test_contrasts_separate_format_from_correctness_and_common_shift():
    assert contrasts(np.array([2.0, 2.0, 0.0, 0.0])).tolist() == [2, 0]
    assert contrasts(np.array([3.0, 0.0, 3.0, 0.0])).tolist() == [0, 3]
    assert contrasts(np.full(4, 12.0)).tolist() == [0, 0]
    np.testing.assert_allclose(choice_probabilities(np.zeros((5, 4))), 0.5)
    p = choice_probabilities(np.array([1000.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(p, [1, 1])


def test_generation_checker_does_not_confuse_mentioned_values_with_answers():
    q = {"correct": "A10"}
    assert classify_generation(' {"answer": "A10"}\nQuestion: more', q)["correct"]
    assert classify_generation(' {"answer": "B20"}', q)["json"]
    assert not classify_generation(' {"answer": "B20"}', q)["correct"]
    assert not classify_generation("The answer is A10 or B20", q)["correct"]
    assert not classify_generation('```json\n{"answer": "A10"}', q)["json"]


def test_sum_score_forecast_restores_completion_length():
    torch.manual_seed(4)
    backend = LanguageModel(ModelConfig(width=8, max_length=64, trainable_pattern="head.weight"))
    e = Example("test", "Question: code? Answer:", ' {"answer":"A10"}')
    row = capture_linear_example(backend, e, "head", verify_weight_gradient=True)
    length = backend.encode(e.prompt, e.completion)[2]
    delta = torch.randn_like(row["weight_gradient"]) * 1e-3
    forecast = (-row["adjoints"] * length * (row["inputs"] @ delta.T)).sum()
    expected = -(row["weight_gradient"] * delta).sum() * length
    assert forecast.item() == pytest.approx(expected.item(), abs=1e-7)
    with torch.no_grad():
        before = backend.log_probability(e.prompt, e.completion, "sum").item()
    with backend.space.temporary_update(delta.flatten()):
        with torch.no_grad():
            after = backend.log_probability(e.prompt, e.completion, "sum").item()
    assert abs(after - before - forecast.item()) < 1e-4
