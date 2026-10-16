import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_collect as collect


def test_completion_loss_ignores_prompt_padding_and_weights_documents_equally():
    logits = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [999.0, -999.0]],
            [[0.0, 1.0], [1.0, 0.0], [-999.0, 999.0], [-999.0, 999.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, -100, 1, 0], [-100, 0, -100, -100]])
    losses = collect.per_answer_loss(logits, labels)
    expected = torch.stack(
        [
            (
                torch.nn.functional.cross_entropy(logits[0, 1:2], torch.tensor([1]))
                + torch.nn.functional.cross_entropy(logits[0, 2:3], torch.tensor([0]))
            )
            / 2,
            torch.nn.functional.cross_entropy(logits[1, 0:1], torch.tensor([0])),
        ]
    )
    torch.testing.assert_close(losses, expected)
    losses.mean().backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 3]) == 0
    assert torch.count_nonzero(logits.grad[1, 1:]) == 0
    assert torch.count_nonzero(logits.grad[0, 2]) > 0  # EOS is scored.


def test_empty_or_misaligned_answer_fails():
    with pytest.raises(ValueError, match="no scored"):
        collect.per_answer_loss(torch.zeros(1, 4, 3), torch.full((1, 4), -100))
    with pytest.raises(ValueError, match="dimensions"):
        collect.per_answer_loss(torch.zeros(1, 4, 3), torch.zeros(1, 3))


def test_packing_scores_completion_and_eos_only(monkeypatch):
    class Tokenizer:
        pad_token_id = 9

    monkeypatch.setattr(
        collect,
        "encoded",
        lambda _, prompt, answer: ([1, 2, 3, 9], 2) if prompt == "long" else ([2, 4, 9], 1),
    )
    ids, mask, labels = collect.packed(
        Tokenizer(), [{"prompt": "long", "answer": "x"}, {"prompt": "short", "answer": "y"}], "cpu"
    )
    assert ids.tolist() == [[1, 2, 3, 9], [2, 4, 9, 9]]
    assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert labels.tolist() == [[-100, -100, 3, 9], [-100, 4, 9, -100]]


def test_independent_adamw_step_and_zero_b_invariance(monkeypatch):
    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
            self.b = torch.nn.Parameter(torch.zeros(2, 2))

    def losses(model, tokenizer, rows):
        return ((model.b @ model.a - torch.tensor([[1.0, 0.0], [0.0, -1.0]])) ** 2).mean(1)

    monkeypatch.setattr(collect, "losses", losses)
    model = TinyAdapter()
    a = model.a.detach().clone()
    first = collect.one_step(model, None, [], 0.0002)
    after = model.b.detach().clone()
    assert torch.equal(model.a, a)
    assert torch.count_nonzero(after) > 0
    assert first["optimizer_steps"] == 1
    with torch.no_grad():
        model.b.zero_()
    collect.one_step(model, None, [], 0.0002)
    assert torch.equal(model.b, after)  # The previous Adam moments were not retained.


def test_nonfinite_gradient_stops_before_update(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()
    monkeypatch.setattr(collect, "losses", lambda m, t, r: m.weight.flatten() * np.nan)
    with pytest.raises(ValueError, match="Nonfinite training"):
        collect.one_step(model, None, [], 0.0002)
    assert torch.equal(before, model.weight)
