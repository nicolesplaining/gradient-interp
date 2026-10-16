from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from gradient_interp.update_loracle_train import (
    MAX_LENGTH,
    installed_hook,
    learning_rate,
    option_token_ids,
    prepared_input,
    shuffled_directions,
    verified_cases,
)


def test_injection_stays_installed_through_checkpoint_backward():
    torch.manual_seed(4)
    layer = torch.nn.Linear(4, 3)
    inputs = torch.randn(2, 4, requires_grad=True)
    calls = []

    def hook(module, inputs, output):
        calls.append(1)
        return output + output.norm(dim=-1, keepdim=True) * 0.2

    with installed_hook(layer, hook):
        expected = layer(inputs).square().mean()
        expected.backward()
    gradient = layer.weight.grad.clone()
    layer.zero_grad(set_to_none=True)
    inputs.grad = None
    calls.clear()
    with installed_hook(layer, hook):
        actual = checkpoint(layer, inputs, use_reentrant=False).square().mean()
        actual.backward()
    assert len(calls) == 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer.weight.grad, gradient)
    assert not layer._forward_hooks


def test_hook_removed_on_failure():
    layer = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError):
        with installed_hook(layer, lambda m, i, o: o):
            assert len(layer._forward_hooks) == 1
            raise RuntimeError("failed step")
    assert not layer._forward_hooks


def test_choice_tokenization_does_not_silently_score_only_part_of_number():
    good = SimpleNamespace(encode=lambda number, **kwargs: [1000 + int(number)])
    assert option_token_ids(good) == list(range(1000, 1033))
    bad = SimpleNamespace(encode=lambda number, **kwargs: list(number))
    with pytest.raises(ValueError, match="single tokens"):
        option_token_ids(bad)
    duplicate = SimpleNamespace(encode=lambda number, **kwargs: [10])
    with pytest.raises(ValueError, match="single tokens"):
        option_token_ids(duplicate)


def test_input_contains_only_question_not_answer_and_rejects_truncation():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "only the question"}]
            assert kwargs["add_generation_prompt"] and not kwargs["enable_thinking"]
            return "chat"

        def __call__(self, text, **kwargs):
            assert text == "chat" and not kwargs["add_special_tokens"]
            return {"input_ids": [8, 9]}

    handle = SimpleNamespace(
        config=SimpleNamespace(use_system_prompt=False),
        tokenizer=Tokenizer(),
        prefix_ids=[7] * 4480,
        prefix_mask=[True] * 4480,
        device="cpu",
    )
    ids, mask = prepared_input(handle, "only the question")
    assert ids.shape == (1, 4482) and mask.sum() == 4480
    assert ids[0, -2:].tolist() == [8, 9]
    assert not mask[0, -2:].any()
    handle.prefix_ids += [1] * MAX_LENGTH
    with pytest.raises(ValueError, match="never truncate"):
        prepared_input(handle, "only the question")


def test_scheduler_warms_then_decays_without_restart():
    values = [learning_rate(s, 10, 2, 1.0) for s in range(10)]
    assert values[:3] == [0.5, 1.0, 1.0]
    assert values[-1] == 0.125
    assert all(a > b for a, b in zip(values[2:], values[3:]))
    with pytest.raises(ValueError):
        learning_rate(10, 10, 2, 1.0)


def test_no_case_access_before_successful_collection_audit(tmp_path):
    (tmp_path / "integrity.json").write_text('{"passed":false}')
    with pytest.raises(ValueError, match="complete audited"):
        verified_cases(tmp_path, "train")
    (tmp_path / "integrity.json").write_text(
        '{"passed":true,"scope":"development_only","files_sha256":{}}'
    )
    with pytest.raises(ValueError, match="reserved test"):
        verified_cases(tmp_path, "reserved")


def test_direction_shuffle_never_moves_questions_norms_or_labels():
    cases = [
        {
            "id": str(i),
            "tokens": f"file-{i}",
            "control": i == 4,
            "text": f"options-{i}",
            "physical_update_norm": i + 1,
            "label": f"label-{i}",
        }
        for i in range(5)
    ]
    moved = shuffled_directions(cases)
    assert moved == shuffled_directions(cases)
    assert sorted(c["tokens"] for c in moved) == sorted(c["tokens"] for c in cases)
    for a, b in zip(cases, moved):
        for key in ("id", "control", "text", "physical_update_norm", "label"):
            assert a[key] == b[key]
        assert (a["tokens"] == b["tokens"]) == a["control"]
