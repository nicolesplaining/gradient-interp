import pytest
import torch

from gradient_interp.data import Example, Probe
from gradient_interp.models import LanguageModel, ModelConfig


@pytest.fixture
def backend():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    return LanguageModel(ModelConfig(width=8, max_length=64))


def test_batch_gradient_matches_mean_of_unequal_length_examples(backend):
    examples = [Example("short", "A:", " x"), Example("long", "B:", " much longer")]
    independent = torch.stack([backend.gradient([example]) for example in examples]).mean(0)
    torch.testing.assert_close(backend.gradient(examples), independent, rtol=2e-5, atol=2e-6)


def test_update_matches_torch_sgd_and_restores_exactly(backend):
    examples = [Example("a", "Q:", " yes"), Example("b", "Q:", " no")]
    probe = Probe("yes", "Q:", " yes", " no")
    initial = backend.space.vector().clone()
    gradient = backend.gradient(examples)
    predicted_step = backend.updated_scores(-0.01 * gradient, [probe])
    assert torch.equal(initial, backend.space.vector())
    optimizer = torch.optim.SGD([p for _, p in backend.space.named], lr=0.01)
    optimizer.zero_grad()
    for example in examples:
        (backend.loss(example) / len(examples)).backward()
    optimizer.step()
    torch.testing.assert_close(backend.scores([probe]), predicted_step, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(backend.space.vector(), initial - 0.01 * gradient)


def test_restore_on_exception(backend):
    initial = backend.space.fingerprint()
    with pytest.raises(RuntimeError):
        with backend.space.temporary_update(torch.ones(backend.space.size) * 0.1):
            raise RuntimeError("evaluation failed")
    assert backend.space.fingerprint() == initial


def test_prompt_tokens_are_not_scored(backend):
    ids, start, length = backend.encode("long prompt:", " x")
    assert length == 2
    assert ids[0, start + 1 :].tolist() == list(b" x")
    assert start == len(b"long prompt:")


def test_no_silent_truncation(backend):
    with pytest.raises(ValueError, match="max_length"):
        backend.encode("Q", "x" * 100)


def test_exact_and_pair_normalized_sequence_probabilities(backend):
    with torch.no_grad():
        backend.model.head.weight.zero_()
    # A uniform 259-token distribution gives a two-token sequence probability 259^-2.
    exact = Probe("exact", "Q:", "xy", metric="probability")
    pair = Probe("pair", "Q:", "x", "xy", metric="probability")
    torch.testing.assert_close(backend.probe_score(exact), torch.tensor(259.0**-2))
    torch.testing.assert_close(backend.probe_score(pair), torch.tensor(259.0 / 260.0))
