import pytest
import torch

from gradient_interp.data import Example
from gradient_interp.feature_updates import (
    capture_linear_example,
    feature_update,
    unwhiten_sae_decoder,
)
from gradient_interp.models import LanguageModel, ModelConfig
from gradient_interp.residual import TopKSAE, Whitening


def test_feature_update_matches_autograd_with_nonuniform_token_loss():
    torch.manual_seed(19)
    inputs = torch.randn(7, 5, dtype=torch.float64)
    weight = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    output = inputs @ weight.T
    loss = (output.sin().square() * torch.arange(1, 8)[:, None]).sum() / 37
    adjoints, weight_gradient = torch.autograd.grad(loss, (output, weight))
    update = feature_update(
        inputs, adjoints, torch.eye(3, dtype=inputs.dtype), inputs.new_zeros(3), 0.017
    )
    torch.testing.assert_close(update.weight_delta(), -0.017 * weight_gradient)
    torch.testing.assert_close(sum(update.component(i) for i in range(4)), update.weight_delta())
    probe_inputs = torch.randn(9, 5, dtype=inputs.dtype)
    probe_adjoints = torch.randn(9, 3, dtype=inputs.dtype)
    direct = (probe_adjoints * (probe_inputs @ update.weight_delta().T)).sum()
    torch.testing.assert_close(update.forecast(probe_inputs, probe_adjoints), direct)


def test_identical_feature_totals_can_produce_opposite_probe_effects():
    # Counts and marginal activations are unchanged; pairing flips the effect.
    inputs = torch.tensor([[1.0], [-1.0]])
    codes = torch.tensor([[1.0], [0.0]])
    directions, offset = torch.ones(1, 1), torch.zeros(1)
    paired = feature_update(inputs, codes, directions, offset, 1)
    permuted = feature_update(
        inputs, codes, directions, offset, 1, pairing="permuted", order=torch.tensor([1, 0])
    )
    independent = feature_update(inputs, codes, directions, offset, 1, pairing="independent")
    assert paired.weight_delta().item() == -1
    assert permuted.weight_delta().item() == 1
    assert independent.weight_delta().item() == 0


def test_decoder_restores_whitening_and_sae_offsets():
    torch.manual_seed(8)
    data = torch.randn(100, 4) * torch.tensor([1, 3, 6, 12]) + 5
    whitening = Whitening.fit(data)
    sae = TopKSAE(4, 9, 3)
    with torch.no_grad():
        sae.center.copy_(torch.randn(4))
        codes = sae.encode(whitening.encode(data))
        expected = whitening.decode(sae(whitening.encode(data)))
        directions, offset = unwhiten_sae_decoder(sae, whitening)
    torch.testing.assert_close(codes @ directions + offset, expected, atol=1e-5, rtol=1e-5)
    inputs = torch.randn(100, 6)
    update = feature_update(inputs, codes, directions, offset, 0.2)
    torch.testing.assert_close(update.weight_delta(), -0.2 * expected.T @ inputs)


def test_rejects_a_shuffle_that_repeats_or_drops_tokens():
    with pytest.raises(ValueError, match="permutation"):
        feature_update(
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.ones(1, 1),
            torch.zeros(1),
            1,
            pairing="permuted",
            order=torch.tensor([0, 0]),
        )


def test_capture_includes_prompt_gradients_and_preserves_model():
    # A later attention block lets completion loss flow into prompt positions.
    torch.manual_seed(11)
    backend = LanguageModel(ModelConfig(width=8, max_length=48))
    backend.model.layers.append(
        torch.nn.TransformerEncoderLayer(8, 2, 16, dropout=0, batch_first=True, norm_first=True)
    )
    backend.model.eval()
    before = {k: v.clone() for k, v in backend.model.state_dict().items()}
    example = Example("capture", "ab cd", " ef gh", "test")
    data = capture_linear_example(backend, example, "layers.0.linear2")
    torch.testing.assert_close(
        data["adjoints"].T @ data["inputs"], data["weight_gradient"], atol=2e-7, rtol=1e-5
    )
    _, start, _ = backend.encode(example.prompt, example.completion)
    assert data["adjoints"][:start].norm() > 0
    for key, tensor in backend.model.state_dict().items():
        torch.testing.assert_close(tensor, before[key], atol=0, rtol=0)
    assert not backend.model.layers[0].linear2._forward_hooks
