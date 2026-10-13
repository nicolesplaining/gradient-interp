from types import SimpleNamespace

import pytest
import torch

from gradient_interp.paper_tangent import direction_dict, tangent_logits


class BilinearLoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor([[0.1, 0.5], [-0.2, 0.6]], dtype=torch.float64))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=input_ids.to(torch.float64) @ (self.a @ self.b))


def test_tangent_has_both_lora_derivatives_but_no_quadratic_term():
    model = BilinearLoRA()
    params = list(model.named_parameters())
    original = {n: p.detach().clone() for n, p in params}
    g = torch.arange(1, 9, dtype=torch.float64) / 10
    direction = direction_dict(params, g)
    x = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)
    primal, derivative = tangent_logits(model, x, original, direction)
    expected = x[:, -1] @ (direction["a"] @ original["b"] + original["a"] @ direction["b"])
    torch.testing.assert_close(derivative, expected.float())
    rate = 0.1
    actual = x[:, -1] @ (
        (original["a"] + rate * direction["a"]) @ (original["b"] + rate * direction["b"])
    )
    quadratic = rate**2 * x[:, -1] @ direction["a"] @ direction["b"]
    torch.testing.assert_close(actual.float(), primal + rate * derivative + quadratic.float())
    assert all(torch.equal(p, original[n]) for n, p in params)


def test_direction_length_mismatch_fails():
    model = BilinearLoRA()
    with pytest.raises(AssertionError):
        direction_dict(list(model.named_parameters()), torch.zeros(9))
