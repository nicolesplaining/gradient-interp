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


def test_frozen_plan_matches_development_controller_contract(tmp_path):
    import json

    from gradient_interp.paper_tangent import freeze
    from gradient_interp.study import atomic_json, file_hash

    atomic_json(
        tmp_path / "specification.json",
        dict(scope="synthetic_development_only", max_hours=12, input_hashes={}),
    )
    atomic_json(
        tmp_path / "preflight.json",
        dict(
            specification_sha256=file_hash(tmp_path / "specification.json"),
            adapter_restored=True,
            sampled_generation_restored=True,
        ),
    )
    atomic_json(tmp_path / "data.json", dict())
    freeze(tmp_path)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["scope"] == "development_only" and plan["max_hours"] == 12
    assert plan["jobs"] == [
        dict(
            id="autoregressive-tangent",
            module="gradient_interp.paper_tangent",
            arguments=["run", "--root", str(tmp_path)],
            manifest=str(tmp_path / "manifest.json"),
        )
    ]
