import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from gradient_interp.loracle import MODULES, effective_delta_factors
from gradient_interp.loracle_fresh_toy_suite import (
    configure_trainables,
    frozen_snapshot,
    restore_frozen,
)
from gradient_interp.loracle_fresh_toy_suite_data import build_data, validate_data
from gradient_interp.loracle_trajectory_toy import fingerprint, restore_modules, snapshot_modules
from gradient_interp.loracle_wide_toy_suite import delta_inputs
from gradient_interp.loracle_wide_toy_suite_data import build_data as old_data


def test_exact_inputs_and_questions_with_only_b_training():
    new, old = build_data(), old_data()
    validate_data(new)
    for key in ("tasks", "controls", "trajectories"):
        assert new[key] == old[key]
    previous = {c["id"]: c for c in old["cases"]}
    assert len(new["cases"]) == 73
    assert len({c["id"] for c in new["cases"]}) == 73
    for c in new["cases"]:
        assert c == previous[c["id"]]
    assert sum(len(t["row_order"][t["tasks"][1]]) // 4 for t in new["trajectories"]) == 2400
    assert len(new["cases"]) * 4 == 292
    rows = [
        r
        for t in new["trajectories"]
        for task in t["tasks"]
        for r in new["tasks"][task]["evaluation"]
    ]
    assert (len(rows) + 24 * 8) * 3 == 2880
    generations = (
        sum(
            len(
                {r["prompt"] for task in t["tasks"] for r in new["tasks"][task]["evaluation"]}
                | {r["prompt"] for r in new["controls"]}
            )
            for t in new["trajectories"]
        )
        * 3
    )
    assert generations == 2016


@pytest.mark.parametrize("change", ["order", "answer", "question", "steps"])
def test_data_mutations_are_rejected(change):
    d = build_data()
    if change == "order":
        t = d["trajectories"][0]
        t["row_order"][t["tasks"][1]].reverse()
    elif change == "answer":
        d["tasks"]["diets"]["train"][0]["answer"] = "changed"
    elif change == "question":
        d["cases"][0]["question"] += " hint"
    else:
        d["segment_b_steps"] = 200
    with pytest.raises(ValueError):
        validate_data(d)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module() for _ in range(40)])
        for layer in self.layers:
            layer.self_attn, layer.mlp = nn.Module(), nn.Module()
            for i, name in enumerate(MODULES):
                setattr(layer.self_attn if i < 4 else layer.mlp, name, nn.Linear(8, 8, bias=False))

    def forward(self, x):
        layer = self.layers[0]
        return layer.self_attn.q_proj(x) + layer.mlp.down_proj(x)


def test_native_additive_boundary_training_and_restoration():
    torch.manual_seed(7)
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0, target_modules=list(MODULES))
    model = get_peft_model(TinyModel(), config)
    initial = snapshot_modules(model)
    # A nonzero prior adapter, not the trivial zero-plus-zero boundary.
    old = {k: (a, torch.randn_like(b) / 10) for k, (a, b) in initial.items()}
    restore_modules(model, old)
    x = torch.randn(3, 8)
    before = model(x).detach().clone()
    model.add_adapter("segment_a", copy.deepcopy(config))
    params = configure_trainables(model)
    restore_frozen(model, old)
    restore_modules(model, initial)
    torch.testing.assert_close(model(x), before, rtol=0, atol=0)
    old_hash = fingerprint(frozen_snapshot(model))
    optimizer = torch.optim.AdamW(params, lr=0.02, weight_decay=0)
    assert not optimizer.state
    loss = model(x).square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    assert fingerprint(frozen_snapshot(model)) == old_hash
    assert fingerprint(snapshot_modules(model)) != fingerprint(initial)
    restore_modules(model, initial)
    torch.testing.assert_close(model(x), before, rtol=0, atol=0)
    assert fingerprint(frozen_snapshot(model)) == old_hash
    assert all(
        (".default." in n) == p.requires_grad for n, p in model.named_parameters() if ".lora_" in n
    )


def test_native_b_represents_exact_increment_without_old_a_or_rank32():
    torch.manual_seed(17)
    a = torch.randn(4, 8)
    b = torch.randn(7, 4)
    initial = {"x": (torch.randn_like(a), torch.zeros_like(b))}
    after = {"x": (a, b)}
    native = delta_inputs(initial, after, True)["x"]
    assert native[0].shape[0] == 4
    torch.testing.assert_close(native[1] @ native[0], b @ a)
    old_a, old_b = torch.randn_like(a), torch.randn_like(b)
    dense_increment = (old_b @ old_a + b @ a) - old_b @ old_a
    torch.testing.assert_close(native[1] @ native[0], dense_increment, rtol=2e-5, atol=2e-6)
    factors = effective_delta_factors(old_a, old_b, a, b, 1.0)
    assert factors[0].shape[0] == 8
    with pytest.raises(ValueError):
        delta_inputs({"x": (old_a, old_b)}, after, True)


def audit_module():
    path = Path("scripts/audit_loracle_fresh_toy_suite.py")
    spec = importlib.util.spec_from_file_location("fresh_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("state", ["running", "failed", "interrupted", "budget_exhausted"])
def test_audit_does_not_release_early_outputs(tmp_path, state):
    import json

    (tmp_path / "controller.json").write_text(json.dumps({"status": state}))
    with pytest.raises(ValueError, match="remain sealed"):
        audit_module().completion_barrier(tmp_path)
