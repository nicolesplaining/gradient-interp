import copy
import json
import random
import re
from pathlib import Path

import pytest
import torch

from gradient_interp.paper_setup import select_docs, validate_factors, validate_layout


def paper_config():
    return json.loads(Path("configs/paper_setup.json").read_text())


def adapter_parameters():
    # Shapes derive independently from Gemma-3 4B: hidden2560, 8 Q heads,
    # 4 KV heads, head_dim256; rank8 adapters on Q/V in34 layers.
    result = []
    for layer in range(34):
        for projection, out_width in (("q_proj", 2048), ("v_proj", 1024)):
            prefix = f"base_model.model.model.language_model.layers.{layer}.self_attn.{projection}"
            result.append(
                (prefix + ".lora_A.default.weight", torch.nn.Parameter(torch.empty(8, 2560)))
            )
            result.append(
                (prefix + ".lora_B.default.weight", torch.nn.Parameter(torch.empty(out_width, 8)))
            )
    result.append(
        (
            "base_model.model.lm_head.weight",
            torch.nn.Parameter(torch.empty(2, 2), requires_grad=False),
        )
    )
    return result


def test_paper_layout_is_all_layer_qv_not_vision_or_full_projection_lora():
    config = paper_config()
    parameters = adapter_parameters()
    layout = validate_layout(parameters, config)
    assert len(layout) == 136
    assert sum(row["numel"] for row in layout) == 2228224
    for row in layout:
        target = row["name"].split(".lora_")[0]
        assert re.fullmatch(config["target_modules"], target)
    assert not re.fullmatch(
        config["target_modules"], "model.vision_tower.layers.0.self_attn.q_proj"
    )
    assert not re.fullmatch(config["target_modules"], "model.layers.0.self_attn.k_proj")
    with pytest.raises(ValueError, match="every paper"):
        validate_layout(parameters[:-3], config)
    changed = list(parameters)
    name, param = changed[0]
    changed[0] = (name.replace("q_proj", "k_proj"), param)
    with pytest.raises(ValueError, match="Unexpected"):
        validate_layout(changed, config)
    changed = list(parameters)
    changed[-1] = ("base_model.model.lm_head.weight", torch.nn.Parameter(torch.empty(2, 2)))
    with pytest.raises(ValueError, match="Trainable base"):
        validate_layout(changed, config)


def test_sampling_matches_released_order_without_changing_global_rng():
    docs = [{"id": i} for i in range(20)]
    for number in (8, 20):
        random.seed(42)
        expected = random.sample(docs, number) if number < len(docs) else list(docs)
        random.shuffle(expected)
        state = random.getstate()
        assert select_docs(docs, number, 42) == expected
        assert random.getstate() == state
    assert docs == [{"id": i} for i in range(20)]
    with pytest.raises(ValueError):
        select_docs(docs, 21, 42)


def test_factor_guard_prevents_silent_offset_corruption():
    layout = [{"name": "module.weight", "shape": [2, 3]}]
    factors = {
        "module": {
            "act_eigenvalues": torch.ones(3),
            "grad_eigenvalues": torch.ones(2),
            "act_eigenvectors": torch.eye(3),
            "grad_eigenvectors": torch.eye(2),
        }
    }
    validate_factors(layout, factors)
    with pytest.raises(ValueError, match="coverage"):
        validate_factors(layout, {})
    bad = copy.deepcopy(factors)
    bad["module"]["act_eigenvalues"] = torch.ones(2)
    with pytest.raises(ValueError, match="Invalid factor"):
        validate_factors(layout, bad)
    bad = copy.deepcopy(factors)
    bad["module"]["grad_eigenvalues"][0] = float("nan")
    with pytest.raises(ValueError, match="Invalid factor"):
        validate_factors(layout, bad)
