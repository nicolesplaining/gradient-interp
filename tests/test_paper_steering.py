import pytest
import torch

from gradient_interp.paper_steering import (
    FAMILIES,
    choose_atoms,
    matched_gaussian,
    restore,
    set_direction,
)


def test_adapter_steps_are_from_original_and_restore_exactly():
    parameters = [
        ("a", torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))),
        ("b", torch.nn.Parameter(torch.arange(6.0, 12.0).reshape(3, 2))),
    ]
    original = {name: p.detach().clone() for name, p in parameters}
    direction = torch.linspace(-2, 2, 12)
    before = torch.cat([p.detach().flatten() for _, p in parameters])
    for scale in (0.5, -2.0, 10.0, 0.0):
        set_direction(parameters, original, direction, scale)
        actual = torch.cat([p.detach().flatten() for _, p in parameters])
        torch.testing.assert_close(actual, before - scale * direction, rtol=0, atol=0)
    restore(parameters, original)
    assert all(torch.equal(p, original[name]) for name, p in parameters)
    with pytest.raises(ValueError, match="unconsumed"):
        set_direction(parameters, original, torch.zeros(13), 1)
    restore(parameters, original)


def test_gaussian_control_matches_each_parameter_block_norm():
    reference = torch.tensor([0.0, 0.0, 0.0, 3.0, 4.0, 0.0, -7.0])
    layout = [{"numel": 3}, {"numel": 2}, {"numel": 2}]
    result = matched_gaussian(reference, layout, torch.Generator().manual_seed(19))
    start = 0
    for row in layout:
        end = start + row["numel"]
        torch.testing.assert_close(result[start:end].norm(), reference[start:end].norm())
        start = end
    assert torch.equal(result[:3], torch.zeros(3))
    assert not torch.equal(reference, result)


def test_training_selection_prioritizes_purity_and_keeps_negative_orientation():
    docs = [
        {"messages": [{"role": "assistant", "content": "hit" if i < 20 else "miss"}]}
        for i in range(40)
    ]
    coefficients = torch.ones(40, 3)
    coefficients[:20, 1] = -1
    info = [
        {"atom_idx": 0, "n_active": 40, "coherence": 0.9, "top_doc_indices": list(range(10, 30))},
        {"atom_idx": 1, "n_active": 20, "coherence": 0.2, "top_doc_indices": list(range(20))},
        {"atom_idx": 2, "n_active": 20, "coherence": 0.1, "top_doc_indices": list(range(20))},
    ]
    checks = {family: lambda text: text == "hit" for family in FAMILIES}
    selected = choose_atoms(info, coefficients, docs, checks)
    for family in FAMILIES:
        assert selected[family]["atom_idx"] == 1
        assert selected[family]["orientation"] == -1
        assert selected[family]["training_detector_hits"] == 20
