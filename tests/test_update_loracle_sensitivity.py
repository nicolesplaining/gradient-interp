import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_sensitivity as sensitivity
from gradient_interp.update_loracle_data import NO_CHANGE, PATTERNS
from gradient_interp.update_loracle_physical_profile import profile_label


def test_calibration_selects_only_first_unique_training_probes():
    def probe(i):
        return {"prompt": str(i), "values": ["abcd", "z", "ef", "ghijk"]}

    data = {
        "cases": [
            {"split": "development", "probes": [probe("forbidden")]},
            {"split": "train", "probes": [probe(0), probe(0), probe(1)]},
            {"split": "reserved", "probes": [probe("also_forbidden")]},
            {"split": "train", "probes": [probe(i) for i in range(2, 10)]},
        ]
    }
    rows = sensitivity.select_calibration(data)
    assert [r["prompt"] for r in rows] == list(map(str, range(8)))
    for pattern in PATTERNS:
        full = sensitivity.calibration_rows(rows, pattern)
        assert len(full) == 8
        assert [r["prompt"] for r in full] == list(map(str, range(8)))


def test_calibration_gradient_averages_all_docs_and_keeps_parameters(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
            self.b = torch.nn.Parameter(torch.zeros(4, 2))

    model = Model()
    monkeypatch.setattr(sensitivity, "module_parameters", lambda _: {"m": (model.a, model.b)})
    monkeypatch.setattr(
        sensitivity,
        "losses",
        lambda m, t, rows: torch.stack([r["value"] * (m.b @ m.a).sum() + 1 for r in rows]),
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    g, scores = sensitivity.gradient_b(model, None, [{"value": i + 1} for i in range(8)])
    expected = model.a.detach().sum(dim=1)[None, :].repeat(4, 1) * 4.5
    torch.testing.assert_close(g["m"], expected, rtol=0, atol=0)
    assert scores == [1.0] * 8
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    assert model.a.grad is None and model.b.grad is None


def test_native_contraction_equals_physical_weight_derivative_with_scaling_once():
    rng = torch.Generator().manual_seed(44)
    a = torch.randn(3, 5, generator=rng)
    b = torch.randn(7, 3, generator=rng)
    physical_gradient = torch.randn(7, 5, generator=rng)
    # PEFT alpha/rank=2 enters dL/dB through the forward model.
    gb = 2 * physical_gradient @ a.T
    initial = {"m": (a, torch.zeros_like(b))}
    snapshots = [{"m": (a, b)}, {"m": (a, -b)}, {"m": (a, 3 * b)}, initial]
    result = sensitivity.contract({"m": gb}, snapshots, initial, device="cpu")
    expected = float((physical_gradient.double() * (2 * b.double() @ a.double())).sum())
    np.testing.assert_allclose(result, [expected, -expected, 3 * expected, 0], rtol=2e-6, atol=2e-6)
    with pytest.raises(ValueError, match="native"):
        sensitivity.contract({"m": gb}, [{"m": (a + 1, b)}], initial, device="cpu")


def test_centered_profile_has_correct_sign_and_invariant_common_offset():
    changes = np.arange(32, dtype=float) / 100
    profile = sensitivity.centered_profile(changes)
    assert profile.argmax() == 0
    np.testing.assert_allclose(profile, sensitivity.centered_profile(changes + 7), atol=1e-15)
    assert profile_label(sensitivity.centered_profile(np.zeros(32))) == NO_CHANGE
    assert profile.mean() == pytest.approx(0, abs=1e-16)
    with pytest.raises(ValueError):
        sensitivity.centered_profile([0] * 31)


def test_matched_shuffle_uses_donor_profile_and_recipient_norm():
    cases = [
        {"id": "a", "control": False, "physical_update_norm": 2.0, "initialization_seed": 1},
        {"id": "b", "control": False, "physical_update_norm": 6.0, "initialization_seed": 2},
        {"id": "zero", "control": True, "physical_update_norm": 0.0, "initialization_seed": 1},
    ]
    values = np.array([np.arange(32), -3 * np.arange(32), np.zeros(32)], dtype=float)
    shuffled, names = sensitivity.matched_shuffle(values, cases)
    assert names == ["b", "a", "zero"]
    np.testing.assert_array_equal(shuffled, [values[1] / 3, values[0] * 3, values[2]])


def test_nonfinite_gradient_and_wrong_initial_B_rejected():
    initial = {"m": (torch.zeros(2, 3), torch.zeros(4, 2))}
    with pytest.raises(ValueError):
        sensitivity.contract(
            {"m": torch.full((4, 2), float("nan"))}, [initial], initial, device="cpu"
        )
    bad = {"m": (initial["m"][0], torch.ones(4, 2))}
    with pytest.raises(ValueError):
        sensitivity.contract({"m": torch.zeros(4, 2)}, [bad], bad, device="cpu")
