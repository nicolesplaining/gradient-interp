import numpy as np
import pytest

from gradient_interp.feature_controls import connected_batch_groups, leave_batch_and_probe_out


def test_connected_batches_prevent_duplicate_example_leakage():
    def ex(i):
        return {"id": str(i), "prompt": str(i), "completion": "answer"}

    groups = connected_batch_groups(
        [
            {"examples": [ex(1)]},
            {"examples": [ex(2)]},
            {"examples": [ex(1), ex(3)]},
            {"examples": [ex(3), ex(4)]},
        ]
    )
    assert groups[0] == groups[2] == groups[3]
    assert groups[1] != groups[0]


def test_scalar_calibration_recovers_known_amplitude_on_disjoint_folds():
    rng = np.random.default_rng(42)
    prediction = rng.normal(size=(12, 4))
    observed = np.repeat((prediction * 0.3)[..., None], 8, axis=-1)
    result = leave_batch_and_probe_out(prediction, observed, list(range(12)))
    assert result["scaled_error"] < 1e-14
    assert result["scales"] == pytest.approx([0.3] * 24)
    assert result["unscaled_error_same_folds"] > 2


def test_calibration_does_not_fit_held_out_batch_outcomes():
    prediction = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    y = np.repeat(prediction[..., None], 4, axis=-1)
    y[0] *= 10
    result = leave_batch_and_probe_out(prediction, y, [0, 1, 2])
    assert result["scales"][:2] == pytest.approx([1, 1])
