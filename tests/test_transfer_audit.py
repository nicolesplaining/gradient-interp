import numpy as np
import pytest

from gradient_interp.transfer_audit import error_decomposition, probe_variance_diagnostic


def test_decomposition_preserves_cancellation_and_common_shift():
    observed = np.array([[[1.0, -1.0, 0.0]]])
    fixed = observed + np.array([[[0.2, -0.2, 0]]])
    calibration = fixed + np.array([[[0.3, -0.3, 0]]])
    # Compression cancels both other errors. Discarding cross terms is wrong.
    result = error_decomposition(observed + 17, calibration, fixed, observed)
    assert result["normalized_squared_error"] == 0
    assert result["reconstruction_squared_error"] == pytest.approx(0, abs=1e-14)
    assert result["max_additive_residual"] < 1e-14
    assert sum(result["term_squared_norms"].values()) > 0
    assert sum(result["twice_cross_products"].values()) < 0


def test_probe_variance_matches_independent_gaussian_simulation():
    rng = np.random.default_rng(76)
    a = rng.normal(size=(1, 20000, 3, 8)) * np.array([1, 2, 3])[None, None, :, None]
    b = rng.normal(size=a.shape) * np.array([1, 2, 3])[None, None, :, None]
    result = probe_variance_diagnostic(a, b)
    assert result["observed_over_plugin"] == pytest.approx(1, abs=0.025)


def test_zero_signal_is_not_reported_as_a_good_forecast():
    with pytest.raises(ValueError, match="zero observed"):
        error_decomposition(*([np.ones((1, 2, 3))] * 4))
