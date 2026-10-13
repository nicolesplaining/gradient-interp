import numpy as np

from gradient_interp.paper_sample import estimators
from gradient_interp.paper_sample_analysis import bootstrap, resampled_forecasts, stats


def test_uniform_resampling_reproduces_locked_estimators():
    rng = np.random.default_rng(37)
    unit = rng.normal(size=(2, 6, 3, 64))
    rewards = rng.integers(0, 2, (3, 64, 4))
    rates = np.array([0.003, 0.03])
    logw = unit[:, None] * rates[None, :, None, None, None]
    weights = np.exp(logw - logw.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    got = resampled_forecasts(unit, rewards, weights, np.full((3, 64), 1 / 64), rates)
    expected = estimators(unit, rewards, rates)
    np.testing.assert_allclose(got, np.stack(expected[:2]), atol=1e-14)


def test_resampling_keeps_pairing_and_constant_effects_across_all_axes():
    cache = {"unit_rate_logp_changes": np.zeros((16, 6, 16, 64)), "rewards": np.ones((16, 64, 4))}
    observed = np.ones((16, 3, 16, 32, 8))
    baseline = np.zeros((16, 32, 8))
    mse, actual, predicted = bootstrap(
        cache, observed, baseline, [0.0003, 0.003, 0.03], repetitions=3
    )
    assert mse.shape == (3, 2, 2, 3, 6, 4)
    np.testing.assert_allclose(mse, 1)
    np.testing.assert_allclose(actual, 1)
    np.testing.assert_allclose(predicted, 0)


def test_extreme_reweighting_handles_bootstrap_omitting_original_mode():
    unit = np.array([[[[0.0, -10000.0]]]])
    rewards = np.array([[[1.0], [0.0]]])
    weights = np.array([[[[[1.0, 0.0]]]]])
    predictions = resampled_forecasts(
        unit, rewards, weights, np.array([[0.0, 1.0]]), np.array([1.0])
    )
    np.testing.assert_allclose(predictions, 0)
    assert stats(np.zeros((2, 3)), np.zeros((2, 3)))["nrmse"] is None
