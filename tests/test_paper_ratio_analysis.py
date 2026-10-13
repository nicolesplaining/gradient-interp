import numpy as np

from gradient_interp.paper_ratio import direct_reweight
from gradient_interp.paper_ratio_analysis import bootstrap, importance_predictions


def test_uniform_proposals_match_direct_reweighting_and_shift_invariance():
    rng = np.random.default_rng(48)
    ratios = rng.normal(size=(2, 3, 4, 64))
    rewards = rng.integers(0, 2, size=(4, 64, 4))
    got = importance_predictions(ratios, rewards)
    expected = np.array([[direct_reweight(r, rewards)["snis_change"] for r in b] for b in ratios])
    np.testing.assert_allclose(got, expected, atol=1e-15)
    np.testing.assert_allclose(importance_predictions(ratios + 1000, rewards), got, atol=1e-12)


def test_bootstrap_pairs_direct_and_taylor_when_ratios_are_identical():
    rng = np.random.default_rng(51)
    unit = rng.normal(size=(16, 6, 16, 64))
    rates = np.array([0.0003, 0.003, 0.03])
    ratios = unit[:, None, 0] * rates[None, :, None, None]
    cache = {"unit_rate_logp_changes": unit, "rewards": rng.integers(0, 2, (16, 64, 4))}
    observed = rng.integers(0, 2, (16, 3, 16, 32, 8))
    base = rng.integers(0, 2, (16, 32, 8))
    mse, _, means = bootstrap(cache, ratios, observed, base, rates, reps=3)
    np.testing.assert_allclose(mse[..., 0, :], mse[..., 1, :], atol=1e-15)
    np.testing.assert_allclose(means[..., 0, :], means[..., 1, :], atol=1e-15)
