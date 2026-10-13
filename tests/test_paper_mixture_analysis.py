import numpy as np

from gradient_interp.paper_mixture import BANKS, weighted_estimators
from gradient_interp.paper_mixture_analysis import nested_counts, resampled_bank


def test_nested_bootstrap_preserves_component_counts_and_shared_subsets():
    c = nested_counts(np.random.default_rng(7), 3, 2)
    for bank, (idx, _) in BANKS.items():
        np.testing.assert_array_equal(c[..., idx].sum(-1), len(idx))
        if bank.startswith("mixture"):
            for component in (0, 1):
                subset = [i for i in idx if i // 256 == component]
                np.testing.assert_array_equal(c[..., subset].sum(-1), len(idx) / 2)


def test_bootstrap_reproduces_estimators_and_explicit_resampled_data():
    rng = np.random.default_rng(16)
    u = rng.normal(size=(2, 3, 2, 8))
    y = rng.normal(size=(2, 8, 4))
    target, warm = rng.normal(size=(2, 8)), rng.normal(size=(2, 8))
    rates = [0.03, 0.3]
    original = weighted_estimators(u, y, target, warm, rates, range(8), "mixture")
    result = resampled_bank(u, y, original["log_base_weights"], rates, np.ones((1, 2, 8)))
    np.testing.assert_allclose(
        result[0], np.stack([original["linear"], original["snis"]]), atol=1e-12
    )
    idx = [0, 0, 2, 3, 3, 5, 6, 6]
    c = np.broadcast_to(np.bincount(idx, minlength=8), (1, 2, 8))
    got = resampled_bank(u, y, original["log_base_weights"], rates, c)[0]
    expected = weighted_estimators(
        u[..., idx], y[:, idx], target[:, idx], warm[:, idx], rates, range(8), "mixture"
    )
    np.testing.assert_allclose(got, np.stack([expected["linear"], expected["snis"]]), atol=1e-12)


def test_omitted_dominant_mode_does_not_underflow():
    got = resampled_bank(
        np.array([[[[0.0, -10000.0]]]]),
        np.array([[[1.0], [0.0]]]),
        np.zeros((1, 2)),
        [1.0],
        np.array([[[0.0, 2.0]]]),
    )
    np.testing.assert_allclose(got, 0)
