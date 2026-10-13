import numpy as np

from gradient_interp.paper_mixture_ratio_analysis import resampled_direct


def test_direct_bootstrap_matches_explicit_weighted_sample():
    rng = np.random.default_rng(32)
    y = rng.normal(size=(2, 8, 4))
    l0 = rng.normal(size=(2, 8))
    lw = rng.normal(size=(2, 3, 2, 8))
    idx = [0, 0, 2, 4, 4, 5, 6, 7]
    counts = np.broadcast_to(np.bincount(idx, minlength=8), (1, 2, 8))
    result = resampled_direct(lw, l0, y, counts)[0]
    for b in range(2):
        for r in range(3):
            for q in range(2):
                a = np.exp(lw[b, r, q, idx])
                a /= a.sum()
                z = np.exp(l0[q, idx])
                z /= z.sum()
                np.testing.assert_allclose(result[b, r, q], (a - z) @ y[q, idx], atol=1e-12)


def test_direct_bootstrap_restoration_and_omitted_mode():
    y = np.array([[[1.0], [0.0]]])
    l0 = np.zeros((1, 2))
    c = np.array([[[0.0, 2.0]]])
    got = resampled_direct(np.array([[[[0.0, -10000.0]]]]), l0, y, c)
    np.testing.assert_array_equal(got, 0)
    restored = resampled_direct(l0[None, None], l0, y, np.ones((1, 1, 2)))
    np.testing.assert_array_equal(restored, 0)
