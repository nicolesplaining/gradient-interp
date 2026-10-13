import numpy as np

from gradient_interp.paper_sample import estimators, stopped_tokens


def test_sampling_trajectory_includes_first_stop_but_no_padding():
    assert stopped_tokens([7, 8, 106, 0, 0], [1, 106]) == [7, 8, 106]
    assert stopped_tokens([7, 8, 9], [1, 106]) == [7, 8, 9]


def test_snis_matches_binary_exponential_tilt_and_tracks_overlap():
    unit = np.array([[[[-0.5, 0.5] * 32]]])
    rewards = np.array([[[0.0], [1.0]] * 32])
    linear, snis, ess = estimators(unit, rewards, [0.0, 0.1, 2.0])
    expected = 1 / (1 + np.exp(-np.array([0.0, 0.1, 2.0]))) - 0.5
    np.testing.assert_allclose(snis[0, :, 0, 0, 0], expected)
    np.testing.assert_allclose(linear[0, :, 0, 0, 0], np.array([0.0, 0.1, 2.0]) * 0.25 * 64 / 63)
    assert ess[0, 0, 0, 0] == 64
    assert 1 < ess[0, 2, 0, 0] < ess[0, 1, 0, 0] < 64


def test_zero_or_constant_trajectory_changes_predict_no_behavior_change():
    rng = np.random.default_rng(17)
    rewards = rng.integers(0, 2, (3, 64, 4))
    unit = np.ones((2, 6, 3, 64)) * 25
    linear, snis, ess = estimators(unit, rewards, [0.001, 0.1])
    np.testing.assert_allclose(linear, 0, atol=1e-14)
    np.testing.assert_allclose(snis, 0, atol=1e-14)
    np.testing.assert_allclose(ess, 64)
