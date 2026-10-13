import numpy as np

from gradient_interp.paper_batch_analysis import (
    bootstrap_mean,
    bootstrap_weights,
    endpoint,
    metrics,
)


def test_contrasts_separate_format_verbosity_and_cancel_common_shift():
    # Common shift 11, numbered increment 3, long increment 7.
    values = np.array([[11, 14, 18, 21]], dtype=float)
    np.testing.assert_allclose(endpoint(values, np.ones_like(values), "summed_logp"), [[3, 7]])
    np.testing.assert_allclose(
        endpoint(values + 100, np.ones_like(values), "summed_logp"), [[3, 7]]
    )
    probs = endpoint(np.zeros((2, 4)), np.ones((2, 4)), "four_choice_probability")
    np.testing.assert_allclose(probs, 0.5)
    np.testing.assert_allclose(
        endpoint(values, np.ones_like(values), "four_choice_probability"),
        endpoint(values + 100, np.ones_like(values), "four_choice_probability"),
    )


def test_crossed_bootstrap_keeps_style_balance_and_pairs_constant_differences():
    styles = ["a", "b", "c", "d"] * 4
    weights = bootstrap_weights(styles, 16, repetitions=100)
    np.testing.assert_allclose(weights[0].sum(1), 1)
    np.testing.assert_allclose(weights[1].sum(1), 1)
    for style in set(styles):
        np.testing.assert_allclose(weights[0][:, np.array(styles) == style].sum(1), 0.25)
    np.testing.assert_allclose(bootstrap_mean(np.full((16, 16), -3), weights), -3)


def test_prediction_error_baselines_and_calibration():
    actual = np.arange(1, 9, dtype=float).reshape(2, 4)
    assert metrics(np.zeros_like(actual), actual)["nrmse"] == 1
    assert metrics(actual, actual)["nrmse"] == 0
    result = metrics(actual * 2, actual)
    assert result["nrmse"] == 1
    assert result["magnitude_ratio"] == 2
    assert result["actual_on_prediction_slope"] == 0.5
