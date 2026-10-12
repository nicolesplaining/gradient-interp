import numpy as np

from gradient_interp.audit import error_metrics, sign_metrics


def test_pooled_error_can_hide_typical_batch_error():
    actual = np.array([[[100.0, 100.0], [1.0, 1.0], [1.0, 1.0]]])
    predicted = np.array([[[100.0, 100.0], [0.0, 0.0], [0.0, 0.0]]])
    result = error_metrics(predicted, actual)
    assert result["pooled_relative_error"]["value"] < 0.02
    assert result["per_seed_batch_error_quantiles"]["p50"] == 1


def test_sign_accuracy_must_be_compared_to_constant_baseline():
    actual = np.array([1.0] * 99 + [-1.0, 0.0])
    result = sign_metrics(np.ones_like(actual), actual)
    assert result["sign_accuracy"] == result["always_positive_accuracy"] == 0.99
    assert result["balanced_sign_accuracy"] == 0.5
    assert result["excluded_below_floor"] == 1
