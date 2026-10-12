import numpy as np

from gradient_interp.analysis import crossed_ratio


def test_crossed_ratio_preserves_pairing_and_sign():
    denominator = np.array([[1.0, 200.0, 3.0], [40.0, 5.0, 600.0]])
    result = crossed_ratio(-2 * denominator, denominator)
    assert result["value"] == -2.0
    np.testing.assert_allclose(result["ci95"], [-2, -2])
    result = crossed_ratio(4 * denominator, denominator, square_root=True)
    assert result["value"] == 2.0
    np.testing.assert_allclose(result["ci95"], [2, 2])
