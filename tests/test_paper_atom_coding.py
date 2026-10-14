import numpy as np
import pytest
from sklearn.linear_model import orthogonal_mp

from gradient_interp.paper_atom_coding import Coder, geometry, omp, ratio, refit, sensitivity


def test_support_refit_removes_shrinkage_without_adding_atoms():
    d = np.eye(5)
    x = np.array([[0.7, 0.6, 0.05, 0.02, 0.0]])
    c = Coder(d).solve(x)
    np.testing.assert_allclose(c[0, 0], [0.6, 0.5, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(c[0, 1], [0.7, 0.6, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(c[0, 3], x[0], atol=1e-12)
    loss = [np.sum((x[0] - a @ d)**2) for a in c[0]]
    assert loss[0] > loss[1] > loss[3]
    assert (loss[0]-loss[1]) + (loss[1]-loss[3]) + loss[3] == pytest.approx(loss[0])


def test_dense_span_rank_deficiency_and_inaccessible_residual():
    d = np.array([[1., 0, 0], [1., 0, 0], [0., 1, 0]])
    x = np.array([[2., 3, 4]])
    coder = Coder(d)
    assert coder.rank == 2
    c = x @ coder.inverse
    np.testing.assert_allclose(c, [[1, 1, 3]], atol=1e-12)
    np.testing.assert_allclose(d @ (x[0] - c[0] @ d), 0, atol=1e-12)
    np.testing.assert_allclose(refit(d, x[0], [0, 1]), [1, 1, 0], atol=1e-12)


def test_omp_matches_independent_solver_and_is_scale_invariant():
    rng = np.random.default_rng(5)
    d = rng.normal(size=(12, 40))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    x = rng.normal(size=40)
    expected = orthogonal_mp(d.T, x, n_nonzero_coefs=4, precompute=False)
    actual = omp(d, x, count=4)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    scaled = omp(d * np.arange(1, 13)[:, None], x, count=4)
    np.testing.assert_allclose(scaled * np.arange(1, 13), actual, atol=1e-12)
    assert np.count_nonzero(actual) == 4


def test_omp_ties_and_zero_signal():
    d = np.eye(3)
    np.testing.assert_array_equal(omp(d, np.ones(3), count=1), [1, 0, 0])
    np.testing.assert_array_equal(omp(d, np.zeros(3)), np.zeros(3))
    with pytest.raises(ValueError):
        omp(np.zeros((2, 3)), np.ones(3))


def test_empty_support_stays_zero_and_fixed_support_is_not_reselected():
    d = np.eye(3)
    x = np.array([[1., 1, 0], [0, 1., 1]])
    result = Coder(d).solve(x, supports=[[], [0]])
    np.testing.assert_array_equal(result[:, 1], 0)
    assert np.any(result[:, 0] != 0)


def test_norm_restoration_geometry_and_undefined_values():
    d = np.eye(3)
    x, c = np.array([3., 4, 0]), np.array([0., 2, 0])
    a, b = geometry(x, c, d), geometry(x*9, c*9, d)
    for key in ('relative_residual', 'relative_sse', 'norm_ratio', 'target_gain', 'cosine'):
        assert a[key] == pytest.approx(b[key])
    assert a['norm_ratio'] == pytest.approx(0.4)
    assert b['coefficient_l2'] == pytest.approx(9*a['coefficient_l2'])
    assert np.isnan(ratio(1, 0))
    assert np.isnan(geometry(x, np.zeros(3), d)['cosine'])


def test_paired_sensitivity_handles_support_and_zero_baseline():
    d = np.eye(3)
    result = sensitivity(np.array([1., 0, 0]), np.array([1., .001, 0]), d, .001)
    assert result['coefficient_amplification'] == pytest.approx(1)
    assert result['reconstruction_amplification'] == pytest.approx(1)
    assert result['support_jaccard'] == .5
    zero = sensitivity(np.zeros(3), np.zeros(3), d, .001)
    assert zero['support_jaccard'] == 1
    assert np.isnan(zero['coefficient_relative_change'])
