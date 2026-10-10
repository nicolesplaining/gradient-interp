import pytest
import torch

from gradient_interp.dictionary import Projection, SparseDictionary, sparse_code
from gradient_interp.forecast import AtomForecaster, prediction_metrics


def test_projection_preserves_zero_and_inverts_whitening():
    torch.manual_seed(3)
    gradients = torch.randn(24, 8) * torch.arange(1, 9)
    projection = Projection.fit(gradients, rank=8, whiten=True)
    assert torch.equal(projection.encode(torch.zeros(8)), torch.zeros(8))
    torch.testing.assert_close(
        projection.decode(projection.encode(gradients)), gradients, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(projection.encode(gradients).square().mean(0), torch.ones(8))


def test_rank_deficiency_does_not_create_whitening_noise():
    gradients = torch.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [3.0, 6.0, 0.0]])
    projection = Projection.fit(gradients, rank=3)
    assert projection.basis.shape == (3, 1)
    torch.testing.assert_close(projection.decode(projection.encode(gradients)), gradients)
    with pytest.raises(ValueError, match="zero gradients"):
        Projection.fit(torch.zeros(4, 3), 2)


@pytest.mark.parametrize("algorithm", ["ksvd", "minibatch"])
def test_sparse_codes_retain_sign_and_update_magnitude(algorithm):
    dictionary = SparseDictionary(torch.eye(3), 2, [], algorithm=algorithm, alpha=0.05)
    value = torch.tensor([0.3, -0.4, 0.0])
    code = dictionary.encode(value)
    assert code[0] > 0 and code[1] < 0 and code[2] == 0
    torch.testing.assert_close(dictionary.encode(-7 * value), -7 * code)
    torch.testing.assert_close(dictionary.encode(torch.zeros(3)), torch.zeros(3))


def test_omp_recovers_known_sparse_signal():
    atoms = torch.eye(5)
    values = torch.tensor([[0.0, 2.0, 0.0, -4.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(sparse_code(values, atoms, 2), values)


def test_ksvd_learns_sparse_data():
    torch.manual_seed(2)
    values = torch.eye(4).repeat(10, 1) * torch.randn(40, 1)
    dictionary = SparseDictionary.fit(values, 4, 1, 8, algorithm="ksvd")
    reconstructed = dictionary.decode(dictionary.encode(values))
    assert (reconstructed - values).norm() / values.norm() < 0.05


def test_calibration_predicts_linear_effects_and_atom_removal():
    projection = Projection(torch.eye(3), torch.tensor([2.0, 0.5, 4.0]))
    dictionary = SparseDictionary(torch.eye(3), 3, [])
    forecaster = AtomForecaster(projection, dictionary)
    jacobian = torch.tensor([[1.0, 2.0, -3.0], [2.0, -1.0, 0.0]])
    forecaster.calibrate(lambda update: jacobian @ update, radius=0.01)
    update = torch.tensor([-0.3, 0.1, 0.2])
    forecast = forecaster.predict(update, target_probe=0)
    torch.testing.assert_close(forecast.predicted, jacobian @ update)
    torch.testing.assert_close(forecast.reconstructed_update, update)
    j = forecast.selected_atom
    removed = forecast.amplitudes[j] * forecaster.directions[j]
    torch.testing.assert_close(
        jacobian @ update - jacobian @ (update - removed), forecast.contributions[j]
    )


def test_metrics_do_not_invent_correlation_or_score_zero_effect_signs():
    metrics = prediction_metrics([0.0, 0.0], [0.0, 0.0])
    assert metrics["pearson"] is None
    assert metrics["sign_accuracy"] is None
    assert metrics["rmse"] == 0
