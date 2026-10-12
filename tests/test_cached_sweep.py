import torch

from gradient_interp.cached_sweep import compressed_coordinates


def test_cached_projection_preserves_units_and_recovers_sparse_signal():
    values = torch.tensor([[0.0, 3.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]])
    scale = torch.tensor([2.0, 3.0, 4.0, 5.0])
    variants, costs = compressed_coordinates(values, scale, torch.eye(4), seed=11)
    torch.testing.assert_close(variants["svd_4"], values)
    torch.testing.assert_close(variants["dictionary_omp_1"], values)
    torch.testing.assert_close(variants["dictionary_omp_4"], values)
    assert costs["dictionary_omp_4"] == 24
    assert costs["svd_4"] == 16
