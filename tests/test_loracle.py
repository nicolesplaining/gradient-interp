import torch

from gradient_interp.loracle import (
    MODULES,
    effective_delta_factors,
    low_rank_svd,
    native_direction_tokens,
)


def test_effective_delta_includes_both_factors_and_cross_term():
    torch.manual_seed(4)
    a, b = torch.randn(3, 7), torch.randn(5, 3)
    after_a, after_b = a + torch.randn_like(a), b + torch.randn_like(b)
    left, right = effective_delta_factors(a, b, after_a, after_b, 2.0)
    expected = 2 * (after_b @ after_a - b @ a)
    torch.testing.assert_close(right @ left, expected)
    u, singular, v = low_rank_svd(left, right)
    torch.testing.assert_close((u * singular) @ v.T, expected, atol=2e-5, rtol=2e-5)


def test_svd_delta_is_invariant_to_lora_factorization():
    torch.manual_seed(5)
    a, b = torch.randn(2, 8), torch.randn(6, 2)
    q = torch.linalg.qr(torch.randn(2, 2)).Q
    u1, s1, v1 = low_rank_svd(a, b)
    u2, s2, v2 = low_rank_svd(q.T @ a, b @ q)
    torch.testing.assert_close(s1, s2)
    torch.testing.assert_close((u1 * s1) @ v1.T, (u2 * s2) @ v2.T, atol=2e-6, rtol=2e-5)


def test_native_token_order_and_missing_modules():
    factors = {(0, "q_proj"): (torch.randn(2, 8), torch.randn(4, 2))}
    tokens, metadata = native_direction_tokens(factors, n_layers=2, width=8, top_k=3)
    assert tokens.shape == (2 * len(MODULES) * 3, 8)
    assert [(row["layer"], row["module"], row["rank"]) for row in metadata[:4]] == [
        (0, "q_proj", 0),
        (0, "q_proj", 1),
        (0, "q_proj", 2),
        (0, "k_proj", 0),
    ]
    assert tokens[:2].norm() > 0
    assert torch.equal(tokens[2:], torch.zeros_like(tokens[2:]))
