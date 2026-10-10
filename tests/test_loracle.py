from types import SimpleNamespace

import pytest
import torch

from gradient_interp.loracle import (
    MODULES,
    configure_interpreter_prefix,
    effective_delta_factors,
    low_rank_svd,
    native_direction_tokens,
    prepare_checkpoint,
    validate_interpreter_tokens,
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


def test_interpreter_layout_is_rank_first_and_checked_before_loading():
    cfg = SimpleNamespace(
        base_model="model",
        n_direction_tokens=28,
        d_model=3,
        token_dir_kind="svd_fixed_k2_mag7_rankfirst",
        encoder_top_k=2,
        prefix_mode="rank_tagged",
    )
    metadata = {"base_model": "model", "n_layers": 2, "top_k": 2, "d_model": 3}
    tokens = torch.arange(84).reshape(28, 3).float()
    arranged = validate_interpreter_tokens(cfg, metadata, tokens)
    torch.testing.assert_close(arranged[:14], tokens[::2])
    torch.testing.assert_close(arranged[14:], tokens[1::2])
    cfg.encoder_top_k = 4
    with pytest.raises(ValueError, match="rank count"):
        validate_interpreter_tokens(cfg, metadata, tokens)
    cfg.encoder_top_k = 2
    cfg.base_model = "different model"
    with pytest.raises(ValueError, match="same base model"):
        validate_interpreter_tokens(cfg, metadata, tokens)


def test_prepare_public_checkpoint_preserves_source(tmp_path):
    yaml = pytest.importorskip("yaml")
    source = tmp_path / "download"
    (source / "interpreter").mkdir(parents=True)
    original = "base_model: /workspace/models/Qwen3-14B\nd_model: 5120\n"
    (source / "loracle_config.yaml").write_text(original)
    (source / "interpreter/adapter_config.json").write_text("{}")
    destination = tmp_path / "prepared"
    prepare_checkpoint(source, destination, "Qwen/Qwen3-14B")
    assert (source / "loracle_config.yaml").read_text() == original
    assert (destination / "best_ao_interpreter").resolve() == source / "interpreter"
    assert yaml.safe_load((destination / "loracle_config.yaml").read_text())["base_model"] == (
        "Qwen/Qwen3-14B"
    )


def test_mag7_prefix_uses_training_rank_count_not_all14_inference():
    handle = SimpleNamespace(
        tokenizer=object(),
        config=SimpleNamespace(
            n_direction_tokens=4480, prefix_mode="rank_tagged", encoder_top_k=16
        ),
        prefix_ids=[],
        prefix_mask=[],
    )

    def builder(tokenizer, *, n_tokens, prefix_mode, top_k, **kwargs):
        # A rank header followed by its direction slots, as in upstream's builder.
        slots = n_tokens // top_k
        ids = [value for rank in range(top_k) for value in [rank] + [-1] * slots]
        mask = [value == -1 for value in ids]
        return ids, mask, {"top_k": top_k, "slots_per_rank": slots}

    layout = configure_interpreter_prefix(handle, builder)
    assert layout == {"top_k": 16, "slots_per_rank": 280}
    assert sum(handle.prefix_mask) == 4480
    assert [value for value, mask in zip(handle.prefix_ids, handle.prefix_mask) if not mask] == (
        list(range(16))
    )
    with pytest.raises(ValueError, match="number of direction tokens"):
        configure_interpreter_prefix(handle, lambda *args, **kwargs: ([1], [True], {}))
