import json

import torch

from gradient_interp.config import ExperimentConfig
from gradient_interp.data import Example, make_demo_data
from gradient_interp.models import LanguageModel, ModelConfig
from gradient_interp.residual import TopKSAE, Whitening, collect_chunks, run_residual_comparison


def test_whitening_round_trip():
    torch.manual_seed(8)
    values = torch.randn(30, 6) * torch.arange(1, 7) + 2
    whitening = Whitening.fit(values)
    torch.testing.assert_close(
        whitening.decode(whitening.encode(values)), values, rtol=2e-5, atol=2e-5
    )


def test_topk_enforces_feature_budget():
    sae = TopKSAE(width=6, n_features=20, k=3)
    encoded = sae.encode(torch.randn(10, 6))
    assert (encoded >= 0).all()
    assert ((encoded > 0).sum(1) <= 3).all()
    torch.testing.assert_close(sae.decoder.weight.norm(dim=0), torch.ones(20))


def test_residual_chunks_align_with_completion_and_do_not_change_weights():
    torch.manual_seed(2)
    backend = LanguageModel(ModelConfig(width=8))
    before = backend.space.fingerprint()
    gradients, activations, chunks = collect_chunks(
        backend, [Example("a", "A long prompt", "abcdefg")], "layers.0", chunk_size=3
    )
    assert [chunk["text"] for chunk in chunks] == ["abc", "def", "g"]
    assert gradients.shape == activations.shape == (3, 8)
    assert gradients.norm() > 0
    assert backend.space.fingerprint() == before
    assert not backend.model.layers[0]._forward_hooks


def test_sae_comparison_uses_document_holdout(tmp_path):
    make_demo_data(tmp_path / "data")
    config = ExperimentConfig(
        fit_path=str(tmp_path / "data/fit.jsonl"), threads=1, model=ModelConfig(width=8)
    )
    metrics = run_residual_comparison(
        config, tmp_path / "run", "layers.0", n_features=16, k=3, epochs=3
    )
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    for name in ["gradient", "activation"]:
        assert metrics[name]["raw_fvu"] is not None
        assert metrics[name]["mean_l0"] <= 3
        features = [
            json.loads(line)
            for line in (tmp_path / f"run/{name}_features.jsonl").read_text().splitlines()
        ]
        assert all(
            chunk["example_id"] not in manifest["validation_example_ids"]
            for feature in features
            for chunk in feature["top_chunks"]
        )
