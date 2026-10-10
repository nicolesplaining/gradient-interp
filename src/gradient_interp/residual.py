"""Matched TopK SAEs on chunked residual gradients and residual activations.

TopK architecture: Gao et al. (2024), https://arxiv.org/abs/2406.04093.
Reference implementation inspected at 4965b941e9eb590b00b253a2c406db1e1b193942:
https://github.com/openai/sparse_autoencoder/blob/4965b941e9eb590b00b253a2c406db1e1b193942/sparse_autoencoder/model.py
This is a small TopK comparison, not a replication of Matryoshka JumpReLU or
the large-scale SAE training recipe. Residual gradients alone are not weight
updates; this command does not pretend their atoms can be removed from weights.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Example, read_jsonl, write_jsonl
from .experiment import save_json
from .models import LanguageModel


def resolve_module(model, name):
    matches = [
        (key, module)
        for key, module in model.named_modules()
        if key == name or key.endswith("." + name)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one module matching {name!r}, found {len(matches)}")
    return matches[0]


def collect_chunks(backend, examples, module_name, chunk_size):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    _, module = resolve_module(backend.model, module_name)
    chunks, activations, gradients = [], [], []
    captured = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            raise ValueError("The residual module must return [batch, position, hidden] tensors")
        # A head-only parameter selection can otherwise leave this tensor detached.
        hidden.requires_grad_(True)
        captured.append(hidden)

    handle = module.register_forward_hook(capture)
    try:
        for example_index, example in enumerate(examples):
            captured.clear()
            loss = backend.loss(example)
            if len(captured) != 1:
                raise ValueError("Residual module must execute exactly once per document")
            hidden = captured[0]
            gradient = torch.autograd.grad(loss, hidden)[0]
            _, start, length = backend.encode(example.prompt, example.completion)
            target_ids = backend.tokenizer.encode(example.completion, add_special_tokens=False)
            for offset in range(0, length, chunk_size):
                end = min(offset + chunk_size, length)
                token_ids = target_ids[offset:end]
                text = (
                    bytes(token_ids).decode("utf-8", errors="replace")
                    if backend.config.kind == "tiny"
                    else backend.tokenizer.decode(token_ids)
                )
                chunks.append(
                    {
                        "example_id": example.id,
                        "group": example.group,
                        "start_token": offset,
                        "end_token": end,
                        "text": text,
                    }
                )
                activations.append(hidden[0, start + offset : start + end].detach().mean(0).cpu())
                gradients.append(gradient[0, start + offset : start + end].detach().mean(0).cpu())
            if (example_index + 1) % 100 == 0:
                print(f"Residual documents {example_index + 1}/{len(examples)}", flush=True)
    finally:
        handle.remove()
        backend.model.zero_grad(set_to_none=True)
    return torch.stack(gradients).float(), torch.stack(activations).float(), chunks


@dataclass
class Whitening:
    mean: torch.Tensor
    basis: torch.Tensor
    scales: torch.Tensor

    @classmethod
    def fit(cls, values):
        mean = values.mean(0)
        centered = (values - mean).double()
        covariance = centered.T @ centered / max(len(values) - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        floor = max(float(eigenvalues.max()) * 1e-4, 1e-20)
        return cls(mean, eigenvectors.float(), eigenvalues.clamp_min(floor).sqrt().float())

    def encode(self, values):
        return ((values - self.mean) @ self.basis) / self.scales

    def decode(self, values):
        return (values * self.scales) @ self.basis.T + self.mean


class TopKSAE(nn.Module):
    def __init__(self, width, n_features, k):
        super().__init__()
        if not 1 <= k <= n_features:
            raise ValueError("SAE k must be between 1 and n_features")
        self.k = k
        self.center = nn.Parameter(torch.zeros(width))
        self.encoder = nn.Linear(width, n_features)
        self.decoder = nn.Linear(n_features, width, bias=False)
        with torch.no_grad():
            self.encoder.bias.zero_()
            self.decoder.weight.copy_(self.encoder.weight.T)
        self.normalize_decoder()

    def encode(self, values):
        pre = self.encoder(values - self.center)
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, top.indices, top.values.relu())

    def forward(self, values):
        return self.decoder(self.encode(values)) + self.center

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))


def fit_sae(train, validation, n_features, k, epochs, learning_rate, seed, device="cpu"):
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("SAE epochs and learning rate must be positive")
    torch.manual_seed(seed)
    whitening = Whitening.fit(train)
    x = whitening.encode(train).to(device)
    sae = TopKSAE(x.shape[1], n_features, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        for indices in torch.randperm(len(x), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = (sae(x[indices]) - x[indices]).square().mean()
            loss.backward()
            # Unit decoder columns prevent shrinking activations via decoder growth.
            with torch.no_grad():
                weight, gradient = sae.decoder.weight, sae.decoder.weight.grad
                gradient.sub_(weight * (weight * gradient).sum(0, keepdim=True))
            optimizer.step()
            sae.normalize_decoder()
        history.append(float(loss.detach()))
    sae.eval()
    with torch.no_grad():
        val_white = whitening.encode(validation).to(device)
        reconstructed_white = sae(val_white)
        reconstructed = whitening.decode(reconstructed_white.cpu())

        def fvu(actual, predicted, mean):
            denominator = (actual - mean).square().sum()
            return (
                float((actual - predicted).square().sum() / denominator)
                if denominator > 0
                else None
            )

        codes = sae.encode(val_white)
        metrics = {
            "raw_fvu": fvu(validation, reconstructed, train.mean(0)),
            "whitened_fvu": fvu(val_white, reconstructed_white, x.mean(0)),
            "mean_l0": float((codes > 0).float().sum(1).mean()),
            "dead_features_on_validation": int((codes == 0).all(0).sum()),
            "train_loss": history,
        }
    return sae, whitening, metrics


def run_residual_comparison(
    config,
    output,
    module_name,
    chunk_size=32,
    n_features=128,
    k=8,
    epochs=100,
    learning_rate=0.003,
    validation_fraction=0.2,
    parameters_path=None,
    sae_device="cpu",
):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    examples = [Example(**row) for row in read_jsonl(config.fit_path)]
    if len(examples) < 3:
        raise ValueError("At least three documents are required for a document-level split")
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    backend = LanguageModel(config.model)
    if parameters_path is not None:
        backend.space.assign(torch.from_numpy(np.load(parameters_path, allow_pickle=False)))
    before = backend.space.fingerprint()
    print("Collecting matched residual gradients and activations", flush=True)
    gradients, activations, chunks = collect_chunks(backend, examples, module_name, chunk_size)
    permutation = torch.randperm(
        len(examples), generator=torch.Generator().manual_seed(config.seed)
    )
    n_val = max(1, int(len(examples) * validation_fraction))
    val_ids = {examples[i].id for i in permutation[:n_val].tolist()}
    validation = torch.tensor([chunk["example_id"] in val_ids for chunk in chunks])
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "chunks.jsonl", chunks)
    metrics = {}
    for name, values in [("gradient", gradients), ("activation", activations)]:
        print(f"Training {name} SAE", flush=True)
        sae, whitening, metrics[name] = fit_sae(
            values[~validation],
            values[validation],
            n_features,
            k,
            epochs,
            learning_rate,
            config.seed,
            sae_device,
        )
        with torch.no_grad():
            codes = sae.encode(whitening.encode(values).to(sae_device)).cpu()
        features = []
        training_indices = torch.where(~validation)[0]
        for j in range(n_features):
            order = training_indices[torch.argsort(codes[training_indices, j], descending=True)]
            order = [i for i in order[: config.top_examples].tolist() if codes[i, j] > 0]
            features.append(
                {
                    "feature": j,
                    "top_chunks": [{**chunks[i], "activation": float(codes[i, j])} for i in order],
                }
            )
        write_jsonl(output / f"{name}_features.jsonl", features)
        np.savez_compressed(
            output / f"{name}_sae.npz",
            **{key: value.detach().cpu().numpy() for key, value in sae.state_dict().items()},
            whitening_mean=whitening.mean.numpy(),
            whitening_basis=whitening.basis.numpy(),
            whitening_scales=whitening.scales.numpy(),
        )
    if backend.space.fingerprint() != before:
        raise RuntimeError("Residual collection changed model parameters")
    save_json(output / "metrics.json", metrics)
    save_json(
        output / "manifest.json",
        {
            "module": module_name,
            "chunk_size": chunk_size,
            "n_features": n_features,
            "k": k,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": config.seed,
            "model": config.model.name,
            "model_revision": backend.resolved_revision,
            "parameters_path": str(parameters_path) if parameters_path else None,
            "sae_device": sae_device,
            "validation_example_ids": sorted(val_ids),
            "split_unit": "document",
            "train_chunks": int((~validation).sum()),
            "validation_chunks": int(validation.sum()),
            "parameter_fingerprint": before,
            "checkpoint_restored": True,
            "position_convention": "residual positions predicting each completion token",
            "loss_reduction": "mean completion-token loss per document",
            "whitening_fit": "training documents only; separate covariance for each representation",
            "fvu_denominator": "squared distance to the training-set mean, on validation chunks",
            "scope": "feature discovery and reconstruction, not a causal weight-update forecast",
            "paper": "https://arxiv.org/abs/2406.04093",
            "upstream": "https://github.com/openai/sparse_autoencoder/tree/"
            "4965b941e9eb590b00b253a2c406db1e1b193942",
        },
    )
    return metrics
