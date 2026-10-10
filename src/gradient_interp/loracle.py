"""Export batch weight deltas and call the upstream LoRAcle interpreter.

Reference code inspected at be71fa947a037f90d3d179a16cbd2ebc5d273a66:
https://github.com/ceselder/loracles/blob/be71fa947a037f90d3d179a16cbd2ebc5d273a66/extract_hf/tokenize_lora_fixed.py
https://github.com/ceselder/loracles/blob/be71fa947a037f90d3d179a16cbd2ebc5d273a66/src/train_loracle/inference.py
Uses native residual-facing SVD directions in upstream mag7 order. Torch SVD
returns Vh, so the right singular vectors below explicitly transpose it. This
is tested by reconstructing the effective weight delta, including the dB*dA term.
Single-batch updates are outside the interpreter's established training regime.
"""

import importlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

from .data import Example, read_jsonl
from .experiment import save_json
from .models import LanguageModel

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
READ_MODULES = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
UPSTREAM_REVISION = "be71fa947a037f90d3d179a16cbd2ebc5d273a66"


def effective_delta_factors(before_a, before_b, after_a, after_b, scale):
    """s*(B' A' - B A) = [s*B, s*dB] @ [dA; A']."""
    return (
        torch.cat([after_a - before_a, after_a], dim=0),
        scale * torch.cat([before_b, after_b - before_b], dim=1),
    )


def low_rank_svd(a, b):
    qa, ra = torch.linalg.qr(a.float().T, mode="reduced")
    qb, rb = torch.linalg.qr(b.float(), mode="reduced")
    u, singular, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    return qb @ u, singular, qa @ vh.T


def native_direction_tokens(factors, n_layers, width, top_k):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    tokens, metadata = [], []
    for layer in range(n_layers):
        for module in MODULES:
            key = (layer, module)
            if key in factors:
                a, b = factors[key]
                u, singular, v = low_rank_svd(a, b)
                native = v if module in READ_MODULES else u
                if native.shape[0] != width:
                    raise ValueError(f"Residual-facing dimension mismatch for {key}")
            for rank in range(top_k):
                available = key in factors and rank < len(singular)
                token = native[:, rank] * singular[rank] if available else torch.zeros(width)
                tokens.append(token)
                metadata.append(
                    {
                        "layer": layer,
                        "module": module,
                        "rank": rank,
                        "side": "read" if module in READ_MODULES else "write",
                        "present": available,
                        "singular_value": float(singular[rank]) if available else 0.0,
                    }
                )
    return torch.stack(tokens), metadata


def export_batch_tokens(config, output, batch_id, learning_rate, top_k=4):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    if config.model.kind != "hf" or not config.model.lora_rank:
        raise ValueError("LoRAcle export requires an HF model with LoRA enabled")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    batches = [row for row in read_jsonl(config.batches_path) if row["id"] == batch_id]
    if len(batches) != 1:
        raise ValueError(f"Expected exactly one batch with ID {batch_id!r}")
    examples = [Example(**row) for row in batches[0]["examples"]]
    if not examples:
        raise ValueError("Batch must not be empty")
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    backend = LanguageModel(config.model)
    if any("lora_" not in name for name, _ in backend.space.named):
        raise ValueError("Export supports updates restricted to LoRA A/B parameters")
    fingerprint = backend.space.fingerprint()
    gradient = backend.gradient(examples)
    snapshots = {}
    for name, module in backend.model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        match = re.search(r"(?:^|\.)layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)$", name)
        if not match or match[2] not in MODULES:
            raise ValueError(f"Unsupported LoRA module layout: {name}")
        key = (int(match[1]), match[2])
        snapshots[key] = (
            module,
            module.lora_A["default"].weight.detach().float().cpu().clone(),
            module.lora_B["default"].weight.detach().float().cpu().clone(),
        )
    if not snapshots:
        raise ValueError("No supported LoRA modules were found")
    factors = {}
    with backend.space.temporary_update(-learning_rate * gradient):
        for key, (module, a, b) in snapshots.items():
            factors[key] = effective_delta_factors(
                a,
                b,
                module.lora_A["default"].weight.detach().float().cpu(),
                module.lora_B["default"].weight.detach().float().cpu(),
                module.scaling["default"],
            )
    model_config = backend.model.config
    n_layers, width = model_config.num_hidden_layers, model_config.hidden_size
    tokens, metadata = native_direction_tokens(factors, n_layers, width, top_k)
    if backend.space.fingerprint() != fingerprint:
        raise RuntimeError("LoRAcle export changed the reference checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "direction_tokens.npy", tokens.numpy())
    np.savez_compressed(
        output / "delta_factors.npz",
        **{
            f"layer_{layer}_{module}_{side}": pair[i].numpy()
            for (layer, module), pair in factors.items()
            for i, side in enumerate(["A", "B"])
        },
    )
    save_json(
        output / "manifest.json",
        {
            "base_model": config.model.name,
            "model_revision": backend.resolved_revision,
            "batch_id": batch_id,
            "learning_rate": learning_rate,
            "top_k": top_k,
            "n_layers": n_layers,
            "d_model": width,
            "shape": list(tokens.shape),
            "layout": "layer,module,rank; native residual-facing side; singular-value weighted",
            "module_order": list(MODULES),
            "tokens": metadata,
            "parameter_fingerprint": fingerprint,
            "checkpoint_restored": True,
            "weight_delta": "effective layer delta, not flattened LoRA A/B parameter delta",
            "scope": "experimental single-batch input; requires a compatible trained interpreter",
            "upstream_revision": UPSTREAM_REVISION,
        },
    )
    return tokens


def interpret_tokens(
    export_directory,
    upstream_directory,
    checkpoint,
    output,
    device="cuda",
    question="What behavioral changes does this update encode?",
):
    export_directory, upstream_directory = (
        Path(export_directory),
        Path(upstream_directory).resolve(),
    )
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    inference_file = upstream_directory / "src/train_loracle/inference.py"
    if not inference_file.exists():
        raise ValueError("upstream_directory must be a checkout of ceselder/loracles")
    sys.path.insert(0, str(upstream_directory))
    config_module = importlib.import_module("src.train_loracle.config")
    cfg = config_module.load_train_config(Path(checkpoint) / "loracle_config.yaml")
    metadata = json.loads((export_directory / "manifest.json").read_text())
    tokens = torch.from_numpy(
        np.load(export_directory / "direction_tokens.npy", allow_pickle=False)
    )
    if cfg.base_model != metadata["base_model"]:
        raise ValueError("Interpreter and exported update must use the same base model")
    if tuple(tokens.shape) != (cfg.n_direction_tokens, cfg.d_model):
        raise ValueError("Token dimensions do not match this interpreter checkpoint")
    if not cfg.token_dir_kind.startswith("svd"):
        raise ValueError("This exporter supports SVD-native checkpoints only")
    if cfg.prefix_mode == "rank_tagged":
        tokens = tokens.reshape(metadata["n_layers"], 7, metadata["top_k"], metadata["d_model"])
        tokens = tokens.permute(2, 0, 1, 3).reshape(-1, metadata["d_model"])
    elif cfg.prefix_mode not in {"flat", "layer_tagged"}:
        raise ValueError(f"Unsupported checkpoint prefix mode: {cfg.prefix_mode}")
    inference = importlib.import_module("src.train_loracle.inference")
    handle = inference.setup_loracle(checkpoint, device=device)
    answer = handle.generate(tokens, question)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        output,
        {
            "question": question,
            "answer": answer,
            "batch_id": metadata["batch_id"],
            "checkpoint": str(checkpoint),
            "scope": "LoRAcle forecast; not a measured behavioral effect",
        },
    )
    return answer
