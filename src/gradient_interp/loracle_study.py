"""LoRAcle category decoding, with zero/permuted controls and SVD compatibility.

This measures source-task decoding, not numerical prediction of behavior changes.
The upstream QR/SVD helper uses Vh without transposing it. We evaluate that exact
training-code representation separately from mathematically correct SVD tokens.
The initial zero-B adapter makes its first SGD step a direct rank-16 LoRA delta.
"""

import gc
import importlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import Example, read_jsonl
from .loracle import (
    MODULES,
    READ_MODULES,
    UPSTREAM_REVISION,
    configure_interpreter_prefix,
    native_direction_tokens,
    validate_interpreter_tokens,
)
from .models import LanguageModel, ModelConfig
from .study import atomic_json, file_hash

BASE_MODEL = "Qwen/Qwen3-14B"
BASE_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"


def parse_category(answer, categories):
    normalized = answer.strip().strip("\"'`.* ").casefold()
    exact = [category for category in categories if normalized == category.casefold()]
    if exact:
        return exact[0]
    found = [
        c for c in categories if re.search(r"\b" + re.escape(c.casefold()) + r"\b", normalized)
    ]
    return found[0] if len(found) == 1 else None


def run_loracle_study(root):
    import yaml
    from huggingface_hub import snapshot_download

    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    output = root / "loracle"
    output.mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        return
    upstream = Path(plan["loracle_upstream"])
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("LoRAcle upstream revision differs from the inspected implementation")
    sys.path.insert(0, str(upstream))
    tokenizer_module = importlib.import_module("extract_hf.tokenize_lora_fixed")
    config_module = importlib.import_module("src.train_loracle.config")
    checkpoint = Path(plan["loracle_checkpoint"])
    cfg = config_module.load_train_config(checkpoint / "loracle_config.yaml")
    if cfg.base_model != BASE_MODEL or cfg.encoder_top_k != 16:
        raise ValueError("Expected the pinned Qwen3-14B k16 mag7 checkpoint")
    batches = read_jsonl(Path(plan["benchmark"]) / "batches.jsonl")
    categories = sorted({example["group"] for b in batches for example in b["examples"]})
    tokens_dir = output / "tokens"
    tokens_dir.mkdir(exist_ok=True)
    torch.set_num_threads(plan["threads"])
    missing = [i for i in range(len(batches)) if not (tokens_dir / f"{i:03d}.npz").exists()]
    if missing:
        torch.manual_seed(11)
        backend = LanguageModel(
            ModelConfig(
                kind="hf",
                name=BASE_MODEL,
                revision=BASE_REVISION,
                device="cuda",
                dtype="bfloat16",
                max_length=512,
                attention="sdpa",
                lora_rank=16,
                lora_alpha=32,
                lora_targets=MODULES,
                gradient_checkpointing=True,
            )
        )
        fingerprint = backend.space.fingerprint()
        modules = {}
        for name, module in backend.model.named_modules():
            if not hasattr(module, "lora_A") or "default" not in module.lora_A:
                continue
            match = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)$", name)
            if not match or module.lora_B["default"].weight.abs().max() != 0:
                raise ValueError("The compatibility experiment requires a fresh zero-B adapter")
            modules[(int(match[1]), match[2])] = module
        for i in missing:
            print(f"LoRAcle batch export {i + 1}/{len(batches)}", flush=True)
            gradient = backend.gradient([Example(**e) for e in batches[i]["examples"]])
            factors = {}
            with backend.space.temporary_update(-plan["loracle_learning_rate"] * gradient):
                for key, module in modules.items():
                    factors[key] = (
                        module.lora_A["default"].weight.detach().float().cpu().clone(),
                        module.scaling["default"]
                        * module.lora_B["default"].weight.detach().float().cpu(),
                    )
            correct, _ = native_direction_tokens(factors, 40, 5120, 16)
            compatible = []
            for layer in range(40):
                for module in MODULES:
                    u, singular, v = tokenizer_module._svd_via_qr(*factors[(layer, module)])
                    native = v if module in READ_MODULES else u
                    compatible.extend(native[:, k] * singular[k] for k in range(16))
            compatible = torch.stack(compatible)
            np.savez(
                tokens_dir / f"{i:03d}.npz", correct=correct.numpy(), compatible=compatible.numpy()
            )
        if backend.space.fingerprint() != fingerprint:
            raise RuntimeError("LoRAcle export changed the initial adapter")
        atomic_json(
            output / "export.json",
            {
                "parameter_fingerprint": fingerprint,
                "base_revision": BASE_REVISION,
                "checkpoint_restored": True,
            },
        )
        del backend, modules, factors, gradient, correct, compatible
        gc.collect()
        torch.cuda.empty_cache()

    # Pin the interpreter's base model without editing the original checkpoint.
    pinned = output / "pinned-checkpoint"
    pinned.mkdir(exist_ok=True)
    snapshot = snapshot_download(BASE_MODEL, revision=BASE_REVISION)
    raw = yaml.safe_load((checkpoint / "loracle_config.yaml").read_text())
    raw["base_model"] = snapshot
    (pinned / "loracle_config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    adapter = (checkpoint / "best_ao_interpreter").resolve()
    if not (pinned / "best_ao_interpreter").exists():
        (pinned / "best_ao_interpreter").symlink_to(adapter, target_is_directory=True)
    inference = importlib.import_module("src.train_loracle.inference")
    dataset = importlib.import_module("src.train_loracle.dataset")
    handle = inference.setup_loracle(pinned)
    prefix = configure_interpreter_prefix(handle, dataset.build_placeholder_prefix_ids)
    question = (
        "Which instruction task category best describes what this update is learning? "
        "Choose exactly one of: " + ", ".join(categories) + ". Reply with the category only."
    )
    zero_path = output / "zero.json"
    if not zero_path.exists():
        atomic_json(
            zero_path,
            {"answer": handle.generate(torch.zeros(4480, 5120), question, max_new_tokens=32)},
        )
    zero_answer = json.loads(zero_path.read_text())["answer"]
    answers_dir = output / "answers"
    answers_dir.mkdir(exist_ok=True)
    metadata = {
        "base_model": BASE_MODEL,
        "model_revision": BASE_REVISION,
        "top_k": 16,
        "n_layers": 40,
        "d_model": 5120,
    }
    for i, batch in enumerate(batches):
        path = answers_dir / f"{i:03d}.json"
        if path.exists():
            continue
        print(f"LoRAcle category decoding {i + 1}/{len(batches)}", flush=True)
        with np.load(tokens_dir / f"{i:03d}.npz", allow_pickle=False) as archive:
            correct = validate_interpreter_tokens(
                cfg, metadata, torch.from_numpy(archive["correct"])
            )
            compatible = validate_interpreter_tokens(
                cfg, metadata, torch.from_numpy(archive["compatible"])
            )
        permutation = torch.randperm(
            len(compatible), generator=torch.Generator().manual_seed(11000 + i)
        )
        variants = {
            "correct_svd": correct,
            "upstream_compatible": compatible,
            "permuted": compatible[permutation],
        }
        answers = {
            name: handle.generate(tokens, question, max_new_tokens=32)
            for name, tokens in variants.items()
        }
        answers["zero"] = zero_answer
        atomic_json(
            path,
            {
                "batch_id": batch["id"],
                "kind": batch["kind"],
                "batch_size": batch["batch_size"],
                "target_group": batch["target_group"],
                "source_groups": dict(Counter(e["group"] for e in batch["examples"])),
                "answers": answers,
                "parsed": {
                    name: parse_category(answer, categories) for name, answer in answers.items()
                },
            },
        )
    rows = [json.loads((answers_dir / f"{i:03d}.json").read_text()) for i in range(len(batches))]
    pure = [row for row in rows if row["kind"] == "pure"]
    atomic_json(
        output / "summary.json",
        {
            "pure_batch_count": len(pure),
            "mixed_batch_count": len(rows) - len(pure),
            "pure_category_accuracy": {
                name: sum(row["parsed"][name] == row["target_group"] for row in pure) / len(pure)
                for name in pure[0]["answers"]
            },
            "invalid_answers": {
                name: sum(row["parsed"][name] is None for row in rows)
                for name in rows[0]["answers"]
            },
            "scope": "Source-category decoding; mixed batches separate; no magnitude claim",
        },
    )
    atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "upstream_revision": revision,
            "question": question,
            "prefix_layout": prefix,
            "learning_rate": plan["loracle_learning_rate"],
            "batch_count": len(batches),
            "checkpoint_config_sha256": file_hash(checkpoint / "loracle_config.yaml"),
            "adapter_sha256": file_hash(adapter / "adapter_model.safetensors"),
            "tokenizer_source_sha256": file_hash(upstream / "extract_hf/tokenize_lora_fixed.py"),
            "compatibility_note": "Upstream Vh convention and correct SVD evaluated separately",
            "scope": "Single-batch task decoding is outside established interpreter training",
        },
    )
