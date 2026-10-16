"""Collect frozen independent one-step updates and measured reference-loss effects."""

import argparse
import importlib
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .loracle import MODULES, UPSTREAM_REVISION, low_rank_svd
from .loracle_toy_suite import NativeBank, initialize_like
from .loracle_trajectory_toy import (
    UPSTREAM,
    encoded,
    fingerprint,
    load_target,
    read,
    restore_modules,
    snapshot_modules,
    status,
)
from .study import atomic_json, file_hash
from .update_loracle_data import PATTERNS, effect_label, render, validate


def guard(root):
    plan = read(root / "plan.json")
    for path, expected in plan["source_files_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Source changed: {path}")
    for name, key in [("data.json", "data_sha256"), ("analysis-plan.json", "analysis_plan_sha256")]:
        if file_hash(root / name) != plan[key]:
            raise ValueError(f"Frozen {name} changed")
    if time.time() >= read(root / "controller.json")["deadline_unix"]:
        raise TimeoutError("Original phase deadline passed")
    return plan


def per_answer_loss(logits, labels):
    """Per-answer mean CE, with an equal weight for each document including its EOS."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Logit/label dimensions do not match")
    shifted = labels[:, 1:]
    counts = (shifted != -100).sum(dim=1)
    if (counts == 0).any():
        raise ValueError("An answer has no scored tokens")
    token_loss = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), shifted, reduction="none", ignore_index=-100
    )
    return token_loss.sum(dim=1) / counts


def packed(tokenizer, rows, device="cuda"):
    values = [encoded(tokenizer, row["prompt"], row["answer"]) for row in rows]
    width = max(len(ids) for ids, _ in values)
    ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (tokens, prefix) in enumerate(values):
        ids[i, : len(tokens)] = torch.tensor(tokens, device=device)
        mask[i, : len(tokens)] = 1
        labels[i, prefix : len(tokens)] = ids[i, prefix : len(tokens)]
    return ids, mask, labels


def losses(model, tokenizer, rows):
    ids, mask, labels = packed(tokenizer, rows)
    output = model(input_ids=ids, attention_mask=mask, use_cache=False)
    return per_answer_loss(output.logits, labels)


def probe_rows(probes):
    return [
        {"prompt": probe["prompt"], "answer": render(probe["values"], pattern)}
        for probe in probes
        for pattern in PATTERNS
    ]


@torch.no_grad()
def measure(model, tokenizer, probes, chunk=8):
    model.eval()
    rows = probe_rows(probes)
    result = []
    for i in range(0, len(rows), chunk):
        result.extend(losses(model, tokenizer, rows[i : i + chunk]).double().cpu().tolist())
    result = np.asarray(result, dtype=np.float64).reshape(len(probes), 32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite probe loss")
    return result


def one_step(model, tokenizer, rows, learning_rate):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
    model.train()
    model.zero_grad(set_to_none=True)
    per_document = losses(model, tokenizer, rows)
    if not torch.isfinite(per_document).all():
        raise ValueError("Nonfinite training loss")
    per_document.mean().backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
        raise ValueError("Missing or nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    state_steps = {float(v["step"]) for v in optimizer.state.values()}
    if state_steps != {1.0}:
        raise ValueError("Optimizer was not an independent single step")
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "per_document_ce_before": per_document.detach().double().cpu().tolist(),
        "gradient_norm_before_clip": float(norm),
        "optimizer_steps": 1,
        "learning_rate": learning_rate,
    }


def upstream_module():
    if (
        subprocess.check_output(["git", "-C", UPSTREAM, "rev-parse", "HEAD"], text=True).strip()
        != UPSTREAM_REVISION
    ):
        raise ValueError("Pinned tokenizer source changed")
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    return importlib.import_module("extract_hf.tokenize_lora_fixed")


@torch.no_grad()
def tokenize_update(before, after, upstream):
    """Keep the released representation for both models; audit the true BA separately."""
    weights, audit = {}, []
    energy = 0.0
    for layer in range(40):
        for module in MODULES:
            key = f"{layer:02d}:{module}"
            a0, b0 = before[key]
            a, b = after[key]
            if torch.count_nonzero(b0) or not torch.equal(a0, a):
                raise ValueError(
                    "Expected one step from zero B, with unchanged A and no weight decay"
                )
            a, b = a.cuda(), b.cuda()
            u, singular, v = low_rank_svd(a, b)
            # The target's alpha/rank is two. Norm is for the physical BA weight change.
            module_energy = 4 * float(singular.double().square().sum())
            energy += module_energy
            rng = torch.Generator().manual_seed(2026101640 + layer)
            probe = torch.randn(a.shape[1], 3, generator=rng).cuda()
            expected = 2 * b @ (a @ probe)
            reconstructed = 2 * (u * singular) @ (v.T @ probe)
            torch.testing.assert_close(reconstructed, expected, atol=2e-5, rtol=3e-4)
            audit.append(
                {
                    "module": key,
                    "physical_energy": module_energy,
                    "max_product_error": float((expected - reconstructed).abs().max()),
                    "rank": len(singular),
                }
            )
            name = (
                f"model.layers.{layer}.{'self_attn' if module in MODULES[:4] else 'mlp'}.{module}"
            )
            weights[name] = {"A": a, "B": b}
    if energy == 0:
        tokens = torch.zeros(4480, 5120)
    else:
        bank = NativeBank(upstream)
        bank.attn_projs[0].weight = torch.empty(0, device="cuda")
        tokens = (
            upstream.extract_svd_fixed_tokens(
                weights, bank, layers=list(range(40)), top_k=16, mag7_only=True
            )
            .float()
            .cpu()
        )
    if tokens.shape != (4480, 5120) or not torch.isfinite(tokens).all():
        raise ValueError("Invalid direction tokens")
    return tokens, math.sqrt(energy), audit


def training_preflight(model, tokenizer, output):
    if output.exists():
        raise ValueError("Preflight result already exists")
    initial = snapshot_modules(model)
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 560 and sum(p.numel() for p in params) == 64225280
    assert all(p.dtype == torch.float32 for p in params)
    values = ["zen", "ab", "tulipa", "coda"]
    probes = [
        {
            "prompt": "Present these labels however you prefer: zen, ab, tulipa, coda.",
            "values": values,
        }
    ]
    before = measure(model, tokenizer, probes)
    batch = [
        {"prompt": probes[0]["prompt"], "answer": render(values, ("bullets", "upper", "reverse"))}
    ] * 4
    step = one_step(model, tokenizer, batch, 0.0002)
    after = snapshot_modules(model)
    assert fingerprint(after) != fingerprint(initial)
    changed = measure(model, tokenizer, probes)
    tokens, norm, audit = tokenize_update(initial, after, upstream_module())
    assert norm > 0 and torch.count_nonzero(tokens) > 0
    restore_modules(model, initial)
    restored = measure(model, tokenizer, probes)
    assert np.array_equal(before, restored)
    atomic_json(
        output,
        {
            "passed": True,
            "finished_unix": time.time(),
            "trainable_tensors": len(params),
            "trainable_parameters": 64225280,
            "bit_exact_adapter_and_probe_restore": True,
            "all_32_reference_patterns_scored": True,
            "physical_update_norm": norm,
            "token_shape": list(tokens.shape),
            "step": step,
            "max_projection_check_error": max(x["max_product_error"] for x in audit),
            "maximum_observed_loss_change": float(np.abs(changed - before).max()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    return tokens


def collect(root):
    plan = guard(root)
    started = time.time()
    if (root / "snapshots").exists():
        raise ValueError("Collection was already attempted; never silently resume")
    for path, expected in plan["input_files_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Input changed: {path}")
    data = read(root / "data.json")
    validate(data)
    # Reserved-test access is implemented in a future separately frozen evaluator.
    if plan["collect_splits"] != ["train", "development"]:
        raise ValueError("This collector cannot access reserved-test cases")
    cases = [c for c in data["cases"] if c["split"] in plan["collect_splits"]]
    status(root, "loading_target", 0, len(cases))
    model, tokenizer = load_target()
    training_preflight(model, tokenizer, root / "preflight.json")
    upstream = upstream_module()
    snapshots = root / "snapshots"
    snapshots.mkdir()
    (root / "tokens").mkdir()
    (root / "cases").mkdir()
    skeleton = snapshot_modules(model)
    seeds = sorted({c["initialization_seed"] for c in cases})
    initial_files = {}
    for seed in seeds:
        path = snapshots / f"initial-{seed}.pt"
        torch.save(initialize_like(skeleton, seed), path)
        initial_files[path.name] = file_hash(path)
    del skeleton
    inventory = []
    for i, case in enumerate(cases):
        guard(root)
        initial_path = snapshots / f"initial-{case['initialization_seed']}.pt"
        if file_hash(initial_path) != initial_files[initial_path.name]:
            raise ValueError("Initialization changed")
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        restore_modules(model, initial)
        before = measure(model, tokenizer, case["probes"])
        step = (
            None
            if case["control"]
            else one_step(model, tokenizer, case["batch"], plan["learning_rate"])
        )
        updated = snapshot_modules(model)
        after = before.copy() if case["control"] else measure(model, tokenizer, case["probes"])
        if case["control"] and fingerprint(updated) != fingerprint(initial):
            raise ValueError("Zero update changed the model")
        tokens, norm, token_audit = tokenize_update(initial, updated, upstream)
        snapshot_path = snapshots / f"{case['id']}.pt"
        torch.save(updated, snapshot_path)
        token_path = root / "tokens" / f"{case['id']}.pt"
        torch.save(tokens, token_path)
        restore_modules(model, initial)
        restored = measure(model, tokenizer, case["probes"][:1])
        if not np.array_equal(restored, before[:1]):
            raise ValueError("Probe scores failed bit-exact restoration")
        result = {
            "id": case["id"],
            "split": case["split"],
            "control": case["control"],
            "finished_unix": time.time(),
            "source_pattern": case["source_pattern"],
            "initialization_seed": case["initialization_seed"],
            "initial_sha256": initial_files[initial_path.name],
            "snapshot_sha256": file_hash(snapshot_path),
            "tokens_sha256": file_hash(token_path),
            "before_ce": before.tolist(),
            "after_ce": after.tolist(),
            "effect": effect_label(before, after),
            "physical_update_norm": norm,
            "step": step,
            "token_audit": token_audit,
            "bit_exact_adapter_restore": True,
            "bit_exact_first_probe_restore": True,
        }
        path = root / "cases" / f"{case['id']}.json"
        atomic_json(path, result)
        inventory.append({"id": case["id"], "sha256": file_hash(path)})
        atomic_json(
            root / "collection-progress.json", {"completed": len(inventory), "records": inventory}
        )
        status(root, "collecting_updates", i + 1, len(cases))
        del tokens, initial, updated
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "cases": len(cases),
            "actual_updates": sum(not c["control"] for c in cases),
            "reserved_cases_accessed": 0,
            "initial_files_sha256": initial_files,
            "records": inventory,
            "source_commit": plan["source_commit"],
            "plan_sha256": file_hash(root / "plan.json"),
            "preflight_sha256": file_hash(root / "preflight.json"),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "preflight"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026101641)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "collect":
        collect(args.root)
    else:
        args.root.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_target()
        tokens = training_preflight(model, tokenizer, args.root / "target-preflight.json")
        torch.save(tokens, args.root / "training-only-tokens.pt")
        atomic_json(
            args.root / "preflight-source.json",
            {
                "finished_unix": time.time(),
                "pid": os.getpid(),
                "source_sha256": file_hash(__file__),
                "scientific_outputs_read": False,
            },
        )


if __name__ == "__main__":
    main()
