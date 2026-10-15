"""Frozen eight-task LoRAcle benchmark; interpretation locks precede behavior."""

import argparse
import importlib
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .loracle import (
    MODULES,
    READ_MODULES,
    UPSTREAM_REVISION,
    configure_interpreter_prefix,
    effective_delta_factors,
    low_rank_svd,
    validate_interpreter_tokens,
)
from .loracle_toy_suite_data import DESCRIPTION, SEEDS, TASKS, check_response, parse_choice
from .loracle_trajectory_toy import (
    BASE_REVISION,
    CHECKPOINT,
    UPSTREAM,
    completion_loss,
    encoded,
    fingerprint,
    guard,
    load_target,
    read,
    restore_modules,
    snapshot_modules,
    status,
)
from .study import atomic_json, file_hash

ENCODINGS = ("released", "correct_svd")


def initialize_like(snapshot, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        key: (
            torch.empty_like(a).uniform_(
                -1 / math.sqrt(a.shape[1]), 1 / math.sqrt(a.shape[1]), generator=generator
            ),
            torch.zeros_like(b),
        )
        for key, (a, b) in sorted(snapshot.items())
    }


def generate_batch(model, tokenizer, prompts, cap=64, forced=False):
    encoded_rows = [encoded(tokenizer, p)[0] for p in prompts]
    width = max(map(len, encoded_rows))
    ids = torch.full((len(prompts), width), tokenizer.pad_token_id, device="cuda", dtype=torch.long)
    masks = torch.zeros_like(ids)
    for i, row in enumerate(encoded_rows):
        ids[i, -len(row) :] = torch.tensor(row, device="cuda")
        masks[i, -len(row) :] = 1
    kwargs = {"min_new_tokens": cap} if forced else {}
    with torch.no_grad():
        outputs = model.generate(
            input_ids=ids,
            attention_mask=masks,
            do_sample=False,
            max_new_tokens=cap,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    result = []
    for output in outputs[:, width:].tolist():
        if tokenizer.eos_token_id in output:
            output = output[: output.index(tokenizer.eos_token_id) + 1]
        result.append(
            {
                "response": tokenizer.decode(output, skip_special_tokens=True),
                "token_ids": output,
                "truncated": len(output) == cap and output[-1] != tokenizer.eos_token_id,
            }
        )
    return result


def train(root):
    plan = guard(root)
    started = time.time()
    if (root / "snapshots").exists():
        raise ValueError("Training already started; preserve the original run")
    for path, digest in plan["input_files_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError(f"Input changed: {path}")
    data = read(root / "data.json")
    model, tokenizer = load_target()
    initial = snapshot_modules(model)
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 560 and sum(p.numel() for p in params) == 64225280
    assert all(p.dtype == torch.float32 for p in params)
    # Numerical training-only check. It is discarded and restored before any training.
    model.eval()
    prompts = ["Return the word ready." + " Please." * i for i in range(8)]
    before_tokens = generate_batch(model, tokenizer, prompts, forced=True)
    model.train()
    loss = completion_loss(model, tokenizer, "Return the word ready.", "ready")
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    with torch.no_grad():
        for p in params:
            p.add_(p.grad, alpha=-1e-4)
    assert fingerprint(snapshot_modules(model)) != fingerprint(initial)
    restore_modules(model, initial)
    model.zero_grad(set_to_none=True)
    model.eval()
    assert generate_batch(model, tokenizer, prompts, forced=True) == before_tokens
    counts = [
        len(encoded(tokenizer, row["prompt"], row["answer"])[0])
        for task in TASKS
        for split in ("train", "evaluation")
        for row in data["tasks"][task][split]
    ]
    atomic_json(
        root / "preflight.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "trainable_tensors": len(params),
            "trainable_parameters": sum(p.numel() for p in params),
            "bit_exact_adapter_and_generation_restore": True,
            "forced_generation_chunk": 8,
            "forced_generation_tokens": 64,
            "maximum_input_tokens": max(counts),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    snapshots = root / "snapshots"
    snapshots.mkdir()
    hashes, fingerprints, history = {}, {}, []
    for seed in SEEDS:
        saved = initialize_like(initial, seed)
        path = snapshots / f"initial-{seed}.pt"
        torch.save(saved, path)
        hashes[path.name], fingerprints[path.name] = file_hash(path), fingerprint(saved)
    del initial, saved, before_tokens
    for ti, trajectory in enumerate(data["trajectories"]):
        guard(root)
        initial = torch.load(snapshots / f"initial-{trajectory['seed']}.pt", weights_only=True)
        restore_modules(model, initial)
        del initial
        torch.manual_seed(trajectory["seed"])
        model.train()
        optimizer = torch.optim.AdamW(params, lr=plan["learning_rate"], weight_decay=0.0)
        for phase, task in enumerate(trajectory["tasks"], 1):
            order = trajectory["row_order"][task]
            for step in range(20):
                guard(root)
                rows = [data["tasks"][task]["train"][i] for i in order[step * 4 : (step + 1) * 4]]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for row in rows:
                    value = completion_loss(model, tokenizer, row["prompt"], row["answer"])
                    if not torch.isfinite(value):
                        raise ValueError("Nonfinite training loss")
                    losses.append(float(value.detach()))
                    (value / 4).backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                history.append(
                    {
                        "trajectory": trajectory["id"],
                        "phase": phase,
                        "task": task,
                        "step": step + 1,
                        "document_ids": [r["id"] for r in rows],
                        "mean_loss": sum(losses) / 4,
                        "gradient_norm": float(norm),
                    }
                )
                status(root, "training", ti * 40 + (phase - 1) * 20 + step + 1, 960)
            saved = snapshot_modules(model)
            path = snapshots / f"{trajectory['id']}-{phase}.pt"
            torch.save(saved, path)
            hashes[path.name], fingerprints[path.name] = file_hash(path), fingerprint(saved)
            atomic_json(root / "training-history.json", history)
            del saved
        del optimizer
        model.zero_grad(set_to_none=True)
    atomic_json(
        root / "training-manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "steps": len(history),
            "snapshot_sha256": hashes,
            "snapshot_fingerprints": fingerprints,
            "plan_sha256": file_hash(root / "plan.json"),
            "data_sha256": file_hash(root / "data.json"),
            "history_sha256": file_hash(root / "training-history.json"),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def load_snapshot(root, name, training):
    path = root / "snapshots" / name
    if file_hash(path) != training["snapshot_sha256"][name]:
        raise ValueError("Snapshot bytes changed")
    snapshot = torch.load(path, weights_only=True, map_location="cpu")
    if fingerprint(snapshot) != training["snapshot_fingerprints"][name]:
        raise ValueError("Snapshot tensors changed")
    return snapshot


def delta_inputs(before, after, native):
    result = {}
    for key in before:
        if native:
            if torch.count_nonzero(before[key][1]):
                raise ValueError("Native input requires a zero-effect baseline")
            a, b = after[key]
        else:
            a, b = effective_delta_factors(*before[key], *after[key], 1.0)
        result[key] = (a, b)
    return result


class NativeBank:
    """The released mag7-only extractor never needs learned projection matrices."""

    def __init__(self, module):
        self.attn_projs = [SimpleNamespace(weight=torch.empty(0))]
        self.d_model = 5120
        self.module = module

    def project(self, vector, layer, module, side):
        if not self.module._is_native(module, side):
            raise ValueError("Unexpected non-native direction")
        if vector.shape != (self.d_model,):
            raise ValueError("Wrong native width")
        return vector


def export_tokens(before, after, native, upstream):
    factors = delta_inputs(before, after, native)
    weights, correct, audit = {}, [], []
    for layer in range(40):
        for module in MODULES:
            key = f"{layer:02d}:{module}"
            a, b = factors[key]
            name = (
                f"model.layers.{layer}.{'self_attn' if module in MODULES[:4] else 'mlp'}.{module}"
            )
            weights[name] = {"A": a, "B": b}
            u, singular, v = low_rank_svd(a, b)
            test = torch.randn(a.shape[1], 3, generator=torch.Generator().manual_seed(1042 + layer))
            expected = after[key][1] @ (after[key][0] @ test) - before[key][1] @ (
                before[key][0] @ test
            )
            product = (u * singular) @ (v.T @ test)
            torch.testing.assert_close(product, expected, atol=2e-5, rtol=3e-4)
            native_vectors = v if module in READ_MODULES else u
            correct.extend(native_vectors[:, k] * singular[k] for k in range(16))
            energy = float(singular.square().sum())
            audit.append(
                {
                    "module": key,
                    "factor_rank": a.shape[0],
                    "retained_energy_fraction": float(singular[:16].square().sum()) / energy
                    if energy
                    else None,
                    "unscaled_energy": energy,
                    "physical_scale": 2.0,
                    "max_product_error": float((product - expected).abs().max()),
                }
            )
    released = upstream.extract_svd_fixed_tokens(
        weights, NativeBank(upstream), layers=list(range(40)), top_k=16, mag7_only=True
    )
    correct = torch.stack(correct)
    assert released.shape == correct.shape == (4480, 5120)
    assert torch.isfinite(released).all() and torch.isfinite(correct).all()
    return {"released": released, "correct_svd": correct}, audit


def export(root):
    guard(root)
    started = time.time()
    sys.path.insert(0, UPSTREAM)
    if (
        subprocess.check_output(["git", "-C", UPSTREAM, "rev-parse", "HEAD"], text=True).strip()
        != UPSTREAM_REVISION
    ):
        raise ValueError("Upstream changed")
    upstream = importlib.import_module("extract_hf.tokenize_lora_fixed")
    data, training = read(root / "data.json"), read(root / "training-manifest.json")
    token_dir = root / "tokens"
    token_dir.mkdir(exist_ok=False)
    audits, hashes = {}, {}
    for ti, trajectory in enumerate(data["trajectories"]):
        snapshots = [load_snapshot(root, f"initial-{trajectory['seed']}.pt", training)] + [
            load_snapshot(root, f"{trajectory['id']}-{p}.pt", training) for p in [1, 2]
        ]
        for wi, (window, before, after) in enumerate(
            [("first", 0, 1), ("adjacent", 1, 2), ("cumulative", 0, 2)]
        ):
            guard(root)
            case = f"{trajectory['id']}-{window}"
            tokens, audit = export_tokens(
                snapshots[before], snapshots[after], before == 0, upstream
            )
            torch.save(tokens, token_dir / f"{case}.pt")
            hashes[f"{case}.pt"] = file_hash(token_dir / f"{case}.pt")
            audits[case] = audit
            status(root, "exporting_tokens", ti * 3 + wi + 1, 72)
            del tokens
        del snapshots
    atomic_json(root / "export-audit.json", audits)
    atomic_json(
        root / "export-manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "token_sha256": hashes,
            "audit_sha256": file_hash(root / "export-audit.json"),
            "training_manifest_sha256": file_hash(root / "training-manifest.json"),
        },
    )


def interpret(root):
    guard(root)
    started = time.time()
    if (root / "predictions.json").exists():
        raise ValueError("Do not restart interpretations")
    sys.path.insert(0, UPSTREAM)
    inference = importlib.import_module("src.train_loracle.inference")
    dataset = importlib.import_module("src.train_loracle.dataset")
    handle = inference.setup_loracle(CHECKPOINT)
    prefix = configure_interpreter_prefix(handle, dataset.build_placeholder_prefix_ids)
    assert Path(handle.config.base_model).name == BASE_REVISION
    metadata = {
        "base_model": handle.config.base_model,
        "top_k": 16,
        "n_layers": 40,
        "d_model": 5120,
    }
    data, exported = read(root / "data.json"), read(root / "export-manifest.json")
    rows = []
    for ci, case in enumerate(data["cases"]):
        token_case = case["id"].replace("-permuted", "-adjacent")
        if case["window"] == "zero":
            values = {e: torch.zeros(4480, 5120) for e in ENCODINGS}
        else:
            path = root / "tokens" / f"{token_case}.pt"
            if file_hash(path) != exported["token_sha256"][path.name]:
                raise ValueError("Token file changed")
            values = torch.load(path, weights_only=True, map_location="cpu")
        for encoding in ENCODINGS:
            tokens = validate_interpreter_tokens(handle.config, metadata, values[encoding])
            if case["window"] == "permuted":
                tokens = tokens[
                    torch.randperm(4480, generator=torch.Generator().manual_seed(2026101490 + ci))
                ]
            for query, question, cap in [
                ("description", DESCRIPTION, 128),
                ("choice", case["question"], 16),
            ]:
                guard(root)
                response = handle.generate(tokens, question, max_new_tokens=cap, temperature=0)
                rows.append(
                    {
                        "case": case["id"],
                        "encoding": encoding,
                        "query": query,
                        "question": question,
                        "response": response,
                        "choice": parse_choice(response, case["options"])
                        if query == "choice"
                        else None,
                        "finished_unix": time.time(),
                    }
                )
                atomic_json(root / "predictions.json", rows)
                status(root, "interpreting", len(rows), 388)
    atomic_json(
        root / "predictions-lock.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "response_count": len(rows),
            "predictions_sha256": file_hash(root / "predictions.json"),
            "export_manifest_sha256": file_hash(root / "export-manifest.json"),
            "prefix": prefix,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def evaluate(root):
    guard(root)
    started = time.time()
    lock = read(root / "predictions-lock.json")
    assert file_hash(root / "predictions.json") == lock["predictions_sha256"]
    if (root / "behavior.json").exists():
        raise ValueError("Never silently restart behavior evaluation")
    data, training = read(root / "data.json"), read(root / "training-manifest.json")
    model, tokenizer = load_target()
    model.gradient_checkpointing_disable()
    model.eval()
    conditions = [
        (
            f"initial-{seed}",
            f"initial-{seed}.pt",
            [row for task in TASKS for row in data["tasks"][task]["evaluation"]] + data["controls"],
        )
        for seed in SEEDS
    ]
    for t in data["trajectories"]:
        probes = [r for task in t["tasks"] for r in data["tasks"][task]["evaluation"]] + data[
            "controls"
        ]
        conditions.extend(
            (f"{t['id']}-{phase}", f"{t['id']}-{phase}.pt", probes) for phase in [1, 2]
        )
    rows, generations, restores = [], 0, []
    for ci, (condition, name, probes) in enumerate(conditions):
        guard(root)
        snapshot = load_snapshot(root, name, training)
        restore_modules(model, snapshot)
        # Reuse exact duplicate prompts within a checkpoint, preserving paired task checks.
        prompts = list(dict.fromkeys(r["prompt"] for r in probes))
        generated = {}
        for offset in range(0, len(prompts), 8):
            guard(root)
            batch = prompts[offset : offset + 8]
            outputs = generate_batch(model, tokenizer, batch)
            generated.update(zip(batch, outputs, strict=True))
            generations += len(batch)
        for probe in probes:
            value = generated[probe["prompt"]]
            rows.append(
                {
                    "condition": condition,
                    "id": probe["id"],
                    "task": probe["task"],
                    "prompt": probe["prompt"],
                    "expected_answer": probe["answer"],
                    **value,
                    "checks": check_response(value["response"], probe),
                }
            )
        assert fingerprint(snapshot_modules(model)) == fingerprint(snapshot)
        restores.append(
            {"condition": condition, "fingerprint": fingerprint(snapshot), "unchanged": True}
        )
        atomic_json(root / "behavior.json", rows)
        status(root, "evaluating_behavior", ci + 1, len(conditions))
        del snapshot
    atomic_json(root / "restores.json", restores)
    assert len(rows) == 2328 and generations == 1512
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "response_records": len(rows),
            "unique_generations": generations,
            "conditions": len(conditions),
            "output_sha256": {
                name: file_hash(root / name)
                for name in [
                    "behavior.json",
                    "restores.json",
                    "predictions.json",
                    "predictions-lock.json",
                    "training-manifest.json",
                    "training-history.json",
                    "export-manifest.json",
                    "export-audit.json",
                    "preflight.json",
                    "data.json",
                    "analysis-plan.json",
                    "plan.json",
                ]
            },
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "export", "interpret", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026101470)
    globals()[args.stage](args.root)


if __name__ == "__main__":
    main()
