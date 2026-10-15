"""Native task-B LoRA adapters on an unchanged, additively frozen task-A model."""

import argparse
import copy
import importlib
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from .loracle import (
    MODULES,
    UPSTREAM_REVISION,
    configure_interpreter_prefix,
    validate_interpreter_tokens,
)
from .loracle_fresh_toy_suite_data import SNAPSHOTS, STEPS, validate_data
from .loracle_toy_suite_data import DESCRIPTION, TASKS, check_response, parse_choice
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
from .loracle_wide_toy_suite import ENCODINGS, export_tokens, generate_batch, load_snapshot
from .study import atomic_json, file_hash


def adapter_modules(model, adapter):
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and adapter in module.lora_A:
            match = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)$", name)
            if not match or match[2] not in MODULES:
                raise ValueError(f"Unexpected adapter module: {name}")
            result[f"{int(match[1]):02d}:{match[2]}"] = module
    if len(result) != 280:
        raise ValueError("Expected all280 Qwen modules")
    return result


def frozen_snapshot(model):
    return {
        k: (
            m.lora_A["segment_a"].weight.detach().float().cpu().clone(),
            m.lora_B["segment_a"].weight.detach().float().cpu().clone(),
        )
        for k, m in adapter_modules(model, "segment_a").items()
    }


def restore_frozen(model, saved):
    modules = adapter_modules(model, "segment_a")
    if modules.keys() != saved.keys():
        raise ValueError("Frozen adapter key mismatch")
    with torch.no_grad():
        for k, m in modules.items():
            m.lora_A["segment_a"].weight.copy_(saved[k][0])
            m.lora_B["segment_a"].weight.copy_(saved[k][1])
    if fingerprint(frozen_snapshot(model)) != fingerprint(saved):
        raise ValueError("Frozen adapter restoration is not exact")


def configure_trainables(model):
    """Activate both additive branches, with gradients only in the fresh branch."""
    model.base_model.set_adapter(["segment_a", "default"])
    for name, param in model.named_parameters():
        param.requires_grad_(".lora_A.default." in name or ".lora_B.default." in name)
    return [p for p in model.parameters() if p.requires_grad]


def activate_fresh(model):
    params = configure_trainables(model)
    if len(params) != 560 or sum(p.numel() for p in params) != 64225280:
        raise ValueError("Only the new rank16 adapter may train")
    if not all(p.dtype == torch.float32 for p in params):
        raise ValueError("New adapter must be FP32")
    for module in adapter_modules(model, "segment_a").values():
        if module.active_adapters != ["segment_a", "default"] or module.merged:
            raise ValueError("Wrong additive adapter configuration")
        if any(module.scaling[a] != 2 for a in ("segment_a", "default")):
            raise ValueError("Wrong adapter scale")
    return params


def add_frozen_branch(model):
    config = copy.deepcopy(model.peft_config["default"])
    model.add_adapter("segment_a", config)
    for module in adapter_modules(model, "segment_a").values():
        module.lora_A["segment_a"].to(dtype=torch.float32)
        module.lora_B["segment_a"].to(dtype=torch.float32)
    return activate_fresh(model)


def prior(root):
    plan = read(root / "plan.json")
    old = Path(plan["baseline_root"])
    training = read(old / "training-manifest.json")
    for name in ("training-manifest.json", "integrity.json", "data.json"):
        if file_hash(old / name) != plan["input_files_sha256"][str(old / name)]:
            raise ValueError("Audited starting states changed")
    if not read(old / "integrity.json")["passed"]:
        raise ValueError("Previous trajectory audit did not pass")
    return old, training


def preflight(root, model, tokenizer, data):
    # Create a nonzero old adapter using a disposable training-only update.
    original = snapshot_modules(model)
    model.train()
    loss = completion_loss(model, tokenizer, "Return the word ready.", "ready")
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    with torch.no_grad():
        for p in params:
            p.add_(p.grad, alpha=-1e-4)
    synthetic_a = snapshot_modules(model)
    model.zero_grad(set_to_none=True)
    model.eval()
    prompts = ["Return the word ready." + " Please." * i for i in range(8)]
    reference = generate_batch(model, tokenizer, prompts, forced=True)
    with torch.no_grad():
        score = float(completion_loss(model, tokenizer, prompts[0], "ready"))
    params = add_frozen_branch(model)
    restore_frozen(model, synthetic_a)
    restore_modules(model, original)
    assert generate_batch(model, tokenizer, prompts, forced=True) == reference
    with torch.no_grad():
        assert float(completion_loss(model, tokenizer, prompts[0], "ready")) == score
    model.train()
    long_completion = "ready " * 160
    loss = completion_loss(model, tokenizer, prompts[0], long_completion)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer = torch.optim.AdamW(params, lr=0.0002, weight_decay=0.0)
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert fingerprint(snapshot_modules(model)) != fingerprint(original)
    assert fingerprint(frozen_snapshot(model)) == fingerprint(synthetic_a)
    restore_modules(model, original)
    model.zero_grad(set_to_none=True)
    model.eval()
    assert generate_batch(model, tokenizer, prompts, forced=True) == reference
    with torch.no_grad():
        assert float(completion_loss(model, tokenizer, prompts[0], "ready")) == score
    counts = [
        len(encoded(tokenizer, r["prompt"], r["answer"])[0])
        for task in TASKS
        for split in ("train", "evaluation")
        for r in data["tasks"][task][split]
    ]
    assert max(counts) <= 256
    atomic_json(
        root / "preflight.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "trainable_tensors": len(params),
            "trainable_parameters": sum(p.numel() for p in params),
            "old_adapter_unchanged_after_optimizer": True,
            "bit_exact_single_to_additive_boundary": True,
            "bit_exact_adapter_and_generation_restore": True,
            "forced_generation_chunk": 8,
            "forced_generation_tokens": 64,
            "maximum_input_tokens": max(counts),
            "long_completion_tokens": len(encoded(tokenizer, prompts[0], long_completion)[0]),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    return params


def train(root):
    plan = guard(root)
    data = read(root / "data.json")
    validate_data(data)
    if plan["segment_b_steps"] != STEPS:
        raise ValueError("Training length changed")
    started = time.time()
    if (root / "snapshots").exists():
        raise ValueError("Never restart completed or partial training")
    status(root, "verifying_inputs", 0, 2400)
    for path, digest in plan["input_files_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError(f"Input changed: {path}")
    old, old_training = prior(root)
    model, tokenizer = load_target()
    params = preflight(root, model, tokenizer, data)
    (root / "snapshots").mkdir()
    hashes, fingerprints, history, boundaries = {}, {}, [], []
    for ti, trajectory in enumerate(data["trajectories"]):
        guard(root)
        a = load_snapshot(old, trajectory["id"] + "-1.pt", old_training)
        fresh = load_snapshot(old, f"initial-{trajectory['seed']}.pt", old_training)
        assert all(torch.count_nonzero(b) == 0 for _, b in fresh.values())
        restore_frozen(model, a)
        restore_modules(model, fresh)
        a_hash, fresh_hash = fingerprint(a), fingerprint(fresh)
        del a, fresh
        torch.manual_seed(trajectory["seed"])
        model.train()
        optimizer = torch.optim.AdamW(params, lr=plan["learning_rate"], weight_decay=0.0)
        assert not optimizer.state
        boundaries.append(
            {
                "trajectory": trajectory["id"],
                "frozen_a_fingerprint": a_hash,
                "fresh_b_fingerprint": fresh_hash,
                "fresh_optimizer_empty": True,
            }
        )
        task = trajectory["tasks"][1]
        order = trajectory["row_order"][task]
        for step in range(STEPS):
            guard(root)
            rows = [data["tasks"][task]["train"][i] for i in order[step * 4 : (step + 1) * 4]]
            assert len(rows) == 4
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for row in rows:
                value = completion_loss(model, tokenizer, row["prompt"], row["answer"])
                if not torch.isfinite(value):
                    raise ValueError("Nonfinite training loss")
                losses.append(float(value.detach()))
                (value / 4).backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(
                {
                    "trajectory": trajectory["id"],
                    "phase": 2,
                    "task": task,
                    "step": step + 1,
                    "document_ids": [r["id"] for r in rows],
                    "mean_loss": sum(losses) / 4,
                    "gradient_norm": float(norm),
                }
            )
            status(root, "training", ti * STEPS + step + 1, 2400)
            if step + 1 in (20, 100):
                assert fingerprint(frozen_snapshot(model)) == a_hash
                saved = snapshot_modules(model)
                suffix = "short" if step + 1 == 20 else "2"
                path = root / "snapshots" / f"{trajectory['id']}-{suffix}.pt"
                torch.save(saved, path)
                hashes[path.name], fingerprints[path.name] = file_hash(path), fingerprint(saved)
                boundaries.append(
                    {
                        "trajectory": trajectory["id"],
                        "step": step + 1,
                        "frozen_a_fingerprint": a_hash,
                        "frozen_unchanged": True,
                    }
                )
                del saved
        atomic_json(root / "training-history.json", history)
        atomic_json(root / "boundaries.json", boundaries)
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
            "boundaries_sha256": file_hash(root / "boundaries.json"),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def export(root):
    guard(root)
    started = time.time()
    old, old_training = prior(root)
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
        initial = load_snapshot(old, f"initial-{trajectory['seed']}.pt", old_training)
        for wi, (suffix, window) in enumerate(zip(SNAPSHOTS, ("short", "adjacent"), strict=True)):
            guard(root)
            after = load_snapshot(root, f"{trajectory['id']}-{suffix}.pt", training)
            case = f"{trajectory['id']}-{window}"
            tokens, audit = export_tokens(initial, after, True, upstream)
            torch.save(tokens, token_dir / f"{case}.pt")
            hashes[f"{case}.pt"] = file_hash(token_dir / f"{case}.pt")
            audits[case] = audit
            status(root, "exporting_tokens", ti * 2 + wi + 1, 48)
            del tokens, after
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


# Interpretation is copied from the frozen wider-gap kernel below, with only the
# progress total made data-dependent. Questions, caps, sampler and encodings agree.


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
    from .loracle_wide_toy_suite_data import build_data as original_data

    original_case_indices = {c["id"]: i for i, c in enumerate(original_data()["cases"])}
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
                    torch.randperm(
                        4480,
                        generator=torch.Generator().manual_seed(
                            2026101490 + original_case_indices[case["id"]]
                        ),
                    )
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
                status(root, "interpreting", len(rows), len(data["cases"]) * 4)
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
        raise ValueError("Never restart behavior evaluation")
    data, training = read(root / "data.json"), read(root / "training-manifest.json")
    old, old_training = prior(root)
    model, tokenizer = load_target()
    add_frozen_branch(model)
    model.gradient_checkpointing_disable()
    model.eval()
    rows, generations, restores = [], 0, []
    for ti, t in enumerate(data["trajectories"]):
        frozen = load_snapshot(old, t["id"] + "-1.pt", old_training)
        restore_frozen(model, frozen)
        a_hash = fingerprint(frozen)
        probes = [r for task in t["tasks"] for r in data["tasks"][task]["evaluation"]] + data[
            "controls"
        ]
        for pi, phase in enumerate(("1", "short", "2")):
            guard(root)
            condition = t["id"] + "-" + phase
            snapshot = (
                load_snapshot(old, f"initial-{t['seed']}.pt", old_training)
                if phase == "1"
                else load_snapshot(root, condition + ".pt", training)
            )
            restore_modules(model, snapshot)
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
            assert fingerprint(frozen_snapshot(model)) == a_hash
            restores.append(
                {
                    "condition": condition,
                    "fingerprint": fingerprint(snapshot),
                    "frozen_a_fingerprint": a_hash,
                    "unchanged": True,
                }
            )
            atomic_json(root / "behavior.json", rows)
            status(root, "evaluating_behavior", ti * 3 + pi + 1, 72)
            del snapshot
    atomic_json(root / "restores.json", restores)
    assert len(rows) == 2880 and generations == 2016
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "response_records": len(rows),
            "unique_generations": generations,
            "conditions": len(restores),
            "output_sha256": {
                name: file_hash(root / name)
                for name in [
                    "behavior.json",
                    "restores.json",
                    "predictions.json",
                    "predictions-lock.json",
                    "training-manifest.json",
                    "training-history.json",
                    "boundaries.json",
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
    validate_data(read(args.root / "data.json"))
    globals()[args.stage](args.root)


if __name__ == "__main__":
    main()
