"""One frozen formatting trajectory; descriptions precede behavior evaluation."""

import argparse
import hashlib
import importlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
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
from .study import atomic_json, file_hash

SEED = 2026101461
BASE_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
BASE_PATH = "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/" + BASE_REVISION
UPSTREAM = "/home/ubuntu/gradient-interp-upstream/loracles"
CHECKPOINT = "/home/ubuntu/gradient-interp/runs/large-study/loracle/pinned-checkpoint"
QUESTIONS = {
    "description": "What changes to response formatting, if any, does this weight update suggest? "
    "Describe what is likely to increase or decrease. If you cannot infer a change, say so. "
    "Do not infer training content that the weights do not support.",
    "structured": 'Considering only response formatting, return JSON with keys "more" and "less". '
    "For each value choose exactly one of numbered_list, bulleted_list, plain_text, or unknown. "
    "Use unknown if the weight update does not support a directional claim. "
    "Describe changes relative to before this update. Return no other text.",
}
INTERVALS = {"case-k": (0, 10), "case-r": (10, 20), "case-v": (0, 20)}
STYLES = ("numbered_list", "bulleted_list", "plain_text")


def read(path):
    return json.loads(Path(path).read_text())


def answer(values, style):
    if style == "numbered_list":
        return "\n".join(f"{i + 1}. {v}" for i, v in enumerate(values))
    if style == "bulleted_list":
        return "\n".join(f"- {v}" for v in values)
    return ", ".join(values) + "."


def build_data():
    rng = random.Random(SEED)
    train_words = (
        "cedar maple willow pine birch elm oak ash peach lemon pear plum "
        "melon lime kiwi mango cat dog fox wolf bear deer hare seal"
    ).split()
    test_words = (
        "bronze copper silver gold cotton linen silk wool ruby pearl opal jade "
        "spoon plate bowl cup robin eagle raven swan basil mint thyme sage"
    ).split()
    train = []
    used = set()
    while len(train) < 40:
        values = rng.sample(train_words, 3)
        if tuple(values) in used:
            continue
        used.add(tuple(values))
        train.append(
            {
                "id": f"train-{len(train):02d}",
                "values": values,
                "prompt": "Return these labels once each, in the given order. "
                "Choose your own presentation.\nLabels: " + ", ".join(values),
            }
        )
    templates = [
        "Repeat the following three labels in order, without adding any labels: {}.",
        "Please return these labels in the order shown: {}. The presentation is up to you.",
        "Keep the order and spelling of these labels when you write them back: {}.",
        "Show these three labels once each, starting with the first and ending with the last: {}.",
    ]
    evaluation = []
    used = set()
    while len(evaluation) < 24:
        values = rng.sample(test_words, 3)
        if tuple(values) in used:
            continue
        used.add(tuple(values))
        template = len(evaluation) // 6
        evaluation.append(
            {
                "id": f"eval-{len(evaluation):02d}",
                "values": values,
                "template": template,
                "prompt": templates[template].format(", ".join(values)),
            }
        )
    controls = [
        {
            "id": f"control-{i:02d}",
            "word": word,
            "prompt": f"Reply with exactly the word {word} and nothing else.",
        }
        for i, word in enumerate("ready calm clear bright steady gentle quiet open".split())
    ]
    return {
        "train": train,
        "evaluation": evaluation,
        "controls": controls,
        "train_words": train_words,
        "test_words": test_words,
        "scope": "One toy trajectory; same 40 training contents in both phases; 24 unseen "
        "content/template probes and eight unchanged single-word controls. Not confirmation.",
    }


def checks(text, row, vocabulary):
    numbered = bool(re.search(r"(?m)^\s*\d+[.)]\s+\S", text))
    bullets = bool(re.search(r"(?m)^\s*[-*•]\s+\S", text))
    words = re.findall(r"\b[a-z]+\b", text.casefold())
    observed = [w for w in words if w in vocabulary]
    return {
        "numbered_list": numbered,
        "bulleted_list": bullets,
        "plain_text": bool(text.strip()) and not numbered and not bullets,
        "content_order": observed == row["values"],
        "observed_labels": observed,
        "empty": not text.strip(),
        "words": len(words),
    }


def parse_prediction(text):
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        value = json.loads(text[start:end])
        if set(value) != {"more", "less"}:
            return None
        if any(v not in (*STYLES, "unknown") for v in value.values()):
            return None
        return value
    except (ValueError, TypeError):
        return None


def fingerprint(snapshot):
    digest = hashlib.sha256()
    for key in sorted(snapshot):
        digest.update(key.encode())
        for value in snapshot[key]:
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def snapshot_modules(model):
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            match = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)$", name)
            if not match or match[2] not in MODULES:
                raise ValueError(f"Unexpected LoRA module {name}")
            result[f"{int(match[1]):02d}:{match[2]}"] = (
                module.lora_A["default"].weight.detach().float().cpu().clone(),
                module.lora_B["default"].weight.detach().float().cpu().clone(),
            )
    if len(result) != 40 * 7:
        raise ValueError("Expected all 280 Qwen LoRA modules")
    return result


def restore_modules(model, snapshot):
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                match = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)$", name)
                a, b = snapshot[f"{int(match[1]):02d}:{match[2]}"]
                module.lora_A["default"].weight.copy_(a)
                module.lora_B["default"].weight.copy_(b)
    if fingerprint(snapshot_modules(model)) != fingerprint(snapshot):
        raise ValueError("Checkpoint restoration was not bit-exact")


def encoded(tokenizer, prompt, completion=None):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if completion is None:
        return prefix, len(prefix)
    suffix = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prefix + suffix
    if len(ids) > 256:
        raise ValueError("Toy input exceeds the frozen token limit")
    return ids, len(prefix)


def completion_loss(model, tokenizer, prompt, completion):
    ids, prefix = encoded(tokenizer, prompt, completion)
    ids = torch.tensor([ids], device="cuda")
    labels = ids.clone()
    labels[:, :prefix] = -100
    return model(
        input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False
    ).loss


def load_target():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=list(MODULES),
            bias="none",
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, tokenizer


def status(root, stage, completed, total):
    atomic_json(
        root / "status.json",
        {
            "stage": stage,
            "completed": completed,
            "total": total,
            "updated_unix": time.time(),
            "pid": os.getpid(),
        },
    )


def guard(root):
    plan = read(root / "plan.json")
    for name, digest in plan["source_files_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"Source changed: {name}")
    if file_hash(root / "data.json") != plan["data_sha256"]:
        raise ValueError("Toy inputs changed")
    if file_hash(root / "analysis-plan.json") != plan["analysis_plan_sha256"]:
        raise ValueError("Toy analysis changed")
    if read(root / "controller.json")["deadline_unix"] <= time.time():
        raise TimeoutError("Original deadline passed")
    return plan


def train(root):
    plan = guard(root)
    started = time.time()
    data = read(root / "data.json")
    if (root / "snapshots").exists():
        raise ValueError("Never silently restart a training trajectory")
    status(root, "loading_target", 0, 20)
    model, tokenizer = load_target()
    initial = snapshot_modules(model)
    model.train()
    # Training-only numerical check, restored before the optimizer is constructed.
    preflight_loss = completion_loss(model, tokenizer, "Reply with the word ready.", "ready")
    preflight_loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
        raise ValueError("Preflight gradient failure")
    with torch.no_grad():
        for p in params:
            p.add_(p.grad, alpha=-1e-4)
    changed = fingerprint(snapshot_modules(model)) != fingerprint(initial)
    restore_modules(model, initial)
    with torch.no_grad():
        restored_loss = completion_loss(model, tokenizer, "Reply with the word ready.", "ready")
    if float(restored_loss) != float(preflight_loss.detach()):
        raise ValueError("Preflight score restoration failed")
    model.zero_grad(set_to_none=True)
    if not changed:
        raise ValueError("Preflight did not change the adapter")
    all_rows = [
        (row["prompt"], answer(row["values"], style))
        for row in data["train"] + data["evaluation"]
        for style in STYLES
    ]
    counts = [len(encoded(tokenizer, p, a)[0]) for p, a in all_rows]
    atomic_json(
        root / "preflight.json",
        {
            "passed": True,
            "adapter_restored_exactly": True,
            "trainable_tensors": len(params),
            "trainable_parameters": sum(p.numel() for p in params),
            "maximum_example_tokens": max(counts),
            "dtype": "bfloat16 base; FP32 LoRA parameters",
            "finished_unix": time.time(),
        },
    )
    snapshots = root / "snapshots"
    snapshots.mkdir()
    torch.save(initial, snapshots / "step-00.pt")
    optimizer = torch.optim.AdamW(params, lr=plan["learning_rate"], weight_decay=0.0)
    history, checkpoint_hashes = [], {"0": fingerprint(initial)}
    for step in range(20):
        guard(root)
        style = "numbered_list" if step < 10 else "bulleted_list"
        rows = data["train"][(step % 10) * 4 : (step % 10 + 1) * 4]
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for row in rows:
            loss = completion_loss(model, tokenizer, row["prompt"], answer(row["values"], style))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss_value += float(loss.detach()) / 4
            (loss / 4).backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "style": style,
                "document_ids": [r["id"] for r in rows],
                "loss": loss_value,
                "gradient_norm_before_clip": float(norm),
            }
        )
        if step + 1 in [10, 20]:
            saved = snapshot_modules(model)
            torch.save(saved, snapshots / f"step-{step + 1:02d}.pt")
            torch.save(optimizer.state_dict(), snapshots / f"optimizer-{step + 1:02d}.pt")
            checkpoint_hashes[str(step + 1)] = fingerprint(saved)
        atomic_json(root / "training-history.json", history)
        status(root, "training", step + 1, 20)
    atomic_json(
        root / "training-manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "steps": 20,
            "checkpoint_fingerprints": checkpoint_hashes,
            "snapshot_sha256": {p.name: file_hash(p) for p in sorted(snapshots.glob("*.pt"))},
            "plan_sha256": file_hash(root / "plan.json"),
            "data_sha256": file_hash(root / "data.json"),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def export_tokens(before, after, upstream_svd):
    correct, compatible, audit = [], [], []
    for layer in range(40):
        for module in MODULES:
            key = f"{layer:02d}:{module}"
            a, b = effective_delta_factors(*before[key], *after[key], 2.0)
            u, singular, v = low_rank_svd(a, b)
            cu, cs, cv = upstream_svd(a, b)
            native, compatible_native = (v, cv) if module in READ_MODULES else (u, cu)
            # Audit the exact matrix difference without materializing a giant dense matrix.
            test = torch.randn(a.shape[1], 3, generator=torch.Generator().manual_seed(101 + layer))
            expected = 2 * (
                after[key][1] @ (after[key][0] @ test) - before[key][1] @ (before[key][0] @ test)
            )
            reconstructed = (u * singular) @ (v.T @ test)
            torch.testing.assert_close(reconstructed, expected, atol=2e-5, rtol=3e-4)
            energy = float((singular**2).sum())
            audit.append(
                {
                    "module": key,
                    "full_energy": energy,
                    "retained_energy_fraction": float((singular[:16] ** 2).sum()) / energy
                    if energy
                    else None,
                    "max_product_check_error": float((reconstructed - expected).abs().max()),
                }
            )
            correct.extend(native[:, k] * singular[k] for k in range(16))
            compatible.extend(compatible_native[:, k] * cs[k] for k in range(16))
    return torch.stack(correct), torch.stack(compatible), audit


def interpret(root):
    guard(root)
    started = time.time()
    if (root / "predictions-lock.json").exists():
        raise ValueError("Descriptions already locked")
    if (
        subprocess.check_output(["git", "-C", UPSTREAM, "rev-parse", "HEAD"], text=True).strip()
        != UPSTREAM_REVISION
    ):
        raise ValueError("LoRAcle source revision changed")
    sys.path.insert(0, UPSTREAM)
    token_module = importlib.import_module("extract_hf.tokenize_lora_fixed")
    inference = importlib.import_module("src.train_loracle.inference")
    dataset = importlib.import_module("src.train_loracle.dataset")
    snapshots = {
        step: torch.load(
            root / "snapshots" / f"step-{step:02d}.pt", map_location="cpu", weights_only=True
        )
        for step in [0, 10, 20]
    }
    training = verified_training(root)
    for step, snapshot in snapshots.items():
        if fingerprint(snapshot) != training["checkpoint_fingerprints"][str(step)]:
            raise ValueError("Saved adapter changed")
    token_dir = root / "tokens"
    token_dir.mkdir(exist_ok=False)
    audits = {}
    for index, (case, (before, after)) in enumerate(INTERVALS.items()):
        status(root, "exporting_intervals", index, 3)
        correct, compatible, audit = export_tokens(
            snapshots[before], snapshots[after], token_module._svd_via_qr
        )
        torch.save(
            {"correct_svd": correct, "upstream_compatible": compatible}, token_dir / f"{case}.pt"
        )
        audits[case] = audit
    del snapshots, correct, compatible
    atomic_json(root / "export-audit.json", audits)
    handle = inference.setup_loracle(CHECKPOINT)
    prefix = configure_interpreter_prefix(handle, dataset.build_placeholder_prefix_ids)
    cfg = handle.config
    if Path(cfg.base_model).name != BASE_REVISION:
        raise ValueError("Wrong interpreter base revision")
    metadata = {"base_model": cfg.base_model, "top_k": 16, "n_layers": 40, "d_model": 5120}
    rows = []
    for case in [*INTERVALS, "case-z", "case-p"]:
        values = torch.load(
            token_dir / f"{case if case in INTERVALS else 'case-r'}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for convention in ["correct_svd", "upstream_compatible"]:
            tokens = validate_interpreter_tokens(cfg, metadata, values[convention])
            if case == "case-z":
                tokens = torch.zeros_like(tokens)
            elif case == "case-p":
                tokens = tokens[
                    torch.randperm(len(tokens), generator=torch.Generator().manual_seed(SEED + 1))
                ]
            for query, question in QUESTIONS.items():
                guard(root)
                status(root, "reading_updates", len(rows), 20)
                response = handle.generate(tokens, question, max_new_tokens=192, temperature=0)
                rows.append(
                    {
                        "case": case,
                        "convention": convention,
                        "query": query,
                        "question": question,
                        "response": response,
                        "parsed": parse_prediction(response) if query == "structured" else None,
                    }
                )
                atomic_json(root / "predictions.json", rows)
    atomic_json(
        root / "predictions-lock.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "predictions_sha256": file_hash(root / "predictions.json"),
            "training_manifest_sha256": file_hash(root / "training-manifest.json"),
            "token_sha256": {p.name: file_hash(p) for p in token_dir.glob("*.pt")},
            "upstream_revision": UPSTREAM_REVISION,
            "prefix": prefix,
            "primary_convention": "correct_svd",
            "secondary_convention": "upstream_compatible",
            "compatibility_limit": "Pinned upstream QR helper uses Vh without transposing. Primary "
            "uses correct effective deltas; secondary preserves upstream convention. "
            "No encoding winner promotion; token magnitude is normalized by the interpreter.",
            "checkpoint_config_sha256": file_hash(Path(CHECKPOINT) / "loracle_config.yaml"),
            "interpreter_adapter_sha256": file_hash(
                Path(CHECKPOINT) / "best_ao_interpreter/adapter_model.safetensors"
            ),
            "upstream_files_sha256": {
                name: file_hash(Path(UPSTREAM) / name)
                for name in [
                    "extract_hf/tokenize_lora_fixed.py",
                    "src/train_loracle/inference.py",
                    "src/train_loracle/encoder_ao.py",
                    "src/train_loracle/dataset.py",
                ]
            },
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def evaluate(root):
    guard(root)
    started = time.time()
    lock = read(root / "predictions-lock.json")
    if file_hash(root / "predictions.json") != lock["predictions_sha256"]:
        raise ValueError("Descriptions changed after locking")
    training = verified_training(root)
    if file_hash(root / "training-manifest.json") != lock["training_manifest_sha256"]:
        raise ValueError("Training changed after predictions")
    if (root / "behavior.json").exists():
        raise ValueError("Behavior evaluation already started; preserve failures")
    data = read(root / "data.json")
    model, tokenizer = load_target()
    model.gradient_checkpointing_disable()
    model.eval()
    rows, scores = [], []
    for step in [0, 10, 20]:
        guard(root)
        snapshot = torch.load(
            root / "snapshots" / f"step-{step:02d}.pt", map_location="cpu", weights_only=True
        )
        if fingerprint(snapshot) != training["checkpoint_fingerprints"][str(step)]:
            raise ValueError("Evaluation checkpoint changed")
        restore_modules(model, snapshot)
        for row in data["evaluation"] + data["controls"]:
            status(root, "evaluating_behavior", len(rows), 96)
            ids, prefix = encoded(tokenizer, row["prompt"])
            ids = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                output = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=96,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output[0, prefix:].tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            values = (
                checks(text, row, set(data["test_words"]))
                if "values" in row
                else {"exact_word": text.strip() == row["word"], "empty": not text.strip()}
            )
            values["truncated"] = len(generated) == 96 and generated[-1] != tokenizer.eos_token_id
            rows.append(
                {
                    "step": step,
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "response": text,
                    "token_ids": generated,
                    "checks": values,
                }
            )
            atomic_json(root / "behavior.json", rows)
        with torch.no_grad():
            for row in data["evaluation"]:
                for style in STYLES:
                    value = completion_loss(
                        model, tokenizer, row["prompt"], answer(row["values"], style)
                    )
                    scores.append(
                        {
                            "step": step,
                            "id": row["id"],
                            "style": style,
                            "mean_answer_ce": float(value),
                        }
                    )
        if fingerprint(snapshot_modules(model)) != fingerprint(snapshot):
            raise ValueError("Evaluation altered adapter")
    atomic_json(root / "scores.json", scores)
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "response_count": len(rows),
            "reference_score_count": len(scores),
            "prediction_count": 20,
            "plan_sha256": file_hash(root / "plan.json"),
            "predictions_lock_sha256": file_hash(root / "predictions-lock.json"),
            "output_sha256": {
                name: file_hash(root / name)
                for name in [
                    "behavior.json",
                    "scores.json",
                    "predictions.json",
                    "training-history.json",
                    "training-manifest.json",
                    "export-audit.json",
                    "preflight.json",
                    "data.json",
                ]
            },
            "all_evaluation_adapters_unchanged": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    status(root, "complete", 96, 96)


def verified_training(root):
    training = read(root / "training-manifest.json")
    if training["status"] != "complete" or training["steps"] != 20:
        raise ValueError("Incomplete training")
    if training["plan_sha256"] != file_hash(root / "plan.json"):
        raise ValueError("Training plan changed")
    for name, digest in training["snapshot_sha256"].items():
        if file_hash(root / "snapshots" / name) != digest:
            raise ValueError("Training snapshot changed")
    return training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "interpret", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = guard(args.root)
    for name, digest in plan["input_files_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"Pinned input changed: {name}")
    {"train": train, "interpret": interpret, "evaluate": evaluate}[args.stage](args.root)


if __name__ == "__main__":
    main()
