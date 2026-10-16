"""Train and score the released interpreter on measured single-update effects."""

import argparse
import gc
import importlib
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .loracle import configure_interpreter_prefix, validate_interpreter_tokens
from .loracle_trajectory_toy import BASE_PATH, CHECKPOINT, UPSTREAM, read, status
from .study import atomic_json, file_hash
from .update_loracle_data import LABELS, question

MAX_LENGTH = 6144


def option_token_ids(tokenizer):
    values = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(33)]
    if any(len(v) != 1 for v in values) or len({v[0] for v in values}) != 33:
        raise ValueError("All 33 choice numbers must be distinct single tokens")
    return [v[0] for v in values]


def prepared_input(handle, text):
    """Exactly the released chat and rank-tagged prefix, without answer leakage."""
    if getattr(handle.config, "use_system_prompt", True):
        raise ValueError("This checkpoint was frozen without a system prompt")
    chat = handle.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    chat_ids = handle.tokenizer(chat, add_special_tokens=False)["input_ids"]
    ids = list(handle.prefix_ids) + chat_ids
    if len(ids) > MAX_LENGTH:
        raise ValueError("Interpreter input exceeds the frozen limit; never truncate")
    mask = list(handle.prefix_mask) + [False] * len(chat_ids)
    if sum(mask) != 4480:
        raise ValueError("Wrong number of injected directions")
    return torch.tensor([ids], device=handle.device), torch.tensor(
        [mask], dtype=torch.bool, device=handle.device
    )


def reordered(tokens, config):
    return validate_interpreter_tokens(
        config,
        {
            "base_model": BASE_PATH,
            "top_k": 16,
            "n_layers": 40,
            "d_model": 5120,
        },
        tokens,
    )


@contextmanager
def installed_hook(layer, hook):
    """Keep the injection installed THROUGH backward checkpoint recomputation."""
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def choice_forward(handle, tokens, text, target_option=None, loss_scale=1.0):
    inference = importlib.import_module("src.train_loracle.inference")
    ids, pmask = prepared_input(handle, text)
    option_ids = option_token_ids(handle.tokenizer)
    direction = handle.encoder.encode_lora(
        reordered(tokens, handle.config).to(device=handle.device, dtype=torch.bfloat16)
    )
    hook = inference._make_injection_hook(direction, pmask)
    calls = [0]

    def observed_hook(*args):
        calls[0] += 1
        return hook(*args)

    with installed_hook(inference._layer1(handle.model), observed_hook):
        output = handle.model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1
        )
        logits = output.logits[:, -1, :].float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite interpreter scores")
        loss = None
        if target_option is not None:
            if target_option not in range(33) or loss_scale <= 0:
                raise ValueError("Invalid supervised target or loss scaling")
            target = torch.tensor([option_ids[target_option]], device=handle.device)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised loss")
            (loss * loss_scale).backward()
    values = logits[0, option_ids].detach().double().cpu().numpy()
    return {
        "choice_logits": values.tolist(),
        "choice": str(int(values.argmax())),
        "loss": float(loss.detach()) if loss is not None else None,
        "hook_calls": calls[0],
        "input_tokens": ids.shape[1],
    }


def load_interpreter():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    inference = importlib.import_module("src.train_loracle.inference")
    dataset = importlib.import_module("src.train_loracle.dataset")
    config = importlib.import_module("src.train_loracle.config").load_train_config(
        Path(CHECKPOINT) / "loracle_config.yaml"
    )
    if config.base_model != BASE_PATH or config.interpreter_rank != 256:
        raise ValueError("Unexpected interpreter checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).to("cuda")
    model = PeftModel.from_pretrained(
        base, Path(CHECKPOINT) / "best_ao_interpreter", is_trainable=True, local_files_only=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    encoder = importlib.import_module("src.train_loracle.encoder_ao").AOEncoder()
    encoder = encoder.to(device="cuda", dtype=torch.bfloat16)
    encoder.requires_grad_(False)
    handle = inference.LoracleHandle(
        config, tokenizer, model, encoder, [], [], torch.device("cuda")
    )
    configure_interpreter_prefix(handle, dataset.build_placeholder_prefix_ids)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(named) != 560 or sum(p.numel() for _, p in named) != 1027604480:
        raise ValueError("Expected all 560 released rank256 LoRA tensors")
    if any("lora_" not in n or p.dtype != torch.float32 for n, p in named):
        raise ValueError("Only FP32 interpreter LoRA parameters may train")
    model.eval()
    return handle


def learning_rate(step, total, warmup, maximum):
    if not 0 <= step < total or not 0 <= warmup < total:
        raise ValueError("Invalid scheduler state")
    if step < warmup:
        return maximum * (step + 1) / max(warmup, 1)
    return maximum * (total - step) / (total - warmup)


def preflight(handle, tokens, output):
    if output.exists():
        raise ValueError("Preflight already attempted")
    params = {n: p for n, p in handle.model.named_parameters() if p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    options = dict(zip(map(str, range(33)), LABELS))
    text = question(options, 1.0)
    handle.model.eval()
    with torch.no_grad():
        before = choice_forward(handle, tokens, text)
    handle.model.train()
    handle.model.zero_grad(set_to_none=True)
    checked = choice_forward(handle, tokens, text, target_option=0)
    if checked["hook_calls"] < 2:
        raise ValueError("Injection hook did not run during checkpointed backward")
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values()):
        raise ValueError("Missing or nonfinite interpreter gradients")
    if any(p.grad is not None for p in handle.model.parameters() if not p.requires_grad):
        raise ValueError("A frozen base parameter received a gradient")
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0, error_if_nonfinite=True)
    optimizer = torch.optim.AdamW(
        list(params.values()), lr=0.00003, weight_decay=0.01, foreach=False
    )
    optimizer.step()
    changed = any(not torch.equal(p.detach().cpu(), initial[n]) for n, p in params.items())
    if not changed:
        raise ValueError("Interpreter optimizer had no effect")
    peak = torch.cuda.max_memory_allocated()
    del optimizer
    handle.model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(initial[n])
    if not all(torch.equal(p.detach().cpu(), initial[n]) for n, p in params.items()):
        raise ValueError("Interpreter adapter did not restore exactly")
    handle.model.eval()
    with torch.no_grad():
        restored = choice_forward(handle, tokens, text)
    if before["choice_logits"] != restored["choice_logits"]:
        raise ValueError("Restored interpreter scores differ")
    atomic_json(
        output,
        {
            "passed": True,
            "finished_unix": time.time(),
            "trainable_tensors": len(params),
            "trainable_parameters": 1027604480,
            "hook_calls_with_backward": checked["hook_calls"],
            "all_gradients_finite": True,
            "gradient_norm": float(norm),
            "bit_exact_adapter_and_scores_restore": True,
            "input_tokens": checked["input_tokens"],
            "choices_single_token": 33,
            "peak_allocated_bytes": peak,
            "loss": checked["loss"],
        },
    )
    del initial
    gc.collect()
    torch.cuda.empty_cache()


def verified_cases(collection, split):
    integrity = read(collection / "integrity.json")
    if not integrity.get("passed") or integrity.get("scope") != "development_only":
        raise ValueError("A complete audited collection is required")
    for name, expected in integrity["files_sha256"].items():
        if file_hash(collection / name) != expected:
            raise ValueError(f"Audited collection changed: {name}")
    if split not in ("train", "development"):
        raise ValueError("Training module cannot access reserved test")
    data = read(collection / "data.json")
    selected = []
    for case in data["cases"]:
        if case["split"] != split:
            continue
        result = read(collection / "cases" / f"{case['id']}.json")
        token_path = collection / "tokens" / f"{case['id']}.pt"
        if file_hash(token_path) != result["tokens_sha256"]:
            raise ValueError("Direction tokens changed after audit")
        selected.append(
            {
                "id": case["id"],
                "tokens": token_path,
                "control": case["control"],
                "text": question(case["options"], result["physical_update_norm"]),
                "options": case["options"],
                "label": result["effect"]["label"],
                "physical_update_norm": result["physical_update_norm"],
            }
        )
    return selected


@torch.no_grad()
def score(handle, cases):
    handle.model.eval()
    result = []
    for case in cases:
        tokens = torch.load(case["tokens"], map_location="cpu", weights_only=True)
        row = choice_forward(handle, tokens, case["text"])
        result.append({"id": case["id"], **row, "prediction": case["options"][row["choice"]]})
    return result


def shuffled_directions(cases, seed=2026101652):
    """Fixed nonzero-case derangement; retain recipient question, norm and options."""
    eligible = [i for i, c in enumerate(cases) if not c["control"]]
    if len(eligible) < 2:
        raise ValueError("Need at least two actual updates for the direction control")
    rng = random.Random(seed)
    while True:
        donors = rng.sample(eligible, len(eligible))
        if all(a != b for a, b in zip(eligible, donors)):
            break
    mapping = dict(zip(eligible, donors))
    return [
        {
            **c,
            "tokens": cases[mapping.get(i, i)]["tokens"],
            "direction_donor": cases[mapping.get(i, i)]["id"],
        }
        for i, c in enumerate(cases)
    ]


def direction_features(tokens, physical_norm):
    """Seven module-wise mean unit vectors plus log1p norm; no measured-effect inputs."""
    if tokens.shape != (4480, 5120) or not torch.isfinite(tokens).all():
        raise ValueError("Invalid direction features")
    if not np.isfinite(physical_norm) or physical_norm < 0:
        raise ValueError("Invalid update magnitude")
    values = tokens.float().reshape(40, 7, 16, 5120)
    values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    means = values.mean(dim=(0, 2)).reshape(-1).double().numpy()
    return np.concatenate([means, [np.log1p(physical_norm)]])


def simple_references(train_cases, dev_cases):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def features(cases):
        return np.stack(
            [
                direction_features(
                    torch.load(c["tokens"], map_location="cpu", weights_only=True),
                    c["physical_update_norm"],
                )
                for c in cases
            ]
        )

    labels = [c["label"] for c in train_cases]
    counts = {label: labels.count(label) for label in LABELS}
    majority = max(LABELS, key=lambda x: counts[x])
    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0, solver="lsqr", tol=1e-8))
    model.fit(features(train_cases), labels)
    predicted = model.predict(features(dev_cases)).tolist()
    shuffled = shuffled_directions(dev_cases)
    shuffled_predictions = model.predict(features(shuffled)).tolist()
    return {
        "recipe": "training-only scaling, ridge alpha1, lsqr tol1e-8; no tuning",
        "feature_count": 35841,
        "majority": majority,
        "training_label_counts": counts,
        "development": [
            {
                "id": c["id"],
                "ridge": p,
                "majority": majority,
                "shuffled_ridge": s,
                "direction_donor": donor["direction_donor"],
            }
            for c, p, s, donor in zip(dev_cases, predicted, shuffled_predictions, shuffled)
        ],
    }


def save_adapter(model, path):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=False)
    weights = {
        n: value.detach().cpu().contiguous()
        for n, value in get_peft_model_state_dict(model).items()
    }
    save_file(weights, str(path / "adapter_model.safetensors"))
    model.peft_config["default"].save_pretrained(path)


def train(root):
    # This is a new phase: its complete plan and input/preflight hashes must be frozen first.
    from .update_loracle_collect import guard

    plan = guard(root)
    for path, expected in plan["input_files_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen training input changed: {path}")
    if (plan["epochs"], plan["effective_batch"], plan["learning_rate"]) != (3, 4, 0.00003):
        raise ValueError("This worker implements only the first frozen development recipe")
    if (root / "epochs").exists():
        raise ValueError("Training was already attempted; no automatic resume")
    collection = Path(plan["collection_root"])
    if file_hash(collection / "integrity.json") != plan["collection_integrity_sha256"]:
        raise ValueError("Collection audit changed")
    train_cases = verified_cases(collection, "train")
    dev_cases = verified_cases(collection, "development")
    atomic_json(root / "simple-references.json", simple_references(train_cases, dev_cases))
    shuffled_dev = shuffled_directions(dev_cases)
    handle = load_interpreter()
    # A deterministic synthetic token tensor tests only the training machinery.
    rng = torch.Generator().manual_seed(2026101650)
    synthetic = torch.randn(4480, 5120, generator=rng)
    preflight(handle, synthetic, root / "preflight.json")
    del synthetic
    status(root, "released_baseline", 0, len(dev_cases))
    atomic_json(root / "released-development.json", score(handle, dev_cases))
    atomic_json(root / "released-shuffled-development.json", score(handle, shuffled_dev))
    atomic_json(
        root / "direction-shuffle.json",
        [{"id": c["id"], "donor": c["direction_donor"]} for c in shuffled_dev],
    )
    params = [p for p in handle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=plan["learning_rate"], weight_decay=0.01, foreach=False
    )
    total_steps = math.ceil(len(train_cases) / plan["effective_batch"]) * plan["epochs"]
    warmup = math.ceil(0.05 * total_steps)
    step, history = 0, []
    (root / "epochs").mkdir()
    for epoch in range(plan["epochs"]):
        shuffled = list(train_cases)
        random.Random(plan["training_seed"] + epoch).shuffle(shuffled)
        handle.model.train()
        for begin in range(0, len(shuffled), plan["effective_batch"]):
            guard(root)
            group = shuffled[begin : begin + plan["effective_batch"]]
            optimizer.zero_grad(set_to_none=True)
            rows = []
            for case in group:
                tokens = torch.load(case["tokens"], map_location="cpu", weights_only=True)
                number = next(int(k) for k, v in case["options"].items() if v == case["label"])
                rows.append(choice_forward(handle, tokens, case["text"], number, 1 / len(group)))
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
                raise ValueError("Missing or nonfinite interpreter gradients")
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            rate = learning_rate(step, total_steps, warmup, plan["learning_rate"])
            for group_setting in optimizer.param_groups:
                group_setting["lr"] = rate
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "ids": [c["id"] for c in group],
                    "loss": float(np.mean([r["loss"] for r in rows])),
                    "gradient_norm": float(norm),
                    "learning_rate": rate,
                }
            )
            step += 1
            status(root, "supervised_training", step, total_steps)
        path = root / "epochs" / str(epoch + 1)
        save_adapter(handle.model, path)
        atomic_json(path / "development.json", score(handle, dev_cases))
        atomic_json(path / "shuffled-development.json", score(handle, shuffled_dev))
        atomic_json(root / "training-history.json", history)
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "finished_unix": time.time(),
            "optimizer_steps": step,
            "epochs": plan["epochs"],
            "training_cases": len(train_cases),
            "development_cases": len(dev_cases),
            "reserved_cases_accessed": 0,
            "plan_sha256": file_hash(root / "plan.json"),
            "source_commit": plan["source_commit"],
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p)
                for p in root.rglob("*")
                if p.is_file()
                and (
                    "epochs" in p.parts
                    or p.name
                    in (
                        "training-history.json",
                        "released-development.json",
                        "preflight.json",
                        "released-shuffled-development.json",
                        "simple-references.json",
                        "direction-shuffle.json",
                    )
                )
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "preflight"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026101650)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "train":
        train(args.root)
    else:
        args.root.mkdir(parents=True, exist_ok=True)
        handle = load_interpreter()
        rng = torch.Generator().manual_seed(2026101650)
        tokens = torch.randn(4480, 5120, generator=rng)
        preflight(handle, tokens, args.root / "preflight.json")
        atomic_json(
            args.root / "source.json",
            {
                "source_sha256": file_hash(__file__),
                "finished_unix": time.time(),
                "pid": os.getpid(),
                "scientific_outputs_read": False,
            },
        )


if __name__ == "__main__":
    main()
