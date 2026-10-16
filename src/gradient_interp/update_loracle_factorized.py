"""Balanced component questions for the unchanged full single-update effect target."""

import argparse
import collections
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch

from .loracle_trajectory_toy import read, status
from .study import atomic_json, file_hash
from .update_loracle_analysis import require
from .update_loracle_collect import guard
from .update_loracle_data import CASES, FORMATS, LABELS, NO_CHANGE, ORDERS
from .update_loracle_train import (
    OPTION_SYMBOLS,
    choice_forward,
    learning_rate,
    load_interpreter,
    save_adapter,
    shuffled_directions,
    verified_cases,
)

HEAD_LABELS = {
    "change": ("change", NO_CHANGE),
    "format": FORMATS,
    "case": CASES,
    "order": ORDERS,
}
DESCRIPTIONS = {
    "change": "A detectable relative preference change.",
    NO_CHANGE: "No detectable relative preference change.",
    "plain": "A comma-separated list.",
    "bullets": "A bullet list.",
    "numbered": "A numbered list.",
    "json": "A JSON array.",
    "lower": "Lowercase labels.",
    "upper": "Uppercase labels.",
    "original": "The original input order.",
    "reverse": "Reverse input order.",
    "alphabetical": "Alphabetical order.",
    "length": "Increasing word-length order, with alphabetical ties.",
}
OPTION_SEED = 2026101682
TRAINING_SEED = 2026101681


def target_components(label):
    require(label in LABELS, "Unknown measured effect")
    if label == NO_CHANGE:
        return {"change": NO_CHANGE}
    fmt, case, order = label.split("/")
    return {"change": "change", "format": fmt, "case": case, "order": order}


def join_components(answers):
    require(set(answers) == set(HEAD_LABELS), "Every component question must be answered")
    require(
        all(answers[h] in labels for h, labels in HEAD_LABELS.items()), "Invalid component answer"
    )
    if answers["change"] == NO_CHANGE:
        return NO_CHANGE
    label = "/".join(answers[h] for h in ("format", "case", "order"))
    require(label in LABELS, "Invalid combined effect")
    return label


def option_map(case_id, head):
    digest = hashlib.sha256(f"{OPTION_SEED}|{case_id}|{head}".encode()).digest()
    labels = list(HEAD_LABELS[head])
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(labels)
    return dict(zip(OPTION_SYMBOLS[: len(labels)], labels, strict=True))


def question(head, options, norm):
    require(
        head in HEAD_LABELS
        and set(options) == set(OPTION_SYMBOLS[: len(HEAD_LABELS[head])])
        and sorted(options.values()) == sorted(HEAD_LABELS[head]),
        "Incomplete head choices",
    )
    require(np.isfinite(norm) and norm >= 0, "Invalid physical update norm")
    common = (
        "These weights encode one local optimizer update to a fixed Qwen3-14B model. "
        f"Its effective weight-change Frobenius norm is {norm:.6g}. "
        "On new label-list prompts, compare 32 response patterns: comma-separated, bullet, "
        "numbered or JSON lists; lowercase or uppercase; original, reverse, alphabetical "
        "or increasing word-length order with alphabetical ties. A pattern's score is its "
        "average change in mean completion log probability, centered across all 32 patterns. "
    )
    if head == "change":
        task = (
            "Is there a detectable change? Choose detectable when the largest and smallest "
            "pattern scores differ by at least 0.02 nats per token, otherwise choose no change. "
        )
    else:
        aspect = {"format": "list formatting", "case": "letter case", "order": "label ordering"}[
            head
        ]
        task = (
            f"If there is a detectable change, what {aspect} does the single pattern with "
            "the highest score use? Identify this part of the most-improved full pattern, "
            "rather than averaging scores over all patterns sharing that part. "
        )
    return (
        common
        + task
        + "Reply with only the option ID.\n"
        + "\n".join(f"{symbol}. {DESCRIPTIONS[label]}" for symbol, label in options.items())
    )


def question_views(case):
    """Explicitly exclude measured labels, source tasks, training text and probes."""
    result = []
    for head in HEAD_LABELS:
        options = option_map(case["id"], head)
        result.append(
            {
                "id": f"{case['id']}:{head}",
                "case_id": case["id"],
                "head": head,
                "options": options,
                "tokens": case["tokens"],
                "physical_update_norm": case["physical_update_norm"],
                "text": question(head, options, case["physical_update_norm"]),
            }
        )
    return result


def training_instances(cases):
    rows = []
    for case in cases:
        targets = target_components(case["label"])
        for row in question_views(case):
            if row["head"] not in targets:
                continue  # Axes are undefined for no-change; it remains in joint evaluation.
            label = targets[row["head"]]
            symbol = next(k for k, v in row["options"].items() if v == label)
            rows.append(
                {**row, "target_label": label, "target_option": OPTION_SYMBOLS.index(symbol)}
            )
    counts = {
        h: collections.Counter(r["target_label"] for r in rows if r["head"] == h)
        for h in HEAD_LABELS
    }
    require(
        all(set(counts[h]) == set(labels) for h, labels in HEAD_LABELS.items()),
        "Every component class needs training support",
    )
    weights = {
        h: {label: len(rows) / (4 * len(labels) * counts[h][label]) for label in labels}
        for h, labels in HEAD_LABELS.items()
    }
    for row in rows:
        row["weight"] = weights[row["head"]][row["target_label"]]
    return rows, {
        "counts": {h: dict(v) for h, v in counts.items()},
        "weights": weights,
        "question_instances": len(rows),
        "weight_sum": sum(r["weight"] for r in rows),
        "formula": "N_total / (4 * number_of_head_classes * training_class_count)",
    }


def forward(handle, tokens, row, train=False, loss_scale=1.0):
    # The inherited function keeps injection installed through checkpointed backward.
    # Full-vocabulary CE still trains the single ID; inference constrains only valid head IDs.
    target = row["target_option"] if train else None
    count = len(HEAD_LABELS[row["head"]])
    require(set(row["options"]) == set(OPTION_SYMBOLS[:count]), "Invalid answer IDs")
    if train:
        require(
            target in range(count)
            and row["options"][OPTION_SYMBOLS[target]] == row["target_label"],
            "Training label and displayed answer disagree",
        )
    raw = choice_forward(handle, tokens, row["text"], target, loss_scale)
    logits = np.asarray(raw["choice_logits"][:count])
    choice = OPTION_SYMBOLS[int(logits.argmax())]
    return {
        "choice_id": choice,
        "prediction": row["options"][choice],
        "choice_logits": logits.tolist(),
        "full_33_choice_logits": raw["choice_logits"],
        "loss": raw["loss"],
        "input_tokens": raw["input_tokens"],
        "hook_calls": raw["hook_calls"],
    }


@torch.no_grad()
def score(handle, cases, progress=None):
    handle.model.eval()
    results = []
    for i, case in enumerate(cases):
        tokens = torch.load(case["tokens"], map_location="cpu", weights_only=True)
        heads = {row["head"]: forward(handle, tokens, row) for row in question_views(case)}
        results.append(
            {
                "id": case["id"],
                "heads": heads,
                "prediction": join_components({h: r["prediction"] for h, r in heads.items()}),
            }
        )
        if progress:
            progress(i + 1, len(cases))
    return results


def preflight(handle, root):
    require(not (root / "preflight.json").exists(), "Preflight already attempted")
    params = {n: p for n, p in handle.model.named_parameters() if p.requires_grad}
    require(
        len(params) == 560 and sum(p.numel() for p in params.values()) == 1027604480,
        "Unexpected interpreter architecture",
    )
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    frozen_versions = {
        n: p._version for n, p in handle.model.named_parameters() if not p.requires_grad
    }
    generator = torch.Generator().manual_seed(2026101680)
    tokens = torch.randn(4480, 5120, generator=generator)
    case = {"id": "training-only-preflight", "physical_update_norm": 1.0, "tokens": "unused"}
    rows = question_views(case)
    targets = target_components("numbered/upper/reverse")
    for row in rows:
        target = targets[row["head"]]
        symbol = next(k for k, v in row["options"].items() if v == target)
        row.update(target_label=target, target_option=OPTION_SYMBOLS.index(symbol))
    handle.model.eval()
    with torch.no_grad():
        before = [forward(handle, tokens, row)["full_33_choice_logits"] for row in rows]
    handle.model.train()
    handle.model.zero_grad(set_to_none=True)
    values = [forward(handle, tokens, row, train=True, loss_scale=0.25) for row in rows]
    require(all(v["hook_calls"] >= 2 for v in values), "Injection missing during backward")
    require(
        all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values()),
        "Missing or nonfinite interpreter gradients",
    )
    require(
        not any(p.grad is not None for p in handle.model.parameters() if not p.requires_grad),
        "Frozen base received gradients",
    )
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0, error_if_nonfinite=True)
    optimizer = torch.optim.AdamW(list(params.values()), lr=3e-5, weight_decay=0.01, foreach=False)
    optimizer.step()
    require(
        any(not torch.equal(p.detach().cpu(), initial[n]) for n, p in params.items()),
        "Synthetic optimizer step did not change interpreter",
    )
    require(all(torch.isfinite(p).all() for p in params.values()), "Nonfinite interpreter update")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(initial[name].to(parameter.device))
    require(
        all(torch.equal(p.detach().cpu(), initial[n]) for n, p in params.items()),
        "Interpreter adapter did not restore exactly",
    )
    require(
        frozen_versions
        == {n: p._version for n, p in handle.model.named_parameters() if not p.requires_grad},
        "A frozen parameter changed",
    )
    del optimizer, initial
    handle.model.zero_grad(set_to_none=True)
    handle.model.eval()
    with torch.no_grad():
        restored = [forward(handle, tokens, row)["full_33_choice_logits"] for row in rows]
    require(before == restored, "Interpreter scores did not restore exactly")
    atomic_json(
        root / "preflight.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "training_only": True,
            "heads": list(HEAD_LABELS),
            "trainable_tensors": 560,
            "trainable_parameters": 1027604480,
            "all_gradients_finite": True,
            "frozen_base_unchanged": True,
            "bit_exact_score_restore": True,
            "bit_exact_adapter_restore": True,
            "hook_calls": [v["hook_calls"] for v in values],
            "input_tokens": [v["input_tokens"] for v in values],
            "gradient_norm": float(norm),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def training_majorities(cases, balance):
    components = {
        head: max(labels, key=lambda label: balance["counts"][head][label])
        for head, labels in HEAD_LABELS.items()
    }
    counts = collections.Counter(c["label"] for c in cases)
    return {
        "flat": max(LABELS, key=lambda x: counts[x]),
        "factorized": join_components(components),
        "components": components,
        "label_counts": dict(counts),
    }


def train(root):
    plan = guard(root)
    require(
        plan["recipe"] == "balanced_factorized_v2"
        and plan["epochs"] == 3
        and plan["effective_batch"] == 4
        and plan["learning_rate"] == 0.00003
        and plan["training_seed"] == TRAINING_SEED,
        "Training recipe changed",
    )
    require(not (root / "epochs").exists(), "Training already attempted; no automatic resume")
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, "Frozen training input changed")
    collection = Path(plan["collection_root"])
    require(
        file_hash(collection / "integrity.json") == plan["collection_integrity_sha256"],
        "Collection audit changed",
    )
    train_cases = verified_cases(collection, "train")
    dev_cases = verified_cases(collection, "development")
    instances, balance = training_instances(train_cases)
    require(len(instances) == 1040, "Training question grid changed")
    shuffled = shuffled_directions(dev_cases)
    handle = load_interpreter()
    preflight(handle, root)
    atomic_json(root / "balance.json", balance)
    atomic_json(root / "training-majorities.json", training_majorities(train_cases, balance))
    simple = read(Path(plan["prior_training_root"]) / "simple-references.json")
    require(
        [r["id"] for r in simple["development"]] == [c["id"] for c in dev_cases],
        "Frozen simple baseline grid differs",
    )
    atomic_json(root / "simple-references.json", simple)
    for name, cases in (("released", dev_cases), ("released-shuffled", shuffled)):
        atomic_json(
            root / f"{name}-development.json",
            score(
                handle,
                cases,
                lambda n, total: status(root, name, n, total),
            ),
        )
    atomic_json(
        root / "direction-shuffle.json",
        [{"id": c["id"], "donor": c["direction_donor"]} for c in shuffled],
    )
    params = [p for p in handle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.00003, weight_decay=0.01, foreach=False)
    total, warmup = 780, 39
    step, history = 0, []
    (root / "epochs").mkdir()
    for epoch in range(3):
        instances_shuffled = list(instances)
        random.Random(TRAINING_SEED + epoch).shuffle(instances_shuffled)
        handle.model.train()
        for begin in range(0, len(instances_shuffled), 4):
            guard(root)
            group = instances_shuffled[begin : begin + 4]
            optimizer.zero_grad(set_to_none=True)
            scored = []
            for instance in group:
                tokens = torch.load(instance["tokens"], map_location="cpu", weights_only=True)
                scored.append(
                    forward(handle, tokens, instance, True, instance["weight"] / len(group))
                )
            require(
                all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),
                "Missing or nonfinite interpreter gradients",
            )
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            rate = learning_rate(step, total, warmup, 0.00003)
            for settings in optimizer.param_groups:
                settings["lr"] = rate
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "instances": [r["id"] for r in group],
                    "weights": [r["weight"] for r in group],
                    "unweighted_losses": [r["loss"] for r in scored],
                    "weighted_loss": float(
                        np.mean([r["loss"] * q["weight"] for r, q in zip(scored, group)])
                    ),
                    "gradient_norm": float(norm),
                    "learning_rate": rate,
                }
            )
            step += 1
            status(root, "factorized_training", step, total)
        path = root / "epochs" / str(epoch + 1)
        save_adapter(handle.model, path)
        atomic_json(
            path / "development.json",
            score(
                handle,
                dev_cases,
                lambda n, total: status(root, f"epoch{epoch + 1}_development", n, total),
            ),
        )
        atomic_json(
            path / "shuffled-development.json",
            score(
                handle,
                shuffled,
                lambda n, total: status(root, f"epoch{epoch + 1}_shuffled", n, total),
            ),
        )
        atomic_json(root / "training-history.json", history)
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "finished_unix": time.time(),
            "optimizer_steps": step,
            "epochs": 3,
            "training_cases": 272,
            "training_questions": 1040,
            "development_cases": 72,
            "questions_per_evaluation_case": 4,
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
                        "preflight.json",
                        "balance.json",
                        "training-majorities.json",
                        "simple-references.json",
                        "released-development.json",
                        "released-shuffled-development.json",
                        "direction-shuffle.json",
                        "training-history.json",
                    )
                )
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026101680)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train(args.root)
