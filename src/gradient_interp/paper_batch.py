"""Synthetic development pilot: prospective all-layer LoRA batch forecasts."""

# ruff: noqa: E501
# Explicit protocol strings are saved verbatim in the frozen specification.

import argparse
import itertools
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .confirmation import verify_files
from .paper_setup import load_model, read_json, upstream
from .paper_steering import generate, parameter_state, restore, set_direction
from .study import atomic_json, file_hash

METHODS = ("full_gradient", "ekfac_projection", "dictionary", "svd16", "mean_direction", "zero")
STYLES = ("plain_short", "numbered_short", "plain_long", "numbered_long")


def alternatives(tags):
    long = [
        f"The {word} catalog tag is {tag}."
        for word, tag in zip(("first", "second", "third"), tags, strict=True)
    ]
    return [
        ", ".join(tags) + ".",
        "\n".join(f"{i + 1}. {x}" for i, x in enumerate(tags)),
        " ".join(long),
        "\n".join(f"{i + 1}. {x}" for i, x in enumerate(long)),
    ]


def make_data():
    rng = random.Random(20261013)
    used = set()
    rows = []
    for i in range(96):
        tags = []
        while len(tags) < 3:
            tag = f"Z{rng.randrange(16**6):06X}"
            if tag not in used:
                used.add(tag)
                tags.append(tag)
        if i < 80:
            prompt = f"Catalog entry {i:03d} contains these three tags: {', '.join(tags)}. Return all three tags exactly once, in the given order, with no other tags. Choose your own presentation."
        else:
            prompt = f"Read this inventory record: {' / '.join(tags)}. Copy its three tags once each, preserving their order. Do not add any tags. You may choose the presentation."
        rows.append(
            {
                "id": f"catalog-{i:03d}",
                "prompt": prompt,
                "tags": tags,
                "answers": alternatives(tags),
            }
        )
    batches = [
        {
            "id": f"batch-{i:02d}",
            "style": STYLES[i % 4],
            "style_index": i % 4,
            "documents": rows[4 * i : 4 * i + 4],
        }
        for i in range(16)
    ]
    return {"batches": batches, "calibration": rows[64:80], "evaluation": rows[80:96]}


def generated_checks(answer, tags):
    found = re.findall(r"Z[0-9A-F]{6}", answer)
    return {
        "exact_content_and_order": found == tags,
        "all_tags_present": all(tag in found for tag in tags),
        "numbered": len(re.findall(r"^\s*\d+[.)]\s+", answer, re.M)) >= 2,
        "words": len(answer.split()),
        "empty": not answer.strip(),
    }


def scalar_logp(model, tokenizer, tokenize, prompt, answer):
    encoded = tokenize(
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        },
        tokenizer,
        max_length=500,
    )
    batch = {key: torch.tensor([value], device="cuda") for key, value in encoded.items()}
    logits = (
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        .logits[:, :-1]
        .float()
    )
    labels = batch["labels"][:, 1:]
    n = int((labels != -100).sum())
    if not n:
        raise ValueError("Empty scored completion")
    return -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum"
    ), n


def flat_grad(value, parameters):
    grads = torch.autograd.grad(value, [p for _, p in parameters])
    result = torch.cat([g.detach().float().reshape(-1).cpu() for g in grads])
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite gradient")
    return result


def prepare(root, source):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "specification.json").exists():
        raise ValueError("Already prepared")
    source = source.resolve()
    if not read_json(source / "integrity.json")["complete"]:
        raise ValueError("Completed source audit required")
    data = make_data()
    atomic_json(root / "data.json", data)
    config = read_json(source / "configuration.json")
    spec = {
        "scope": "synthetic_development_only",
        "source_root": str(source),
        "source_config": config,
        "training_seed": config["seed"],
        "max_hours": 12,
        "rates": [0.0003, 0.003, 0.03],
        "methods": list(METHODS),
        "batches": 16,
        "batch_size": 4,
        "calibration_questions": 16,
        "evaluation_questions": 16,
        "max_new_tokens": 120,
        "update": "One SGD step on the mean of per-document mean completion CE, all136 LoRA A/B tensors, model.eval, dropout disabled; unchanged starting adapter for each batch and rate",
        "scores": "Four fixed completions crossing plain/numbered and short/long, identical requested tags. Report summed sequence log probabilities and per-token averages including template ending. Scores and four-way normalized probabilities do not equal unrestricted generated-behavior probabilities.",
        "forecast": "Lock all per-question four-completion Taylor forecasts before any updates. Full raw gradient, retained6800dim EKFAC, frozen500-atom dictionary lasso alpha.1 on normalized batch projection with norm restored, SVD16 of normalized fitting projections, normalized mean fitting projection, zero. Reconstruction reverses original preconditioning; no learned outcome regression.",
        "svd": {"components": 16, "iterations": 7, "seed": 8173},
        "generations": "Baseline and after all48 updates on16 evaluation questions. Score exact tags and order, numbered detector, length, truncation. Both calibration-template and unseen-template probes are synthetic development only.",
        "limitations": [
            "Selected after completed steering development, not confirmation",
            "Two reused training seeds with different GPUs",
            "16 batches and16 evaluation prompts are small",
            "Four-choice sequence scores are surrogates; free generations checked separately",
            "SVD16 is a simple baseline, not a universal equal-compute or equal-capacity match",
            "Synthetic copying with known labels does not establish general atom semantics",
            "New synthetic prompts are unique internally; no claim of untouched confirmation or exhaustive historical deduplication",
        ],
        "input_hashes": {
            str(p): file_hash(p)
            for p in (
                source / "configuration.json",
                source / "integrity.json",
                source / "factors-manifest.json",
                source / "atoms.pt",
                source / "adapter/adapter_model.safetensors",
                source / "adapter/adapter_config.json",
            )
        },
    }
    atomic_json(root / "specification.json", spec)


def setup(spec):
    from transformers import AutoTokenizer

    helpers = upstream(spec["source_config"])
    model, _ = load_model(spec["source_config"], Path(spec["source_root"]) / "adapter")
    model.eval()
    scoring = helpers.get_tokenizer(spec["source_config"]["base_directory"])
    generation = AutoTokenizer.from_pretrained(
        spec["source_config"]["base_directory"], local_files_only=True, padding_side="left"
    )
    payload = torch.load(
        Path(spec["source_root"]) / "atoms.pt", map_location="cpu", weights_only=True
    )
    parameters = parameter_state(model, payload["lora_names"])
    return model, scoring, generation, helpers.tokenize_chat, payload, parameters


def preflight(root):
    spec = read_json(root / "specification.json")
    row = read_json(root / "data.json")["batches"][0]["documents"][0]
    model, tokenizer, gen_tok, tokenize, payload, parameters = setup(spec)
    original = {name: p.detach().clone() for name, p in parameters}
    value, n = scalar_logp(model, tokenizer, tokenize, row["prompt"], row["answers"][0])
    gradient = flat_grad(-value / n, parameters)
    _, before = generate(model, gen_tok, [row["prompt"]], 12)
    set_direction(parameters, original, gradient, spec["rates"][0])
    assert any(not torch.equal(p, original[name]) for name, p in parameters)
    restore(parameters, original)
    _, after = generate(model, gen_tok, [row["prompt"]], 12)
    assert before == after
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "gradient_norm": gradient.norm().item(),
            "parameter_tensors": len(parameters),
            "adapter_restored": True,
            "generation_restored": True,
        },
    )


def freeze(root):
    spec = read_json(root / "specification.json")
    if (root / "plan.json").exists():
        raise ValueError("Already frozen")
    assert read_json(root / "preflight.json")["specification_sha256"] == file_hash(
        root / "specification.json"
    )
    atomic_json(
        root / "plan.json",
        {
            "scope": "development_only",
            "frozen_unix": time.time(),
            "max_hours": spec["max_hours"],
            "source_files_sha256": {
                str(p): file_hash(p) for p in Path("src/gradient_interp").glob("*.py")
            },
            "input_hashes": {
                str(root / name): file_hash(root / name)
                for name in ("specification.json", "data.json", "preflight.json")
            },
            "jobs": [
                {
                    "id": "batch-forecast",
                    "module": "gradient_interp.paper_batch",
                    "arguments": ["run", "--root", str(root.resolve())],
                    "manifest": str(root.resolve() / "manifest.json"),
                }
            ],
        },
    )


def run(root):
    from sklearn.decomposition import TruncatedSVD, sparse_encode

    plan, spec, data = [
        read_json(root / name) for name in ("plan.json", "specification.json", "data.json")
    ]
    verify_files(Path.cwd(), plan["source_files_sha256"])
    verify_files(Path.cwd(), plan["input_hashes"])
    verify_files(Path.cwd(), spec["input_hashes"])
    verify_files(
        Path(spec["source_root"]),
        read_json(Path(spec["source_root"]) / "factors-manifest.json")["files_sha256"],
    )
    started = time.time()

    def status(stage, **extra):
        atomic_json(root / "status.json", {"stage": stage, "updated_unix": time.time(), **extra})
        print(stage, extra, flush=True)

    model, tokenizer, gen_tok, tokenize, payload, parameters = setup(spec)
    from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac, unproject_atom

    original = {name: p.detach().clone() for name, p in parameters}
    probes = data["calibration"] + data["evaluation"]
    cache = root / "forecasts.pt"
    lock = root / "forecast-lock.json"
    if lock.exists():
        verify_files(root, read_json(lock)["files_sha256"])
        saved = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        # All gradient and representation choices are frozen before this pass.
        scores, jacobians, lengths = [], [], []
        for i, row in enumerate(probes):
            values, grads, counts = [], [], []
            for answer in row["answers"]:
                value, n = scalar_logp(model, tokenizer, tokenize, row["prompt"], answer)
                values.append(value.detach().cpu())
                counts.append(n)
                grads.append(flat_grad(value, parameters))
            scores.append(torch.stack(values))
            lengths.append(counts)
            jacobians.append(torch.stack(grads))
            status("probe_gradients", completed=i + 1, total=len(probes))
        baseline = torch.stack(scores)
        jacobian = torch.stack(jacobians)
        gradients = []
        for i, batch in enumerate(data["batches"]):
            grads = []
            for row in batch["documents"]:
                value, n = scalar_logp(
                    model, tokenizer, tokenize, row["prompt"], row["answers"][batch["style_index"]]
                )
                grads.append(flat_grad(-value / n, parameters))
            gradients.append(torch.stack(grads).mean(0))
            status("batch_gradients", completed=i + 1, total=len(data["batches"]))
        gradients = torch.stack(gradients)
        factors = load_ekfac_eigen(
            str(Path(spec["source_root"]) / "factors/paper_ekfac/factors_training")
        )
        projected, module_info = project_gradients_ekfac(
            gradients, payload["lora_names"], factors, 50
        )
        fitting = payload["projected_gradients"].numpy()
        fitting = fitting / np.maximum(np.linalg.norm(fitting, axis=1, keepdims=True), 1e-12)
        svd = TruncatedSVD(n_components=16, n_iter=7, random_state=8173).fit(fitting)
        basis = torch.from_numpy(svd.components_).float()
        mean = torch.from_numpy(fitting.mean(0)).float()
        mean /= mean.norm().clamp_min(1e-12)
        norms = projected.norm(dim=1, keepdim=True).clamp_min(1e-12)
        codes = sparse_encode(
            (projected / norms).numpy(),
            payload["dictionary"].numpy(),
            algorithm="lasso_lars",
            alpha=0.1,
        )
        variants = {
            "ekfac_projection": projected,
            "dictionary": torch.from_numpy(codes).float() @ payload["dictionary"].float() * norms,
            "svd16": projected @ basis.T @ basis,
            "mean_direction": (projected @ mean)[:, None] * mean,
            "zero": torch.zeros_like(projected),
        }
        forecasts = torch.empty(len(gradients), len(METHODS), len(probes), 4)
        for i, gradient in enumerate(gradients):
            for j, method in enumerate(METHODS):
                approx = (
                    gradient
                    if method == "full_gradient"
                    else unproject_atom(
                        variants[method][i].numpy(), module_info, factors, len(gradient)
                    )
                )
                forecasts[i, j] = -(jacobian @ approx)
            status("forecasting", completed=i + 1, total=len(gradients))
        saved = {
            "baseline_logp": baseline,
            "completion_tokens": torch.tensor(lengths),
            "unit_rate_forecasts": forecasts,
            "batch_gradients": gradients,
            "sparse_codes": torch.from_numpy(codes).float(),
            "projected_norms": norms,
            "svd_basis": basis,
            "mean_projected": mean,
        }
        torch.save(saved, cache)
        atomic_json(
            lock,
            {
                "locked_unix": time.time(),
                "plan_sha256": file_hash(root / "plan.json"),
                "methods": list(METHODS),
                "files_sha256": {cache.name: file_hash(cache)},
                "all_forecasts_before_updates": True,
            },
        )
        del jacobian, jacobians, factors
    directory = root / "outcomes"
    directory.mkdir(exist_ok=True)

    def responses():
        texts, tokens = generate(
            model, gen_tok, [p["prompt"] for p in data["evaluation"]], spec["max_new_tokens"]
        )
        return [
            {
                "id": p["id"],
                "response": text,
                "token_ids": ids,
                "checks": generated_checks(text, p["tags"]),
            }
            for p, text, ids in zip(data["evaluation"], texts, tokens, strict=True)
        ]

    baseline_path = directory / "baseline.json"
    if not baseline_path.exists():
        atomic_json(
            baseline_path,
            {"plan_sha256": file_hash(root / "plan.json"), "generations": responses()},
        )
    completed = [baseline_path]
    for i, rate in itertools.product(range(len(data["batches"])), spec["rates"]):
        out = directory / f"batch-{i:02d}-rate-{rate:g}.json"
        if out.exists():
            assert read_json(out)["forecast_lock_sha256"] == file_hash(lock)
            completed.append(out)
            continue
        restore(parameters, original)
        set_direction(parameters, original, saved["batch_gradients"][i], rate)
        try:
            scores = []
            with torch.no_grad():
                for row in probes:
                    scores.append(
                        [
                            scalar_logp(model, tokenizer, tokenize, row["prompt"], answer)[0].item()
                            for answer in row["answers"]
                        ]
                    )
            generations = responses()
        finally:
            restore(parameters, original)
        atomic_json(
            out,
            {
                "plan_sha256": file_hash(root / "plan.json"),
                "forecast_lock_sha256": file_hash(lock),
                "batch_index": i,
                "style": data["batches"][i]["style"],
                "rate": rate,
                "actual_logp": scores,
                "generations": generations,
                "adapter_restored": True,
            },
        )
        completed.append(out)
        status("updates", completed=len(completed) - 1, total=48)
    restore(parameters, original)
    assert len(completed) == 49
    atomic_json(
        root / "manifest.json",
        {
            "started_unix": started,
            "completed_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "forecast_lock_sha256": file_hash(lock),
            "updates": 48,
            "baseline_generations": 16,
            "post_update_generations": 768,
            "adapter_restored": True,
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p) for p in [cache, lock, *completed]
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "freeze", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "prepare":
        prepare(args.root.resolve(), args.source)
    else:
        globals()[args.command](args.root.resolve())


if __name__ == "__main__":
    main()
