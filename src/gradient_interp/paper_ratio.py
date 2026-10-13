"""Post-update likelihood-ratio diagnostic; deliberately not a forecasting method."""

# ruff: noqa: E501

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .confirmation import verify_files
from .paper_batch import setup
from .paper_sample import prompt_ids, trajectory_logp
from .paper_setup import read_json
from .paper_steering import restore, set_direction
from .study import atomic_json, file_hash


def pack_trajectories(prefixes, tokens, pad_id, device):
    if (
        not tokens
        or len(prefixes) != len(tokens)
        or any(not p or not t for p, t in zip(prefixes, tokens, strict=True))
    ):
        raise ValueError("Nonempty paired prompts and trajectories required")
    length = max(len(p) + len(t) for p, t in zip(prefixes, tokens, strict=True))
    ids = torch.full((len(tokens), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    spans = []
    for i, (prefix, output) in enumerate(zip(prefixes, tokens, strict=True)):
        sequence = prefix + output
        ids[i, : len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, : len(sequence)] = 1
        spans.append((len(prefix), len(output)))
    return ids, mask, spans


def score_logits(logits, ids, spans, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    # Exclude prompt likelihood and right padding; include the actual final token.
    return torch.stack(
        [
            -F.cross_entropy(
                logits[i, start - 1 : start + count - 1].float() / temperature,
                ids[i, start : start + count],
                reduction="sum",
            )
            for i, (start, count) in enumerate(spans)
        ]
    )


def score_bank(model, tokenizer, trajectories, temperature, batch_size=1):
    scores = []
    with torch.inference_mode():
        for start in range(0, len(trajectories), batch_size):
            rows = trajectories[start : start + batch_size]
            ids, mask, spans = pack_trajectories(
                [prompt_ids(tokenizer, r["prompt"]) for r in rows],
                [r["token_ids"] for r in rows],
                tokenizer.pad_token_id,
                "cuda",
            )
            result = model(input_ids=ids, attention_mask=mask, use_cache=False)
            scores.extend(score_logits(result.logits, ids, spans, temperature).cpu().tolist())
    result = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite trajectory likelihood")
    return result


def unique_trajectories(data, proposals):
    probes = {r["id"]: r["prompt"] for r in data["probes"]}
    expected = [(q["id"], s) for q in data["probes"] for s in range(64)]
    if [(r["id"], r["sample"]) for r in proposals["rows"]] != expected:
        raise ValueError("Unexpected proposal order")
    unique, mapping, inverse = [], {}, []
    for row in proposals["rows"]:
        key = (row["id"], tuple(row["token_ids"]))
        if key not in mapping:
            mapping[key] = len(unique)
            unique.append({"prompt": probes[row["id"]], "token_ids": row["token_ids"]})
        inverse.append(mapping[key])
    return unique, np.asarray(inverse).reshape(len(probes), 64)


def direct_reweight(log_ratio, rewards):
    shift = log_ratio.max(axis=-1, keepdims=True)
    weights = np.exp(log_ratio - shift)
    log_mean_weight = shift[..., 0] + np.log(weights.mean(axis=-1))
    weights /= weights.sum(axis=-1, keepdims=True)
    levels = np.einsum("qs,qsk->qk", weights, rewards)
    return {
        "snis_level": levels,
        "snis_change": levels - rewards.mean(axis=1),
        "ess": 1 / np.sum(weights**2, axis=-1),
        "log_mean_weight": log_mean_weight,
    }


def prepare(root, prior):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "specification.json").exists():
        raise ValueError("Already prepared")
    prior = prior.resolve()
    old = read_json(prior / "specification.json")
    if read_json(prior / "controller.json")["status"] != "complete":
        raise ValueError("Only completed pilot outputs may be diagnosed")
    assert read_json(prior / "integrity.json")["complete"]
    verify_files(prior, read_json(prior / "manifest.json")["files_sha256"])
    spec = {
        "scope": "posthoc_development_diagnostic_not_forecasting",
        "max_hours": 4,
        "prior_root": str(prior),
        "source_root": old["source_root"],
        "source_config": old["source_config"],
        "training_seed": old["training_seed"],
        "gradient_cache": str(Path(old["prior_root"]) / "forecasts.pt"),
        "rates": old["rates"],
        "temperature": old["temperature"],
        "max_new_tokens": old["max_new_tokens"],
        "stop_ids": old["stop_ids"],
        "score_batch_size": 1,
        "rationale": "The completed pilot had no evaluation proposal events for numbering or truncation; full-gradient forecasts of copying improved locally but failed at the large step and transferred poorly. Direct forward likelihoods on the unchanged proposal bank separate Taylor error from sampling limitations. No new responses or method tuning.",
        "procedure": "Re-score every saved baseline proposal at baseline and all 16 batches x 3 steps using actual all-layer LoRA updates. Deduplicate identical token sequences only within prompt, then restore every original sample multiplicity. Score exact stopped/censored tokens with temperature .8; no appended EOS. Compare direct-likelihood SNIS with all frozen Taylor methods and existing independent response evaluation. Direct ratios use post-update access, not a prospective forecast. Restore every adapter after every condition.",
        "analysis": "Read no new likelihood results until both diagnostic runs complete or original phase deadlines expire. Report every rate and seed, calibration/evaluation splits, all four endpoints, same-question/transfer errors, all batches/styles/tails, direct-versus-Taylor log-ratio discrepancy, ESS and log mean weights. Compare with fixed completed sampled-pilot observations; zero proposal support remains a failure. Bootstrap proposal and paired evaluation samples with crossed batches/questions. No confirmation, significance, or novelty claim; data and previous outcomes already read. Future broader proposals need a separate phase.",
        "input_hashes": {
            **old["input_hashes"],
            **{
                str(prior / n): file_hash(prior / n)
                for n in (
                    "specification.json",
                    "integrity.json",
                    "manifest.json",
                    "forecast-lock.json",
                    "data.json",
                    "forecasts.npz",
                    "proposals.json",
                )
            },
        },
    }
    atomic_json(root / "specification.json", spec)


def preflight(root):
    spec = read_json(root / "specification.json")
    row = read_json(Path(spec["prior_root"]) / "data.json")["batches"][0]["documents"][0]
    model, _, tokenizer, _, _, parameters = setup(spec)
    original = {n: p.detach().clone() for n, p in parameters}
    examples = [
        {
            "prompt": row["prompt"],
            "token_ids": tokenizer(answer, add_special_tokens=False)["input_ids"]
            + [spec["stop_ids"][1]],
        }
        for answer in row["answers"]
    ]
    batch = score_bank(model, tokenizer, examples, spec["temperature"])
    with torch.inference_mode():
        single = np.array(
            [
                trajectory_logp(
                    model, tokenizer, r["prompt"], r["token_ids"], spec["temperature"]
                ).item()
                for r in examples
            ]
        )
    discrepancy = np.abs(batch - single) / np.array([len(r["token_ids"]) for r in examples])
    assert discrepancy.max() < 0.005, discrepancy
    saved = torch.load(spec["gradient_cache"], map_location="cpu", weights_only=True)
    set_direction(parameters, original, saved["batch_gradients"][0], spec["rates"][1])
    assert any(not torch.equal(p, original[n]) for n, p in parameters)
    restore(parameters, original)
    restored = score_bank(model, tokenizer, examples, spec["temperature"])
    np.testing.assert_array_equal(batch, restored)
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "parameter_tensors": len(parameters),
            "single_vs_batched_abs_error_per_token": discrepancy.tolist(),
            "adapter_restored": True,
            "likelihoods_restored_exactly": True,
        },
    )


def freeze(root):
    if (root / "plan.json").exists():
        raise ValueError("Already frozen")
    spec = read_json(root / "specification.json")
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
                str(root / n): file_hash(root / n) for n in ("specification.json", "preflight.json")
            },
            "jobs": [
                {
                    "id": "direct-likelihood",
                    "module": "gradient_interp.paper_ratio",
                    "arguments": ["run", "--root", str(root.resolve())],
                    "manifest": str(root.resolve() / "manifest.json"),
                }
            ],
        },
    )


def run(root):
    started = time.time()
    spec, plan = read_json(root / "specification.json"), read_json(root / "plan.json")
    for hashes in (plan["source_files_sha256"], plan["input_hashes"], spec["input_hashes"]):
        verify_files(Path.cwd(), hashes)
    prior = Path(spec["prior_root"])
    data, proposals = read_json(prior / "data.json"), read_json(prior / "proposals.json")
    cache = dict(np.load(prior / "forecasts.npz"))
    trajectories, inverse = unique_trajectories(data, proposals)
    model, _, tokenizer, _, _, parameters = setup(spec)
    assert len(parameters) == 136
    original = {n: p.detach().clone() for n, p in parameters}
    saved = torch.load(spec["gradient_cache"], map_location="cpu", weights_only=True)
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    directory = root / "likelihoods"
    directory.mkdir(exist_ok=True)
    basepath = directory / "baseline.npz"
    if basepath.exists():
        baseline = np.load(basepath)["logp"]
    else:
        atomic_json(
            root / "status.json", {"stage": "baseline_scoring", "completed": 0, "total": 48}
        )
        baseline = score_bank(
            model, tokenizer, trajectories, spec["temperature"], spec["score_batch_size"]
        )[inverse]
        np.savez_compressed(
            basepath, logp=baseline, original_cache_logp=cache["baseline_logp"], inverse=inverse
        )
    completed = [basepath]
    for bi, rate in itertools.product(range(16), spec["rates"]):
        path = directory / f"batch-{bi:02d}-rate-{rate:g}.npz"
        metadata = path.with_suffix(".json")
        if path.exists() and metadata.exists():
            record = read_json(metadata)
            assert record["plan_sha256"] == file_hash(root / "plan.json") and record[
                "sha256"
            ] == file_hash(path)
            assert record["adapter_restored"]
            completed.extend((path, metadata))
            continue
        update_started = time.time()
        set_direction(parameters, original, saved["batch_gradients"][bi], rate)
        try:
            logp = score_bank(
                model, tokenizer, trajectories, spec["temperature"], spec["score_batch_size"]
            )[inverse]
        finally:
            restore(parameters, original)
        ratios = logp - baseline
        np.savez_compressed(
            path, logp=logp, log_ratio=ratios, **direct_reweight(ratios, cache["rewards"])
        )
        atomic_json(
            metadata,
            {
                "plan_sha256": file_hash(root / "plan.json"),
                "sha256": file_hash(path),
                "batch_index": bi,
                "rate": rate,
                "adapter_restored": True,
                "update_started_unix": update_started,
                "completed_unix": time.time(),
            },
        )
        completed.extend((path, metadata))
        atomic_json(
            root / "status.json",
            {
                "stage": "direct_likelihoods",
                "completed": (len(completed) - 1) // 2,
                "total": 48,
                "checked_unix": time.time(),
            },
        )
    restore(parameters, original)
    verify_files(Path.cwd(), spec["input_hashes"])
    atomic_json(
        root / "manifest.json",
        {
            "started_unix": started,
            "completed_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "scope": spec["scope"],
            "conditions": 49,
            "updates": 48,
            "adapter_restored": True,
            "proposal_trajectories_per_condition": int(inverse.size),
            "unique_sequences_per_condition": len(trajectories),
            "files_sha256": {str(p.relative_to(root)): file_hash(p) for p in completed},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "freeze", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "prepare":
        prepare(args.root.resolve(), args.prior)
    else:
        globals()[args.command](args.root.resolve())


if __name__ == "__main__":
    main()
