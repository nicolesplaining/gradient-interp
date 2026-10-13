"""Post-update oracle diagnostic on frozen float32 mixture trajectories."""

# ruff: noqa: E501

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch

from .confirmation import verify_files
from .paper_mixture import BANKS, normalized, stable_setup, trajectory_terms
from .paper_ratio import score_bank
from .paper_setup import read_json
from .paper_steering import restore, set_direction
from .study import atomic_json, file_hash


def unique_trajectories(data, proposals):
    prompts = {q["id"]: q["prompt"] for q in data["probes"]}
    assert [(r["id"], r["sample"]) for r in proposals["rows"]] == [
        (q["id"], s) for q in data["probes"] for s in range(512)
    ]
    unique, mapping, inverse = [], {}, []
    for row in proposals["rows"]:
        key = (row["id"], tuple(row["token_ids"]))
        if key not in mapping:
            mapping[key] = len(unique)
            unique.append({"prompt": prompts[row["id"]], "token_ids": row["token_ids"]})
        inverse.append(mapping[key])
    return unique, np.asarray(inverse).reshape(len(prompts), 512)


def direct_banks(updated_logp, base_logp, warm_logp, rewards):
    """Actual p_updated/q weighting; no gradients or Taylor approximation here."""
    out = {}
    for bank, (idx, proposal) in BANKS.items():
        base, warm = base_logp[:, idx], warm_logp[:, idx]
        q = {"base": base, "warm": warm, "mixture": np.logaddexp(base, warm) - np.log(2)}[proposal]
        lw = updated_logp[:, idx] - q
        w0, w = normalized(base - q), normalized(lw)
        baseline = np.einsum("qs,qsk->qk", w0, rewards[:, idx])
        level = np.einsum("qs,qsk->qk", w, rewards[:, idx])
        out.update(
            {
                f"{bank}_level": level,
                f"{bank}_change": level - baseline,
                f"{bank}_ess": 1 / (w**2).sum(-1),
                f"{bank}_log_mean_weight": np.logaddexp.reduce(lw, axis=-1) - np.log(len(idx)),
                f"{bank}_max_weight": w.max(-1),
            }
        )
    return out


def prepare(root, prior):
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "specification.json").exists()
    prior = prior.resolve()
    old = read_json(prior / "specification.json")
    assert (
        read_json(prior / "controller.json")["status"] == "complete"
        and read_json(prior / "integrity.json")["complete"]
    )
    verify_files(prior, read_json(prior / "manifest.json")["files_sha256"])
    spec = {
        "scope": "posthoc_development_oracle_not_forecasting",
        "max_hours": 4,
        "prior_root": str(prior),
        "source_root": old["source_root"],
        "source_config": old["source_config"],
        "training_seed": old["training_seed"],
        "gradient_cache": str(prior / "batch-representations.pt"),
        "rates": old["rates"],
        "temperature": old["temperature"],
        "warm_temperature": old["warm_temperature"],
        "max_new_tokens": old["max_new_tokens"],
        "stop_ids": old["stop_ids"],
        "numerics": old["numerics"],
        "score_batch_size": 1,
        "rationale": "Broader proposals improved rare-event coverage only sparsely, with no consistent equal-count improvement. Large-step approximate weights collapse; truncation forecasts remain near zero. Earlier direct-ratio diagnostic used fewer proposals and BF16. Re-score this same larger FP32 bank to separate likelihood approximation error from inadequate weighted coverage, without new responses or tuning.",
        "procedure": "Exactly reuse all 8192 stopped/capped token sequences and all 48 original FP32 LoRA updates. Deduplicate within question only; preserve every multiplicity. Score baseline at both proposal temperatures and every updated model at target .8. Every sequence uses a single forward, no appended EOS. Mixture density remains the ORIGINAL frozen proposal density. Save direct p_updated/q weights for all five banks, levels/changes/ESS/mean weights. Restore all 136 LoRA tensors after every update. Direct ratios use updated model access and are not prospective forecasts.",
        "analysis": "No new likelihood outcome reads until BOTH runs complete or original deadlines expire. Verify all hashes, 49 conditions, 48 restores, 8192 scores/condition and original deadline. Compare direct versus Taylor errors and baseline scoring drift, every bank/rate/seed/style/endpoint; same-question versus transfer. Reuse independent 128-response evaluations. Component-stratified paired proposal/evaluation/batch/question uncertainty as in completed mixture analysis. Report low overlap and absent modes, all negatives and fixed examples. No confirmation or novelty claim.",
        "input_hashes": {
            **old["input_hashes"],
            **{
                str(prior / n): file_hash(prior / n)
                for n in (
                    "specification.json",
                    "manifest.json",
                    "integrity.json",
                    "forecast-lock.json",
                    "data.json",
                    "forecasts.npz",
                    "proposals.json",
                    "batch-representations.pt",
                )
            },
        },
    }
    atomic_json(root / "specification.json", spec)


def preflight(root):
    spec = read_json(root / "specification.json")
    row = read_json(Path(spec["prior_root"]) / "data.json")["batches"][0]["documents"][0]
    model, _, tokenizer, _, _, parameters = stable_setup(spec)
    original = {n: p.detach().clone() for n, p in parameters}
    examples = [
        {
            "prompt": row["prompt"],
            "token_ids": tokenizer(a, add_special_tokens=False)["input_ids"]
            + [spec["stop_ids"][1]],
        }
        for a in row["answers"]
    ]
    discrepancies = []
    for temperature in (spec["temperature"], spec["warm_temperature"]):
        scored = score_bank(model, tokenizer, examples, temperature, batch_size=1)
        with torch.inference_mode():
            reference = np.array(
                [
                    trajectory_terms(model, tokenizer, r["prompt"], r["token_ids"], [temperature])[
                        0
                    ].item()
                    for r in examples
                ]
            )
        discrepancies.extend(
            (
                np.abs(scored - reference) / np.array([len(r["token_ids"]) for r in examples])
            ).tolist()
        )
    assert max(discrepancies) < 0.005 and len(parameters) == 136
    saved = torch.load(spec["gradient_cache"], map_location="cpu", weights_only=True)
    before = score_bank(model, tokenizer, examples, spec["temperature"])
    set_direction(parameters, original, saved["batch_gradients"][0], spec["rates"][1])
    assert any(not torch.equal(p, original[n]) for n, p in parameters)
    restore(parameters, original)
    np.testing.assert_array_equal(
        before, score_bank(model, tokenizer, examples, spec["temperature"])
    )
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "parameter_tensors": len(parameters),
            "numerics": spec["numerics"],
            "single_vs_original_abs_error_per_token": discrepancies,
            "adapter_restored": True,
            "likelihoods_restored_exactly": True,
        },
    )


def freeze(root):
    assert not (root / "plan.json").exists()
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
                    "id": "direct-mixture-likelihood",
                    "module": "gradient_interp.paper_mixture_ratio",
                    "arguments": ["run", "--root", str(root.resolve())],
                    "manifest": str(root.resolve() / "manifest.json"),
                }
            ],
        },
    )


def run(root):
    started = time.time()
    spec, plan = [read_json(root / n) for n in ("specification.json", "plan.json")]
    for hashes in (spec["input_hashes"], plan["input_hashes"], plan["source_files_sha256"]):
        verify_files(Path.cwd(), hashes)
    prior = Path(spec["prior_root"])
    cache = dict(np.load(prior / "forecasts.npz"))
    data = read_json(prior / "data.json")
    trajectories, inverse = unique_trajectories(data, read_json(prior / "proposals.json"))
    model, _, tokenizer, _, _, parameters = stable_setup(spec)
    assert len(parameters) == 136
    original = {n: p.detach().clone() for n, p in parameters}
    saved = torch.load(spec["gradient_cache"], map_location="cpu", weights_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    directory = root / "likelihoods"
    directory.mkdir(exist_ok=True)
    basepath = directory / "baseline.npz"
    if not basepath.exists():
        atomic_json(
            root / "status.json",
            {"stage": "baseline_scoring", "completed": 0, "total": 48, "checked_unix": time.time()},
        )
        baseline = score_bank(model, tokenizer, trajectories, spec["temperature"])[inverse]
        warm = score_bank(model, tokenizer, trajectories, spec["warm_temperature"])[inverse]
        np.savez_compressed(
            basepath,
            logp=baseline,
            warm_logp=warm,
            original_cache_logp=cache["baseline_logp"],
            original_warm_logp=cache["warm_logp"],
            inverse=inverse,
        )
    baseline = np.load(basepath)["logp"]
    completed = [basepath]
    for b, rate in itertools.product(range(16), spec["rates"]):
        path = directory / f"batch-{b:02d}-rate-{rate:g}.npz"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            record = read_json(meta)
            assert (
                record["plan_sha256"] == file_hash(root / "plan.json")
                and record["sha256"] == file_hash(path)
                and record["adapter_restored"]
            )
            completed.extend((path, meta))
            continue
        update_started = time.time()
        set_direction(parameters, original, saved["batch_gradients"][b], rate)
        try:
            logp = score_bank(model, tokenizer, trajectories, spec["temperature"])[inverse]
        finally:
            restore(parameters, original)
        np.savez_compressed(
            path,
            logp=logp,
            log_ratio=logp - baseline,
            **direct_banks(logp, cache["baseline_logp"], cache["warm_logp"], cache["rewards"]),
        )
        atomic_json(
            meta,
            {
                "plan_sha256": file_hash(root / "plan.json"),
                "sha256": file_hash(path),
                "batch_index": b,
                "rate": rate,
                "adapter_restored": True,
                "update_started_unix": update_started,
                "completed_unix": time.time(),
            },
        )
        completed.extend((path, meta))
        atomic_json(
            root / "status.json",
            {
                "stage": "direct_mixture_likelihoods",
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
            "scope": spec["scope"],
            "plan_sha256": file_hash(root / "plan.json"),
            "conditions": 49,
            "updates": 48,
            "adapter_restored": True,
            "proposal_trajectories_per_condition": int(inverse.size),
            "unique_sequences_per_condition": len(trajectories),
            "files_sha256": {str(p.relative_to(root)): file_hash(p) for p in completed},
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "preflight", "freeze", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prior", type=Path)
    args = p.parse_args()
    torch.set_num_threads(8)
    if args.command == "prepare":
        prepare(args.root.resolve(), args.prior)
    else:
        globals()[args.command](args.root.resolve())


if __name__ == "__main__":
    main()
