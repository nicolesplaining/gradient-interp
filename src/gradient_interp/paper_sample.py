"""Development: forecast sampled behavior from baseline response trajectories."""

# ruff: noqa: E501

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .confirmation import verify_files
from .paper_batch import METHODS, flat_grad, generated_checks, setup
from .paper_setup import read_json
from .paper_steering import restore, set_direction
from .paper_steering_analysis import quality
from .study import atomic_json, file_hash

ENDPOINTS = ("numbered", "exact_content_and_order", "tokens", "truncated")


def stopped_tokens(ids, stop_ids):
    for i, token in enumerate(ids):
        if token in stop_ids:
            return ids[: i + 1]
    return ids


def estimators(unit_changes, rewards, rates):
    """Inputs B,M,Q,S and Q,S,K; centered unbiased linear and SNIS estimates."""
    rewards = np.asarray(rewards, dtype=np.float64)
    unit_changes = np.asarray(unit_changes, dtype=np.float64)
    n = rewards.shape[-2]
    center = rewards - rewards.mean(axis=-2, keepdims=True)
    derivative = np.einsum("bmqs,qsk->bmqk", unit_changes, center) / (n - 1)
    linear = derivative[:, None] * np.asarray(rates)[None, :, None, None, None]
    logw = unit_changes[:, None] * np.asarray(rates)[None, :, None, None, None]
    logw -= logw.max(axis=-1, keepdims=True)
    weights = np.exp(logw)
    weights /= weights.sum(axis=-1, keepdims=True)
    snis = np.einsum("brmqs,qsk->brmqk", weights, rewards) - rewards.mean(axis=-2)
    ess = 1 / np.sum(weights**2, axis=-1)
    return linear, snis, ess


def prompt_ids(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def sample(model, tokenizer, row, count, seed, spec, return_scores=False):
    from transformers import GenerationConfig

    ids = prompt_ids(tokenizer, row["prompt"])
    config = GenerationConfig(
        do_sample=True,
        temperature=spec["temperature"],
        top_k=0,
        top_p=1.0,
        max_new_tokens=spec["max_new_tokens"],
        eos_token_id=spec["stop_ids"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=return_scores,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = torch.tensor([ids] * count, device="cuda")
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            generation_config=config,
            use_model_defaults=False,
        )
    tokens = [stopped_tokens(t, spec["stop_ids"]) for t in output.sequences[:, len(ids) :].tolist()]
    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    rows = []
    for i, (text, token_ids) in enumerate(zip(texts, tokens, strict=True)):
        record = {
            "id": row["id"],
            "sample": i,
            "response": text,
            "token_ids": token_ids,
            "checks": generated_checks(text, row["tags"]),
        }
        record["quality"] = quality(record, spec["max_new_tokens"])
        rows.append(record)
    generated_logp = None
    if return_scores:
        generated_logp = [
            sum(
                float(F.log_softmax(output.scores[t][i].float(), dim=-1)[token])
                for t, token in enumerate(token_ids)
            )
            for i, token_ids in enumerate(tokens)
        ]
    return rows, generated_logp


def trajectory_logp(model, tokenizer, prompt, tokens, temperature):
    prefix = prompt_ids(tokenizer, prompt)
    ids = torch.tensor([prefix + tokens], device="cuda")
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    # Score exactly the sampled trajectory, including its stop token, without an appended EOS.
    logits = logits[0, len(prefix) - 1 : -1].float() / temperature
    targets = ids[0, len(prefix) :]
    assert len(targets) == len(tokens) == logits.shape[0]
    return -F.cross_entropy(logits, targets, reduction="sum")


def prepare(root, prior):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "specification.json").exists():
        raise ValueError("Already prepared")
    prior = prior.resolve()
    old = read_json(prior / "specification.json")
    assert read_json(prior / "integrity.json")["complete"]
    verify_files(prior, read_json(prior / "forecast-lock.json")["files_sha256"])
    data = read_json(prior / "data.json")
    probes = [dict(r, split="calibration") for r in data["calibration"][:8]] + [
        dict(r, split="evaluation") for r in data["evaluation"][:8]
    ]
    atomic_json(root / "data.json", {"batches": data["batches"], "probes": probes})
    spec = {
        "scope": "synthetic_development_only",
        "max_hours": 8,
        "prior_root": str(prior),
        "source_root": old["source_root"],
        "source_config": old["source_config"],
        "training_seed": old["training_seed"],
        "rates": old["rates"],
        "methods": list(METHODS),
        "endpoints": list(ENDPOINTS),
        "temperature": 0.8,
        "max_new_tokens": 96,
        "stop_ids": [1, 106],
        "proposal_samples": 64,
        "evaluation_samples": 32,
        "proposal_seed": 202610130,
        "evaluation_seed": 202610131,
        "hypothesis": "Scoring naturally sampled baseline trajectories may forecast changes in sampled formatting and exact-content rates more directly than four hand-written answer scores. Standard score-function and importance-sampling estimators; no novelty or confirmation claim.",
        "data": "Reuse all16 prior synthetic batches and first8 calibration plus first8 evaluation prompts. No untouched-data claim. Independent evaluation sampling stream; common random seeds across restored updates. Temperature.8, full-vocabulary sampling, top_k0/top_p1, cap96; not greedy behavior.",
        "forecast": "Before updates, sample64 trajectories perprobe from the original adapter and differentiate their exact temperature-adjusted log probabilities, including actual terminal token or cap, without extra template ending. Contract with full and5compressed batch gradients. Predict centered score-function changes with N/(N-1) correction and self-normalized exponential Taylor reweighting. Save perprobe forecasts for allrates; calibration-mean transfer assessed separately. No fitted outcome regression or clipping of linear probabilities.",
        "evaluation": "Baseline and all48 updates:32 independently sampled responses perprobe,16probes,512responses/condition.25088 evaluation generations plus1024 proposals perseed. Actual update identical to old batch pilot, mean-perdocument mean-CE SGD across136LoRA tensors, including original scoring EOS convention. Reset checkpoint each condition. Sampling endpoints differ explicitly from old greedy endpoint.",
        "analysis": "Wait until BOTH seeds finish/deadline. Verify locks/hash/restoration/grid. Report allmethods/rates/estimators/seeds/endpoints, sameprobe and independenttemplate transfer, pooled/perbatch errors and tails; paired batch/probe uncertainty plus proposal/evaluation MonteCarlo uncertainty. ESS and proposal event counts: no event in proposal means no support for predicting its emergence. ESS<16/64 is a predeclared low-overlap flag, report results without omitting. Linear values outside[0,1] flagged, not clipped. No significance or generalized interpretation claims. Fixed examples batch0..3/probe0,8/sample0 at everyrate.",
        "limitations": [
            "Two reused adapters and hardware confound",
            "Synthetic copying and few prompts",
            "Importance weights use approximate Taylor ratios and may miss new modes",
            "No clean intervention or atom-description validation",
            "Support and MonteCarlo noise may defeat quantitative prediction",
            "Known score-function and importance-sampling machinery, not a discovery",
        ],
        "references": [
            "https://doi.org/10.1007/BF00992696",
            "https://proceedings.mlr.press/v119/huang20b.html",
            "https://papers.neurips.cc/paper/7789-policy-optimization-via-importance-sampling.pdf",
            "https://arxiv.org/abs/2603.14665",
        ],
        "input_hashes": {
            **old["input_hashes"],
            **{
                str(prior / n): file_hash(prior / n)
                for n in (
                    "specification.json",
                    "integrity.json",
                    "forecast-lock.json",
                    "forecasts.pt",
                    "data.json",
                )
            },
        },
    }
    atomic_json(root / "specification.json", spec)


def preflight(root):
    spec = read_json(root / "specification.json")
    model, _, tokenizer, _, _, parameters = setup(spec)
    original = {n: p.detach().clone() for n, p in parameters}
    # Only a training document is used for preflight; main proposal/evaluation outcomes remain unread.
    row = read_json(root / "data.json")["batches"][0]["documents"][0]
    trial = {**spec, "max_new_tokens": 16}
    effective_settings = []
    original_processor = model._get_logits_processor

    def audited_processor(*args, **kwargs):
        config = kwargs["generation_config"]
        effective_settings.append(
            {
                key: getattr(config, key)
                for key in ("do_sample", "temperature", "top_p", "top_k", "eos_token_id")
            }
        )
        assert config.do_sample and config.top_p == 1 and config.top_k == 0
        assert config.temperature == spec["temperature"] and config.eos_token_id == spec["stop_ids"]
        processors = original_processor(*args, **kwargs)
        assert not any(
            type(p).__name__ in ("TopPLogitsWarper", "TopKLogitsWarper") for p in processors
        )
        return processors

    model._get_logits_processor = audited_processor
    before, logps = sample(model, tokenizer, row, 2, 18173, trial, return_scores=True)
    discrepancies = []
    for record, generated in zip(before, logps, strict=True):
        score = trajectory_logp(
            model, tokenizer, row["prompt"], record["token_ids"], spec["temperature"]
        )
        discrepancies.append(abs(score.item() - generated) / len(record["token_ids"]))
        gradient = flat_grad(score, parameters)
        assert gradient.norm() > 0 and torch.isfinite(gradient).all()
    assert max(discrepancies) < 0.1, discrepancies
    old = torch.load(
        Path(spec["prior_root"]) / "forecasts.pt", map_location="cpu", weights_only=True
    )
    set_direction(parameters, original, old["batch_gradients"][0], spec["rates"][1])
    assert any(not torch.equal(p, original[n]) for n, p in parameters)
    restore(parameters, original)
    after, _ = sample(model, tokenizer, row, 2, 18173, trial)
    assert [r["token_ids"] for r in before] == [r["token_ids"] for r in after]
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "adapter_restored": True,
            "sampled_generation_restored": True,
            "parameter_tensors": len(parameters),
            "temperature_score_mean_abs_error_per_token": discrepancies,
            "effective_sampling_settings": effective_settings,
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
                str(root / n): file_hash(root / n)
                for n in ("specification.json", "data.json", "preflight.json")
            },
            "jobs": [
                {
                    "id": "sampled-behavior",
                    "module": "gradient_interp.paper_sample",
                    "arguments": ["run", "--root", str(root.resolve())],
                    "manifest": str(root.resolve() / "manifest.json"),
                }
            ],
        },
    )


def gradient_bank(spec, saved, payload):
    from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac, unproject_atom

    factors = load_ekfac_eigen(
        str(Path(spec["source_root"]) / "factors/paper_ekfac/factors_training")
    )
    gradients = saved["batch_gradients"]
    projected, module_info = project_gradients_ekfac(gradients, payload["lora_names"], factors, 50)
    variants = {
        "ekfac_projection": projected,
        "dictionary": saved["sparse_codes"]
        @ payload["dictionary"].float()
        * saved["projected_norms"],
        "svd16": projected @ saved["svd_basis"].T @ saved["svd_basis"],
        "mean_direction": (projected @ saved["mean_projected"])[:, None] * saved["mean_projected"],
        "zero": torch.zeros_like(projected),
    }
    return torch.stack(
        [
            torch.stack(
                [
                    g
                    if method == "full_gradient"
                    else unproject_atom(variants[method][i].numpy(), module_info, factors, len(g))
                    for method in METHODS
                ]
            )
            for i, g in enumerate(gradients)
        ]
    )


def run(root):
    started = time.time()
    plan, spec, data = [
        read_json(root / n) for n in ("plan.json", "specification.json", "data.json")
    ]
    verify_files(Path.cwd(), plan["source_files_sha256"])
    verify_files(Path.cwd(), plan["input_hashes"])
    verify_files(Path.cwd(), spec["input_hashes"])
    verify_files(
        Path(spec["source_root"]),
        read_json(Path(spec["source_root"]) / "factors-manifest.json")["files_sha256"],
    )

    def status(stage, **extra):
        atomic_json(root / "status.json", {"stage": stage, "updated_unix": time.time(), **extra})
        print(stage, extra, flush=True)

    model, _, tokenizer, _, payload, parameters = setup(spec)
    original = {n: p.detach().clone() for n, p in parameters}
    old = torch.load(
        Path(spec["prior_root"]) / "forecasts.pt", map_location="cpu", weights_only=True
    )
    cache, lock = root / "forecasts.npz", root / "forecast-lock.json"
    if lock.exists():
        verify_files(root, read_json(lock)["files_sha256"])
    else:
        bank = gradient_bank(spec, old, payload).to("cuda")
        proposals, changes, rewards, logps = [], [], [], []
        for qi, row in enumerate(data["probes"]):
            records, _ = sample(
                model, tokenizer, row, spec["proposal_samples"], spec["proposal_seed"] + qi, spec
            )
            units, values, scores = [], [], []
            for si, record in enumerate(records):
                score = trajectory_logp(
                    model, tokenizer, row["prompt"], record["token_ids"], spec["temperature"]
                )
                grad = flat_grad(score, parameters).to("cuda")
                units.append(-(bank @ grad).cpu().numpy())
                values.append([{**record["checks"], **record["quality"]}[k] for k in ENDPOINTS])
                scores.append(score.item())
                if (si + 1) % 16 == 0:
                    status(
                        "trajectory_gradients",
                        completed=qi * spec["proposal_samples"] + si + 1,
                        total=len(data["probes"]) * spec["proposal_samples"],
                    )
            proposals.extend(records)
            changes.append(np.stack(units, axis=-1))
            rewards.append(values)
            logps.append(scores)
        unit = np.stack(changes, axis=2)
        rewards = np.asarray(rewards, dtype=float)
        linear, snis, ess = estimators(unit, rewards, spec["rates"])
        np.savez_compressed(
            cache,
            unit_rate_logp_changes=unit,
            rewards=rewards,
            baseline_logp=np.array(logps),
            linear=linear,
            snis=snis,
            ess=ess,
        )
        atomic_json(
            root / "proposals.json",
            {"plan_sha256": file_hash(root / "plan.json"), "rows": proposals},
        )
        atomic_json(
            lock,
            {
                "locked_unix": time.time(),
                "plan_sha256": file_hash(root / "plan.json"),
                "all_forecasts_before_updates": True,
                "files_sha256": {p.name: file_hash(p) for p in (cache, root / "proposals.json")},
            },
        )
        del bank
    directory = root / "outcomes"
    directory.mkdir(exist_ok=True)

    def collect():
        records = []
        for qi, row in enumerate(data["probes"]):
            sampled, _ = sample(
                model,
                tokenizer,
                row,
                spec["evaluation_samples"],
                spec["evaluation_seed"] + 1000 + qi,
                spec,
            )
            records.extend(sampled)
        return records

    baseline = directory / "baseline.json"
    if not baseline.exists():
        atomic_json(
            baseline,
            {
                "plan_sha256": file_hash(root / "plan.json"),
                "forecast_lock_sha256": file_hash(lock),
                "rows": collect(),
                "adapter_restored": True,
            },
        )
    completed = [baseline]
    for bi, rate in itertools.product(range(16), spec["rates"]):
        path = directory / f"batch-{bi:02d}-rate-{rate:g}.json"
        if path.exists():
            assert read_json(path)["forecast_lock_sha256"] == file_hash(lock)
            completed.append(path)
            continue
        update_started = time.time()
        restore(parameters, original)
        set_direction(parameters, original, old["batch_gradients"][bi], rate)
        try:
            records = collect()
        finally:
            restore(parameters, original)
        atomic_json(
            path,
            {
                "plan_sha256": file_hash(root / "plan.json"),
                "forecast_lock_sha256": file_hash(lock),
                "batch_index": bi,
                "style": data["batches"][bi]["style"],
                "rate": rate,
                "rows": records,
                "adapter_restored": True,
                "update_started_unix": update_started,
                "completed_unix": time.time(),
            },
        )
        completed.append(path)
        status("updates", completed=len(completed) - 1, total=48)
    restore(parameters, original)
    atomic_json(
        root / "manifest.json",
        {
            "started_unix": started,
            "completed_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "forecast_lock_sha256": file_hash(lock),
            "updates": 48,
            "evaluation_generations": 49 * 16 * spec["evaluation_samples"],
            "proposal_generations": 16 * spec["proposal_samples"],
            "adapter_restored": True,
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p)
                for p in (cache, lock, root / "proposals.json", *completed)
            },
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
