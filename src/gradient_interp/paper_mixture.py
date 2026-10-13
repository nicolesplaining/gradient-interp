"""Development comparison of proposal count and temperature-mixture coverage."""

# ruff: noqa: E501

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .confirmation import verify_files
from .paper_batch import METHODS, flat_grad, scalar_logp, setup
from .paper_sample import ENDPOINTS, gradient_bank, prompt_ids, sample
from .paper_setup import read_json
from .paper_steering import restore, set_direction
from .study import atomic_json, file_hash

BANKS = {
    "base64": (list(range(64)), "base"),
    "base256": (list(range(256)), "base"),
    "warm256": (list(range(256, 512)), "warm"),
    "mixture256": (list(range(128)) + list(range(256, 384)), "mixture"),
    "mixture512": (list(range(512)), "mixture"),
}


def stable_setup(spec):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adjusted = {**spec, "source_config": {**spec["source_config"], "attention": "sdpa"}}
    result = setup(adjusted)
    result[0].float()
    assert all(p.dtype == torch.float32 for p in result[0].parameters())
    torch.cuda.empty_cache()
    return result


def batch_representations(model, scoring, tokenize, parameters, payload, data, spec, old):
    from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac
    from sklearn.decomposition import sparse_encode

    gradients = []
    for batch in data["batches"]:
        grads = []
        for row in batch["documents"]:
            value, n = scalar_logp(
                model, scoring, tokenize, row["prompt"], row["answers"][batch["style_index"]]
            )
            grads.append(flat_grad(-value / n, parameters))
        gradients.append(torch.stack(grads).mean(0))
    gradients = torch.stack(gradients)
    factors = load_ekfac_eigen(
        str(Path(spec["source_root"]) / "factors/paper_ekfac/factors_training")
    )
    projected, _ = project_gradients_ekfac(gradients, payload["lora_names"], factors, 50)
    norms = projected.norm(dim=1, keepdim=True).clamp_min(1e-12)
    codes = sparse_encode(
        (projected / norms).numpy(),
        payload["dictionary"].numpy(),
        algorithm="lasso_lars",
        alpha=0.1,
    )
    return {
        "batch_gradients": gradients,
        "projected_norms": norms,
        "sparse_codes": torch.from_numpy(codes).float(),
        "svd_basis": old["svd_basis"],
        "mean_projected": old["mean_projected"],
    }


def normalized(logw):
    w = np.exp(logw - logw.max(axis=-1, keepdims=True))
    return w / w.sum(axis=-1, keepdims=True)


def weighted_estimators(unit, rewards, target_logp, warm_logp, rates, indices, proposal):
    indices = np.asarray(indices)
    target, warm = target_logp[:, indices], warm_logp[:, indices]
    logq = {"base": target, "warm": warm, "mixture": np.logaddexp(target, warm) - np.log(2)}[
        proposal
    ]
    logw0 = target - logq
    w0 = normalized(logw0)
    rewards, unit = rewards[:, indices], unit[..., indices]
    base = np.einsum("qs,qsk->qk", w0, rewards)
    # Derivative of the self-normalized estimator, not an unbiased covariance estimator.
    derivative = np.einsum("qs,bmqs,qsk->bmqk", w0, unit, rewards - base[:, None], optimize=True)
    linear = derivative[:, None] * np.asarray(rates)[None, :, None, None, None]
    logw = logw0 + unit[:, None] * np.asarray(rates)[None, :, None, None, None]
    w = normalized(logw)
    snis = np.einsum("brmqs,qsk->brmqk", w, rewards) - base
    return {
        "linear": linear,
        "snis": snis,
        "baseline": base,
        "baseline_ess": 1 / (w0**2).sum(-1),
        "ess": 1 / (w**2).sum(-1),
        "log_base_weights": logw0,
    }


def trajectory_terms(model, tokenizer, prompt, tokens, temperatures):
    prefix = prompt_ids(tokenizer, prompt)
    ids = torch.tensor([prefix + tokens], device="cuda")
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    logits = logits[0, len(prefix) - 1 : -1].float()
    targets = ids[0, len(prefix) :]
    assert len(targets) == len(tokens) == logits.shape[0]
    return [-F.cross_entropy(logits / t, targets, reduction="sum") for t in temperatures]


def prepare(root, prior):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "specification.json").exists():
        raise ValueError("Already prepared")
    prior = prior.resolve()
    old = read_json(prior / "specification.json")
    assert read_json(prior / "controller.json")["status"] == "complete"
    assert read_json(prior / "integrity.json")["complete"]
    source = Path(old["prior_root"])
    sample_spec = read_json(source / "specification.json")
    data = read_json(source / "data.json")
    atomic_json(root / "data.json", data)
    spec = {
        "scope": "synthetic_development_only",
        "max_hours": 8,
        "diagnostic_root": str(prior),
        "sample_root": str(source),
        "prior_root": sample_spec["prior_root"],
        "source_root": old["source_root"],
        "source_config": old["source_config"],
        "training_seed": old["training_seed"],
        "rates": old["rates"],
        "methods": list(METHODS),
        "endpoints": list(ENDPOINTS),
        "numerics": "float32_sdpa_tf32_disabled",
        "temperature": 0.8,
        "warm_temperature": 1.2,
        "stop_ids": [1, 106],
        "max_new_tokens": 96,
        "proposal_samples_per_component": 256,
        "proposal_chunk": 64,
        "evaluation_samples": 128,
        "evaluation_chunk": 32,
        "proposal_seed": 2026101310,
        "evaluation_seed": 2026101311,
        "banks": {k: {"indices": v[0], "proposal_density": v[1]} for k, v in BANKS.items()},
        "hypothesis": "Broader pre-update proposal coverage may improve generation-change forecasting beyond merely drawing more target-temperature samples. Direct-ratio diagnostics did not rescue old proposal samples. Compare equal256sample base versus warm versus half-base/half-warm mixture, with base64 and mixture512 as fixed sample-count controls. No assumed positive result.",
        "development": "All16 prior synthetic training batches and16 prior prompts are reused; previous results already inspected. New stochastic trajectories and larger128sample independent evaluation streams do not make this untouched-data confirmation. Same2adapters/hardware. No rate or endpoint selection.",
        "proposals": "Draw256 samples perquestion at temperature.8 and256 at1.2 from unchanged original adapter, top_k0/top_p1, stopIDs1/106 andcap96. Fixed temperatures chosen before new trajectories, no adaptation. Compute both component sequence log probabilities on exact stopped/censored IDs using single-trajectory forward calls. Deduplicate identical token IDs only within question to reuse derivatives, preserving every sample multiplicity.",
        "weighting": "For deterministic equal allocation mixture banks, q(y)=.5p_0.8(y)+.5p_1.2(y). Weight by p_0.8/q before applying exp(eta*directional_derivative). Warm-only uses q=p_1.2, base-only q=p_0.8. SNIS change is updated weighted reward minus baseline weighted reward. Linear is derivative of this self-normalized estimator ateta0, with no N/(N-1)correction and no unbiasedness claim. Include both component log probabilities, unit derivatives, rewards, allbanks/estimators/rates/methods andESS in forecast lock before ANYupdate.",
        "evaluation": "Same16batches and rates .0003/.003/.03, mean-perdocument mean-completion CE gradients recomputed in float32 before proposals. Dictionary, EKFAC factors, SVD16 basis and mean basis remain from completed discovery/development; re-encode the new gradients with unchanged alpha.1. All136LoRA A/B tensors updated.128independent response samples perprobe at baseline and after48updates; common RNG acrossconditions, distinct fromproposals.100352evaluation responses and8192proposals perseed. Restore checkpoint aftereveryupdate. Temperature.8/cap96 endpoints unchanged. No updating full model baseweights.",
        "analysis": "Wait until BOTH runsfinish ororiginaldeadlines. Verify fullhashes/forecastlock-before-updates/restoration/counts/immutabledeadline. Report everybank/rate/method/estimator/endpoint/seed/style andbatchtails, actualquality andfixedexamples. Compare base256 against mixture256 as equal-proposal-count development contrast; mixture512 not equal-cost. Measure totalforward/backward/generationcost honestly. Bootstrap separatelywithin eachproposalcomponent preserving deterministic allocation and sharedbanksubsets; pair evaluationstreams acrossbaseline/updates andresamplequestions/batches. Includeproposal/evaluationMonteCarlo uncertainty, ESSfractionbelowN/4, eventcoverage, weightconcentration and linearprobabilityrange violations; no clipping/dropping badcases. Sameprobe and calibration-template transfer separately.128evalsamples reduces noise but8promptsand2reusedseeds remain limited. Fixedexamples batch0..3/probe0,8/sample0/allrates. No novelty or confirmatoryclaim; standard multipleimportance sampling.",
        "numeric_limit": "Pre-freeze bfloat16 warm-sampling preflight failed the .005 nats/token generation-score check. This development phase uses float32 for generation/scoring/gradients, SDPA attention, highest matmul precision and TF32 disabled. Base and LoRA checkpoint values are preserved by exact upcasting; batch gradients and sparse codes are recomputed. This is an explicit numerical-backend variation, not identical to previous bfloat16 results. Original failed spec/code/logs retained. No scientific outcomes were generated before amendment; unchanged .005 tolerance.",
        "references": [
            "https://arxiv.org/abs/1511.03095",
            "https://proceedings.mlr.press/v119/huang20b.html",
            "https://arxiv.org/abs/2603.14665",
        ],
        "input_hashes": {
            **old["input_hashes"],
            **{
                str(prior / n): file_hash(prior / n)
                for n in ("specification.json", "manifest.json", "integrity.json")
            },
        },
    }
    atomic_json(root / "specification.json", spec)


def preflight(root):
    spec = read_json(root / "specification.json")
    model, scoring, tokenizer, tokenize, payload, parameters = stable_setup(spec)
    original = {n: p.detach().clone() for n, p in parameters}
    row = read_json(root / "data.json")["batches"][0]["documents"][0]
    generator = model.get_base_model()
    processor = generator._get_logits_processor
    audited = []

    def intercept(*args, **kwargs):
        c = kwargs["generation_config"]
        assert c.do_sample and c.top_p == 1 and c.top_k == 0 and c.eos_token_id == spec["stop_ids"]
        assert c.temperature in (spec["temperature"], spec["warm_temperature"])
        processors = processor(*args, **kwargs)
        assert not any(
            type(p).__name__ in ("TopPLogitsWarper", "TopKLogitsWarper") for p in processors
        )
        audited.append(
            {
                k: getattr(c, k)
                for k in ("temperature", "top_p", "top_k", "eos_token_id", "do_sample")
            }
        )
        return processors

    generator._get_logits_processor = intercept
    discrepancies, before = [], []
    for ti, temperature in enumerate((spec["temperature"], spec["warm_temperature"])):
        trial = {**spec, "temperature": temperature, "max_new_tokens": 8}
        rows, logps = sample(model, tokenizer, row, 64, 17217 + ti, trial, return_scores=True)
        before.append(rows)
        for si in (0, 1, 62, 63):
            terms = trajectory_terms(
                model,
                tokenizer,
                row["prompt"],
                rows[si]["token_ids"],
                [spec["temperature"], spec["warm_temperature"]],
            )
            discrepancy = abs(terms[ti].item() - logps[si]) / len(rows[si]["token_ids"])
            discrepancies.append(discrepancy)
            gradient = flat_grad(terms[0], parameters)
            assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert max(discrepancies) < 0.005, discrepancies
    old = torch.load(
        Path(spec["prior_root"]) / "forecasts.pt", map_location="cpu", weights_only=True
    )
    bank = gradient_bank(spec, old, payload).cuda()
    contracted = bank @ gradient.cuda()
    assert contracted.shape == (16, 6) and torch.isfinite(contracted).all()
    del bank
    value, n = scalar_logp(model, scoring, tokenize, row["prompt"], row["answers"][0])
    train_gradient = flat_grad(-value / n, parameters)
    set_direction(parameters, original, train_gradient, spec["rates"][1])
    assert any(not torch.equal(p, original[n]) for n, p in parameters)
    restore(parameters, original)
    after, _ = sample(model, tokenizer, row, 64, 17217, {**spec, "max_new_tokens": 8})
    assert [r["token_ids"] for r in after] == [r["token_ids"] for r in before[0]]
    assert len(audited) == 3
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "parameter_tensors": len(parameters),
            "numerics": "float32_sdpa_tf32_disabled",
            "adapter_restored": True,
            "sampled_generation_restored": True,
            "effective_sampling_settings": audited,
            "generation_score_abs_error_per_token": discrepancies,
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
                str(root / n): file_hash(root / n)
                for n in ("specification.json", "preflight.json", "data.json")
            },
            "jobs": [
                {
                    "id": "mixture-proposals",
                    "module": "gradient_interp.paper_mixture",
                    "arguments": ["run", "--root", str(root.resolve())],
                    "manifest": str(root.resolve() / "manifest.json"),
                }
            ],
        },
    )


def run(root):
    started = time.time()
    spec, plan, data = [
        read_json(root / n) for n in ("specification.json", "plan.json", "data.json")
    ]
    for hashes in (plan["source_files_sha256"], plan["input_hashes"], spec["input_hashes"]):
        verify_files(Path.cwd(), hashes)
    verify_files(
        Path(spec["source_root"]),
        read_json(Path(spec["source_root"]) / "factors-manifest.json")["files_sha256"],
    )
    model, scoring, tokenizer, tokenize, payload, parameters = stable_setup(spec)
    assert len(parameters) == 136
    original = {n: p.detach().clone() for n, p in parameters}
    old = torch.load(
        Path(spec["prior_root"]) / "forecasts.pt", map_location="cpu", weights_only=True
    )
    cache, lock = root / "forecasts.npz", root / "forecast-lock.json"

    def status(stage, completed, total):
        atomic_json(
            root / "status.json",
            {"stage": stage, "completed": completed, "total": total, "checked_unix": time.time()},
        )

    representation_path = root / "batch-representations.pt"
    if lock.exists():
        verify_files(root, read_json(lock)["files_sha256"])
        old = torch.load(representation_path, map_location="cpu", weights_only=True)
    else:
        status("batch_gradients_fp32", 0, 16)
        old = batch_representations(model, scoring, tokenize, parameters, payload, data, spec, old)
        torch.save(old, representation_path)

    if lock.exists():
        assert read_json(lock)["plan_sha256"] == file_hash(root / "plan.json")
        verify_files(root, read_json(lock)["files_sha256"])
    else:
        bank = gradient_bank(spec, old, payload).cuda()
        allrows, allunits, allrewards, allbase, allwarm = [], [], [], [], []
        forward_unique = 0
        for qi, row in enumerate(data["probes"]):
            rows = []
            for ci, temperature in enumerate((spec["temperature"], spec["warm_temperature"])):
                for chunk in range(4):
                    items, _ = sample(
                        model,
                        tokenizer,
                        row,
                        64,
                        spec["proposal_seed"] + qi * 10000 + ci * 100 + chunk,
                        {**spec, "temperature": temperature},
                    )
                    for j, item in enumerate(items):
                        item.update(
                            sample=ci * 256 + chunk * 64 + j,
                            component=ci,
                            component_sample=chunk * 64 + j,
                        )
                    rows.extend(items)
            unique = {}
            units, rewards, base, warm = [], [], [], []
            for si, record in enumerate(rows):
                key = tuple(record["token_ids"])
                if key not in unique:
                    terms = trajectory_terms(
                        model,
                        tokenizer,
                        row["prompt"],
                        record["token_ids"],
                        [spec["temperature"], spec["warm_temperature"]],
                    )
                    b, w = (t.item() for t in terms)
                    g = flat_grad(terms[0], parameters).cuda()
                    u = -(bank @ g).cpu().numpy()
                    unique[key] = (u, b, w)
                    forward_unique += 1
                u, b, w = unique[key]
                units.append(u)
                base.append(b)
                warm.append(w)
                values = {**record["checks"], **record["quality"]}
                rewards.append([values[k] for k in ENDPOINTS])
                if (si + 1) % 64 == 0:
                    status("proposal_gradients", qi * 512 + si + 1, 8192)
            allrows.extend(rows)
            allunits.append(np.stack(units, axis=-1))
            allrewards.append(rewards)
            allbase.append(base)
            allwarm.append(warm)
        arrays = {
            "unit_rate_logp_changes": np.stack(allunits, axis=2),
            "rewards": np.asarray(allrewards, dtype=float),
            "baseline_logp": np.asarray(allbase),
            "warm_logp": np.asarray(allwarm),
        }
        for name, (indices, proposal) in BANKS.items():
            estimates = weighted_estimators(
                arrays["unit_rate_logp_changes"],
                arrays["rewards"],
                arrays["baseline_logp"],
                arrays["warm_logp"],
                spec["rates"],
                indices,
                proposal,
            )
            arrays.update({f"{name}_{k}": v for k, v in estimates.items()})
        np.savez_compressed(cache, **arrays)
        atomic_json(
            root / "proposals.json",
            {
                "plan_sha256": file_hash(root / "plan.json"),
                "rows": allrows,
                "unique_gradient_forwards": forward_unique,
            },
        )
        atomic_json(
            lock,
            {
                "locked_unix": time.time(),
                "plan_sha256": file_hash(root / "plan.json"),
                "all_forecasts_before_updates": True,
                "files_sha256": {
                    p.name: file_hash(p)
                    for p in (cache, root / "proposals.json", representation_path)
                },
            },
        )
        del bank
    directory = root / "outcomes"
    directory.mkdir(exist_ok=True)

    def collect():
        rows = []
        for qi, row in enumerate(data["probes"]):
            for chunk in range(4):
                items, _ = sample(
                    model,
                    tokenizer,
                    row,
                    32,
                    spec["evaluation_seed"] + 1000000 + qi * 100 + chunk,
                    spec,
                )
                for j, item in enumerate(items):
                    item["sample"] = chunk * 32 + j
                rows.extend(items)
        return rows

    baseline = directory / "baseline.json"
    if not baseline.exists():
        status("baseline_evaluation", 0, 49)
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
        set_direction(parameters, original, old["batch_gradients"][bi], rate)
        try:
            rows = collect()
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
                "rows": rows,
                "adapter_restored": True,
                "update_started_unix": update_started,
                "completed_unix": time.time(),
            },
        )
        completed.append(path)
        status("updates", len(completed) - 1, 48)
    restore(parameters, original)
    atomic_json(
        root / "manifest.json",
        {
            "started_unix": started,
            "completed_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "forecast_lock_sha256": file_hash(lock),
            "updates": 48,
            "evaluation_generations": 49 * 16 * 128,
            "proposal_generations": 16 * 512,
            "adapter_restored": True,
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p)
                for p in (cache, lock, root / "proposals.json", representation_path, *completed)
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
