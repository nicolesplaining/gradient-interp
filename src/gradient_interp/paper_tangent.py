"""Development: sample a frozen parameter-logit linearization autoregressively."""

# ruff: noqa: E501
import argparse
import itertools
import time
from pathlib import Path

import torch
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from .confirmation import verify_files
from .paper_batch import METHODS, generated_checks
from .paper_mixture import stable_setup
from .paper_sample import ENDPOINTS, gradient_bank, prompt_ids, stopped_tokens
from .paper_setup import read_json
from .paper_steering import restore, set_direction
from .paper_steering_analysis import quality
from .study import atomic_json, file_hash

TANGENT_METHODS = ("full_gradient", "ekfac_projection", "dictionary", "svd16", "zero")


def direction_dict(parameters, gradient):
    result, offset = {}, 0
    for name, p in parameters:
        result[name] = -gradient[offset : offset + p.numel()].reshape_as(p).to(p.device)
        offset += p.numel()
    assert offset == gradient.numel()
    return result


def last_logits(model, ids, parameters=None):
    kwargs = dict(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1
    )
    out = model(**kwargs) if parameters is None else functional_call(model, parameters, (), kwargs)
    return out.logits[:, -1].float()


def tangent_logits(model, ids, original, direction):
    return jvp(lambda p: last_logits(model, ids, p), (original,), (direction,))


def autoregressive(
    model, tokenizer, row, count, seed, spec, original=None, direction=None, rate=0.0
):
    """Identical uncached math-SDPA sampling for tangent forecasts and actual models."""
    ids = torch.tensor(
        [prompt_ids(tokenizer, row["prompt"])] * count, device=next(model.parameters()).device
    )
    prefix = ids.shape[1]
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    done = torch.zeros(count, device=ids.device, dtype=torch.bool)
    calls, max_derivative = 0, 0.0
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for _ in range(spec["max_new_tokens"]):
            if direction is None:
                logits = last_logits(model, ids, original)
            else:
                logits, derivative = tangent_logits(model, ids, original, direction)
                assert torch.isfinite(derivative).all()
                max_derivative = max(max_derivative, float(derivative.abs().max()))
                logits = logits + rate * derivative
            assert torch.isfinite(logits).all()
            probabilities = torch.softmax(logits / spec["temperature"], dim=-1)
            nxt = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            nxt = torch.where(done, tokenizer.pad_token_id, nxt)
            ids = torch.cat((ids, nxt[:, None]), dim=-1)
            done |= torch.isin(nxt, torch.tensor(spec["stop_ids"], device=ids.device))
            calls += 1
            if done.all():
                break
    token_lists = [stopped_tokens(x, spec["stop_ids"]) for x in ids[:, prefix:].tolist()]
    rows = []
    for i, (tokens, text) in enumerate(
        zip(token_lists, tokenizer.batch_decode(token_lists, skip_special_tokens=True), strict=True)
    ):
        r = dict(
            id=row["id"],
            sample=i,
            response=text,
            token_ids=tokens,
            checks=generated_checks(text, row["tags"]),
        )
        r["quality"] = quality(r, spec["max_new_tokens"])
        rows.append(r)
    return rows, dict(
        forward_calls=calls,
        jvp_calls=calls if direction is not None else 0,
        max_abs_logit_derivative=max_derivative,
    )


def prepare(root, prior):
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "specification.json").exists()
    prior = prior.resolve()
    old = read_json(prior / "specification.json")
    source = read_json(prior / "data.json")
    assert (
        read_json(prior / "controller.json")["status"] == "complete"
        and read_json(prior / "integrity.json")["complete"]
    )
    data = dict(batches=source["batches"][:4], probes=[source["probes"][i] for i in (0, 1, 8, 9)])
    atomic_json(root / "data.json", data)
    spec = dict(
        scope="synthetic_development_only",
        max_hours=12,
        prior_root=str(prior),
        source_root=old["source_root"],
        source_config=old["source_config"],
        training_seed=old["training_seed"],
        gradient_cache=str(prior / "batch-representations.pt"),
        methods=list(TANGENT_METHODS),
        rates=[0.0003, 0.003, 0.03],
        endpoints=list(ENDPOINTS),
        batch_indices=list(range(4)),
        probe_indices=[0, 1, 8, 9],
        temperature=0.8,
        stop_ids=[1, 106],
        max_new_tokens=96,
        forecast_samples=32,
        evaluation_samples=128,
        chunk=4,
        forecast_seed=2026101317,
        evaluation_seed=2026101318,
        numerics="float32_math_sdpa_tf32_disabled_no_kv_cache",
        hypothesis="Sampling the original-model logit tangent along its own generated prefixes can represent emergent events absent from a fixed proposal bank. Curvature may still defeat the approximation, and compressed gradients may be worse than full gradients. No positive result assumed.",
        selection="First four prior batches, one per style, and first two questions from each old template by index. Chosen after completed development evidence; not untouched data. Same adapters and hardware; only new response RNG.",
        forecast="For each batch/rate/representation, sample from softmax((logits(theta,prefix)-rate*J_logits(theta,prefix)*gradient_representation)/temperature). Use forward-mode AD at ORIGINAL parameters and newly sampled prefixes, including every A/B derivative. Linearize final model logits after any softcap. No finite differences, post-update parameters, cached base trajectories, clipping or importance weighting in forecasts. All forecast generations and checks locked before any actual update. Zero uses shared original-model forecast generations, hence exactly zero predicted change. Full/dictionary/EKFAC/SVD directions preserve actual batch-gradient magnitude.",
        updates="Reuse corresponding FP32 mean-per-document mean-completion CE batch gradients from the completed mixture experiment, including its original training EOS convention. Apply 12 actual one-step SGD updates across all136LoRA tensors; restore after each. Regenerate independent128responses/question with same uncached math-SDPA sampler, cap96,T.8,full vocabulary and EOS1/106. Common evaluation RNG across all conditions, separate from forecast stream.",
        counts=dict(
            forecast_conditions=49,
            forecast_generations=6272,
            actual_updates=12,
            evaluation_conditions=13,
            evaluation_generations=6656,
        ),
        cost="This is an expensive autoregressive surrogate, not an efficiency claim. Every generated prefix requires original model forward plus directional derivative, without KV cache. Record JVP/forward calls, tokens, runtime and GPU peak memory separately from actual evaluations. Compare against simply evaluating real updates.",
        analysis="No outcome reading until BOTH runs complete or original deadlines. Verify all source/input/cache/forecast/result hashes,49forecastconditions/6272forecastresponses,13evaluationconditions/6656responses,12updates,lock-beforeeveryupdate and136tensorrestoration. Report every method/rate/seed/style and four endpoints, full generated content/order/format/length/truncation/empty/repetition. Samequestion and calibrationmean template transfer. Independent forecast/evaluation MonteCarlo uncertainty with common-RNG pairing; question resampling within template; only one batch/style means no within-style batch-population claims. Fixed examples all4batches,question0/2,sample0,allrates/methods/seeds. Compare full/EKFAC/dictionary/SVD/no-change and original reweighting only as prior-context due to new sampler/data subset. No significance/novelty/confirmation claim.",
        references=[
            "https://docs.pytorch.org/docs/2.9/generated/torch.func.jvp.html",
            "https://arxiv.org/abs/2603.14665",
        ],
        input_hashes={
            **old["input_hashes"],
            **{
                str(prior / n): file_hash(prior / n)
                for n in (
                    "specification.json",
                    "data.json",
                    "integrity.json",
                    "batch-representations.pt",
                )
            },
        },
    )
    atomic_json(root / "specification.json", spec)


def preflight(root):
    spec = read_json(root / "specification.json")
    data = read_json(root / "data.json")
    model, _, tok, _, payload, parameters = stable_setup(spec)
    for p in model.parameters():
        p.requires_grad_(False)
    original = {n: p.detach().clone() for n, p in parameters}
    saved = torch.load(spec["gradient_cache"], map_location="cpu", weights_only=True)
    bank = gradient_bank(spec, saved, payload)
    assert bank.shape == (16, 6, 2228224) and torch.isfinite(bank).all()
    direction = direction_dict(parameters, bank[0, 0])
    row = data["batches"][0]["documents"][0]
    ids = torch.tensor([prompt_ids(tok, row["prompt"])] * spec["chunk"], device="cuda")
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        started = time.time()
        v, d = tangent_logits(model, ids, original, direction)
        torch.cuda.synchronize()
        jvp_time = time.time() - started
        h = 0.0001
        plus = last_logits(model, ids, {n: p + h * direction[n] for n, p in original.items()})
        minus = last_logits(model, ids, {n: p - h * direction[n] for n, p in original.items()})
        finite = (plus - minus) / (2 * h)
        relative = float((finite - d).norm() / d.norm())
        cosine = float(torch.nn.functional.cosine_similarity(finite, d).mean())
        assert relative < 0.01 and cosine > 0.999, (relative, cosine)
        z, zd = tangent_logits(
            model, ids, original, {n: torch.zeros_like(p) for n, p in original.items()}
        )
        assert torch.equal(z, v) and torch.count_nonzero(zd) == 0
        assert torch.equal(v, last_logits(model, ids))
        # Bound memory at a long deterministic training prefix before freezing.
        longest = (
            max(len(prompt_ids(tok, q["prompt"])) for q in data["probes"]) + spec["max_new_tokens"]
        )
        extra = torch.full(
            (spec["chunk"], max(0, longest - ids.shape[1])),
            int(ids[0, -1]),
            device=ids.device,
            dtype=ids.dtype,
        )
        long_ids = torch.cat((ids, extra), dim=-1)
        lv, ld = tangent_logits(model, long_ids, original, direction)
        assert torch.isfinite(lv).all() and torch.isfinite(ld).all()
    trial = {**spec, "max_new_tokens": 8}
    before, _ = autoregressive(model, tok, row, spec["chunk"], 731, trial, original)
    zero, _ = autoregressive(
        model,
        tok,
        row,
        spec["chunk"],
        731,
        trial,
        original,
        {n: torch.zeros_like(p) for n, p in original.items()},
        0.03,
    )
    assert [r["token_ids"] for r in before] == [r["token_ids"] for r in zero]
    assert all(torch.equal(p, original[n]) for n, p in parameters)
    set_direction(parameters, original, saved["batch_gradients"][0], 0.003)
    assert any(not torch.equal(p, original[n]) for n, p in parameters)
    restore(parameters, original)
    after, _ = autoregressive(model, tok, row, spec["chunk"], 731, trial)
    assert [r["token_ids"] for r in before] == [r["token_ids"] for r in after]
    atomic_json(
        root / "preflight.json",
        dict(
            completed_unix=time.time(),
            specification_sha256=file_hash(root / "specification.json"),
            parameter_tensors=len(parameters),
            gradient_bank_shape=list(bank.shape),
            finite_difference_step=h,
            relative_jvp_error=relative,
            cosine=cosine,
            chunk_jvp_seconds=jvp_time,
            zero_jvp_exact=True,
            zero_generation_exact=True,
            adapter_restored=True,
            sampled_generation_restored=True,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            training_only=True,
        ),
    )


def freeze(root):
    assert not (root / "plan.json").exists()
    spec = read_json(root / "specification.json")
    pre = read_json(root / "preflight.json")
    assert (
        pre["specification_sha256"] == file_hash(root / "specification.json")
        and pre["adapter_restored"]
        and pre["sampled_generation_restored"]
    )
    verify_files(Path.cwd(), spec["input_hashes"])
    atomic_json(
        root / "plan.json",
        dict(
            scope="development_only",
            frozen_unix=time.time(),
            max_hours=spec["max_hours"],
            source_files_sha256={
                str(p): file_hash(p) for p in sorted(Path("src/gradient_interp").glob("*.py"))
            },
            input_hashes={
                str(root / n): file_hash(root / n)
                for n in ("specification.json", "preflight.json", "data.json")
            },
            jobs=[
                dict(
                    id="autoregressive-tangent",
                    module="gradient_interp.paper_tangent",
                    arguments=["run", "--root", str(root)],
                    manifest=str(root / "manifest.json"),
                )
            ],
        ),
    )


def run(root):
    spec = read_json(root / "specification.json")
    data = read_json(root / "data.json")
    plan = read_json(root / "plan.json")
    started = time.time()
    for hashes in (spec["input_hashes"], plan["source_files_sha256"], plan["input_hashes"]):
        verify_files(Path.cwd(), hashes)
    model, _, tok, _, payload, parameters = stable_setup(spec)
    for p in model.parameters():
        p.requires_grad_(False)
    original = {n: p.detach().clone() for n, p in parameters}
    saved = torch.load(spec["gradient_cache"], map_location="cpu", weights_only=True)
    bank = gradient_bank(spec, saved, payload)
    ph = file_hash(root / "plan.json")
    forecasts = root / "forecasts"
    outcomes = root / "outcomes"
    forecasts.mkdir(exist_ok=True)
    outcomes.mkdir(exist_ok=True)

    def status(stage, n, total):
        atomic_json(
            root / "status.json",
            dict(stage=stage, completed=n, total=total, checked_unix=time.time()),
        )

    def responses(count, stream, primal=None, direction=None, rate=0.0):
        rows = []
        cost = dict(forward_calls=0, jvp_calls=0, max_abs_logit_derivative=0.0)
        begin = time.time()
        for qi, q in enumerate(data["probes"]):
            for offset in range(0, count, spec["chunk"]):
                chunk, c = autoregressive(
                    model,
                    tok,
                    q,
                    min(spec["chunk"], count - offset),
                    stream + qi * 100000 + offset,
                    spec,
                    primal,
                    direction,
                    rate,
                )
                for row in chunk:
                    row["sample"] += offset
                rows.extend(chunk)
                for key in ("forward_calls", "jvp_calls"):
                    cost[key] += c[key]
                cost["max_abs_logit_derivative"] = max(
                    cost["max_abs_logit_derivative"], c["max_abs_logit_derivative"]
                )
        cost["elapsed_seconds"] = time.time() - begin
        return rows, cost

    lock = root / "forecast-lock.json"
    fp = []

    def forecast(path, b=None, rate=None, method="zero"):
        if path.exists():
            rec = read_json(path)
            assert rec["plan_sha256"] == ph and rec["original_adapter_unchanged"]
            return
        status("forecast_generations", len(fp), 49)
        direction = (
            None if b is None else direction_dict(parameters, bank[b, METHODS.index(method)])
        )
        begin = time.time()
        rows, cost = responses(
            spec["forecast_samples"], spec["forecast_seed"], original, direction, rate or 0.0
        )
        assert all(torch.equal(p, original[n]) for n, p in parameters)
        atomic_json(
            path,
            dict(
                plan_sha256=ph,
                batch_index=b,
                rate=rate,
                method=method,
                rows=rows,
                cost=cost,
                started_unix=begin,
                completed_unix=time.time(),
                original_adapter_unchanged=True,
            ),
        )

    forecast(forecasts / "baseline.json")
    fp.append(forecasts / "baseline.json")
    for b, rate, method in itertools.product(
        spec["batch_indices"], spec["rates"], TANGENT_METHODS[:-1]
    ):
        path = forecasts / f"batch-{b:02d}-rate-{rate:g}-{method}.json"
        forecast(path, b, rate, method)
        fp.append(path)
    assert len(fp) == 49
    if not lock.exists():
        atomic_json(
            lock,
            dict(
                locked_unix=time.time(),
                plan_sha256=ph,
                files_sha256={str(p.relative_to(root)): file_hash(p) for p in fp},
                gradient_cache_sha256=file_hash(Path(spec["gradient_cache"])),
                adapter_unchanged=True,
            ),
        )
    locked = read_json(lock)
    assert locked["plan_sha256"] == ph
    verify_files(root, locked["files_sha256"])
    assert locked["gradient_cache_sha256"] == file_hash(Path(spec["gradient_cache"]))
    done = []
    baseline = outcomes / "baseline.json"
    if not baseline.exists():
        status("evaluation_baseline", 0, 13)
        rows, cost = responses(spec["evaluation_samples"], spec["evaluation_seed"])
        atomic_json(
            baseline,
            dict(
                plan_sha256=ph,
                forecast_lock_sha256=file_hash(lock),
                rows=rows,
                cost=cost,
                completed_unix=time.time(),
            ),
        )
    done.append(baseline)
    for b, rate in itertools.product(spec["batch_indices"], spec["rates"]):
        path = outcomes / f"batch-{b:02d}-rate-{rate:g}.json"
        if path.exists():
            rec = read_json(path)
            assert (
                rec["plan_sha256"] == ph
                and rec["adapter_restored"]
                and rec["forecast_lock_sha256"] == file_hash(lock)
            )
            done.append(path)
            continue
        status("actual_updates", len(done) - 1, 12)
        begin = time.time()
        set_direction(parameters, original, saved["batch_gradients"][b], rate)
        try:
            rows, cost = responses(spec["evaluation_samples"], spec["evaluation_seed"])
        finally:
            restore(parameters, original)
        atomic_json(
            path,
            dict(
                plan_sha256=ph,
                forecast_lock_sha256=file_hash(lock),
                batch_index=b,
                rate=rate,
                update_started_unix=begin,
                completed_unix=time.time(),
                rows=rows,
                cost=cost,
                adapter_restored=True,
            ),
        )
        done.append(path)
    restore(parameters, original)
    verify_files(Path.cwd(), spec["input_hashes"])
    atomic_json(
        root / "manifest.json",
        dict(
            started_unix=started,
            completed_unix=time.time(),
            scope=spec["scope"],
            plan_sha256=ph,
            forecast_lock_sha256=file_hash(lock),
            adapter_restored=True,
            updates=12,
            forecast_conditions=49,
            forecast_generations=6272,
            evaluation_conditions=13,
            evaluation_generations=6656,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            files_sha256={str(p.relative_to(root)): file_hash(p) for p in [*fp, lock, *done]},
        ),
    )
    status("complete", 12, 12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "preflight", "freeze", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prior", type=Path)
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.command == "prepare":
        prepare(a.root.resolve(), a.prior)
    else:
        globals()[a.command](a.root.resolve())


if __name__ == "__main__":
    main()
