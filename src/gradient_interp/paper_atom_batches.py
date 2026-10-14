"""Prospective atom explanations, matched semantic assays and selective batch edits."""

# ruff: noqa: E501
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .confirmation import verify_files
from .paper_atom_data import FORMATS, all_items, canonical, checks, make_data, prompt_strings
from .paper_batch import flat_grad, scalar_logp
from .paper_mixture import stable_setup
from .paper_sample import sample, trajectory_logp
from .paper_setup import read_docs, read_json
from .paper_steering import restore, set_direction
from .study import atomic_json, file_hash

ENDPOINTS = (
    "numbered",
    "bullets",
    "answer_values_and_order",
    "tokens",
    "truncated",
    "words",
    "empty",
    "possible_degeneration",
)
FORECAST_METHODS = (
    "dictionary",
    "named_atoms_only",
    "shuffled_atom_profiles",
    "full_gradient",
    "svd16",
    "zero",
)


def reward(row):
    return [float(row["checks"].get(k, row["quality"].get(k))) for k in ENDPOINTS]


def sample_rows(model, tokenizer, row, count, seed, spec, scores=False):
    rows, logp = sample(model, tokenizer, row, count, seed, spec, return_scores=scores)
    for r in rows:
        r["checks"] = checks(r["response"], row["expected"], row["task"])
        r["task"] = row["task"]
    return rows, logp


def covariance_effects(unit, rewards):
    """Q,S,D and Q,S,E -> Q,D,E. Paired, unbiased sample covariance."""
    if unit.shape[:2] != rewards.shape[:2] or unit.shape[1] < 2:
        raise ValueError("Mismatched paired proposal samples")
    centered = rewards - rewards.mean(1, keepdims=True)
    return np.einsum("qsd,qse->qde", unit, centered, optimize=True) / (unit.shape[1] - 1)


def atom_forecasts(coefficients, effects, rate):
    return rate * np.einsum("bd,qde->bqe", coefficients, effects, optimize=True)


def prepare(root, prior, steering):
    assert not (root / "specification.json").exists()
    root.mkdir(parents=True, exist_ok=True)
    prior, steering = prior.resolve(), steering.resolve()
    old = read_json(prior / "specification.json")
    source = Path(old["source_root"])
    assert read_json(prior / "integrity.json")["complete"]
    selection = read_json(steering / "selection.json")
    cards = {}
    for label in ("numbered", "bullets"):
        s = selection[label]
        cards[label] = dict(
            atom_index=s["atom_idx"],
            orientation=s["orientation"],
            description=f"A descent contribution associated with answers presented as {'numbered' if label == 'numbered' else 'bulleted'} lists.",
            source="Prior discovery-training top20 detector purity, then coherence; no new outcomes used.",
            status="Candidate hypothesis; prior steering suggested formatting effects with length confounding. Neither semantics nor selectivity is assumed.",
            training_examples=s["training_examples"],
        )
    data = make_data()
    items = all_items(data)
    assert len({canonical(q["prompt"]) for q in items}) == len(items) == 48
    assert len({tuple(sorted(q["expected"])) for q in items}) == len(items)
    # Enumerate only input-data filenames, explicitly excluding all output directories.
    prior_paths = sorted(
        set(
            list(Path("runs").glob("paper-*/data.json"))
            + list(Path("runs").glob("paper-*/questions.json"))
            + [source / "training.jsonl", steering / "selection.json"]
        )
    )
    old_prompts, scanned = set(), {}
    for p in prior_paths:
        if not p.exists() or p.parent == root:
            continue
        value = read_docs(p) if p.suffix == ".jsonl" else read_json(p)
        old_prompts.update(canonical(t) for t in prompt_strings(value))
        scanned[str(p)] = file_hash(p)
    collisions = [q["id"] for q in items if canonical(q["prompt"]) in old_prompts]
    assert not collisions, collisions
    atomic_json(root / "data.json", data)
    atomic_json(root / "atom-cards.json", cards)
    atomic_json(
        root / "data-audit.json",
        dict(
            written_unix=time.time(),
            input_files_sha256=scanned,
            unique_prior_prompts=len(old_prompts),
            new_unique_items=len(items),
            exact_prompt_collisions=collisions,
            disjoint_new_answer_triples=True,
            scope="Exact normalized prompt check against enumerated paper-phase inputs and discovery training. Not full semantic/source deduplication of all historical corpora; no confirmation claim.",
        ),
    )
    spec = dict(
        scope="development_only",
        source_root=str(source),
        source_config=old["source_config"],
        training_seed=old["training_seed"],
        prior_root=str(prior),
        steering_root=str(steering),
        max_hours=12,
        rate=0.003,
        methods=list(FORECAST_METHODS),
        endpoints=list(ENDPOINTS),
        proposal_samples=128,
        evaluation_samples=64,
        chunk=16,
        max_new_tokens=96,
        temperature=0.8,
        stop_ids=[1, 106],
        proposal_seed=2026101403,
        evaluation_seed=2026101404,
        control_seed=2026101405,
        numerics="float32_sdpa_tf32_disabled",
        counts=dict(
            batches=16,
            semantic_gradients=48,
            proposal_responses=3072,
            batch_updates=64,
            single_atom_updates=8,
            evaluation_conditions=73,
            evaluation_responses=74752,
        ),
        priority="Interpretable sparse atom explanations are the method of interest. Beating full-gradient or SVD is not a success requirement. Retain honest accuracy, cost and failure comparisons.",
        semantics="Freeze two discovery-selected labels before new measurements. Evaluate oriented coefficients and gradient alignment across content-matched plain/numbered/bulleted and short/long answers on8separate items. Report every task/length and both atoms, including wrong or absent preferences. Training purity alone is not validation.",
        forecast="Estimate all500atoms' score-function effects from128original-model sampled trajectories perquestion. Preserve each sample multiplicity and exact stop/cap token. Primary: transfer mean eight calibration-question profiles to16evaluation questions, then sum magnitude-bearing batch coefficients times profiles. Secondary same-question predictions also locked. Covariance uses N/(N-1); no clipping. Save full atom contributions, named-only share, and fixed random profile permutation. Full-gradient/SVD16/zero are references, not winner requirements. No fitted outcome regression or new atom learning.",
        selectivity="16batches crossing four shared content blocks and fourstyles. Perbatch actualfull, remove_named, remove_equal_norm_random_active_atom, and rescale_full_to_removed_norm. Target is bullets forbulletedbatches andnumbered otherwise. Random activeatom drawn withfixedseed excluding both namedatoms; if none, fixed noncandidateatom. Retain zero namedcontributions as zero edits. Two namedatoms alone at±.5 and±1times medianbatchgradientnorm, all atfixedrate. The training orientation defines positive descent. These8controls test signandselectivity, not selectedbeststrength.",
        endpoints_note="Requested vocabulary values/order is a controlled content check, not unrestricted semantic correctness. Format must be assessed with content, length, truncation, emptiness and repetition; no formatting-only success claim.",
        analysis="No new outcome inspection until BOTH runs finish or original deadlines. All semantic assays/atomcards/methods/tasks/styles/conditions included. Forecast and evaluation MC streams independent; pair samples across conditions within each question, bootstrap questions within task/split and paired content blocks. Report pointwise descriptive intervals and tails, not seed population/significance claims. Four blocks/four task families and reused adapters are development only. Separate descriptor validation, selective causal effects and quantitative forecast claims. Fixed examples first evaluation question per task/sample0, firstblock allstyles/allconditions and allsingleatomconditions. No secondary winner promotion.",
        decision="Use complete evidence to decide whether a fresh-seed, source/prompt-deduplicated confirmation with explicit effect thresholds is justified. Strong positive results cannot be promised. Do not expand proposal banks as a route to large-step reliability.",
        input_hashes={
            **old["input_hashes"],
            **{
                str(p): file_hash(p)
                for p in [
                    prior / "integrity.json",
                    prior / "batch-representations.pt",
                    steering / "selection.json",
                    source / "training.jsonl",
                ]
            },
        },
    )
    atomic_json(root / "specification.json", spec)


def preflight(root):
    spec, data = [read_json(root / n) for n in ("specification.json", "data.json")]
    model, scoring, tok, tokenize, payload, parameters = stable_setup(spec)
    from gradient_atoms.projection import load_ekfac_eigen, unproject_atom

    factors = load_ekfac_eigen(
        str(Path(spec["source_root"]) / "factors/paper_ekfac/factors_training")
    )
    atom_bank = (
        torch.stack(
            [
                unproject_atom(d.numpy(), payload["module_info"], factors, 2228224)
                for d in payload["dictionary"]
            ]
        )
        .float()
        .cuda()
    )
    assert atom_bank.shape == (500, 2228224) and torch.isfinite(atom_bank).all()
    original = {n: p.detach().clone() for n, p in parameters}
    q = data["batches"][0]["documents"][0]
    value, n = scalar_logp(model, scoring, tokenize, q["prompt"], q["answer"])
    g = flat_grad(-value / n, parameters)
    contractions = atom_bank @ g.cuda()
    assert torch.isfinite(contractions).all()
    # A deterministic long completion exercises the generation-cap gradient allocation.
    value, n = scalar_logp(model, scoring, tokenize, q["prompt"], q["answers"][5] * 8)
    long_gradient = flat_grad(-value / n, parameters)
    assert torch.isfinite(atom_bank @ long_gradient.cuda()).all()
    before, generated = sample_rows(model, tok, q, spec["chunk"], 917, spec, scores=True)
    errors = []
    for row, gen in zip(before, generated, strict=True):
        logp = trajectory_logp(model, tok, q["prompt"], row["token_ids"], spec["temperature"])
        errors.append(abs(float(logp.detach()) - gen) / len(row["token_ids"]))
    assert max(errors) < 0.005
    set_direction(parameters, original, g, spec["rate"])
    assert any(not torch.equal(p, original[n]) for n, p in parameters)
    restore(parameters, original)
    after, _ = sample_rows(model, tok, q, spec["chunk"], 917, spec)
    assert [r["token_ids"] for r in before] == [r["token_ids"] for r in after]
    assert len(parameters) == 136 and g.numel() == 2228224
    atomic_json(
        root / "preflight.json",
        dict(
            completed_unix=time.time(),
            specification_sha256=file_hash(root / "specification.json"),
            tensors=len(parameters),
            gradient_elements=g.numel(),
            finite_gradient=bool(torch.isfinite(g).all()),
            max_score_error_per_token=max(errors),
            adapter_restored=True,
            generation_restored=True,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            outputs_not_inspected=True,
            atom_bank_shape=list(atom_bank.shape),
            long_training_gradient_checked=True,
        ),
    )


def freeze(root):
    assert not (root / "plan.json").exists()
    spec, pre = [read_json(root / n) for n in ("specification.json", "preflight.json")]
    assert pre["specification_sha256"] == file_hash(root / "specification.json")
    assert pre["adapter_restored"] and pre["generation_restored"] and pre["finite_gradient"]
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
                for n in (
                    "specification.json",
                    "preflight.json",
                    "data.json",
                    "atom-cards.json",
                    "data-audit.json",
                )
            },
            jobs=[
                dict(
                    id="interpretable-atom-batches",
                    module="gradient_interp.paper_atom_batches",
                    arguments=["run", "--root", str(root.resolve())],
                    manifest=str(root.resolve() / "manifest.json"),
                )
            ],
        ),
    )


def intervention_conditions(gradients, coefficients, directions, batches, cards, seed):
    """Describe edits without reading any outcomes; coefficients have restored magnitude."""
    rng = np.random.default_rng(seed)
    named = {v["atom_index"] for v in cards.values()}
    result = []
    norms = directions.norm(dim=1).numpy()
    for b, batch in enumerate(batches):
        target = cards[batch["target"]]["atom_index"]
        coefficient = float(coefficients[b, target])
        active = [i for i, x in enumerate(coefficients[b]) if x != 0 and i not in named]
        eligible = active or [i for i in range(len(directions)) if i not in named]
        other = int(rng.choice(eligible))
        random_sign = np.sign(coefficients[b, other]) or 1
        other_coefficient = float(random_sign * abs(coefficient) * norms[target] / norms[other])
        removed = gradients[b] - coefficient * directions[target]
        norm_scale = float(removed.norm() / gradients[b].norm())
        for mode in ("full", "remove_named", "remove_random", "rescale"):
            result.append(
                dict(
                    id=f"batch-{b:02d}-{mode}",
                    kind="batch",
                    batch_index=b,
                    mode=mode,
                    target=target,
                    target_label=batch["target"],
                    named_coefficient=coefficient,
                    random_atom=other,
                    random_coefficient=other_coefficient,
                    random_from_active=bool(active),
                    norm_scale=norm_scale,
                    named_component_norm=float(abs(coefficient) * norms[target]),
                )
            )
    median = float(gradients.norm(dim=1).median())
    for label, card in cards.items():
        idx = card["atom_index"]
        for strength in (-1.0, -0.5, 0.5, 1.0):
            result.append(
                dict(
                    id=f"atom-{label}-{strength:g}",
                    kind="atom",
                    atom_index=idx,
                    target_label=label,
                    strength=strength,
                    coefficient=float(strength * card["orientation"] * median / norms[idx]),
                    median_batch_gradient_norm=median,
                )
            )
    assert len(result) == 72
    return result


def condition_direction(condition, gradients, directions):
    c = condition
    if c["kind"] == "atom":
        return c["coefficient"] * directions[c["atom_index"]]
    g = gradients[c["batch_index"]]
    return {
        "full": lambda: g,
        "remove_named": lambda: g - c["named_coefficient"] * directions[c["target"]],
        "remove_random": lambda: g - c["random_coefficient"] * directions[c["random_atom"]],
        "rescale": lambda: g * c["norm_scale"],
    }[c["mode"]]()


def run(root):
    from sklearn.decomposition import sparse_encode

    spec, data, cards, plan = [
        read_json(root / n)
        for n in ("specification.json", "data.json", "atom-cards.json", "plan.json")
    ]
    started = time.time()
    for hashes in (spec["input_hashes"], plan["source_files_sha256"], plan["input_hashes"]):
        verify_files(Path.cwd(), hashes)
    source = Path(spec["source_root"])
    verify_files(source, read_json(source / "factors-manifest.json")["files_sha256"])
    model, scoring, tok, tokenize, payload, parameters = stable_setup(spec)
    from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac, unproject_atom

    original = {n: p.detach().clone() for n, p in parameters}
    ph = file_hash(root / "plan.json")
    factors = load_ekfac_eigen(str(source / "factors/paper_ekfac/factors_training"))

    def status(stage, completed, total):
        atomic_json(
            root / "status.json",
            dict(stage=stage, completed=completed, total=total, checked_unix=time.time()),
        )

    directions = torch.stack(
        [
            unproject_atom(d.numpy(), payload["module_info"], factors, 2228224)
            for d in payload["dictionary"]
        ]
    ).float()
    assert directions.shape == (500, 2228224) and torch.isfinite(directions).all()
    assert (directions.norm(dim=1) > 0).all()

    def encode(g):
        projected, _ = project_gradients_ekfac(g, payload["lora_names"], factors, 50)
        norm = projected.norm(dim=1, keepdim=True).clamp_min(1e-12)
        codes = sparse_encode(
            (projected / norm).numpy(),
            payload["dictionary"].numpy(),
            algorithm="lasso_lars",
            alpha=0.1,
        )
        return projected, norm, codes

    # Cache immutable, source-hashed gradient work; resumes retain prior computations.
    gp = root / "representations.pt"
    if not gp.exists():
        gradients, document_cache = [], {}
        for b, batch in enumerate(data["batches"]):
            gs = []
            for q in batch["documents"]:
                key = (q["prompt"], q["answer"])
                if key not in document_cache:
                    value, n = scalar_logp(model, scoring, tokenize, *key)
                    document_cache[key] = flat_grad(-value / n, parameters)
                gs.append(document_cache[key])
            gradients.append(torch.stack(gs).mean(0))
            status("batch_gradients", b + 1, 16)
        gradients = torch.stack(gradients)
        projected, norms, codes = encode(gradients)
        old = torch.load(
            Path(spec["prior_root"]) / "batch-representations.pt",
            map_location="cpu",
            weights_only=True,
        )
        basis = old["svd_basis"]
        svd = projected @ basis.T @ basis
        svd_gradients = torch.stack(
            [unproject_atom(x.numpy(), payload["module_info"], factors, 2228224) for x in svd]
        )
        torch.save(
            dict(
                batch_gradients=gradients,
                projected=projected,
                projected_norms=norms,
                sparse_codes=torch.from_numpy(codes).float(),
                svd_gradients=svd_gradients,
                plan_sha256=ph,
                unique_training_gradient_calls=len(document_cache),
            ),
            gp,
        )
        del document_cache
    saved = torch.load(gp, map_location="cpu", weights_only=True)
    assert saved["plan_sha256"] == ph
    gradients, svd_gradients = saved["batch_gradients"], saved["svd_gradients"]
    coefficients = (saved["sparse_codes"] * saved["projected_norms"]).numpy()
    assert coefficients.shape == (16, 500)

    assay_path = root / "semantic-assay.json"
    if not assay_path.exists():
        assay = []
        for qi, q in enumerate(data["semantic"]):
            for style, answer in enumerate(q["answers"]):
                value, n = scalar_logp(model, scoring, tokenize, q["prompt"], answer)
                g = flat_grad(-value / n, parameters)
                projected, norm, codes = encode(g[None])
                row = dict(
                    id=q["id"],
                    task=q["task"],
                    format=FORMATS[style % 3],
                    long=style >= 3,
                    gradient_norm=float(g.norm()),
                    projected_norm=float(norm[0, 0]),
                    sparse_codes=codes[0].tolist(),
                    candidates={},
                )
                for label, card in cards.items():
                    idx, orientation = card["atom_index"], card["orientation"]
                    row["candidates"][label] = dict(
                        oriented_code=float(orientation * codes[0, idx]),
                        gradient_cosine=float(
                            orientation
                            * torch.dot(g, directions[idx])
                            / (g.norm() * directions[idx].norm())
                        ),
                    )
                assay.append(row)
                status("semantic_gradients", len(assay), 48)
        atomic_json(assay_path, dict(plan_sha256=ph, rows=assay, completed_unix=time.time()))

    conditions = intervention_conditions(
        gradients, coefficients, directions, data["batches"], cards, spec["control_seed"]
    )
    condition_path = root / "conditions.json"
    if not condition_path.exists():
        atomic_json(
            condition_path, dict(plan_sha256=ph, conditions=conditions, chosen_before_outcomes=True)
        )
    assert read_json(condition_path)["conditions"] == conditions
    cache, proposals_path = root / "forecasts.npz", root / "proposals.json"
    if not cache.exists():
        bank = torch.cat([directions, gradients, svd_gradients]).cuda()
        units = np.empty((24, spec["proposal_samples"], 532), dtype=np.float64)
        rewards = np.empty((24, spec["proposal_samples"], len(ENDPOINTS)), dtype=np.float64)
        rows, unique_calls = [], 0
        for qi, q in enumerate(data["probes"]):
            trajectories = []
            for offset in range(0, spec["proposal_samples"], spec["chunk"]):
                chunk, _ = sample_rows(
                    model, tok, q, spec["chunk"], spec["proposal_seed"] + qi * 100000 + offset, spec
                )
                for row in chunk:
                    row["sample"] += offset
                trajectories.extend(chunk)
            dedup = {}
            for si, row in enumerate(trajectories):
                key = tuple(row["token_ids"])
                if key not in dedup:
                    value = trajectory_logp(
                        model, tok, q["prompt"], row["token_ids"], spec["temperature"]
                    )
                    h = flat_grad(value, parameters).cuda()
                    dedup[key] = (-(bank @ h)).double().cpu().numpy()
                    unique_calls += 1
                units[qi, si] = dedup[key]
                rewards[qi, si] = reward(row)
            rows.extend(trajectories)
            status("proposal_gradients", len(rows), 3072)
        del bank
        torch.cuda.empty_cache()
        effects = covariance_effects(units, rewards)
        assert np.isfinite(units).all() and np.isfinite(effects).all()
        atoms = effects[:, :500]
        perm = np.random.default_rng(spec["control_seed"]).permutation(500)
        named_coefficients = np.zeros_like(coefficients)
        for c in cards.values():
            named_coefficients[:, c["atom_index"]] = coefficients[:, c["atom_index"]]
        predictions = np.stack(
            [
                atom_forecasts(coefficients, atoms, spec["rate"]),
                atom_forecasts(named_coefficients, atoms, spec["rate"]),
                atom_forecasts(coefficients, atoms[:, perm], spec["rate"]),
                spec["rate"] * effects[:, 500:516].transpose(1, 0, 2),
                spec["rate"] * effects[:, 516:].transpose(1, 0, 2),
                np.zeros((16, 24, len(ENDPOINTS))),
            ],
            axis=1,
        )
        # Explicit contributions make named-atom explanations independently inspectable.
        contributions = spec["rate"] * np.einsum("bd,qde->bqde", coefficients, atoms)
        np.testing.assert_allclose(contributions.sum(2), predictions[:, 0], rtol=1e-7, atol=1e-10)
        condition_predictions = []
        for c in conditions:
            if c["kind"] == "atom":
                d = spec["rate"] * c["coefficient"] * atoms[:, c["atom_index"]]
                condition_predictions.append(np.stack([d, d]))
                continue
            b = c["batch_index"]
            d = np.stack([predictions[b, 0], predictions[b, 3]])
            if c["mode"] in ("remove_named", "remove_random"):
                idx = c["target"] if c["mode"] == "remove_named" else c["random_atom"]
                coef = (
                    c["named_coefficient"]
                    if c["mode"] == "remove_named"
                    else c["random_coefficient"]
                )
                d = d - spec["rate"] * coef * atoms[:, idx]
            elif c["mode"] == "rescale":
                d = d * c["norm_scale"]
            condition_predictions.append(d)
        atomic_json(
            proposals_path,
            dict(
                plan_sha256=ph,
                rows=rows,
                unique_gradient_calls=unique_calls,
                completed_unix=time.time(),
            ),
        )
        np.savez_compressed(
            cache,
            units=units,
            rewards=rewards,
            effects=effects,
            coefficients=coefficients,
            predictions=predictions,
            atom_contributions=contributions,
            permutation=perm,
            condition_predictions=np.stack(condition_predictions),
            baseline=rewards.mean(1),
        )
    lock_path = root / "forecast-lock.json"
    if not lock_path.exists():
        assert all(torch.equal(p, original[n]) for n, p in parameters)
        atomic_json(
            lock_path,
            dict(
                locked_unix=time.time(),
                plan_sha256=ph,
                original_adapter_unchanged=True,
                files_sha256={
                    str(p.relative_to(root)): file_hash(p)
                    for p in [
                        gp,
                        assay_path,
                        condition_path,
                        cache,
                        proposals_path,
                        root / "atom-cards.json",
                    ]
                },
            ),
        )
    lock = read_json(lock_path)
    assert lock["plan_sha256"] == ph and lock["original_adapter_unchanged"]
    verify_files(root, lock["files_sha256"])
    lh = file_hash(lock_path)
    outputs = root / "outcomes"
    outputs.mkdir(exist_ok=True)
    eval_probes = [q for q in data["probes"] if q["split"] == "evaluation"]

    def collect():
        rows = []
        for qi, q in enumerate(eval_probes):
            for offset in range(0, spec["evaluation_samples"], spec["chunk"]):
                chunk, _ = sample_rows(
                    model,
                    tok,
                    q,
                    spec["chunk"],
                    spec["evaluation_seed"] + qi * 100000 + offset,
                    spec,
                )
                for row in chunk:
                    row["sample"] += offset
                rows.extend(chunk)
        return rows

    output_paths = []
    for i, c in enumerate([dict(id="baseline", kind="baseline")] + conditions):
        path = outputs / (c["id"] + ".json")
        if path.exists():
            r = read_json(path)
            assert (
                r["forecast_lock_sha256"] == lh and r["plan_sha256"] == ph and r["adapter_restored"]
            )
            output_paths.append(path)
            continue
        status("evaluation_conditions", i, 73)
        begin = time.time()
        if c["kind"] != "baseline":
            set_direction(
                parameters, original, condition_direction(c, gradients, directions), spec["rate"]
            )
        try:
            rows = collect()
        finally:
            restore(parameters, original)
        atomic_json(
            path,
            dict(
                plan_sha256=ph,
                forecast_lock_sha256=lh,
                condition=c,
                update_started_unix=begin,
                completed_unix=time.time(),
                rows=rows,
                adapter_restored=True,
            ),
        )
        output_paths.append(path)
    verify_files(root, lock["files_sha256"])
    atomic_json(
        root / "manifest.json",
        dict(
            plan_sha256=ph,
            forecast_lock_sha256=lh,
            completed_unix=time.time(),
            elapsed_seconds=time.time() - started,
            adapter_restored=True,
            counts=spec["counts"],
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            files_sha256={
                str(p.relative_to(root)): file_hash(p) for p in [lock_path, *output_paths]
            },
        ),
    )
    status("complete", 73, 73)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "preflight", "freeze", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prior", type=Path)
    p.add_argument("--steering", type=Path)
    args = p.parse_args()
    torch.set_num_threads(8)
    args.root = args.root.resolve()
    if args.command == "prepare":
        prepare(args.root, args.prior, args.steering)
    else:
        globals()[args.command](args.root)


if __name__ == "__main__":
    main()
