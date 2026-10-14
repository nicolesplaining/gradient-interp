"""Separate description, intervention and prospective forecast evidence for frozen atoms."""

# ruff: noqa: E501
import argparse
import html
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .paper_atom_batches import ENDPOINTS, FORECAST_METHODS, reward
from .paper_atom_data import TASKS, checks
from .paper_batch_analysis import write_csv
from .paper_steering_analysis import quality
from .paper_tangent_analysis import interval_fields
from .study import atomic_json, file_hash

STYLES = ("plain", "numbered", "bullets", "mixed")
GROUPS = ("all", *STYLES, *TASKS)
MODES = ("proposal_mc", "evaluation_mc", "both_mc", "both_mc_questions_blocks")
SCOPES = ("calibration_transfer", "same_question")
CONTRASTS = (
    "remove_named_minus_full",
    "remove_random_minus_full",
    "rescale_minus_full",
    "remove_named_minus_random",
    "remove_named_minus_rescale",
)
BINARY_ENDPOINTS = tuple(e for e in ENDPOINTS if e not in ("tokens", "words"))


def read(p):
    return json.loads(p.read_text())


def guard(roots, plan):
    assert len(roots) == 2 and not read(plan)["outcomes_read_before_plan"]
    for root in roots:
        a, c, m, lock = [
            read(root / n)
            for n in ("integrity.json", "controller.json", "manifest.json", "forecast-lock.json")
        ]
        assert (
            a["complete"] and a["source_input_output_hashes_verified"] and c["status"] == "complete"
        )
        assert a["physical_control_formula_verified"]
        assert m["adapter_restored"] and lock["original_adapter_unchanged"]
        assert m["completed_unix"] < c["deadline_unix"]
        for name in ("plan", "specification", "manifest", "forecast-lock"):
            assert file_hash(root / f"{name}.json") == a[name.replace("-", "_") + "_sha256"]
        for files in (m["files_sha256"], lock["files_sha256"]):
            for name, digest in files.items():
                # The large physical-gradient cache remains on the audited original host.
                if name != "representations.pt":
                    assert file_hash(root / name) == digest, name


def rows_array(rows, probes, count):
    assert [(r["id"], r["sample"]) for r in rows] == [
        (q["id"], s) for q in probes for s in range(count)
    ]
    for qi, q in enumerate(probes):
        for r in rows[qi * count : (qi + 1) * count]:
            assert r["checks"] == checks(r["response"], q["expected"], q["task"])
            assert r["quality"] == quality(r, 96)
    return np.array([reward(r) for r in rows]).reshape(len(probes), count, len(ENDPOINTS))


def combined_units(z, cards, conditions):
    """Combine scores before covariance; this exactly equals summing all atom effects."""
    coeff = z["coefficients"]
    atom = z["units"][:, :, :500]
    named = np.zeros_like(coeff)
    for card in cards.values():
        idx = card["atom_index"]
        named[:, idx] = coeff[:, idx]
    method_units = np.stack(
        [
            np.einsum("bd,qsd->bqs", coeff, atom),
            np.einsum("bd,qsd->bqs", named, atom),
            np.einsum("bd,qsd->bqs", coeff, atom[:, :, z["permutation"]]),
            z["units"][:, :, 500:516].transpose(2, 0, 1),
            z["units"][:, :, 516:].transpose(2, 0, 1),
            np.zeros((16, 24, 128)),
        ],
        axis=1,
    )
    condition_units = []
    for c in conditions:
        if c["kind"] == "atom":
            unit = c["coefficient"] * atom[:, :, c["atom_index"]]
            condition_units.append(np.stack([unit, unit]))
            continue
        unit = method_units[c["batch_index"], [0, 3]].copy()
        if c["mode"] in ("remove_named", "remove_random"):
            idx = c["target"] if c["mode"] == "remove_named" else c["random_atom"]
            factor = (
                c["named_coefficient"] if c["mode"] == "remove_named" else c["random_coefficient"]
            )
            unit -= factor * atom[:, :, idx]
        elif c["mode"] == "rescale":
            unit *= c["norm_scale"]
        condition_units.append(unit)
    return np.concatenate(
        [method_units.reshape(96, 24, 128), np.array(condition_units).reshape(144, 24, 128)]
    ).transpose(1, 2, 0)


def weighted_covariance(units, rewards, weights):
    mean = np.einsum("qs,qse->qe", weights, rewards)
    centered = weights[:, :, None] * (rewards - mean[:, None])
    return (
        np.einsum("qsd,qse->dqe", units, centered, optimize=True)
        * weights.shape[1]
        / (weights.shape[1] - 1)
    )


def stratified_questions(rng, per_task):
    counts = np.empty(4 * per_task)
    for t in range(4):
        counts[t::4] = rng.multinomial(per_task, np.full(per_task, 1 / per_task))
    return counts / (4 * per_task)


def grouped_weights(bw, qw):
    b = np.tile(bw, (9, 1))
    q = np.tile(qw, (9, 1))
    for i in range(4):
        b[i + 1, np.arange(16) % 4 != i] = 0
        b[i + 1] /= b[i + 1].sum()
        q[i + 5, np.arange(16) % 4 != i] = 0
        q[i + 5] /= q[i + 5].sum()
    return b, q


def relative_error(mse, denominator):
    denom = np.broadcast_to(denominator, mse.shape)
    return np.sqrt(np.divide(mse, denom, out=np.full_like(mse, np.nan), where=denom > 0))


def metrics(pred, target, bw, qw, baseline=None):
    error = pred - target[:, None]
    mse = np.einsum("bmqe,gb,gq->gme", error**2, bw, qw, optimize=True)
    zero = np.einsum("bqe,gb,gq->ge", target**2, bw, qw, optimize=True)
    pmean = np.einsum("bmqe,gb,gq->gme", pred, bw, qw, optimize=True)
    ymean = np.einsum("bqe,gb,gq->ge", target, bw, qw, optimize=True)
    pred2 = np.einsum("bmqe,gb,gq->gme", pred**2, bw, qw, optimize=True)
    cross = np.einsum("bmqe,bqe,gb,gq->gme", pred, target, bw, qw, optimize=True)
    batch = np.sqrt(np.einsum("bmqe,gq->gbme", error**2, qw, optimize=True))
    worst = np.max(np.where(bw[:, :, None, None] > 0, batch, -np.inf), axis=1)
    result = dict(
        rmse=np.sqrt(mse),
        mae=np.einsum("bmqe,gb,gq->gme", np.abs(error), bw, qw, optimize=True),
        bias=pmean - ymean[:, None],
        nrmse=relative_error(mse, zero[:, None]),
        mse_difference_vs_zero=mse - zero[:, None],
        mse_difference_vs_shuffled=mse - mse[:, 2:3],
        predicted_mean=pmean,
        actual_mean=np.broadcast_to(ymean[:, None], pmean.shape),
        slope=np.divide(cross, pred2, out=np.full_like(cross, np.nan), where=pred2 > 1e-30),
        worst_batch_rmse=worst,
    )
    if baseline is not None:
        levels = pred + baseline[None, None]
        outside = np.einsum("bmqe,gb,gq->gme", ((levels < 0) | (levels > 1)).astype(float), bw, qw)
        for ei, endpoint in enumerate(ENDPOINTS):
            if endpoint not in BINARY_ENDPOINTS:
                outside[:, :, ei] = np.nan
        result["outside_probability_range_fraction"] = outside
    return result


def scopes(pred, cq=None):
    same = pred[..., 8:, :]
    if cq is None:
        cq = np.full(8, 1 / 8)
    transfer = np.einsum("...qe,q->...e", pred[..., :8, :], cq)
    return (np.broadcast_to(transfer[..., None, :], same.shape), same)


def contrasts(delta):
    # delta has C72,Q16,E8; first64conditions cross16batches and4edits.
    b = delta[:64].reshape(16, 4, 16, 8)
    return np.stack(
        (
            b[:, 1] - b[:, 0],
            b[:, 2] - b[:, 0],
            b[:, 3] - b[:, 0],
            b[:, 1] - b[:, 2],
            b[:, 1] - b[:, 3],
        )
    )


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def analyze(root, out, bootstrap_out, reps=1000):
    spec, data, cards = [
        read(root / n) for n in ("specification.json", "data.json", "atom-cards.json")
    ]
    seed = spec["training_seed"]
    rate = spec["rate"]
    z = dict(np.load(root / "forecasts.npz"))
    conditions = read(root / "conditions.json")["conditions"]
    evalq = data["probes"][8:]
    records = {
        c["id"]: read(root / "outcomes" / (c["id"] + ".json"))
        for c in [dict(id="baseline"), *conditions]
    }
    baseline = rows_array(records["baseline"]["rows"], evalq, 64)
    observations = np.array([rows_array(records[c["id"]]["rows"], evalq, 64) for c in conditions])
    difference = observations - baseline
    actual = difference.mean(2)
    target = actual[:64:4]
    pu = combined_units(z, cards, conditions)
    uniform = np.full((24, 128), 1 / 128)
    unit_effect = rate * weighted_covariance(pu, z["rewards"], uniform)
    np.testing.assert_allclose(
        unit_effect[:96].reshape(16, 6, 24, 8), z["predictions"], rtol=3e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        unit_effect[96:].reshape(72, 2, 24, 8), z["condition_predictions"], rtol=3e-6, atol=1e-9
    )
    bw, qw = grouped_weights(np.full(16, 1 / 16), np.full(16, 1 / 16))
    point = [metrics(p, target, bw, qw, z["baseline"][8:]) for p in scopes(z["predictions"])]
    boots = {(scope, mode): {k: [] for k in point[0]} for scope in SCOPES for mode in MODES}
    edit_point = np.einsum("ibqe,gb,gq->ige", contrasts(actual), bw, qw, optimize=True)
    edits = {mode: [] for mode in ("evaluation_mc", "evaluation_mc_questions_blocks")}
    single_point = np.einsum("cqe,gq->cge", actual[64:], qw, optimize=True)
    singles = {mode: [] for mode in edits}
    condition_boot = {
        (scope, mode): [] for scope in SCOPES for mode in ("both_mc", "both_mc_questions")
    }
    rng = np.random.default_rng(2026101407)
    for ri in range(reps):
        pw = rng.multinomial(128, np.full(128, 1 / 128), size=24) / 128
        ew = rng.multinomial(64, np.full(64, 1 / 64), size=16) / 64
        sampled_all = rate * weighted_covariance(pu, z["rewards"], pw)
        sampled_baseline = np.einsum("qse,qs->qe", z["rewards"], pw)
        sampled = sampled_all[:96].reshape(16, 6, 24, 8)
        yd = np.einsum("cqse,qs->cqe", difference, ew, optimize=True)
        yy = yd[:64:4]
        cq = stratified_questions(rng, 2)
        eq = stratified_questions(rng, 4)
        block = rng.multinomial(4, np.full(4, 0.25)) / 4
        bpop, qpop = grouped_weights(np.repeat(block / 4, 4), eq)
        for mode in ("both_mc", "both_mc_questions"):
            for scope, cp in zip(
                SCOPES,
                scopes(
                    sampled_all[96:].reshape(72, 2, 24, 8),
                    cq if mode.endswith("questions") else None,
                ),
                strict=True,
            ):
                ww = eq if mode.endswith("questions") else np.full(16, 1 / 16)
                cmse = np.einsum("cmqe,q->cme", (cp - yd[:, None]) ** 2, ww)
                czero = np.einsum("cqe,q->ce", yd**2, ww)
                condition_boot[scope, mode].append(
                    np.stack([np.sqrt(cmse), relative_error(cmse, czero[:, None])])
                )
        for mode in MODES:
            pop = mode == "both_mc_questions_blocks"
            preds = z["predictions"] if mode == "evaluation_mc" else sampled
            observed = target if mode == "proposal_mc" else yy
            bww, qww = (bpop, qpop) if pop else (bw, qw)
            pb = z["baseline"][8:] if mode == "evaluation_mc" else sampled_baseline[8:]
            for scope, p in zip(SCOPES, scopes(preds, cq if pop else None), strict=True):
                for key, val in metrics(p, observed, bww, qww, pb).items():
                    boots[scope, mode][key].append(val)
        cc = contrasts(yd)
        for mode, (bww, qww) in zip(edits, ((bw, qw), (bpop, qpop)), strict=True):
            edits[mode].append(np.einsum("ibqe,gb,gq->ige", cc, bww, qww, optimize=True))
            singles[mode].append(np.einsum("cqe,gq->cge", yd[64:], qww, optimize=True))
        if (ri + 1) % 100 == 0:
            print(f"seed{seed}: bootstrap {ri + 1}/{reps}", flush=True)
    boots = {key: {k: np.asarray(v) for k, v in d.items()} for key, d in boots.items()}
    edits = {k: np.asarray(v) for k, v in edits.items()}
    singles = {k: np.asarray(v) for k, v in singles.items()}
    condition_boot = {k: np.asarray(v) for k, v in condition_boot.items()}
    np.savez_compressed(
        bootstrap_out / f"seed{seed}-bootstrap.npz",
        **{f"{scope}_{mode}_{k}": v for (scope, mode), d in boots.items() for k, v in d.items()},
        **{f"edit_{k}": v for k, v in edits.items()},
        **{f"single_{k}": v for k, v in singles.items()},
        **{f"conditions_{scope}_{mode}": v for (scope, mode), v in condition_boot.items()},
    )
    metric_rows = []
    for si, scope in enumerate(SCOPES):
        for gi, mi, ei in itertools.product(range(9), range(6), range(8)):
            r = dict(
                seed=seed,
                scope=scope,
                group=GROUPS[gi],
                method=FORECAST_METHODS[mi],
                endpoint=ENDPOINTS[ei],
                **{k: v[gi, mi, ei] for k, v in point[si].items()},
            )
            for mode in MODES:
                for key in point[si]:
                    r.update(
                        interval_fields(boots[scope, mode][key][:, gi, mi, ei], f"{mode}_{key}")
                    )
            metric_rows.append(r)
    edit_rows = []
    single_rows = []
    for ci, gi, ei in itertools.product(range(5), range(9), range(8)):
        r = dict(
            seed=seed,
            contrast=CONTRASTS[ci],
            group=GROUPS[gi],
            endpoint=ENDPOINTS[ei],
            effect=edit_point[ci, gi, ei],
        )
        for mode in edits:
            r.update(interval_fields(edits[mode][:, ci, gi, ei], mode))
        edit_rows.append(r)
    for ci, gi, ei in itertools.product(range(8), (0, 5, 6, 7, 8), range(8)):
        r = dict(
            seed=seed,
            condition=conditions[64 + ci]["id"],
            label=conditions[64 + ci]["target_label"],
            strength=conditions[64 + ci]["strength"],
            group=GROUPS[gi],
            endpoint=ENDPOINTS[ei],
            effect=single_point[ci, gi, ei],
        )
        for mode in singles:
            r.update(interval_fields(singles[mode][:, ci, gi, ei], mode))
        single_rows.append(r)
    # Preserve every condition forecast, batch/question error, and quality result.
    pairs = []
    condition_rows = []
    generation = []
    for scope, p in zip(SCOPES, scopes(z["predictions"]), strict=True):
        for b, mi, qi, ei in itertools.product(range(16), range(6), range(16), range(8)):
            pairs.append(
                dict(
                    seed=seed,
                    scope=scope,
                    batch=b,
                    block=b // 4,
                    style=STYLES[b % 4],
                    method=FORECAST_METHODS[mi],
                    question=evalq[qi]["id"],
                    task=evalq[qi]["task"],
                    endpoint=ENDPOINTS[ei],
                    predicted=p[b, mi, qi, ei],
                    actual=target[b, qi, ei],
                    error=p[b, mi, qi, ei] - target[b, qi, ei],
                    baseline_proposal=z["baseline"][8 + qi, ei],
                    predicted_level=z["baseline"][8 + qi, ei] + p[b, mi, qi, ei],
                    outside_probability_range=(
                        bool(not 0 <= z["baseline"][8 + qi, ei] + p[b, mi, qi, ei] <= 1)
                        if ENDPOINTS[ei] in BINARY_ENDPOINTS
                        else None
                    ),
                )
            )
    for scope, p in zip(SCOPES, scopes(z["condition_predictions"]), strict=True):
        for ci, mi, ei in itertools.product(range(72), range(2), range(8)):
            pred = p[ci, mi, :, ei]
            obs = actual[ci, :, ei]
            error = pred - obs
            condition_rows.append(
                dict(
                    seed=seed,
                    scope=scope,
                    condition=conditions[ci]["id"],
                    model=("dictionary", "full_gradient")[mi],
                    endpoint=ENDPOINTS[ei],
                    rmse=np.sqrt(np.mean(error**2)),
                    nrmse=None if not np.any(obs) else np.sqrt(np.mean(error**2) / np.mean(obs**2)),
                    actual_mean=obs.mean(),
                    predicted_mean=pred.mean(),
                )
            )
            for mode in ("both_mc", "both_mc_questions"):
                for k, key in enumerate(("rmse", "nrmse")):
                    condition_rows[-1].update(
                        interval_fields(
                            condition_boot[scope, mode][:, k, ci, mi, ei], f"{mode}_{key}"
                        )
                    )
    for ci, c in enumerate(conditions):
        for group, ix in [("all", np.arange(16))] + [
            (task, np.arange(t, 16, 4)) for t, task in enumerate(TASKS)
        ]:
            for ei, endpoint in enumerate(ENDPOINTS):
                generation.append(
                    dict(
                        seed=seed,
                        condition=c["id"],
                        kind=c["kind"],
                        batch=c.get("batch_index"),
                        style=STYLES[c["batch_index"] % 4] if c["kind"] == "batch" else None,
                        group=group,
                        endpoint=endpoint,
                        baseline=baseline[ix, :, ei].mean(),
                        updated=observations[ci, ix, :, ei].mean(),
                        change=actual[ci, ix, ei].mean(),
                    )
                )
    examples = []
    for name in [
        "baseline",
        *[c["id"] for c in conditions[:16]],
        *[c["id"] for c in conditions[64:]],
    ]:
        for qi in range(4):
            r = records[name]["rows"][qi * 64]
            examples.append(
                dict(
                    seed=seed,
                    condition=name,
                    prompt=evalq[qi]["prompt"],
                    expected=evalq[qi]["expected"],
                    **r,
                )
            )
    shares = []
    named = [c["atom_index"] for c in cards.values()]
    for scope, contrib in zip(
        SCOPES,
        (
            np.broadcast_to(z["atom_contributions"][:, :8].mean(1)[:, None], (16, 16, 500, 8)),
            z["atom_contributions"][:, 8:],
        ),
        strict=True,
    ):
        for b, ei in itertools.product(range(16), range(8)):
            total = contrib[b, :, :, ei]
            absolute = np.abs(total).sum()
            named_absolute = np.abs(total[:, named]).sum()
            net = total.sum()
            named_net = total[:, named].sum()
            shares.append(
                dict(
                    seed=seed,
                    scope=scope,
                    batch=b,
                    style=STYLES[b % 4],
                    endpoint=ENDPOINTS[ei],
                    active_atoms=int(np.count_nonzero(z["coefficients"][b])),
                    named_absolute_share=named_absolute / absolute if absolute else None,
                    named_net_share=named_net / net if net else None,
                    named_net_mean=named_net / 16,
                    dictionary_net_mean=net / 16,
                    zero_named_target=conditions[4 * b]["named_coefficient"] == 0,
                )
            )
    coverage = []
    for qi, q in enumerate(data["probes"]):
        for ei, endpoint in enumerate(ENDPOINTS):
            coverage.append(
                dict(
                    seed=seed,
                    question=q["id"],
                    task=q["task"],
                    split=q["split"],
                    endpoint=endpoint,
                    nonzero_samples=int(np.count_nonzero(z["rewards"][qi, :, ei])),
                    baseline=z["baseline"][qi, ei],
                )
            )
    np.savez_compressed(
        out / f"seed{seed}-atom-decomposition.npz",
        coefficients=z["coefficients"],
        effects=z["effects"][:, :500],
        contributions=z["atom_contributions"],
        predictions=z["predictions"],
        condition_predictions=z["condition_predictions"],
        permutation=z["permutation"],
    )
    semantic = semantic_analysis(root, seed, reps)
    m = read(root / "manifest.json")
    lock = read(root / "forecast-lock.json")
    controller = read(root / "controller.json")
    audit = read(root / "integrity.json")
    provenance = dict(
        seed=seed,
        elapsed_seconds=m["elapsed_seconds"],
        forecast_stage_seconds=lock["locked_unix"] - controller["started_unix"],
        evaluation_stage_seconds=m["completed_unix"] - lock["locked_unix"],
        peak_memory_bytes=m["peak_memory_bytes"],
        proposal_gradient_calls=read(root / "proposals.json")["unique_gradient_calls"],
        training_gradient_calls=audit["unique_training_gradient_calls"],
        semantic_gradient_calls=48,
        proposal_responses=3072,
        evaluation_responses=74752,
        source_snapshot_sha256=audit["source_snapshot_sha256"],
        control_norm_residual_max=max(
            r["fp64_remaining_norm_relative_error"] for r in audit["physical_norm_residuals"]
        ),
        bootstrap_sha256=file_hash(bootstrap_out / f"seed{seed}-bootstrap.npz"),
    )
    return clean(
        dict(
            metrics=metric_rows,
            edits=edit_rows,
            singles=single_rows,
            pairs=pairs,
            condition_forecasts=condition_rows,
            generation=generation,
            examples=examples,
            shares=shares,
            coverage=coverage,
            semantic=semantic,
            conditions=[dict(seed=seed, **c) for c in conditions],
            provenance=provenance,
        )
    )


def semantic_analysis(root, seed, reps):
    rows = read(root / "semantic-assay.json")["rows"]
    items = list(dict.fromkeys(r["id"] for r in rows))
    output = []
    rng = np.random.default_rng(2026101407)
    weights = np.array([stratified_questions(rng, 2) for _ in range(reps)])
    for label, other in (("numbered", "bullets"), ("bullets", "numbered")):
        for long, metric, comparison in itertools.product(
            (False, True), ("oriented_code", "gradient_cosine"), ("plain", other)
        ):
            v = []
            for item in items:
                a = next(
                    r
                    for r in rows
                    if r["id"] == item and r["long"] == long and r["format"] == label
                )
                b = next(
                    r
                    for r in rows
                    if r["id"] == item and r["long"] == long and r["format"] == comparison
                )
                v.append(a["candidates"][label][metric] - b["candidates"][label][metric])
            v = np.array(v)
            for task, idx in [("all", np.arange(8))] + [
                (task, np.arange(t, 8, 4)) for t, task in enumerate(TASKS)
            ]:
                w = weights[:, idx]
                w = w / w.sum(1, keepdims=True)
                r = dict(
                    seed=seed,
                    label=label,
                    long=long,
                    metric=metric,
                    comparison=comparison,
                    task=task,
                    mean_preference=v[idx].mean(),
                    positive_pairs=int(np.count_nonzero(v[idx] > 0)),
                    zero_pairs=int(np.count_nonzero(v[idx] == 0)),
                    pairs=len(idx),
                )
                r.update(interval_fields(w @ v[idx], "item_bootstrap"))
                output.append(r)
    return dict(contrasts=output, all_rows=rows)


def plots(out, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for scope in SCOPES:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, seed in zip(axes, (42, 43), strict=True):
            vals = np.full((6, 8), np.nan)
            for mi, method in enumerate(FORECAST_METHODS):
                for ei, endpoint in enumerate(ENDPOINTS):
                    r = next(
                        r
                        for r in summary["metrics"]
                        if r["seed"] == seed
                        and r["scope"] == scope
                        and r["group"] == "all"
                        and r["method"] == method
                        and r["endpoint"] == endpoint
                    )
                    vals[mi, ei] = r["nrmse"] if r["nrmse"] is not None else np.nan
            ax.imshow(np.minimum(vals, 2), cmap="RdYlBu_r", vmin=0, vmax=2, aspect="auto")
            for i, j in itertools.product(range(6), range(8)):
                ax.text(
                    j,
                    i,
                    "undef" if np.isnan(vals[i, j]) else f"{vals[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            ax.set_yticks(range(6), FORECAST_METHODS, fontsize=9)
            ax.set_xticks(range(8), ENDPOINTS, rotation=40, ha="right", fontsize=8)
            ax.set_title(f"Seed {seed}")
        fig.suptitle(
            f"{scope.replace('_', ' ')}: error relative to no change\nOne equals no change; lower is better. Color saturates at 2; labels retain full values."
        )
        fig.tight_layout()
        fig.savefig(out / f"forecast-{scope}.png", dpi=170, bbox_inches="tight")
        fig.savefig(out / f"forecast-{scope}.pdf", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for col, seed in enumerate((42, 43)):
        for ri, metric in enumerate(("oriented_code", "gradient_cosine")):
            ax = axes[ri, col]
            rr = [
                r
                for r in summary["semantic_contrasts"]
                if r["seed"] == seed and r["metric"] == metric and r["task"] == "all"
            ]
            for i, r in enumerate(rr):
                ax.plot(
                    i,
                    r["mean_preference"],
                    "o",
                    color="#a8431e" if r["label"] == "numbered" else "#286793",
                )
                ax.vlines(i, r["item_bootstrap_lo"], r["item_bootstrap_hi"], lw=2)
            ax.axhline(0, color="gray", ls=":")
            ax.set_xticks(
                range(len(rr)),
                [
                    f"{r['label']}\n{'long' if r['long'] else 'short'} vs {r['comparison']}"
                    for r in rr
                ],
                rotation=40,
                ha="right",
                fontsize=7,
            )
            ax.set_title(f"Seed {seed}: {metric.replace('_', ' ')}")
            ax.set_ylabel("Target minus comparator")
    fig.suptitle(
        "Fixed atom descriptions: matched-content and length tests\nPositive values favor the named format; pointwise item bootstrap, development only"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5, w_pad=2.5)
    fig.savefig(out / "descriptions.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / "descriptions.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    for col, seed in enumerate((42, 43)):
        for ei, endpoint in enumerate(("numbered", "bullets", "answer_values_and_order", "tokens")):
            ax = axes[ei, col]
            scale = 1 if endpoint == "tokens" else 100
            for ci, contrast in enumerate(CONTRASTS[:3]):
                rr = [
                    next(
                        r
                        for r in summary["edits"]
                        if r["seed"] == seed
                        and r["contrast"] == contrast
                        and r["group"] == style
                        and r["endpoint"] == endpoint
                    )
                    for style in STYLES
                ]
                x = np.arange(4) + (ci - 1) * 0.18
                y = np.array([r["effect"] for r in rr]) * scale
                ax.plot(x, y, "o", label=contrast.replace("_minus_full", ""), ms=4)
                ax.vlines(
                    x,
                    [r["evaluation_mc_questions_blocks_lo"] * scale for r in rr],
                    [r["evaluation_mc_questions_blocks_hi"] * scale for r in rr],
                    lw=1,
                )
            ax.axhline(0, color="gray", ls=":")
            ax.set_xticks(range(4), STYLES, fontsize=8)
            ax.set_title(f"Seed {seed}: {endpoint}", fontsize=10)
            ax.set_ylabel(
                "Change from full update" + (" in tokens" if endpoint == "tokens" else " in pp")
            )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Selective removal versus two controls\nPaired evaluation, questions and content blocks; all edits including zero contributions"
    )
    fig.tight_layout()
    fig.savefig(out / "selective-removal.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / "selective-removal.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    for col, seed in enumerate((42, 43)):
        for ei, endpoint in enumerate(("numbered", "bullets", "answer_values_and_order", "tokens")):
            ax = axes[ei, col]
            scale = 1 if endpoint == "tokens" else 100
            for label in ("numbered", "bullets"):
                rr = sorted(
                    (
                        r
                        for r in summary["singles"]
                        if r["seed"] == seed
                        and r["label"] == label
                        and r["group"] == "all"
                        and r["endpoint"] == endpoint
                    ),
                    key=lambda r: r["strength"],
                )
                x = [r["strength"] for r in rr]
                y = [r["effect"] * scale for r in rr]
                ax.plot(x, y, "o-", label=label)
                ax.vlines(
                    x,
                    [r["evaluation_mc_questions_blocks_lo"] * scale for r in rr],
                    [r["evaluation_mc_questions_blocks_hi"] * scale for r in rr],
                    lw=1,
                )
            ax.axhline(0, color="gray", ls=":")
            ax.set_title(f"Seed {seed}: {endpoint}", fontsize=10)
            ax.set_ylabel(
                "Change from baseline" + (" in tokens" if endpoint == "tokens" else " in pp")
            )
            ax.set_xlabel("Signed strength × median batch-gradient norm")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Single-atom interventions: every fixed sign and strength\nRate .003; pointwise paired evaluation and question intervals"
    )
    fig.tight_layout()
    fig.savefig(out / "single-atoms.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / "single-atoms.pdf", bbox_inches="tight")
    plt.close(fig)


def table(rows, fields):
    def fmt(v):
        if v is None:
            return "undefined"
        if isinstance(v, float):
            return f"{v:.4g}"
        return html.escape(str(v))

    return (
        '<div class="scroll"><table><tr>'
        + "".join(f"<th>{html.escape(f.replace('_', ' '))}</th>" for f in fields)
        + "</tr>"
        + "".join(
            "<tr>" + "".join(f"<td>{fmt(r.get(f))}</td>" for f in fields) + "</tr>" for r in rows
        )
        + "</table></div>"
    )


def report(out, s):
    parts = [
        """<!doctype html><html><head><meta charset="utf-8"><title>Atom-centered batch explanations</title><style>body{max-width:1200px;margin:auto;padding:35px;font:16px/1.5 system-ui;color:#182d40;background:#fafbfc}table{border-collapse:collapse;font-size:12px;margin:18px 0}td,th{padding:8px;border-bottom:1px solid #cbd4dc;text-align:left}th{background:#e5edf4}.scroll{overflow:auto}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf1f5;padding:16px}details{margin:15px 0}summary{cursor:pointer;font-weight:600}h1,h2{line-height:1.2}.note{border-left:4px solid #b1542c;padding-left:18px}</style></head><body><h1>Atom-centered batch explanations</h1><p class="note">The goal is a small set of meaningfully described atoms that explains and approximately predicts batch-update effects. Beating full gradients or SVD is not required. Description validity, causal selectivity and forecast accuracy are separate questions. This is a development experiment with two reused Gemma-3 4B rank-8 query/value LoRA adapters, 136 tensors, four simple task families and four batch-content blocks.</p>
<p>All candidate descriptions, examples, methods, rates and interventions were fixed before new results. Each seed completed 48 matched semantic gradients, 3,072 proposal responses and 74,752 independent evaluation responses across 72 updates plus baseline. All forecasts were saved and locked before updates. The atom dictionary was learned from normalized EKFAC-projected gradients; batch codes restore the projected norm. It was not learned from raw magnitude-bearing gradients.</p>
<h2>Fixed descriptions</h2><p>The two candidate labels are numbered lists and bulleted lists, selected earlier from discovery-training examples. They were hypotheses, not established meanings. The assay compares sparse codes and physical gradient cosines for the named format against plain text and the other list format on eight separate items, each in short and long versions with matching requested content. Positive differences favor the description. All items, tasks, lengths and 500-dimensional sparse codes are retained in the semantic assay files. No new atom is promoted from this assay.</p><img src="descriptions.png" alt="Matched atom description tests">"""
    ]
    interpretation = out / "interpretation.json"
    if interpretation.exists():
        finding = read(interpretation)
        interpretation_html = "<h2>Interpretation of the complete study</h2>" + "".join(
            f"<p><strong>{html.escape(label)}:</strong> {html.escape(finding[key])}</p>"
            for label, key in (
                ("Descriptions", "description_conclusion"),
                ("Selective interventions", "causal_conclusion"),
                ("Primary forecasts", "forecast_conclusion"),
                ("Secondary evidence", "secondary_conclusion"),
                ("Next development", "next_development"),
            )
        )
        parts[0] = parts[0].replace(
            "<h2>Fixed descriptions</h2>", interpretation_html + "<h2>Fixed descriptions</h2>"
        )
    parts.append(
        table(
            [r for r in s["semantic_contrasts"] if r["task"] == "all"],
            (
                "seed",
                "label",
                "long",
                "metric",
                "comparison",
                "mean_preference",
                "positive_pairs",
                "zero_pairs",
                "pairs",
                "item_bootstrap_lo",
                "item_bootstrap_hi",
            ),
        )
    )
    parts.append(
        '<h2>Selective interventions and their costs</h2><p>Every batch is evaluated after its full update, after removing its named atom contribution, after removing an equal-norm contribution along a fixed random nonnamed atom, and after rescaling the full update to match the remaining named-removal norm. The target is bullets for bulleted batches and numbered for the other styles. Zero contributions remain zero edits. Removing an atom is tested against both controls, with content, length, truncation, emptiness and possible degeneration retained. The random comparison is one fixed draw per batch, not an average over random atoms.</p><img src="selective-removal.png" alt="Selective removal controls">'
    )
    zero = []
    for seed in (42, 43):
        for style in STYLES:
            rr = [
                r
                for r in s["conditions"]
                if r["seed"] == seed
                and r["kind"] == "batch"
                and r["mode"] == "full"
                and STYLES[r["batch_index"] % 4] == style
            ]
            zero.append(
                dict(
                    seed=seed,
                    style=style,
                    batches=len(rr),
                    zero_named_contributions=sum(r["named_coefficient"] == 0 for r in rr),
                    named_coefficients=[r["named_coefficient"] for r in rr],
                )
            )
    parts.append(
        table(zero, ("seed", "style", "batches", "zero_named_contributions", "named_coefficients"))
    )
    parts.append(
        '<p>All signed single-atom strengths are shown, with no selected best strength. Positive strength follows the discovery-training descent orientation. A format shift accompanied by worse content or altered length is not sufficient evidence of a selective formatting concept.</p><img src="single-atoms.png" alt="All signed single atom effects">'
    )
    parts.append(
        "<h2>Prospective batch forecasts</h2><p>Before updates, baseline sampled responses estimate every atom's behavioral effect through the standard score-function covariance. The primary forecast sums batch coefficients times mean atom effects on eight separate calibration questions and transfers that prediction to 16 evaluation questions with different templates. Same-question estimates are secondary. Dictionary, named-only, shuffled atom-profile, full-gradient, SVD16 and zero forecasts are all retained. Rates stay fixed at .003; no regression, clipping or outcome fitting is used.</p><p>One unit of relative RMSE means the same error as predicting no change. Lower is better. A zero observed effect makes relative error undefined. Binary-endpoint absolute errors and changes are fractions; multiply by 100 for percentage points. Tokens and words use their own units.</p>"
    )
    for scope in SCOPES:
        parts.append(
            f'<h3>{scope.replace("_", " ")}</h3><img src="forecast-{scope}.png" alt="{scope} forecast errors">'
        )
        rr = [r for r in s["metrics"] if r["scope"] == scope and r["group"] == "all"]
        parts.append(
            "<details><summary>All pooled methods and endpoints with uncertainty</summary>"
            + table(
                rr,
                (
                    "seed",
                    "method",
                    "endpoint",
                    "rmse",
                    "nrmse",
                    "actual_mean",
                    "predicted_mean",
                    "worst_batch_rmse",
                    "bias",
                    "outside_probability_range_fraction",
                    "both_mc_questions_blocks_nrmse_lo",
                    "both_mc_questions_blocks_nrmse_hi",
                    "both_mc_questions_blocks_nrmse_defined_replicates",
                    "both_mc_questions_blocks_mse_difference_vs_zero_lo",
                    "both_mc_questions_blocks_mse_difference_vs_zero_hi",
                ),
            )
            + "</details>"
        )
    parts.append(
        "<h2>Decomposition, coverage and uncertainty</h2><p>The downloadable atom-decomposition arrays retain all 500 coefficients, question-specific effect profiles, individual contributions, named-only forecasts and shuffled permutation. Named absolute share measures absolute contribution mass; named net share can exceed one because other contributions cancel. Undefined zero-denominator shares remain undefined. These are numerical decompositions and do not supply semantic descriptions for unnamed atoms. Missing proposal events yield zero derivatives and remain forecast failures. Linear predictions outside the probability range are not clipped.</p><p>There are 1,000 paired bootstrap replicates, fixed seed 2026101407. Proposal samples are resampled independently from evaluation samples, with common draws across conditions inside each question. Calibration and evaluation questions are resampled separately within each of four task families. Four content blocks are drawn jointly across all styles. Forecast tables separate proposal-only, evaluation-only, both-stream Monte Carlo variation, and both streams plus questions and blocks. Intervention intervals use paired evaluation samples, with and without question/block variation. Semantic intervals resample two matched items within each task while preserving format and length pairings. With only four blocks, four task families and two reused adapters, these are descriptive pointwise intervals, not broad population or multiplicity-adjusted significance claims. Empirical bootstrap intervals cannot recover missing events. Undefined ratios are counted without dropping replicates.</p>"
    )
    parts.append(
        "<details><summary>Every batch's named-atom contribution shares</summary>"
        + table(
            s["shares"],
            (
                "seed",
                "scope",
                "batch",
                "style",
                "endpoint",
                "active_atoms",
                "named_absolute_share",
                "named_net_share",
                "named_net_mean",
                "dictionary_net_mean",
                "zero_named_target",
            ),
        )
        + "</details>"
    )
    parts.append(
        "<details><summary>All proposal-event counts by question and split</summary>"
        + table(
            s["coverage"],
            ("seed", "question", "task", "split", "endpoint", "nonzero_samples", "baseline"),
        )
        + "</details>"
    )
    parts.append(
        "<h2>Audit and computation</h2><p>Both original source snapshots were archived before any checkout changes. Source, input, cache, forecast and output hashes, the entire condition grid, response counts, pre-update locks and all adapter restores passed. The first norm auditor used a two-parts-per-million comparison of float32 reductions and failed on H100. Its code and log are preserved. The corrected auditor reproduces the frozen scaling formula exactly and reports higher-precision physical-norm residuals, at most 3.911 parts per million. No worker code, protocol, trajectories or updates changed.</p><p>Computation includes a full 500-direction bank and full gradients. Unique proposal gradient calls deduplicate exact token sequences within question while preserving sample multiplicities. Generation also requires token-decoding forwards; those forward counts were not instrumented in this study and are not inferred from gradient-call counts. No efficiency or compression claim is made.</p>"
    )
    parts.append(
        table(
            s["provenance"],
            (
                "seed",
                "elapsed_seconds",
                "forecast_stage_seconds",
                "evaluation_stage_seconds",
                "peak_memory_bytes",
                "proposal_gradient_calls",
                "training_gradient_calls",
                "semantic_gradient_calls",
                "proposal_responses",
                "evaluation_responses",
                "control_norm_residual_max",
            ),
        )
    )
    names = (
        "metrics",
        "edits",
        "singles",
        "pairs",
        "condition_forecasts",
        "generation",
        "shares",
        "coverage",
        "semantic_contrasts",
        "conditions",
    )
    parts.append(
        "<h2>Complete results</h2><p>"
        + " · ".join(f'<a href="{n}.csv">{n.replace("_", " ")}</a>' for n in names)
        + ' · <a href="analysis-plan.json">Pre-reading analysis plan</a> · <a href="summary.json">Summary JSON</a> · <a href="seed42-atom-decomposition.npz">Seed 42 decomposition</a> · <a href="seed43-atom-decomposition.npz">Seed 43 decomposition</a></p>'
    )
    parts.append(
        "<h2>Fixed examples</h2><p>The first evaluation question per task, sample zero: baseline, first block with every style and edit, and all eight single-atom conditions. All examples are retained. Content checks require the specified vocabulary values and their order, plus lowercase on that task; they are not unrestricted semantic correctness.</p>"
    )
    for r in s["examples"]:
        parts.append(
            f"<details><summary>Seed {r['seed']} | {html.escape(r['condition'])} | {html.escape(r['task'])}</summary><p>{html.escape(r['prompt'])}</p><p>Expected values: {html.escape(str(r['expected']))}</p><pre>{html.escape(r['response'])}</pre><pre>{html.escape(json.dumps(dict(checks=r['checks'], quality=r['quality'])))}</pre></details>"
        )
    (out / "report.html").write_text("\n".join(parts) + "</body></html>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        type=Path,
        nargs=2,
        default=[Path(f"runs/paper-atom-batches-seed{s}") for s in (42, 43)],
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("runs/paper-atom-batches-analysis/analysis-plan.json")
    )
    parser.add_argument("--out", type=Path, default=Path("results/paper-atom-batches"))
    args = parser.parse_args()
    guard(args.roots, args.plan)
    stamp = args.plan.parent / "first-analysis-reading.json"
    if not stamp.exists():
        atomic_json(
            stamp,
            dict(
                written_unix=time.time(),
                analysis_plan_sha256=file_hash(args.plan),
                analysis_source_sha256=file_hash(Path(__file__)),
                both_completed_audits_verified=True,
            ),
        )
    assert read(args.plan)["written_unix"] < read(stamp)["written_unix"]
    assert read(stamp)["analysis_plan_sha256"] == file_hash(args.plan)
    args.out.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(args.plan, args.out / "analysis-plan.json")
    results = []
    for root in args.roots:
        results.append(analyze(root, args.out, args.plan.parent))
    keys = (
        "metrics",
        "edits",
        "singles",
        "pairs",
        "condition_forecasts",
        "generation",
        "examples",
        "shares",
        "coverage",
        "conditions",
    )
    summary = {k: [r for result in results for r in result[k]] for k in keys}
    summary["provenance"] = [r["provenance"] for r in results]
    summary["semantic_contrasts"] = [
        r for result in results for r in result["semantic"]["contrasts"]
    ]
    for seed, result in zip((42, 43), results, strict=True):
        atomic_json(args.out / f"seed{seed}-semantic-assay.json", result["semantic"])
    for key in (*[k for k in keys if k != "examples"], "semantic_contrasts"):
        write_csv(args.out / f"{key}.csv", summary[key])
    atomic_json(args.out / "summary.json", {k: v for k, v in summary.items() if k != "pairs"})
    plots(args.out, summary)
    report(args.out, summary)
    print(
        json.dumps(
            dict(
                report=str(args.out / "report.html"),
                forecast_rows=len(summary["metrics"]),
                complete=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
