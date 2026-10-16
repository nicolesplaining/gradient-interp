"""Predeclared paired tests for untouched single-update interpreter evaluations."""

import itertools

import numpy as np

from .update_loracle_data import LABELS, NO_CHANGE


def checked_vectors(cases, targets, predictions):
    if len(cases) != len(targets) or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Incomplete or duplicated held-out cases")
    if any(t not in LABELS for t in targets):
        raise ValueError("Unexpected measured effect")
    needed = {"adapted", "released", "adapted_shuffled", "majority", "ridge"}
    if set(predictions) != needed:
        raise ValueError("Every prespecified comparator is required")
    if any(len(p) != len(cases) or any(v not in LABELS for v in p) for p in predictions.values()):
        raise ValueError("Incomplete prediction grid")
    active = np.asarray([not c["control"] for c in cases])
    if active.sum() != 128 or (~active).sum() != 8:
        raise ValueError("Require the full128-update/eight-control frozen test grid")
    if any(targets[i] != NO_CHANGE for i in np.flatnonzero(~active)):
        raise ValueError("A zero control has a measured change")
    return active, {
        name: np.asarray([p == t for p, t in zip(values, targets)], dtype=float)
        for name, values in predictions.items()
    }


def crossed_bootstrap(cases, values, replicates=4000, seed=2026101665):
    """Shared composition/initialization/case resamples for every method and contrast."""
    if replicates < 1000:
        raise ValueError("At least 1000 replicates required for final intervals")
    active = [i for i, c in enumerate(cases) if not c["control"]]
    groups = sorted({cases[i]["source_pattern"] for i in active})
    seeds = sorted({cases[i]["initialization_seed"] for i in active})
    cells = {
        (g, s): [
            i
            for i in active
            if cases[i]["source_pattern"] == g and cases[i]["initialization_seed"] == s
        ]
        for g in groups
        for s in seeds
    }
    sizes = {len(cell) for cell in cells.values()}
    if len(groups) != 8 or len(seeds) != 4 or len(sizes) != 1 or 0 in sizes:
        raise ValueError("Expected complete balanced eight-composition/four-seed design")
    if any(len(v) != len(cases) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Invalid scored outcomes")
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(replicates) for name in values}
    for b in range(replicates):
        indices = []
        drawn_groups = rng.choice(groups, len(groups))
        drawn_seeds = rng.choice(seeds, len(seeds))
        for group in drawn_groups:
            for initial in drawn_seeds:
                cell = cells[group, initial]
                indices.extend(rng.choice(cell, len(cell)).tolist())
        for name, vector in values.items():
            samples[name][b] = float(np.asarray(vector)[indices].mean())
    return samples


def exact_cluster_method_swap(cases, adapted_correct, released_correct):
    """One-sided paired method-swap reference distribution, by all eight task compositions."""
    active = [i for i, c in enumerate(cases) if not c["control"]]
    groups = sorted({cases[i]["source_pattern"] for i in active})
    if len(groups) != 8:
        raise ValueError("Exact test requires all eight prespecified composition clusters")
    first, second = np.asarray(adapted_correct), np.asarray(released_correct)
    if first.shape != (len(cases),) or second.shape != first.shape:
        raise ValueError("Invalid paired predictions")
    if not np.isin(first, [0, 1]).all() or not np.isin(second, [0, 1]).all():
        raise ValueError("Expected binary correctness, with no missing cases")
    # The numerator sums every case, preserving the exact benchmark weighting.
    totals = np.asarray(
        [
            sum(first[i] - second[i] for i in active if cases[i]["source_pattern"] == group)
            for group in groups
        ]
    )
    observed = float(totals.sum() / len(active))
    null = np.asarray(
        [np.dot(signs, totals) / len(active) for signs in itertools.product([-1.0, 1.0], repeat=8)]
    )
    return {
        "observed_advantage": observed,
        "one_sided_p": float(np.mean(null >= observed - 1e-12)),
        "assignments": len(null),
        "cluster_count": 8,
        "minimum_possible_p": 1 / 256,
        "assumption": "Method differences can have their signs exchanged by composition under "
        "the null; this is not a 128-independent-trial binomial test.",
    }


def evaluate(cases, targets, predictions, spans, replicates=4000):
    active, correct = checked_vectors(cases, targets, predictions)
    spans = np.asarray(spans)
    if spans.shape != (len(cases),) or not np.isfinite(spans).all() or (spans < 0).any():
        raise ValueError("Every case needs a valid measured effect span")
    if any((target == NO_CHANGE) != (span < .02) for target, span in zip(targets, spans)):
        raise ValueError("No-change targets and frozen effect floor disagree")
    # Weak actual updates remain in the accuracy denominator, including no-change targets.
    accuracy = {name: float(values[active].mean()) for name, values in correct.items()}
    controls = {name: float(values[~active].mean()) for name, values in correct.items()}
    samples = crossed_bootstrap(cases, correct, replicates)
    interval = np.quantile(samples["adapted"], [0.0125, 0.9875]).tolist()
    advantage_draws = samples["adapted"] - samples["released"]
    advantage_interval = np.quantile(advantage_draws, [0.0125, 0.9875]).tolist()
    advantage = accuracy["adapted"] - accuracy["released"]
    coverage = float((spans[active] >= 0.02).mean())
    gates = {
        "accuracy_at_least_70_percent": accuracy["adapted"] >= 0.7,
        "released_advantage_at_least_20pp": advantage >= 0.2,
        "simultaneous_accuracy_lower_bound_above_60_percent": interval[0] > 0.6,
        "simultaneous_advantage_lower_bound_above_10pp": advantage_interval[0] > 0.1,
        "measurable_effects_at_least_80_percent": coverage >= 0.8,
        "zero_control_accuracy_at_least_90_percent": controls["adapted"] >= 0.9,
        "majority_advantage_at_least_10pp": accuracy["adapted"] - accuracy["majority"] >= 0.1,
        "shuffled_direction_advantage_at_least_10pp": (
            accuracy["adapted"] - accuracy["adapted_shuffled"] >= 0.1
        ),
    }
    result = {
        "n_actual_updates": int(active.sum()),
        "n_controls": int((~active).sum()),
        "accuracy": accuracy,
        "zero_control_accuracy": controls,
        "measurable_effect_fraction": coverage,
        "primary_accuracy_interval": interval,
        "primary_advantage_interval": advantage_interval,
        "interval_definition": "Two 97.5% marginal two-sided percentile intervals, "
        "using1.25/98.75 percentiles; Bonferroni target95% simultaneous coverage.",
        "paired_exact_reference": exact_cluster_method_swap(
            cases, correct["adapted"], correct["released"]
        ),
        "gates": gates,
        "all_numerical_gates_pass": all(gates.values()),
        "replicates": replicates,
        "replicate_seed": 2026101665,
        "completion_warning": "Passing these numeric checks alone does not prove completion. "
        "Separate audits must establish untouched data, a model-selection lock before test access, "
        "complete outputs and the stated generalization scope.",
    }
    return result, samples
