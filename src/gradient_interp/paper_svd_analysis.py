"""Guarded fixed-plan comparison of SVD and dictionary interpretation/steering."""

# ruff: noqa: E501

import argparse
import csv
import html
import json
import time
from pathlib import Path

import numpy as np

from .paper_steering import FAMILIES
from .paper_svd_replication import METHODS, STRENGTHS
from .study import atomic_json, file_hash

METRICS = (
    *FAMILIES,
    "controlled_content_check",
    "required_content_present",
    "tokens",
    "words",
    "truncated",
    "empty",
    "possible_degeneration",
)
BOOTSTRAP_SEED = 2026101421
FROZEN_COMMIT = "e12ecc1989531ce48b1352e2f76985c6a7cf1a23"


def read(path):
    return json.loads(Path(path).read_text())


def guard(roots, analysis_plan, barrier_path):
    """Check the two-run barrier and complete audits before opening compact outcomes."""
    plan, barrier = read(analysis_plan), read(barrier_path)
    if (
        plan["scientific_outcomes_read"]
        or plan["source_commit"] != FROZEN_COMMIT
        or barrier["type"] != "paper_svd_pair_barrier"
        or barrier["analysis_plan_sha256"] != file_hash(analysis_plan)
    ):
        raise ValueError("A frozen pre-reading plan and pair barrier are required")
    receipts = barrier["receipts"]
    if len(receipts) != 2 or {r["seed"] for r in receipts} != {42, 43} or len(roots) != 2:
        raise ValueError("Both seeds are required")
    if not plan["written_unix"] < barrier["created_unix"] <= time.time():
        raise ValueError("Analysis plan must precede output release")
    for receipt in receipts:
        if (
            receipt["type"] != "paper_svd_completion_receipt"
            or receipt["status"] != "complete"
            or receipt["release_reason"] != "complete"
            or not receipt["all_processes_dead"]
        ):
            raise ValueError("Complete dead-worker receipts are required for full analysis")
        if abs(receipt["deadline_unix"] - receipt["started_unix"] - 43200) > 0.01:
            raise ValueError("Invalid original phase duration")
        if (
            not receipt["started_unix"]
            <= receipt["completed_unix"]
            <= receipt["deadline_unix"]
            <= receipt["started_unix"] + 43200.01
        ):
            raise ValueError("Invalid original deadline")
        if (
            not receipt["completed_unix"] <= receipt["observed_unix"] <= barrier["created_unix"]
            or plan["written_unix"] >= receipt["observed_unix"]
        ):
            raise ValueError("Invalid completion receipt time")
    seen = set()
    for root in roots:
        audit = read(root / "integrity.json")
        required = (
            "complete",
            "source_input_output_hashes_verified",
            "selection_and_orientations_verified",
            "physical_norm_matching_verified",
            "descriptor_coefficients_recomputed",
            "all_saved_generation_metrics_recomputed",
            "original_deadline_verified",
            "all_adapter_restores_verified",
        )
        if not all(audit.get(key) is True for key in required):
            raise ValueError("A complete scientific integrity audit is required")
        seed = audit["seed"]
        seen.add(seed)
        receipt = next(r for r in receipts if r["seed"] == seed)
        if audit["barrier_sha256"] != file_hash(barrier_path) or audit[
            "analysis_plan_sha256"
        ] != file_hash(analysis_plan):
            raise ValueError("Audit and barrier do not match")
        if audit["counts"] != {"conditions": 101, "responses": 14540, "descriptor_gradients": 160}:
            raise ValueError("Incomplete scientific grid")
        expected_metadata = {
            "plan",
            "specification",
            "controller",
            "manifest",
            "selection-lock",
            "descriptor-manifest",
            "selection",
            "candidates",
            "data",
            "fit",
            "preflight",
            "data-overlap-audit",
        }
        if set(audit["metadata_hashes"]) != expected_metadata or set(
            audit["analysis_export_hashes"]
        ) != {"analysis-export.npz", "fixed-examples.json", "analysis-candidates.json"}:
            raise ValueError("Audited metadata or export grid incomplete")
        for name, digest in audit["metadata_hashes"].items():
            if file_hash(root / f"{name}.json") != digest:
                raise ValueError("Audited metadata changed")
        for name in ("controller", "plan", "specification", "manifest"):
            if audit["metadata_hashes"][name] != receipt[name + "_sha256"]:
                raise ValueError("Completion receipt is stale")
        for name, digest in audit["analysis_export_hashes"].items():
            if file_hash(root / name) != digest:
                raise ValueError("Audited compact outcome changed")
    if seen != {42, 43}:
        raise ValueError("Duplicate seed roots")


def interval(values, family_size=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or family_size < 1:
        raise ValueError("Expected one estimand and positive family size")
    undefined = int((~np.isfinite(values)).sum())
    bounds = (
        None
        if undefined
        else np.quantile(values, [0.025 / family_size, 1 - 0.025 / family_size]).tolist()
    )
    return {
        "lo": None if bounds is None else bounds[0],
        "hi": None if bounds is None else bounds[1],
        "undefined_replicates": undefined,
        "coverage": 1 - 0.05 / family_size,
    }


def paired_weights(replicates=1000, seed=BOOTSTRAP_SEED):
    """Shared weights preserve method/sign/strength and descriptor pair structure."""
    rng = np.random.default_rng(seed)
    published = np.empty((replicates, 5, 100))
    for fi in range(5):
        for start, n in ((0, 60), (60, 40)):
            published[:, fi, start : start + n] = (
                rng.multinomial(n, np.full(n, 1 / n), size=replicates) / 100
            )
    items = rng.multinomial(8, np.full(8, 1 / 8), size=(replicates, 5)) / 8
    return published, items


def signed_response(values):
    """M,F,A,S,Q,E -> M,F,Q,E; retain every fixed strength equally."""
    values = np.asarray(values)
    if values.ndim != 6 or values.shape[2:4] != (5, 2):
        raise ValueError("Expected five strengths and both signs")
    return (values[:, :, :, 0] - values[:, :, :, 1]).mean(axis=2) / 2


def controlled_statistics(controlled, baseline, weights):
    """Condition-level positive-part harm cannot be cancelled by improvements."""
    c = np.asarray(controlled)
    w = np.asarray(weights)
    if (
        c.shape != (2, 5, 5, 2, 40, len(METRICS))
        or baseline.shape != (40, len(METRICS))
        or w.ndim != 2
        or w.shape[1] != 40
    ):
        raise ValueError("Controlled grid shape differs")
    means = np.einsum("rq,mfasqe->rmfase", w, c, optimize=True)
    base = w @ baseline
    delta = means - base[:, None, None, None, None]
    indices = [
        METRICS.index(k) for k in ("controlled_content_check", "truncated", "possible_degeneration")
    ]
    costs = delta[..., indices] * np.array([-1, 1, 1])
    harms = np.maximum(costs, 0)
    signed = np.einsum("rq,mfqe->rmfe", w, signed_response(c), optimize=True)
    token_cost = np.abs(delta[..., METRICS.index("tokens")])
    return {
        "signed": signed,
        "mean_harm": harms.mean(axis=(3, 4)),
        "worst_harm": harms.max(axis=(3, 4)),
        "mean_absolute_length_change": token_cost.mean(axis=(3, 4)),
        "worst_absolute_length_change": token_cost.max(axis=(3, 4)),
    }


def descriptor_differences(data, codes, cosines):
    if codes.shape != (160, 10) or cosines.shape != (160, 10):
        raise ValueError("Descriptor shapes differ")
    values = np.full((2, 5, 5, 2, 8, 2, 2), np.nan)
    rows = []
    for index, row in enumerate(data["descriptor_pairs"]):
        fs, li, ii, label = (
            FAMILIES.index(row["family"]),
            (0 if row["verbosity"] == "short" else 1),
            int(row["item_id"].split("-")[-1]),
            (0 if row["label"] == "target" else 1),
        )
        for mi, method in enumerate(METHODS):
            for fi, family in enumerate(FAMILIES):
                values[mi, fi, fs, li, ii, label] = [
                    codes[index, mi * 5 + fi],
                    cosines[index, mi * 5 + fi],
                ]
                rows.append(
                    {
                        "method": method,
                        "selected_family": family,
                        "stimulus_family": row["family"],
                        "item": ii,
                        "verbosity": row["verbosity"],
                        "label": row["label"],
                        "id": row["id"],
                        "oriented_coefficient": float(codes[index, mi * 5 + fi]),
                        "physical_cosine": float(cosines[index, mi * 5 + fi]),
                    }
                )
    if not np.isfinite(values).all():
        raise ValueError("Descriptor grid incomplete or nonfinite")
    return values[..., 0, :] - values[..., 1, :], rows


def bootstrap(published, controlled, baseline_controlled, descriptor_diff, replicates=1000):
    pw, iw = paired_weights(replicates)
    pub_signed = signed_response(published)[..., :5]
    target = np.stack([pub_signed[:, fi, :, fi] for fi in range(5)], axis=1)
    family = np.einsum("rfq,mfq->rmf", pw, target, optimize=True)
    aggregate = family.mean(axis=2)
    control = controlled_statistics(controlled, baseline_controlled, iw.reshape(replicates, 40) / 5)
    descriptor = np.einsum("rsi,mfslik->rmfslk", iw, descriptor_diff, optimize=True)
    return {
        "family": family,
        "aggregate": aggregate,
        "family_advantage": family[:, 1] - family[:, 0],
        "aggregate_advantage": aggregate[:, 1] - aggregate[:, 0],
        "descriptor_mean": descriptor,
        **{"controlled_" + key: value for key, value in control.items()},
    }


def development_gates(family, aggregate, boots, descriptor_diff, controlled):
    descriptor_pass = np.empty((2, 5), dtype=bool)
    selectivity = np.empty((2, 5), dtype=bool)
    for mi in range(2):
        for fi in range(5):
            d = descriptor_diff[mi, fi, fi]
            descriptor_pass[mi, fi] = bool(
                (d.mean(axis=1) > 0).all() and ((d > 0).sum(axis=1) >= 6).all()
            )
            signed = controlled["signed"][0, mi, fi, :5]
            selectivity[mi, fi] = abs(signed[fi]) > np.max(np.abs(np.delete(signed, fi)))
    quality = (controlled["mean_harm"][0] <= 0.05).all(axis=-1) & (
        controlled["worst_harm"][0] <= 0.15
    ).all(axis=-1)
    svd_interval, advantage_interval = (
        interval(boots["aggregate"][:, 1]),
        interval(boots["aggregate_advantage"]),
    )
    numerical = (
        svd_interval["lo"] is not None
        and svd_interval["lo"] >= 0.10
        and advantage_interval["lo"] >= 0.05
    )
    return {
        "aggregate_svd_response": float(aggregate[1]),
        "aggregate_svd_advantage": float(aggregate[1] - aggregate[0]),
        "aggregate_svd_interval": svd_interval,
        "aggregate_advantage_interval": advantage_interval,
        "numerical_thresholds_pass": bool(numerical),
        "descriptor_pass": descriptor_pass.tolist(),
        "controlled_selectivity_pass": selectivity.tolist(),
        "controlled_quality_pass": quality.tolist(),
        "all_svd_requirements_pass": bool(
            numerical and descriptor_pass[1].all() and selectivity[1].all() and quality[1].all()
        ),
        "scope": "Development gates; both reused seeds must pass. No untouched confirmation claim.",
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("Do not silently write empty scientific tables")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, work, replicates):
    data, spec, fit = [read(root / f"{name}.json") for name in ("data", "specification", "fit")]
    seed = spec["training_seed"]
    export = np.load(root / "analysis-export.npz")
    if export["metrics"].tolist() != list(METRICS):
        raise ValueError("Metric order changed")
    pub, ctrl, bp, bc = [
        export[k] for k in ("published", "controlled", "baseline_published", "baseline_controlled")
    ]
    if pub.shape != (2, 5, 5, 2, 100, len(METRICS)) or bp.shape != (5, 100, len(METRICS)):
        raise ValueError("Published grid incomplete")
    applicable = [
        i
        for i, name in enumerate(METRICS)
        if name not in {"controlled_content_check", "required_content_present"}
    ]
    if not all(np.isfinite(x).all() for x in (pub[..., applicable], bp[..., applicable], ctrl, bc)):
        raise ValueError("Applicable scientific endpoints must be finite")
    diff, desc_rows = descriptor_differences(
        data, export["selected_oriented_coefficients"], export["selected_physical_cosines"]
    )
    boots = bootstrap(pub, ctrl, bc, diff, replicates)
    np.savez_compressed(work / f"seed{seed}-bootstrap.npz", **boots)
    signed = signed_response(pub)
    family = np.stack([signed[:, fi, :, fi].mean(axis=-1) for fi in range(5)], axis=1)
    aggregate = family.mean(axis=1)
    controlled = controlled_statistics(ctrl, bc, np.full((1, 40), 1 / 40))
    primary = []
    for mi, method in enumerate(METHODS):
        primary.append(
            {
                "seed": seed,
                "method": method,
                "scope": "all_families",
                "signed_response": float(aggregate[mi]),
                **interval(boots["aggregate"][:, mi]),
            }
        )
        for fi, name in enumerate(FAMILIES):
            primary.append(
                {
                    "seed": seed,
                    "method": method,
                    "scope": name,
                    "signed_response": float(family[mi, fi]),
                    **interval(boots["family"][:, mi, fi], 5),
                }
            )
    primary.append(
        {
            "seed": seed,
            "method": "svd_minus_dictionary",
            "scope": "all_families",
            "signed_response": float(aggregate[1] - aggregate[0]),
            **interval(boots["aggregate_advantage"]),
        }
    )
    for fi, name in enumerate(FAMILIES):
        primary.append(
            {
                "seed": seed,
                "method": "svd_minus_dictionary",
                "scope": name,
                "signed_response": float(family[1, fi] - family[0, fi]),
                **interval(boots["family_advantage"][:, fi], 5),
            }
        )
    conditions, signed_rows, quality_rows, descriptor_rows, length_rows = [], [], [], [], []
    for mi, method in enumerate(METHODS):
        for fi, family_name in enumerate(FAMILIES):
            pub_strata = {
                "all": np.arange(100),
                "inviting": np.arange(60),
                "control": np.arange(60, 100),
            }
            ctrl_strata = {
                "all": np.arange(40),
                **{f: np.arange(i * 8, (i + 1) * 8) for i, f in enumerate(FAMILIES)},
            }
            for population, values, baseline, strata in (
                ("published", pub, bp[fi], pub_strata),
                ("controlled", ctrl, bc, ctrl_strata),
            ):
                for stratum, indices in strata.items():
                    for ei, endpoint in enumerate(METRICS):
                        if population == "published" and endpoint in {
                            "controlled_content_check",
                            "required_content_present",
                        }:
                            continue
                        base = float(baseline[indices, ei].mean())
                        sr = float(signed_response(values)[mi, fi, indices, ei].mean())
                        signed_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "direction_family": family_name,
                                "population": population,
                                "stratum": stratum,
                                "endpoint": endpoint,
                                "signed_response": sr,
                                "baseline": base,
                            }
                        )
                        for ai, strength in enumerate(STRENGTHS):
                            for si, sign in enumerate((1, -1)):
                                mean = float(values[mi, fi, ai, si, indices, ei].mean())
                                conditions.append(
                                    {
                                        "seed": seed,
                                        "method": method,
                                        "direction_family": family_name,
                                        "population": population,
                                        "stratum": stratum,
                                        "endpoint": endpoint,
                                        "strength": strength,
                                        "sign": sign,
                                        "baseline": base,
                                        "actual": mean,
                                        "change": mean - base,
                                    }
                                )
            for ki, endpoint in enumerate(
                ("content_loss", "extra_truncation", "extra_degeneration")
            ):
                for statistic in ("mean_harm", "worst_harm"):
                    quality_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "direction_family": family_name,
                            "endpoint": endpoint,
                            "statistic": statistic,
                            "value": float(controlled[statistic][0, mi, fi, ki]),
                            **interval(boots["controlled_" + statistic][:, mi, fi, ki]),
                        }
                    )
            for statistic in ("mean_absolute_length_change", "worst_absolute_length_change"):
                quality_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "direction_family": family_name,
                        "endpoint": "tokens",
                        "statistic": statistic,
                        "value": float(controlled[statistic][0, mi, fi]),
                        **interval(boots["controlled_" + statistic][:, mi, fi]),
                    }
                )
            # Fixed baseline quartile cut points; no adjustment for post-update length.
            for population, values, baseline in (
                ("published", pub, bp[fi]),
                ("controlled", ctrl, bc),
            ):
                lengths = baseline[:, METRICS.index("tokens")]
                cuts = np.quantile(lengths, [0.25, 0.5, 0.75])
                labels = np.searchsorted(cuts, lengths, side="right")
                for quartile in range(4):
                    indices = np.flatnonzero(labels == quartile)
                    length_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "direction_family": family_name,
                            "population": population,
                            "baseline_length_quartile": quartile,
                            "questions": len(indices),
                            "signed_target_response": float(
                                signed_response(values)[mi, fi, indices, fi].mean()
                            )
                            if len(indices)
                            else None,
                            "cutpoints": cuts.tolist(),
                        }
                    )
            for fs, stimulus in enumerate(FAMILIES):
                for li, verbosity in enumerate(("short", "long")):
                    for ki, metric in enumerate(("oriented_coefficient", "physical_cosine")):
                        pairs = diff[mi, fi, fs, li, :, ki]
                        descriptor_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "selected_family": family_name,
                                "stimulus_family": stimulus,
                                "verbosity": verbosity,
                                "metric": metric,
                                "mean_difference": float(pairs.mean()),
                                "positive_pairs": int((pairs > 0).sum()),
                                "zero_pairs": int((pairs == 0).sum()),
                                "pairs": 8,
                                **interval(boots["descriptor_mean"][:, mi, fi, fs, li, ki]),
                            }
                        )
    for row in desc_rows:
        row["seed"] = seed
    candidates = read(root / "analysis-candidates.json")
    for row in candidates:
        row["seed"] = seed
    gates = development_gates(family, aggregate, boots, diff, controlled)
    return {
        "seed": seed,
        "primary": primary,
        "conditions": conditions,
        "signed": signed_rows,
        "quality": quality_rows,
        "descriptor_summary": descriptor_rows,
        "descriptor_values": desc_rows,
        "baseline_length_strata": length_rows,
        "candidates": candidates,
        "fit": fit,
        "gates": gates,
        "execution_amendment": read(root / "integrity.json").get("execution_amendment"),
        "selection": read(root / "selection.json"),
        "examples": read(root / "fixed-examples.json"),
    }


def plots(summaries, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, summary in zip(axes, summaries, strict=True):
        for mi, method in enumerate(METHODS):
            rows = [
                r for r in summary["primary"] if r["method"] == method and r["scope"] in FAMILIES
            ]
            y = np.asarray([r["signed_response"] for r in rows]) * 100
            lower, upper = (
                np.asarray([r["lo"] for r in rows]) * 100,
                np.asarray([r["hi"] for r in rows]) * 100,
            )
            x = np.arange(5) + (mi - 0.5) * 0.15
            line = ax.plot(x, y, "o", label=method)[0]
            ax.vlines(x, lower, upper, color=line.get_color())
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(
            xticks=range(5),
            xticklabels=FAMILIES,
            title=f"Seed {summary['seed']}",
            ylabel="Signed target response (percentage points)",
        )
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
    fig.suptitle("All-strength signed response, 99% marginal intervals (five-family Bonferroni)")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"signed-response.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(17, 6), constrained_layout=True)
    for ri, summary in enumerate(summaries):
        for fi, family in enumerate(FAMILIES):
            ax = axes[ri, fi]
            for method in METHODS:
                for sign, style in ((1, "-"), (-1, "--")):
                    rows = [
                        r
                        for r in summary["conditions"]
                        if r["method"] == method
                        and r["direction_family"] == family
                        and r["population"] == "published"
                        and r["stratum"] == "all"
                        and r["endpoint"] == family
                        and r["sign"] == sign
                    ]
                    ax.plot(
                        [r["strength"] for r in rows],
                        [100 * r["change"] for r in rows],
                        style,
                        marker="o",
                        label=f"{method} {sign:+d}",
                    )
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.set(
                title=f"{family}, seed {summary['seed']}",
                xlabel="Fixed strength",
                ylabel="Change (pp)",
            )
    axes[0, 0].legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"complete-strength-grid.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, summary in zip(axes, summaries, strict=True):
        for method in METHODS:
            values = [r["coherence"] for r in summary["candidates"] if r["method"] == method]
            ax.hist(values, bins=30, alpha=0.5, label=method)
        ax.set(
            title=f"All 500 candidates, seed {summary['seed']}",
            xlabel="Training raw-gradient coherence",
            ylabel="Candidates",
        )
        ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"candidate-coherence.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for si, summary in enumerate(summaries):
        for ki, metric in enumerate(("oriented_coefficient", "physical_cosine")):
            ax = axes[si, ki]
            for mi, method in enumerate(METHODS):
                for li, verbosity in enumerate(("short", "long")):
                    rows = [
                        r
                        for r in summary["descriptor_summary"]
                        if r["method"] == method
                        and r["selected_family"] == r["stimulus_family"]
                        and r["verbosity"] == verbosity
                        and r["metric"] == metric
                    ]
                    x = np.arange(5) + (mi * 2 + li - 1.5) * 0.12
                    line = ax.plot(
                        x,
                        [r["mean_difference"] for r in rows],
                        "o" if li == 0 else "s",
                        label=f"{method} {verbosity}",
                    )[0]
                    ax.vlines(
                        x, [r["lo"] for r in rows], [r["hi"] for r in rows], color=line.get_color()
                    )
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.set(
                xticks=range(5),
                xticklabels=FAMILIES,
                title=f"{metric.replace('_', ' ')}, seed {summary['seed']}",
                ylabel="Target minus matched control",
            )
            ax.tick_params(axis="x", rotation=20)
    axes[0, 0].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"descriptor-preference.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for si, summary in enumerate(summaries):
        for ei, endpoint in enumerate(("content_loss", "extra_truncation", "extra_degeneration")):
            ax = axes[si, ei]
            for mi, method in enumerate(METHODS):
                for ki, statistic in enumerate(("mean_harm", "worst_harm")):
                    rows = [
                        r
                        for r in summary["quality"]
                        if r["method"] == method
                        and r["endpoint"] == endpoint
                        and r["statistic"] == statistic
                    ]
                    ax.plot(
                        np.arange(5) + (mi * 2 + ki - 1.5) * 0.12,
                        [100 * r["value"] for r in rows],
                        "o" if ki == 0 else "x",
                        label=f"{method} {statistic.replace('_', ' ')}",
                    )
            ax.axhline(5, color="gray", linewidth=0.7, linestyle="--")
            ax.axhline(15, color="gray", linewidth=0.7, linestyle=":")
            ax.set(
                xticks=range(5),
                xticklabels=FAMILIES,
                title=f"{endpoint.replace('_', ' ')}, seed {summary['seed']}",
                ylabel="Harm (percentage points)",
            )
            ax.tick_params(axis="x", rotation=20)
    axes[0, 0].legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"controlled-quality.{ext}", dpi=170)
    plt.close(fig)


def render_report(summaries, output):
    def table(rows, columns):
        return (
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(c)}</th>" for c in columns)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
                + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )

    success = all(s["gates"]["all_svd_requirements_pass"] for s in summaries)
    text = [
        "<!doctype html><meta charset='utf-8'><title>SVD versus Gradient Atoms</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;color:#17202a}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}pre{white-space:pre-wrap;font:13px system-ui}details{margin:12px 0}</style>",
        "<h1>SVD versus Gradient Atoms: matched development replication</h1>",
        f"<p>The frozen joint development gates {'passed' if success else 'did not all pass'}. This comparison uses two reused adapters and public paper prompts. It is not untouched confirmation.</p>",
        "<p>The fixed primary combines every signed strength and all five families. Component selection uses only training documents. Coherence and detector purity alone do not validate semantics. All quality costs, failed descriptor pairs and zero codes remain reported. Missing-input clarification is not safety refusal; fenced-code detection is not code correctness.</p>",
    ]
    if any(s.get("execution_amendment") for s in summaries):
        text.append(
            "<p><strong>Parallel execution limitation.</strong> An amendment fixed condition ownership before the moved conditions ran; scientific choices and original deadlines were unchanged. Each family retains its dictionary and SVD conditions on the same device. For seed 42, the shared baseline and first three families ran on A100, while bullets and numbered interventions ran on H100. For seed 43, numbered interventions ran on a second H100. Identical software and generation settings do not guarantee identical greedy responses across devices. In particular, seed 42 baseline-relative quality comparisons for the moved families include this device difference. The reported memory peak covers the original training preflight and tail worker; the original prefix generation peak was not recorded. Final assembly time is not summed GPU time. This study makes no equal-compute or efficiency claim.</p>"
        )
    text.append(
        table(
            [r for s in summaries for r in s["primary"] if r["scope"] == "all_families"],
            ["seed", "method", "signed_response", "lo", "hi"],
        )
    )
    text.extend(
        f"<img src='{name}.png' alt='{name}'>"
        for name in (
            "signed-response",
            "complete-strength-grid",
            "descriptor-preference",
            "controlled-quality",
            "candidate-coherence",
        )
    )
    text.append("<h2>Separate numerical, descriptor and quality gates</h2>")
    for summary in summaries:
        text.append(
            f"<h3>Seed {summary['seed']}</h3><pre>{html.escape(json.dumps(summary['gates'], indent=2))}</pre>"
        )
        text.append(
            f"<details><summary>Fitting and selected components</summary><pre>{html.escape(json.dumps({'fit': summary['fit'], 'selection': summary['selection']}, indent=2))}</pre></details>"
        )
    text.append("<h2>Complete tables</h2><ul>")
    for name in (
        "primary",
        "conditions",
        "signed",
        "quality",
        "descriptor_summary",
        "descriptor_values",
        "baseline_length_strata",
        "candidates",
    ):
        text.append(f"<li><a href='{name}.csv'>{name.replace('_', ' ')}</a></li>")
    text.append(
        "</ul><p>Intervals use 1,000 paired draws. Published prompts are resampled within family and inviting/control strata. Controlled item draws are shared across methods, signs, strengths, descriptor labels and lengths. Family comparisons use 99% Bonferroni intervals; secondary intervals are pointwise. Baseline-length strata are descriptive, not causal adjustments.</p><h2>All fixed examples</h2>"
    )
    for summary in summaries:
        for example in summary["examples"]:
            row, condition = example["row"], example["condition"]
            text.append(
                f"<details><summary>Seed {summary['seed']} · {html.escape(condition['id'])} · {html.escape(row['id'])}</summary><pre>{html.escape(row['prompt'])}</pre><pre>{html.escape(row['response'])}</pre><pre>{html.escape(json.dumps(row['checks'], sort_keys=True))}</pre></details>"
            )
    (output / "report.html").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs=2, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    guard(args.roots, args.analysis_plan, args.barrier)
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    first = args.work / "first-analysis-reading.json"
    if first.exists():
        raise ValueError("Analysis already read outcomes; preserve the original before any repair")
    atomic_json(
        first,
        {
            "created_unix": time.time(),
            "analysis_plan_sha256": file_hash(args.analysis_plan),
            "barrier_sha256": file_hash(args.barrier),
            "analyzer_sha256": file_hash(Path(__file__)),
            "replicates": 1000,
        },
    )
    summaries = [summarize(root, args.work, 1000) for root in args.roots]
    for name in (
        "primary",
        "conditions",
        "signed",
        "quality",
        "descriptor_summary",
        "descriptor_values",
        "baseline_length_strata",
        "candidates",
    ):
        write_csv(args.output / f"{name}.csv", [r for s in summaries for r in s[name]])
    atomic_json(
        args.output / "summary.json",
        [{k: v for k, v in s.items() if k != "examples"} for s in summaries],
    )
    atomic_json(
        args.output / "fixed-examples.json",
        [{"seed": s["seed"], **e} for s in summaries for e in s["examples"]],
    )
    plots(summaries, args.output)
    render_report(summaries, args.output)
    atomic_json(
        args.output / "analysis-manifest.json",
        {
            "completed_unix": time.time(),
            "replicates": 1000,
            "analyzer_sha256": file_hash(Path(__file__)),
            "analysis_plan_sha256": file_hash(args.analysis_plan),
            "barrier_sha256": file_hash(args.barrier),
            "files_sha256": {
                p.name: file_hash(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name != "analysis-manifest.json"
            },
        },
    )


if __name__ == "__main__":
    main()
