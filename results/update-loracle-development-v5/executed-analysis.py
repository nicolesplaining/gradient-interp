"""Audit and evaluate exact calibration effects against the unchanged measured target."""

import argparse
import hashlib
import html
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from .loracle_trajectory_toy import load_target, read
from .study import atomic_json, file_hash
from .update_loracle_analysis import metrics, paired_cluster_intervals, require
from .update_loracle_exact_calibration import (
    RECIPE,
    donor_assignments,
    effect_profile,
    physical_norm,
    scaled_native,
    score_and_restore,
    validate_case_grid,
)
from .update_loracle_factorized_analysis import completion_barrier
from .update_loracle_physical_profile import source_keys
from .update_loracle_sensitivity import select_calibration
from .update_loracle_sensitivity_analysis import development_gates, profile_metrics


def checked_profile(record, baseline):
    after = np.asarray(record["after_ce"], dtype=np.float64)
    require(after.shape == (8, 32) and np.isfinite(after).all(), "Incomplete after-score grid")
    # Independently average the eight document changes, retaining every response pattern.
    raw = np.asarray([sum(after[q, p] - baseline[q, p] for q in range(8)) / 8 for p in range(32)])
    profile = raw.mean() - raw
    np.testing.assert_allclose(record["ce_change"], raw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(record["profile"], profile, rtol=0, atol=1e-12)
    require(
        record["prediction"] == effect_profile(baseline, after)["prediction"],
        "Changed category/floor",
    )


def audit(root):
    completion_barrier(read(root / "controller.json"))
    controller, plan, manifest = [
        read(root / name) for name in ("controller.json", "plan.json", "manifest.json")
    ]
    require(
        plan["recipe"] == manifest["recipe"] == RECIPE and manifest["status"] == "complete",
        "Wrong completed recipe",
    )
    require(
        file_hash(root / "plan.json") == controller["plan_sha256"] == manifest["plan_sha256"],
        "Plan changed",
    )
    require(manifest["source_commit"] == plan["source_commit"], "Source commit changed")
    with tarfile.open(root / "source_snapshot.tar") as archive:
        require(
            set(archive.getnames()) == set(plan["source_files_sha256"]), "Source archive incomplete"
        )
        for path, expected in plan["source_files_sha256"].items():
            require(
                hashlib.sha256(archive.extractfile(path).read()).hexdigest() == expected,
                "Archived source changed",
            )
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Input changed: {path}")
    for name, field in (
        ("data.json", "data_sha256"),
        ("analysis-plan.json", "analysis_plan_sha256"),
    ):
        require(file_hash(root / name) == plan[field], "Data/analysis plan changed")
    data = read(root / "data.json")
    prior = Path(plan["prior_root"])
    require(data == read(prior / "data.json"), "Calibration/case selection differs from v4")
    cases, calibration = data["cases"], data["calibration"]
    validate_case_grid(cases)
    collection = Path(plan["collection_root"])
    require(
        calibration == select_calibration(read(collection / "data.json")),
        "Training calibration changed",
    )
    expected_grid = {
        "cases": 344,
        "scored_adapter_conditions": 688,
        "calibration_questions": 8,
        "patterns": 32,
        "initializations": 8,
        "target_optimizer_updates": 0,
        "reserved_cases_accessed": 0,
    }
    require(
        all(manifest.get(k) == v for k, v in expected_grid.items()), "Scientific grid incomplete"
    )
    require(
        len(cases) == 344 and {c["split"] for c in cases} == {"train", "development"},
        "Wrong case scope",
    )
    expected_files = {
        f"scores/{c['id']}-{kind}.json" for c in cases for kind in ("own", "shuffled")
    }
    expected_files |= {
        "started.json",
        "preflight.json",
        "baselines.json",
        "predictions.json",
        "predictions-lock.json",
    }
    require(set(manifest["files_sha256"]) == expected_files, "Missing or extra outputs")
    for name, expected in manifest["files_sha256"].items():
        require(file_hash(root / name) == expected, "Scientific output changed")
    lock, preflight, baselines = [
        read(root / name) for name in ("predictions-lock.json", "preflight.json", "baselines.json")
    ]
    require(
        lock["files_sha256"]
        == {k: v for k, v in manifest["files_sha256"].items() if k != "predictions-lock.json"},
        "Lock differs",
    )
    require(
        lock["cases"] == 344
        and lock["scored_adapter_conditions"] == 688
        and lock["reserved_cases_accessed"] == lock["target_optimizer_updates"] == 0
        and lock["evaluation_probes_accessed_for_prediction"] is False,
        "Lock scope differs",
    )
    require(
        all(
            preflight.get(k) is True
            for k in (
                "passed",
                "training_only",
                "synthetic_only_no_target_updates",
                "all_256_scores_finite",
                "bit_exact_adapter_and_full_score_restore",
                "frozen_base_unchanged",
            )
        )
        and preflight["modules"] == 280
        and preflight["trainable_tensors"] == 560
        and preflight["parameters"] == 64225280
        and preflight["physical_norm_scale_relative_error"] < 5e-6,
        "Failed training-only preflight",
    )
    attempt = controller["attempts"][0]
    require(
        plan["max_hours"] == 4
        and controller["deadline_unix"] == controller["started_unix"] + 14400
        and controller["started_unix"]
        <= attempt["started_unix"]
        <= read(root / "started.json")["started_unix"]
        < preflight["finished_unix"]
        < baselines["finished_unix"]
        < lock["locked_unix"]
        <= manifest["finished_unix"]
        <= attempt["finished_unix"]
        <= controller["checked_unix"]
        <= controller["deadline_unix"],
        "Original execution deadline/order changed",
    )
    seeds = {c["initialization_seed"] for c in cases}
    require(
        set(baselines["scores"]) == set(map(str, seeds))
        and baselines["all_zero_B_scores_bit_exact"] is True,
        "Wrong baseline grid",
    )
    gradient_rows = read(prior / "gradient-manifest.json")["records"]
    for seed in seeds:
        old_scores = np.asarray([r["calibration_ce"] for r in gradient_rows if r["seed"] == seed]).T
        require(
            old_scores.shape == (8, 32)
            and np.array_equal(baselines["scores"][str(seed)], old_scores),
            "Original zero-B calibration differs",
        )
    initial = {
        seed: torch.load(
            collection / "snapshots" / f"initial-{seed}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for seed in seeds
    }
    require(
        all(
            set(value) == set(source_keys())
            and sum(t.numel() for pair in value.values() for t in pair) == 64225280
            for value in initial.values()
        ),
        "Original native architecture differs",
    )
    donors, lookup = donor_assignments(cases), {c["id"]: c for c in cases}
    rows = read(root / "predictions.json")
    require([r["id"] for r in rows] == [c["id"] for c in cases], "Prediction case order changed")
    previous, rechecks = baselines["finished_unix"], []
    chosen = [
        next(c["id"] for c in cases if c["split"] == split and not c["control"])
        for split in ("train", "development")
    ]
    for case, prediction in zip(cases, rows, strict=True):
        require(
            all(prediction[k] == v for k, v in case.items())
            and prediction["direction_donor"] == donors[case["id"]],
            "Prediction metadata differs",
        )
        for kind, donor_id in (("own", case["id"]), ("shuffled", donors[case["id"]])):
            record = read(root / "scores" / f"{case['id']}-{kind}.json")
            donor = lookup[donor_id]
            seed = donor["initialization_seed"]
            norm, donor_norm = case["physical_update_norm"], donor["physical_update_norm"]
            scale = norm / donor_norm if donor_norm else 1.0
            require((norm > 0) == (donor_norm > 0), "Zero and actual donor mixed")
            require(
                record["id"] == case["id"]
                and record["kind"] == kind
                and record["direction_donor"] == donor_id
                and record["initialization_seed"] == seed
                and record["donor_scale"] == scale
                and record["physical_update_norm"] == norm
                and record["bit_exact_restore"] is True
                and record["frozen_base_unchanged"] is True
                and previous < record["finished_unix"] < lock["locked_unix"],
                "Score identity/timeline/restore differs",
            )
            previous = record["finished_unix"]
            path = collection / "snapshots" / f"{donor_id}.pt"
            require(
                record["snapshot_sha256"] == plan["input_files_sha256"][str(path)],
                "Native input hash differs",
            )
            native = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            scaled = scaled_native(initial[seed], native, scale)
            measured_norm = physical_norm(scaled)
            require(
                np.isclose(measured_norm, norm, rtol=5e-6, atol=1e-10)
                and np.isclose(record["actual_scaled_norm"], measured_norm, rtol=1e-12, atol=1e-12),
                "Native physical norm differs",
            )
            baseline = np.asarray(baselines["scores"][str(seed)])
            checked_profile(record, baseline)
            if case["control"]:
                require(np.array_equal(record["after_ce"], baseline), "Zero effect changed")
            if kind == "own":
                for k in ("ce_change", "profile", "prediction"):
                    require(prediction[k] == record[k], "Own prediction differs")
            else:
                require(
                    prediction["shuffled_profile"] == record["profile"]
                    and prediction["shuffled_prediction"] == record["prediction"],
                    "Shuffle prediction differs",
                )
            if case["id"] in chosen:
                rechecks.append((case["id"], kind, donor_id, seed, scale))
            del native, scaled
    model, tokenizer = load_target()
    model.eval()
    checks = []
    for case_id, kind, donor_id, seed, scale in rechecks:
        native = torch.load(
            collection / "snapshots" / f"{donor_id}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        scaled = scaled_native(initial[seed], native, scale)
        result = score_and_restore(
            model,
            tokenizer,
            calibration,
            initial[seed],
            scaled,
            np.asarray(baselines["scores"][str(seed)]),
        )
        require(
            np.array_equal(result, read(root / "scores" / f"{case_id}-{kind}.json")["after_ce"]),
            "Fixed exact scoring recheck differs",
        )
        checks.append({"id": case_id, "kind": kind, "passed": True})
    hashes = {
        **manifest["files_sha256"],
        **{
            name: file_hash(root / name)
            for name in (
                "plan.json",
                "data.json",
                "analysis-plan.json",
                "manifest.json",
                "controller.json",
                "preparation.json",
                "source_snapshot.tar",
            )
        },
    }
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "recipe": RECIPE,
            "finished_unix": time.time(),
            "cases": 344,
            "scored_adapter_conditions": 688,
            "all_profiles_and_native_norms_verified": True,
            "fixed_scoring_rechecks": checks,
            "files_sha256": hashes,
            "reserved_cases_accessed": 0,
        },
    )


def gates(all_metrics):
    result = development_gates(
        {
            **all_metrics,
            "sensitivity": all_metrics["functional"],
            "sensitivity_shuffled": all_metrics["functional_shuffled"],
        }
    )
    result["functional_shuffled_advantage_10pp"] = result.pop("sensitivity_shuffled_advantage_10pp")
    return result


def error_decomposition(linear, exact, target):
    linear, exact, target = [np.asarray(v, dtype=np.float64) for v in (linear, exact, target)]
    require(
        linear.shape == exact.shape == target.shape and linear.shape[-1] == 32,
        "Profile shapes differ",
    )
    require(all(np.isfinite(v).all() for v in (linear, exact, target)), "Nonfinite profiles")
    a, b = linear - exact, exact - target
    return {
        "total_linear_to_evaluation_mse": float(np.mean((linear - target) ** 2)),
        "linear_to_exact_calibration_mse": float(np.mean(a * a)),
        "exact_calibration_to_evaluation_mse": float(np.mean(b * b)),
        "signed_cross_term": float(2 * np.mean(a * b)),
        "identity_residual": float(
            np.mean((linear - target) ** 2) - np.mean(a * a) - np.mean(b * b) - 2 * np.mean(a * b)
        ),
        "interpretation": (
            "Algebraic decomposition with covariance, not independent causal attribution"
        ),
    }


def analyze(root, output):
    completion_barrier(read(root / "controller.json"))
    integrity = read(root / "integrity.json")
    require(
        integrity["passed"]
        and integrity["recipe"] == RECIPE
        and integrity["cases"] == 344
        and integrity["scored_adapter_conditions"] == 688
        and integrity["all_profiles_and_native_norms_verified"] is True
        and len(integrity["fixed_scoring_rechecks"]) == 4
        and all(c["passed"] for c in integrity["fixed_scoring_rechecks"]),
        "Complete audit required",
    )
    require(not output.exists(), "Preserve first analysis")
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited output changed")
    plan = read(root / "plan.json")
    prior = Path(plan["prior_root"])
    prior_cases = prior / "analysis/all-development-cases.json"
    require(
        file_hash(prior_cases) == plan["input_files_sha256"][str(prior_cases)], "Baseline changed"
    )
    cases = read(prior_cases)
    all_rows = read(root / "predictions.json")
    rows = [r for r in all_rows if r["split"] == "development"]
    require(
        len(all_rows) == 344 and sum(r["split"] == "train" for r in all_rows) == 272,
        "Missing original training/development predictions",
    )
    require(
        len(cases) == len(rows) == 72 and [c["id"] for c in cases] == [r["id"] for r in rows],
        "Complete development grid required",
    )
    predictions = {
        name: [c["comparison_predictions"][name] for c in cases]
        for name in cases[0]["comparison_predictions"]
    }
    predictions.update(
        functional=[r["prediction"] for r in rows],
        functional_shuffled=[r["shuffled_prediction"] for r in rows],
    )
    effects = {c["id"]: c["measured"] for c in cases}
    measured = {k: metrics(v, cases, effects) for k, v in predictions.items()}
    correct = {
        k: np.asarray([p == c["label"] for p, c in zip(v, cases, strict=True)])
        for k, v in predictions.items()
    }
    boot = paired_cluster_intervals(correct, cases)
    differences = {
        k: {
            "point": measured["functional"]["actual_update_accuracy"] - v["actual_update_accuracy"],
            "descriptive_95_interval": np.quantile(
                boot["functional"] - boot[k], [0.025, 0.975]
            ).tolist(),
        }
        for k, v in measured.items()
        if k != "functional"
    }
    actual = np.asarray([not c["control"] for c in cases])
    target = np.asarray([c["measured"]["effect"]["centered_improvements"] for c in cases])
    profiles = np.asarray([r["profile"] for r in rows])
    linear = np.asarray([c["sensitivity_decoder"]["profile"] for c in cases])
    strata = {
        field: {
            str(value): {
                name: {
                    "n": sum(not c["control"] and c[field] == value for c in cases),
                    "correct": sum(
                        bool(v[i])
                        for i, c in enumerate(cases)
                        if not c["control"] and c[field] == value
                    ),
                }
                for name, v in correct.items()
            }
            for value in sorted({c[field] for c in cases if not c["control"]})
        }
        for field in ("source_pattern", "initialization_seed")
    }
    train = [r for r in all_rows if r["split"] == "train"]
    collection = Path(plan["collection_root"])
    train_effects = [read(collection / "cases" / f"{r['id']}.json")["effect"] for r in train]
    checks = gates(measured)
    summary = {
        "recipe": RECIPE,
        "scope": "reused development, not confirmation",
        "metrics": measured,
        "paired_differences": differences,
        "development_gates": checks,
        "all_development_gates_pass": all(checks.values()),
        "profile_metrics": profile_metrics(profiles[actual], target[actual]),
        "error_decomposition": error_decomposition(
            linear[actual], profiles[actual], target[actual]
        ),
        "strata": strata,
        "training_diagnostic": {
            "cases": len(train),
            "correct": sum(
                r["prediction"] == e["label"] for r, e in zip(train, train_effects, strict=True)
            ),
            "includes_zero_controls": True,
            "profile_metrics": profile_metrics(
                [r["profile"] for r in train], [e["centered_improvements"] for e in train_effects]
            ),
        },
        "replicates": 2000,
        "reserved_test_read": False,
        "limitations": [
            (
                "Scores each applied update with a pinned base model and fixed "
                "task-specific calibration library"
            ),
            (
                "Extra access and compute beyond released LoRAcle; not a trained "
                "improved LoRAcle checkpoint"
            ),
            "Finite-reference mean CE preference; not unrestricted generated behavior",
            (
                "Reused development chosen after linear predictor failure; "
                "untouched generalization untested"
            ),
            (
                "Both source and calibration template/word transfer can matter; "
                "small top-two margins remain in the primary endpoint"
            ),
        ],
    }
    records = [
        {
            **c,
            "functional_decoder": r,
            "exact_comparison_predictions": {k: v[i] for k, v in predictions.items()},
        }
        for i, (c, r) in enumerate(zip(cases, rows, strict=True))
    ]
    output.mkdir(parents=True)
    for name, value in (
        ("summary.json", summary),
        ("all-development-cases.json", records),
        ("all-training-and-development-predictions.json", all_rows),
    ):
        atomic_json(output / name, value)
    np.savez_compressed(root / "development-bootstrap.npz", **boot)
    (output / "executed-analysis.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(
        output / "analysis-provenance.json",
        {
            "finished_unix": time.time(),
            "integrity_sha256": file_hash(root / "integrity.json"),
            "executed_source_sha256": file_hash(__file__),
            "replicate_seed": 2026101653,
        },
    )
    render(output, summary, records)


def render(output, summary, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["metrics"])
    values = [100 * summary["metrics"][n]["actual_update_accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(names)), values, color=["#8796a2"] * (len(names) - 2) + ["#21658c", "#b88138"])
    ax.set(
        xticks=range(len(names)),
        xticklabels=[n.replace("_", "\n") for n in names],
        ylim=(0, 105),
        ylabel="Complete effect pattern correct (%)",
        title="All64 development updates; fixed complete-description endpoint",
    )
    ax.axhline(70, color="#555", linestyle="--", label="Fixed70% development criterion")
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "development-accuracy.png", dpi=180)
    plt.close(fig)
    actual = [c for c in records if not c["control"]]
    x = np.asarray([c["measured"]["effect"]["centered_improvements"] for c in actual])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, label in zip(
        axes,
        ("sensitivity_decoder", "functional_decoder"),
        ("Linear calibration", "Exact calibration"),
        strict=True,
    ):
        y = np.asarray([c[key]["profile"] for c in actual])
        ax.scatter(x.ravel(), y.ravel(), s=6, alpha=0.3)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#555")
        ax.set(
            xlabel="Measured evaluation effect (nats/token)",
            ylabel="Calibration effect (nats/token)",
            title=label,
        )
    fig.tight_layout()
    fig.savefig(output / "profile-predictions.png", dpi=180)
    plt.close(fig)
    details = "".join(
        f"<details><summary>{html.escape(c['id'])}: {html.escape(c['label'])}</summary>"
        f"<pre>{html.escape(json.dumps(c, indent=2))}</pre></details>"
        for c in records
    )
    (output / "report.html").write_text(
        """<!doctype html><meta charset='utf-8'><title>Exact calibration update descriptions</title>
<style>body{font:17px system-ui;max-width:1200px;margin:35px auto;padding:20px;line-height:1.55}
img{max-width:100%}pre{white-space:pre-wrap;font-size:12px}details{padding:8px}</style>
<h1>Can exact calibration effects describe small updates?</h1>
<p>Apply each saved native update to Qwen3-14B and score 32 fixed answer patterns on the same
8 training calibration prompts used by the linear predictor. Predict the complete format/case/order
pattern, then compare with previously measured effects on separate evaluation prompts.
No evaluation prompts enter the predictor.</p>
<p>This functional predictor uses extra base-model and task-library access. It is not an improved
LoRAcle checkpoint. All cases, negative results, controls and original thresholds are retained.
Untouched generalization remains untested.</p>
<img src='development-accuracy.png' alt='All method accuracies'>
<img src='profile-predictions.png' alt='Linear and exact calibration profiles'>
<h2>Complete results</h2><pre>"""
        + html.escape(json.dumps(summary, indent=2))
        + "</pre><h2>Every development case</h2>"
        + details
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026101711)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "audit":
        audit(args.root)
    else:
        analyze(args.root, args.output)
