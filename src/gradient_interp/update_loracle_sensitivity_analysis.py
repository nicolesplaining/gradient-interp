"""Complete-run audit and unchanged full-category analysis of base sensitivities."""

import argparse
import hashlib
import html
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from .loracle_trajectory_toy import (
    fingerprint,
    load_target,
    read,
    restore_modules,
    snapshot_modules,
)
from .study import atomic_json, file_hash
from .update_loracle_analysis import metrics, paired_cluster_intervals, require
from .update_loracle_data import PATTERNS
from .update_loracle_factorized_analysis import completion_barrier
from .update_loracle_physical_profile import profile_label, source_keys
from .update_loracle_sensitivity import (
    calibration_rows,
    centered_profile,
    frozen_versions,
    gradient_b,
    matched_shuffle,
    select_calibration,
)


def independent_contract(gradients, snapshots, initial, device="cuda"):
    """Elementwise FP64 sums independently check the worker's matrix multiplication."""
    require(set(gradients) == set(initial), "Gradient inventory changed")
    result = np.zeros(len(snapshots), dtype=np.float64)
    for key in sorted(initial):
        a0, b0 = initial[key]
        g = gradients[key]
        require(
            a0.dtype == b0.dtype == g.dtype == torch.float32
            and not torch.count_nonzero(b0)
            and g.shape == b0.shape
            and torch.isfinite(g).all(),
            "Invalid native calibration gradient",
        )
        values = []
        for snap in snapshots:
            require(
                set(snap) == set(initial)
                and torch.equal(snap[key][0], a0)
                and snap[key][1].dtype == torch.float32
                and snap[key][1].shape == b0.shape
                and torch.isfinite(snap[key][1]).all(),
                "Native initialization mismatch",
            )
            values.append(snap[key][1])
        block = torch.stack(values).to(device=device, dtype=torch.float64)
        product = block * g.to(device=device, dtype=torch.float64)[None]
        result += product.flatten(1).sum(dim=1).cpu().numpy()
    return result


def verify_outputs(raw, rows, cases):
    require(
        raw.shape == (len(cases), 32)
        and np.isfinite(raw).all()
        and [r["id"] for r in rows] == [c["id"] for c in cases],
        "Wrong complete prediction grid",
    )
    profile = centered_profile(raw)
    np.testing.assert_allclose([r["profile"] for r in rows], profile, rtol=1e-10, atol=1e-12)
    require(
        [r["prediction"] for r in rows] == [profile_label(v) for v in profile],
        "Changed primary category or effect floor",
    )
    for split in ("train", "development"):
        ids = [i for i, c in enumerate(cases) if c["split"] == split]
        group = [cases[i] for i in ids]
        shuffled, donors = matched_shuffle(profile[ids], group)
        saved = [rows[i] for i in ids]
        np.testing.assert_allclose(
            [r["shuffled_profile"] for r in saved], shuffled, rtol=1e-10, atol=1e-12
        )
        require(
            [r["shuffled_prediction"] for r in saved] == [profile_label(v) for v in shuffled]
            and [r["direction_donor"] for r in saved] == donors
            and [r["split"] for r in saved] == [split] * len(ids)
            and [r["physical_update_norm"] for r in saved]
            == [c["physical_update_norm"] for c in group],
            "Incorrect direction shuffle or metadata",
        )


def audit(root):
    controller = read(root / "controller.json")
    completion_barrier(controller)
    plan, manifest = read(root / "plan.json"), read(root / "manifest.json")
    require(
        plan["recipe"] == manifest["recipe"] == "base_sensitivity_v4"
        and plan["max_hours"] == 4
        and manifest["status"] == "complete"
        and manifest["source_commit"] == plan["source_commit"],
        "Wrong execution recipe",
    )
    require(
        file_hash(root / "plan.json") == controller["plan_sha256"] == manifest["plan_sha256"],
        "Changed frozen plan",
    )
    attempt = controller["attempts"][0]
    lock = read(root / "predictions-lock.json")
    preflight = read(root / "preflight.json")
    require(
        controller["deadline_unix"] == controller["started_unix"] + 14400
        and controller["started_unix"]
        <= attempt["started_unix"]
        <= read(root / "started.json")["started_unix"]
        < preflight["finished_unix"]
        < lock["locked_unix"]
        <= manifest["finished_unix"]
        <= attempt["finished_unix"]
        <= controller["checked_unix"]
        <= controller["deadline_unix"],
        "Original deadline/timeline differs",
    )
    for k, v in {
        "cases": 344,
        "gradient_banks": 256,
        "calibration_questions": 8,
        "output_patterns": 32,
        "initializations": 8,
        "target_optimizer_updates": 0,
        "reserved_cases_accessed": 0,
    }.items():
        require(manifest[k] == v, "Incomplete scientific grid")
    for name, key in [("data.json", "data_sha256"), ("analysis-plan.json", "analysis_plan_sha256")]:
        require(file_hash(root / name) == plan[key], "Changed frozen data/analysis")
    with tarfile.open(root / "source_snapshot.tar") as archive:
        require(set(archive.getnames()) == set(plan["source_files_sha256"]), "Archive differs")
        for name, expected in plan["source_files_sha256"].items():
            require(
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                "Archived scientific source changed",
            )
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Input changed: {path}")
    data = read(root / "data.json")
    cases, calibration = data["cases"], data["calibration"]
    collection = Path(plan["collection_root"])
    original = read(collection / "data.json")
    require(calibration == select_calibration(original), "Calibration selection changed")
    originals = [c for c in original["cases"] if c["split"] in ("train", "development")]
    effects = [read(collection / "cases" / f"{c['id']}.json") for c in originals]
    require(
        cases
        == [
            {
                **{k: c[k] for k in ("id", "split", "control", "initialization_seed")},
                "physical_update_norm": e["physical_update_norm"],
            }
            for c, e in zip(originals, effects, strict=True)
        ],
        "Original case inventory changed",
    )
    seeds = sorted({c["initialization_seed"] for c in cases})
    names = {f"gradients/seed{s}-pattern{p:02d}.pt" for s in seeds for p in range(32)}
    names |= {
        "started.json",
        "preflight.json",
        "raw-ce-predictions.npy",
        "predictions.json",
        "gradient-manifest.json",
        "predictions-lock.json",
    }
    require(set(manifest["files_sha256"]) == names, "Output inventory differs")
    for name, expected in manifest["files_sha256"].items():
        require(file_hash(root / name) == expected, f"Output changed: {name}")
    require(
        lock["files_sha256"]
        == {k: v for k, v in manifest["files_sha256"].items() if k != "predictions-lock.json"}
        and lock["development_effects_used_for_fitting"] is False
        and lock["reserved_cases_accessed"] == 0
        and lock["predicted_cases"] == 344,
        "Prediction lock changed",
    )
    require(
        all(
            preflight[k] is True
            for k in (
                "passed",
                "training_only",
                "all_gradients_finite",
                "all_A_gradients_zero",
                "bit_exact_adapter_and_score_restore",
                "frozen_base_unchanged",
                "synthetic_perturbation_only_no_optimizer",
            )
        )
        and preflight["modules"] == 280
        and preflight["trainable_tensors"] == 560
        and preflight["trainable_parameters"] == 64225280
        and preflight["directional_contraction"] < 0,
        "GPU preflight failed",
    )
    np.testing.assert_allclose(
        preflight["directional_contraction"],
        preflight["independent_scalar_contraction"],
        rtol=1e-7,
        atol=1e-8,
    )
    gm = read(root / "gradient-manifest.json")
    records = gm["records"]
    require(
        [(r["seed"], r["pattern_index"]) for r in records]
        == [(s, p) for s in seeds for p in range(32)]
        and [r["seed"] for r in gm["restores"]] == seeds
        and gm["all_zero_B_calibration_scores_bit_exact"] is True,
        "Incomplete gradient/restore grid",
    )
    raw = np.load(root / "raw-ce-predictions.npy")
    recomputed = np.zeros_like(raw)
    previous = preflight["finished_unix"]
    for si, seed in enumerate(seeds):
        initial_path = collection / "snapshots" / f"initial-{seed}.pt"
        initial = torch.load(initial_path, map_location="cpu", weights_only=True, mmap=True)
        require(set(initial) == set(source_keys()), "Wrong native module grid")
        require(
            file_hash(initial_path) == gm["initialization_files_sha256"][str(seed)],
            "Initialization changed",
        )
        ids = [i for i, c in enumerate(cases) if c["initialization_seed"] == seed]
        snapshots = []
        for i in ids:
            p = collection / "snapshots" / f"{cases[i]['id']}.pt"
            require(
                file_hash(p) == effects[i]["snapshot_sha256"]
                and file_hash(initial_path) == effects[i]["initial_sha256"],
                "Native files differ from audited collection",
            )
            snapshots.append(torch.load(p, map_location="cpu", weights_only=True, mmap=True))
        for p in range(32):
            row = records[si * 32 + p]
            name = f"gradients/seed{seed}-pattern{p:02d}.pt"
            require(
                row["path"] == name
                and row["sha256"] == manifest["files_sha256"][name]
                and row["case_ids"] == [cases[i]["id"] for i in ids]
                and all(
                    row[k] is True
                    for k in ("all_A_gradients_zero", "adapter_unchanged", "frozen_base_unchanged")
                )
                and previous <= row["finished_unix"] < lock["locked_unix"],
                "Gradient timeline/invariance differs",
            )
            previous = row["finished_unix"]
            gradients = torch.load(root / name, map_location="cpu", weights_only=True, mmap=True)
            recomputed[ids, p] = independent_contract(gradients, snapshots, initial)
        r = gm["restores"][si]
        require(
            r["bit_exact_restore"] is True
            and r["fingerprint"] == fingerprint(initial)
            and previous <= r["finished_unix"] < lock["locked_unix"],
            "Restore differs",
        )
        previous = r["finished_unix"]
        del snapshots
    np.testing.assert_allclose(recomputed, raw, rtol=1e-8, atol=1e-9)
    scores = np.asarray([r["calibration_ce"] for r in records]).reshape(8, 32, 8)
    require(
        np.isfinite(scores).all()
        and np.array_equal(scores, np.broadcast_to(scores[:1], scores.shape)),
        "Zero-B calibration scores differ",
    )
    # Fixed derivative recomputation checks two seeds, never an outcome-selected bank.
    chosen = [
        next(c["initialization_seed"] for c in cases if c["split"] == split)
        for split in ("train", "development")
    ]
    model, tokenizer = load_target()
    model.eval()
    versions = frozen_versions(model)
    derivative_checks = []
    for seed in chosen:
        initial = torch.load(
            collection / "snapshots" / f"initial-{seed}.pt", map_location="cpu", weights_only=True
        )
        restore_modules(model, initial)
        actual, ce = gradient_b(model, tokenizer, calibration_rows(calibration, PATTERNS[0]))
        saved = torch.load(
            root / f"gradients/seed{seed}-pattern00.pt", map_location="cpu", weights_only=True
        )
        for key in source_keys():
            torch.testing.assert_close(actual[key], saved[key], rtol=1e-5, atol=1e-7)
        require(
            ce == records[seeds.index(seed) * 32]["calibration_ce"]
            and fingerprint(snapshot_modules(model)) == fingerprint(initial)
            and frozen_versions(model) == versions,
            "Derivative recomputation differs",
        )
        derivative_checks.append({"seed": seed, "pattern_index": 0, "passed": True})
    del model
    verify_outputs(raw, read(root / "predictions.json"), cases)
    hashes = {
        **manifest["files_sha256"],
        **{
            n: file_hash(root / n)
            for n in (
                "plan.json",
                "data.json",
                "analysis-plan.json",
                "manifest.json",
                "controller.json",
                "source_snapshot.tar",
                "preparation.json",
            )
        },
    }
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "recipe": "base_sensitivity_v4",
            "finished_unix": time.time(),
            "files_sha256": hashes,
            "cases": 344,
            "gradient_banks": 256,
            "all_native_contractions_verified": True,
            "max_contraction_error": float(np.abs(raw - recomputed).max()),
            "derivative_recomputations": derivative_checks,
            "reserved_cases_accessed": 0,
        },
    )


def profile_metrics(predicted, target):
    predicted, target = np.asarray(predicted), np.asarray(target)
    require(
        predicted.shape == target.shape and predicted.shape[-1] == 32, "Profile dimensions differ"
    )
    error = predicted - target
    denom = float(np.sum(target**2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nrmse_vs_zero": float(np.sqrt(np.sum(error**2) / denom)) if denom > 0 else None,
        "mae": float(np.abs(error).mean()),
        "maximum_error": float(np.abs(error).max()),
        "actual_mean_span": float(np.ptp(target, axis=1).mean()),
        "predicted_mean_span": float(np.ptp(predicted, axis=1).mean()),
    }


def development_gates(all_metrics):
    m = all_metrics["sensitivity"]
    accuracy = m["actual_update_accuracy"]
    return {
        "accuracy_70": accuracy >= 0.7,
        "released_advantage_20pp": accuracy - all_metrics["released"]["actual_update_accuracy"]
        >= 0.2,
        "effect_coverage_80": m["above_effect_floor_fraction"] >= 0.8,
        "zero_accuracy_90": m["zero_accuracy"] >= 0.9,
        **{
            f"{name}_advantage_10pp": accuracy - all_metrics[name]["actual_update_accuracy"] >= 0.1
            for name in ("flat_majority", "factorized_majority", "sensitivity_shuffled")
        },
    }


def analyze(root, output):
    require(not output.exists(), "Preserve earlier analyses")
    completion_barrier(read(root / "controller.json"))
    integrity = read(root / "integrity.json")
    require(
        integrity["passed"] and integrity["recipe"] == "base_sensitivity_v4",
        "Full sensitivity audit required",
    )
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited output changed")
    plan = read(root / "plan.json")
    baseline_path = Path(plan["baseline_root"]) / "analysis/all-development-cases.json"
    prior_path = Path(plan["prior_root"]) / "predictions.json"
    for path in (baseline_path, prior_path):
        require(file_hash(path) == plan["input_files_sha256"][str(path)], "Baseline changed")
    cases = read(baseline_path)
    old = read(prior_path)
    all_rows = read(root / "predictions.json")
    rows = [r for r in all_rows if r["split"] == "development"]
    require(
        [c["id"] for c in cases] == [r["id"] for r in rows] == [r["id"] for r in old],
        "Complete comparison grid differs",
    )
    effects = {c["id"]: c["measured"] for c in cases}
    predictions = {
        n: [c["predictions"][n] for c in cases]
        for n in ("released", "flat_majority", "factorized_majority", "ridge", "epoch1")
    }
    predictions["physical_profile"] = [r["prediction"] for r in old]
    predictions["sensitivity"] = [r["prediction"] for r in rows]
    predictions["sensitivity_shuffled"] = [r["shuffled_prediction"] for r in rows]
    measured = {k: metrics(v, cases, effects) for k, v in predictions.items()}
    correct = {
        k: np.asarray([p == c["label"] for p, c in zip(v, cases, strict=True)])
        for k, v in predictions.items()
    }
    boot = paired_cluster_intervals(correct, cases)
    differences = {
        k: {
            "point": measured["sensitivity"]["actual_update_accuracy"]
            - v["actual_update_accuracy"],
            "descriptive_95_interval": np.quantile(
                boot["sensitivity"] - boot[k], [0.025, 0.975]
            ).tolist(),
        }
        for k, v in measured.items()
        if k != "sensitivity"
    }
    gates = development_gates(measured)
    actual = np.asarray([not c["control"] for c in cases])
    target = np.asarray([c["measured"]["effect"]["centered_improvements"] for c in cases])
    profile = np.asarray([r["profile"] for r in rows])
    strata = {}
    for field in ("source_pattern", "initialization_seed"):
        strata[field] = {}
        for value in sorted({c[field] for c in cases if not c["control"]}):
            ids = [i for i, c in enumerate(cases) if not c["control"] and c[field] == value]
            strata[field][str(value)] = {
                k: {"n": len(ids), "correct": int(v[ids].sum())} for k, v in correct.items()
            }
    collection = Path(plan["collection_root"])
    train = [r for r in all_rows if r["split"] == "train"]
    training_effects = [read(collection / "cases" / f"{r['id']}.json")["effect"] for r in train]
    summary = {
        "scope": "reused development, not confirmation",
        "recipe": "base_sensitivity_v4",
        "metrics": measured,
        "paired_differences": differences,
        "development_gates": gates,
        "all_development_gates_pass": all(gates.values()),
        "strata": strata,
        "profile_metrics": profile_metrics(profile[actual], target[actual]),
        "training_diagnostic": {
            "cases": len(train),
            "correct": sum(r["prediction"] == e["label"] for r, e in zip(train, training_effects)),
            "includes_zero_controls": True,
            "profile_metrics": profile_metrics(
                [r["profile"] for r in train],
                [e["centered_improvements"] for e in training_effects],
            ),
        },
        "replicates": 2000,
        "reserved_test_read": False,
        "limitations": [
            "Standard first-order loss prediction; not a LoRAcle fine-tune or new identity",
            "Uses a base model and task-specific answer library beyond ordinary LoRAcle inputs",
            "New initializations need their own gradients with the same fixed prompts",
            "One native optimizer update; no arbitrary-checkpoint or unrestricted-behavior claim",
            "Development reused after prior failures; untouched generalization is still untested",
        ],
    }
    records = [
        {
            **c,
            "sensitivity_decoder": r,
            "comparison_predictions": {k: v[i] for k, v in predictions.items()},
        }
        for i, (c, r) in enumerate(zip(cases, rows, strict=True))
    ]
    output.mkdir(parents=True)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "all-development-cases.json", records)
    atomic_json(output / "all-training-and-development-predictions.json", all_rows)
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
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(names)), values, color=["#8796a2"] * 6 + ["#21658c", "#b88138"])
    ax.set(
        xticks=range(len(names)),
        xticklabels=[n.replace("_", "\n") for n in names],
        ylim=(0, 105),
        ylabel="Complete effect pattern correct (%)",
        title="All 64 development updates; unchanged complete-description target",
    )
    ax.axhline(70, color="#555", linestyle="--", label="Fixed development criterion")
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "development-accuracy.png", dpi=180)
    plt.close(fig)
    actual = [c for c in records if not c["control"]]
    x = np.array([c["measured"]["effect"]["centered_improvements"] for c in actual])
    y = np.array([c["sensitivity_decoder"]["profile"] for c in actual])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x.ravel(), y.ravel(), s=7, alpha=0.3)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="#555", linestyle="--")
    ax.set(
        xlabel="Measured centered CE improvement (nats/token)",
        ylabel="Predicted centered CE improvement (nats/token)",
        title="All 32 answer patterns for all 64 updates",
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
        """<!doctype html><meta charset='utf-8'><title>Base-model sensitivities</title>
<style>body{font:17px system-ui;max-width:1100px;margin:35px auto;padding:20px;line-height:1.55}
img{max-width:100%}pre{white-space:pre-wrap;font-size:12px}details{padding:8px}</style>
<h1>Describing small updates with fixed base-model sensitivities</h1>
<p>Eight fixed training prompts provide calibration gradients for 32 answer patterns. We contract
those gradients with each update's native adapter weights, preserving sign and magnitude.
The predicted complete pattern includes formatting, letter case and ordering. The same calibration
prompts are used for every initialization. The gradient bank uses its exact initial A.</p>
<p>This is a base-model-assisted predictor, not an improved LoRAcle checkpoint. It uses more
task-specific infrastructure. These results do not establish untouched generalization.</p>
<img src='development-accuracy.png' alt='Full pattern accuracy across all comparisons'>
<img src='profile-predictions.png' alt='Predicted versus actual centered loss changes'>
<h2>Complete results</h2><pre>"""
        + html.escape(json.dumps(summary, indent=2))
        + "</pre><h2>Every development input, answer and failure</h2>"
        + details
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026101701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "audit":
        audit(args.root)
    else:
        require(args.output is not None, "Output required")
        analyze(args.root, args.output)
