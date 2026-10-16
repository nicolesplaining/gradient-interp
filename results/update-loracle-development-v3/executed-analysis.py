"""Audit the complete fixed physical decoder and retain all development outcomes."""

import argparse
import hashlib
import html
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from .loracle_trajectory_toy import read
from .study import atomic_json, file_hash
from .update_loracle_analysis import metrics, paired_cluster_intervals, require
from .update_loracle_factorized_analysis import completion_barrier
from .update_loracle_physical_profile import (
    fit_profile,
    predict_profiles,
    shuffled_cross_gram,
    source_keys,
)
from .update_loracle_train import shuffled_directions


def audit(root):
    controller = read(root / "controller.json")
    completion_barrier(controller)
    plan, manifest = read(root / "plan.json"), read(root / "manifest.json")
    require(plan["recipe"] == manifest["recipe"] == "physical_profile_v3", "Wrong decoder recipe")
    require(
        plan["max_hours"] == 2
        and manifest["status"] == "complete"
        and manifest["source_commit"] == plan["source_commit"],
        "Wrong frozen execution",
    )
    require(
        file_hash(root / "plan.json") == controller["plan_sha256"] == manifest["plan_sha256"],
        "Plan changed",
    )
    attempt = controller["attempts"][0]
    require(
        controller["deadline_unix"] == controller["started_unix"] + 7200
        and controller["started_unix"]
        <= attempt["started_unix"]
        <= read(root / "started.json")["started_unix"]
        < read(root / "preflight.json")["finished_unix"]
        < read(root / "predictions-lock.json")["locked_unix"]
        <= manifest["finished_unix"]
        <= attempt["finished_unix"]
        <= controller["deadline_unix"],
        "Original execution timeline differs",
    )
    require(
        all(
            manifest[k] == v
            for k, v in {
                "cases": 344,
                "training_cases": 272,
                "development_cases": 72,
                "modules": 280,
                "new_target_updates": 0,
                "reserved_cases_accessed": 0,
            }.items()
        ),
        "Incomplete kernel/fit grid",
    )
    for name, key in [("data.json", "data_sha256"), ("analysis-plan.json", "analysis_plan_sha256")]:
        require(file_hash(root / name) == plan[key], "Frozen data/analysis plan changed")
    with tarfile.open(root / "source_snapshot.tar") as archive:
        require(
            set(archive.getnames()) == set(plan["source_files_sha256"]), "Archive inventory changed"
        )
        for name, expected in plan["source_files_sha256"].items():
            require(
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                "Archived source changed",
            )
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Frozen input changed: {path}")
    expected_files = {
        "started.json",
        "preflight.json",
        "gram.npy",
        "kernel-audit.json",
        "model.npz",
        "predictions.json",
        "predictions-lock.json",
    }
    require(set(manifest["files_sha256"]) == expected_files, "Output inventory differs")
    for name, expected in manifest["files_sha256"].items():
        require(file_hash(root / name) == expected, "Scientific output changed")
    lock = read(root / "predictions-lock.json")
    require(
        lock["development_effects_used_for_fitting"] is False
        and lock["reserved_cases_accessed"] == 0
        and lock["files_sha256"]
        == {k: v for k, v in manifest["files_sha256"].items() if k != "predictions-lock.json"},
        "Prediction lock differs",
    )
    preflight = read(root / "preflight.json")
    require(
        preflight["passed"]
        and preflight["training_only"]
        and preflight["dense_synthetic_identity"],
        "Physical algebra preflight failed",
    )
    cases = read(root / "data.json")["cases"]
    collection = Path(plan["collection_root"])
    original = read(collection / "data.json")["cases"]
    require(
        cases
        == [
            {k: c[k] for k in ("id", "split", "control", "initialization_seed")}
            for c in original
            if c["split"] in ("train", "development")
        ],
        "Original train/development grid changed",
    )
    effects = [read(collection / "cases" / f"{c['id']}.json") for c in cases]
    gram = np.load(root / "gram.npy")
    require(
        gram.shape == (344, 344)
        and np.isfinite(gram).all()
        and np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12),
        "Invalid Gram matrix",
    )
    require(
        np.linalg.eigvalsh(gram).min() >= -1e-9 * max(1.0, np.diag(gram).max()),
        "Physical Gram matrix is not positive semidefinite",
    )
    norms = np.array([r["physical_update_norm"] for r in effects])
    np.testing.assert_allclose(np.diag(gram), norms**2, rtol=1e-5, atol=1e-12)
    kernel = read(root / "kernel-audit.json")
    require(
        kernel["case_ids"] == [c["id"] for c in cases]
        and [r["module"] for r in kernel["modules"]] == source_keys(),
        "Module inventory changed",
    )
    for j, row in enumerate(kernel["modules"]):
        expected = [r["token_audit"][j]["physical_energy"] for r in effects]
        np.testing.assert_allclose(row["diagonal"], expected, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(
        np.sum([r["diagonal"] for r in kernel["modules"]], axis=0),
        np.diag(gram),
        rtol=1e-12,
        atol=1e-12,
    )
    # Independent fixed off-diagonal verification using batched per-pair products.
    rng = np.random.default_rng(2026101692)
    pairs = rng.integers(0, len(cases), size=(64, 2))
    used = sorted(set(pairs.flatten().tolist()))
    snapshots = {
        i: torch.load(
            collection / "snapshots" / f"{cases[i]['id']}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for i in used
    }
    values = np.zeros(64, dtype=np.float64)
    for key in source_keys():
        ai = torch.stack([snapshots[int(i)][key][0] for i in pairs[:, 0]]).cuda().double()
        aj = torch.stack([snapshots[int(i)][key][0] for i in pairs[:, 1]]).cuda().double()
        bi = torch.stack([snapshots[int(i)][key][1] for i in pairs[:, 0]]).cuda().double()
        bj = torch.stack([snapshots[int(i)][key][1] for i in pairs[:, 1]]).cuda().double()
        values += (
            (4 * ((ai @ aj.transpose(1, 2)) * (bi.transpose(1, 2) @ bj)).sum((1, 2))).cpu().numpy()
        )
    np.testing.assert_allclose(values, gram[pairs[:, 0], pairs[:, 1]], rtol=1e-9, atol=1e-10)
    train = [i for i, c in enumerate(cases) if c["split"] == "train"]
    dev = [i for i, c in enumerate(cases) if c["split"] == "development"]
    targets = np.asarray([effects[i]["effect"]["centered_improvements"] for i in train])
    alpha, penalty = fit_profile(
        gram[np.ix_(train, train)], targets, [not cases[i]["control"] for i in train]
    )
    model = np.load(root / "model.npz")
    require(
        set(model.files)
        == {"alpha", "regularization", "training_targets", "training_indices", "training_gram"},
        "Model fields differ",
    )
    np.testing.assert_array_equal(model["training_indices"], train)
    np.testing.assert_array_equal(model["training_targets"], targets)
    np.testing.assert_array_equal(model["training_gram"], gram[np.ix_(train, train)])
    require(float(model["regularization"]) == penalty, "Training-only regularization changed")
    np.testing.assert_allclose(model["alpha"], alpha, rtol=1e-10, atol=1e-12)
    cross = gram[np.ix_(dev, train)]
    profile, labels = predict_profiles(cross, alpha)
    dev_cases = [{**cases[i], "tokens": cases[i]["id"]} for i in dev]
    donors = shuffled_directions(dev_cases)
    lookup = {c["id"]: i for i, c in enumerate(dev_cases)}
    indices = [lookup[c["direction_donor"]] for c in donors]
    shuffled, shuffled_labels = predict_profiles(
        shuffled_cross_gram(cross, np.sqrt(np.diag(gram)[dev]), indices), alpha
    )
    predictions = read(root / "predictions.json")
    require(
        [r["id"] for r in predictions] == [c["id"] for c in dev_cases], "Prediction grid differs"
    )
    np.testing.assert_allclose([r["profile"] for r in predictions], profile, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        [r["shuffled_profile"] for r in predictions], shuffled, rtol=1e-10, atol=1e-12
    )
    require(
        [r["prediction"] for r in predictions] == labels
        and [r["shuffled_prediction"] for r in predictions] == shuffled_labels
        and [r["direction_donor"] for r in predictions] == [r["direction_donor"] for r in donors],
        "Prediction or matched shuffle differs",
    )
    hashes = {
        **manifest["files_sha256"],
        **{
            name: file_hash(root / name)
            for name in [
                "plan.json",
                "data.json",
                "analysis-plan.json",
                "manifest.json",
                "controller.json",
                "source_snapshot.tar",
                "preparation.json",
            ]
        },
    }
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "recipe": "physical_profile_v3",
            "files_sha256": hashes,
            "cases": 344,
            "all_native_input_hashes_verified": True,
            "all_module_norms_verified": True,
            "independent_cross_pairs": pairs.tolist(),
            "cross_pair_max_abs_error": float(
                np.abs(values - gram[pairs[:, 0], pairs[:, 1]]).max()
            ),
            "training_only_fit_recomputed": True,
            "reserved_cases_accessed": 0,
        },
    )


def analyze(root, output):
    require(not output.exists(), "Preserve previous analysis")
    integrity = read(root / "integrity.json")
    require(
        integrity["passed"] and integrity["recipe"] == "physical_profile_v3",
        "Complete audit required",
    )
    completion_barrier(read(root / "controller.json"))
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited output changed")
    plan = read(root / "plan.json")
    prior_path = Path(plan["prior_root"]) / "analysis/all-development-cases.json"
    require(
        file_hash(prior_path) == plan["input_files_sha256"][str(prior_path)],
        "Frozen baseline changed",
    )
    cases = read(prior_path)
    rows = read(root / "predictions.json")
    require([c["id"] for c in cases] == [r["id"] for r in rows], "Baseline/prediction grid differs")
    effects = {c["id"]: c["measured"] for c in cases}
    predictions = {
        name: [c["predictions"][name] for c in cases]
        for name in ["released", "flat_majority", "factorized_majority", "ridge", "epoch1"]
    }
    predictions["physical_profile"] = [r["prediction"] for r in rows]
    predictions["physical_profile_shuffled"] = [r["shuffled_prediction"] for r in rows]
    all_metrics = {k: metrics(v, cases, effects) for k, v in predictions.items()}
    correct = {
        k: np.array([p == c["label"] for p, c in zip(v, cases, strict=True)])
        for k, v in predictions.items()
    }
    boot = paired_cluster_intervals(correct, cases)
    m = all_metrics["physical_profile"]
    differences = {
        k: {
            "point": m["actual_update_accuracy"] - v["actual_update_accuracy"],
            "descriptive_95_interval": np.quantile(
                boot["physical_profile"] - boot[k], [0.025, 0.975]
            ).tolist(),
        }
        for k, v in all_metrics.items()
        if k != "physical_profile"
    }
    gates = {
        "accuracy_70": m["actual_update_accuracy"] >= 0.7,
        "released_advantage_20pp": differences["released"]["point"] >= 0.2,
        "effect_coverage_80": m["above_effect_floor_fraction"] >= 0.8,
        "zero_accuracy_90": m["zero_accuracy"] >= 0.9,
        "flat_majority_advantage_10pp": differences["flat_majority"]["point"] >= 0.1,
        "factorized_majority_advantage_10pp": differences["factorized_majority"]["point"] >= 0.1,
        "direction_dependence_10pp": differences["physical_profile_shuffled"]["point"] >= 0.1,
    }
    strata = {}
    for field in ["source_pattern", "initialization_seed"]:
        strata[field] = {}
        for value in sorted({c[field] for c in cases if not c["control"]}):
            ids = [i for i, c in enumerate(cases) if not c["control"] and c[field] == value]
            strata[field][str(value)] = {
                k: {"n": len(ids), "correct": int(v[ids].sum())} for k, v in correct.items()
            }
    actual = np.array([not c["control"] for c in cases])
    target = np.array([c["measured"]["effect"]["centered_improvements"] for c in cases])
    profile = np.array([r["profile"] for r in rows])
    error = profile[actual] - target[actual]
    summary = {
        "scope": "reused development, not confirmation",
        "recipe": "physical_profile_v3",
        "metrics": all_metrics,
        "paired_differences": differences,
        "development_gates": gates,
        "all_development_gates_pass": all(gates.values()),
        "strata": strata,
        "profile_metrics": {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "nrmse_vs_zero": float(np.sqrt(np.sum(error**2) / np.sum(target[actual] ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "max_absolute_error": float(np.abs(error).max()),
            "actual_mean_span": float(np.ptp(target[actual], axis=1).mean()),
            "predicted_mean_span": float(np.ptp(profile[actual], axis=1).mean()),
        },
        "replicates": 2000,
        "reserved_test_read": False,
        "limitations": [
            "A different supervised weight-input decoder, not a released-LoRAcle fine-tune",
            "One base and one-step native low-rank updates; no arbitrary-checkpoint claim",
            "Full finite-reference likelihood profiles; no unrestricted generation claim",
            "Training/development data reused after two failures; no confirmation claim",
            "Dense low-rank support set retained; no efficiency or compressed-storage claim",
        ],
    }
    records = [
        {
            **c,
            "physical_decoder": r,
            "comparison_predictions": {k: v[i] for k, v in predictions.items()},
        }
        for i, (c, r) in enumerate(zip(cases, rows, strict=True))
    ]
    output.mkdir(parents=True)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "all-development-cases.json", records)
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
    ax.bar(range(len(names)), values, color=["#8796a2"] * 5 + ["#21658c", "#b88138"])
    ax.set(
        xticks=range(len(names)),
        xticklabels=[n.replace("_", "\n") for n in names],
        ylim=(0, 105),
        ylabel="Complete effect pattern correct (%)",
        title="All 64 development updates; same fixed target",
    )
    ax.axhline(70, color="#555", linestyle="--", label="Fixed development criterion")
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "development-accuracy.png", dpi=180)
    plt.close(fig)
    details = "".join(
        f"<details><summary>{html.escape(c['id'])}: {html.escape(c['label'])}</summary>"
        f"<pre>{html.escape(json.dumps(c, indent=2))}</pre></details>"
        for c in records
    )
    (output / "report.html").write_text(
        """<!doctype html><meta charset='utf-8'><title>Physical update profiles</title>
<style>body{font:17px system-ui;max-width:1100px;margin:35px auto;padding:20px;line-height:1.55}
img{max-width:100%}pre{white-space:pre-wrap;font-size:12px}details{padding:8px}</style>
<h1>Predicting effects from signed weight changes</h1>
<p>This supervised decoder learns all 32 measured
effect scores from exact physical weight changes. It uses every original training example, one fixed
regularization rule, and no development fitting. The complete predicted pattern remains the primary
score. It is a different weight-based method, not a successful LoRAcle fine-tune.</p>
<p>All cases and controls remain. Untouched tests have not been run.</p>
<img src='development-accuracy.png' alt='Complete pattern accuracy across all methods'>
<h2>Complete development results</h2><pre>"""
        + html.escape(json.dumps(summary, indent=2))
        + "</pre><h2>All inputs, profiles, answers and failures</h2>"
        + details
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "audit":
        audit(args.root)
    else:
        require(args.output is not None, "Output required")
        analyze(args.root, args.output)
