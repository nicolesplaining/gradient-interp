"""One fixed training-only map from exact calibration profiles to probe effects."""

import argparse
import json
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

from .loracle_trajectory_toy import read
from .study import atomic_json, file_hash
from .update_loracle_analysis import metrics, paired_cluster_intervals, require
from .update_loracle_exact_calibration_analysis import gates
from .update_loracle_factorized_analysis import completion_barrier
from .update_loracle_physical_profile import fit_profile, predict_profiles, profile_label
from .update_loracle_sensitivity_analysis import profile_metrics

RECIPE = "functional_transfer_v6"


def fit_training(records):
    """Never accept a development record, even if its other fields look plausible."""
    require(records and all(r["split"] == "train" for r in records), "Training split only")
    require(len({r["id"] for r in records}) == len(records), "Duplicate training rows")
    x = np.asarray([r["profile"] for r in records], dtype=np.float64)
    y = np.asarray([r["target"] for r in records], dtype=np.float64)
    mask = np.asarray([not r["control"] for r in records])
    require(x.shape == y.shape == (len(records), 32), "Expected complete profiles")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite training profiles")
    require(np.allclose(x.mean(axis=1), 0, atol=1e-12), "Features must be centered")
    require(np.all(x[~mask] == 0), "Zero controls must have zero features")
    alpha, penalty = fit_profile(x @ x.T, y, mask)
    beta = x.T @ alpha
    # Independent primal solution checks the inherited dual implementation.
    primal = np.linalg.solve(x.T @ x + penalty * np.eye(32), x.T @ y)
    require(np.allclose(beta, primal, atol=2e-10, rtol=2e-10), "Primal/dual mismatch")
    require(np.isfinite(beta).all(), "Nonfinite fitted mapping")
    return {
        "training_features": x,
        "alpha": alpha,
        "beta": beta,
        "regularization": np.asarray(penalty),
    }


def infer(model, features):
    x = np.asarray(features, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == 32 and np.isfinite(x).all(), "Invalid features")
    profiles, labels = predict_profiles(x @ model["training_features"].T, model["alpha"])
    require(
        np.allclose(profiles, x @ model["beta"], atol=2e-10, rtol=2e-10),
        "Inference contraction mismatch",
    )
    require(np.all(profiles[np.all(x == 0, axis=1)] == 0), "Zero mapping changed")
    return profiles, labels


def checked_v5(v5):
    completion_barrier(read(v5 / "controller.json"))
    audit = read(v5 / "integrity.json")
    require(
        audit["passed"]
        and audit["recipe"] == "exact_calibration_v5"
        and audit["cases"] == 344
        and audit["scored_adapter_conditions"] == 688
        and audit["all_profiles_and_native_norms_verified"]
        and audit["reserved_cases_accessed"] == 0,
        "Full prior audit required",
    )
    require(
        len(audit["fixed_scoring_rechecks"]) == 4
        and all(r["passed"] for r in audit["fixed_scoring_rechecks"]),
        "Rechecks missing",
    )
    receipt = read(v5 / "local-copy-receipt.json")
    for name, sha in receipt["files_sha256"].items():
        require(file_hash(v5 / name) == sha, "Copied v5 evidence changed")
    for name, sha in audit["files_sha256"].items():
        require(receipt["files_sha256"].get(name) == sha, "Prior audit/copy mismatch")
    require(receipt["handoff"]["stage"] == "complete", "Prior analysis incomplete")
    return read(v5 / "plan.json")


def collect_training(rows, targets, prior_plan):
    train = [r for r in rows if r["split"] == "train"]
    require(
        len(train) == 272 and sum(not r["control"] for r in train) == 256,
        "Original complete training grid required",
    )
    result, hashes = [], {}
    for r in train:
        path = targets / f"{r['id']}.json"
        remote = str(Path(prior_plan["collection_root"]) / "cases" / path.name)
        sha = file_hash(path)
        require(sha == prior_plan["input_files_sha256"][remote], "Original target changed")
        t = read(path)
        require(
            t["id"] == r["id"] and t["split"] == "train" and t["control"] == r["control"],
            "Training target identity mismatch",
        )
        ce = np.asarray(t["effect"]["mean_ce_changes"], dtype=np.float64)
        require(ce.shape == (32,) and np.isfinite(ce).all(), "Invalid target losses")
        y = ce.mean() - ce
        require(
            np.allclose(y, t["effect"]["centered_improvements"], atol=1e-12, rtol=0)
            and profile_label(y) == t["effect"]["label"],
            "Target derivation changed",
        )
        result.append(
            {
                "id": r["id"],
                "split": "train",
                "control": r["control"],
                "profile": r["profile"],
                "target": y.tolist(),
            }
        )
        hashes[str(path.resolve())] = sha
    return result, hashes


def run(v5, targets, root, output):
    require(not root.exists() and not output.exists(), "Never overwrite an executed recipe")
    started = time.time()
    prior_plan = checked_v5(v5)
    rows = read(v5 / "predictions.json")
    require(len(rows) == 344 and len({r["id"] for r in rows}) == 344, "Incomplete input grid")
    require(
        [r["id"] for r in rows] == [r["id"] for r in read(v5 / "data.json")["cases"]],
        "Original case ordering changed",
    )
    require(all(r["split"] in ("train", "development") for r in rows), "Unexpected split")
    training, target_hashes = collect_training(rows, targets, prior_plan)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    proposal = Path(
        "results/update-loracle-program/provenance/functional-transfer-development-proposal.json"
    )
    source_hashes = {str(p.resolve()): file_hash(p) for p in source_files}
    root.mkdir(parents=True)
    with tarfile.open(root / "source_snapshot.tar", "w") as archive:
        for path in source_files:
            archive.add(path, arcname=path.name)
    with tarfile.open(root / "source_snapshot.tar") as archive:
        import hashlib

        for path in source_files:
            require(
                hashlib.sha256(archive.extractfile(path.name).read()).hexdigest()
                == source_hashes[str(path.resolve())],
                "Source archive mismatch",
            )
    atomic_json(
        root / "plan.json",
        {
            "recipe": RECIPE,
            "started_unix": started,
            "deadline_unix": started + 600,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "sources_sha256": source_hashes,
            "proposal_sha256": file_hash(proposal),
            "prior_integrity_sha256": file_hash(v5 / "integrity.json"),
            "prior_copy_receipt_sha256": file_hash(v5 / "local-copy-receipt.json"),
            "source_archive_sha256": file_hash(root / "source_snapshot.tar"),
            "numpy_version": np.__version__,
            "training_target_hashes": target_hashes,
            "training_count": 272,
            "development_count": 72,
            "reserved_cases_accessed": 0,
            "regularization_rule": ".01*mean squared feature norm over actual training updates",
            "previous_development_familiarity": True,
            "no_new_gpu_or_target_updates": True,
        },
    )
    model = fit_training(training)
    np.savez(root / "model.npz", **model)
    atomic_json(
        root / "model-lock.json",
        {
            "locked_unix": time.time(),
            "model_sha256": file_hash(root / "model.npz"),
            "plan_sha256": file_hash(root / "plan.json"),
            "training_ids": [r["id"] for r in training],
            "no_development_target_fitting": True,
        },
    )
    own, labels = infer(model, [r["profile"] for r in rows])
    shuffled, shuffled_labels = infer(model, [r["shuffled_profile"] for r in rows])
    predictions = [
        {
            "id": r["id"],
            "split": r["split"],
            "control": r["control"],
            "profile": own[i].tolist(),
            "prediction": labels[i],
            "shuffled_profile": shuffled[i].tolist(),
            "shuffled_prediction": shuffled_labels[i],
        }
        for i, r in enumerate(rows)
    ]
    atomic_json(root / "predictions.json", predictions)
    atomic_json(
        root / "predictions-lock.json",
        {
            "locked_unix": time.time(),
            "model_lock_sha256": file_hash(root / "model-lock.json"),
            "predictions_sha256": file_hash(root / "predictions.json"),
            "all344_own_and_shuffled_predictions_locked": True,
        },
    )
    # This is the first development-target join for the new model, after both locks.
    cases = read(v5 / "analysis/all-development-cases.json")
    dev = [r for r in predictions if r["split"] == "development"]
    require(
        len(cases) == len(dev) == 72 and [r["id"] for r in dev] == [c["id"] for c in cases],
        "Full development grid mismatch",
    )
    p = {
        k: [c["exact_comparison_predictions"][k] for c in cases]
        for k in cases[0]["exact_comparison_predictions"]
    }
    p.update(
        transfer=[r["prediction"] for r in dev],
        transfer_shuffled=[r["shuffled_prediction"] for r in dev],
    )
    effects = {c["id"]: c["measured"] for c in cases}
    measured = {k: metrics(v, cases, effects) for k, v in p.items()}
    correct = {
        k: np.asarray([x == c["label"] for x, c in zip(v, cases, strict=True)])
        for k, v in p.items()
    }
    boot = paired_cluster_intervals(correct, cases)
    differences = {
        k: {
            "point": measured["transfer"]["actual_update_accuracy"] - v["actual_update_accuracy"],
            "descriptive_95_interval": np.quantile(
                boot["transfer"] - boot[k], [0.025, 0.975]
            ).tolist(),
        }
        for k, v in measured.items()
        if k != "transfer"
    }
    checks = gates(
        {
            **measured,
            "functional": measured["transfer"],
            "functional_shuffled": measured["transfer_shuffled"],
        }
    )
    target = np.asarray([c["measured"]["effect"]["centered_improvements"] for c in cases])
    actual = np.asarray([not c["control"] for c in cases])
    summary = {
        "recipe": RECIPE,
        "scope": "reused development only; base-assisted learned profile map",
        "metrics": measured,
        "paired_differences": differences,
        "development_gates": checks,
        "all_development_gates_pass": all(checks.values()),
        "profile_metrics": profile_metrics(
            np.asarray([r["profile"] for r in dev])[actual], target[actual]
        ),
        "shuffled_profile_metrics": profile_metrics(
            np.asarray([r["shuffled_profile"] for r in dev])[actual], target[actual]
        ),
        "training_diagnostic": profile_metrics(
            own[[i for i, r in enumerate(rows) if r["split"] == "train"]],
            [r["target"] for r in training],
        ),
        "regularization": float(model["regularization"]),
        "replicates": 2000,
        "reserved_test_read": False,
        "new_gpu_work": False,
        "limitations": [
            "Extra pinned base-model and task-library access beyond LoRAcle.",
            "Not a trained LoRAcle checkpoint or weights-only decoder.",
            "Same reused development cases; no independent generalization claim.",
            "Finite-reference CE preference, not unrestricted generation.",
        ],
    }
    summary["strata"] = {
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
    records = [
        {
            "id": c["id"],
            "control": c["control"],
            "source_pattern": c["source_pattern"],
            "initialization_seed": c["initialization_seed"],
            "batch": c["batch"],
            "probes": c["probes"],
            "label": c["label"],
            "measured_effect": c["measured"]["effect"],
            "predictions": {k: v[i] for k, v in p.items()},
            "transfer": dev[i],
        }
        for i, c in enumerate(cases)
    ]
    require(time.time() <= started + 600, "Original CPU deadline exceeded")
    for path, sha in source_hashes.items():
        require(file_hash(path) == sha, "Executed source changed")
    for path, sha in target_hashes.items():
        require(file_hash(path) == sha, "Training target changed during execution")
    require(
        file_hash(root / "model.npz") == read(root / "model-lock.json")["model_sha256"],
        "Model changed after lock",
    )
    require(
        file_hash(root / "predictions.json")
        == read(root / "predictions-lock.json")["predictions_sha256"],
        "Predictions changed after lock",
    )
    output.mkdir(parents=True)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "all-development-cases.json", records)
    np.savez_compressed(root / "bootstrap.npz", **boot)
    (output / "executed-analysis.py").write_bytes(Path(__file__).read_bytes())
    for name in ("plan.json", "model-lock.json", "predictions-lock.json"):
        (output / name).write_bytes((root / name).read_bytes())
    atomic_json(
        output / "validation.json",
        {
            "finished_unix": time.time(),
            "passed": True,
            "all_original_v5_audit_and_copy_hashes_verified": True,
            "all272_training_targets_hash_verified": True,
            "independent_primal_dual_fit_agreement": True,
            "zero_input_exactly_zero": True,
            "model_and_all_predictions_locked_before_new_development_target_join": True,
            "all12_methods_all64_actual_all8_zeros_retained": True,
            "no_retraining_retuning_or_reserved_cases": True,
            "files_sha256": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
        },
    )
    render(output, summary, records)
    print(
        json.dumps(
            {
                "correct": measured["transfer"]["actual_update_correct"],
                "total": 64,
                "all_gates_pass": all(checks.values()),
            }
        )
    )


def render(output, summary, records):
    import html

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["metrics"])
    values = [100 * summary["metrics"][n]["actual_update_accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(names)), values, color=["#8796a2"] * (len(names) - 2) + ["#21658c", "#b88138"])
    ax.set(
        xticks=range(len(names)),
        xticklabels=[n.replace("_", "\n") for n in names],
        ylim=(0, 105),
        ylabel="Complete effect pattern correct (%)",
        title="One training-only profile map; all 64 development updates",
    )
    ax.axhline(70, color="#555", linestyle="--", label="Original 70% development criterion")
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
        "<!doctype html><meta charset='utf-8'><title>Learned functional transfer</title>"
        "<style>body{font:17px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:20px}"
        "img{width:100%}pre{white-space:pre-wrap;font-size:13px}details{padding:8px}</style>"
        "<h1>Can a learned map transfer calibration effects?</h1><p>One fixed ridge model uses only"
        " the original 272 training profiles and measured targets. No development fitting,"
        " threshold changes or reserved-test evaluation."
        " Original 0.02 no-change floor retained.</p>"
        "<p>This method scores the actual update on a fixed library with the base model. It adds"
        " a learned profile map; it is not an improved learned LoRAcle checkpoint.</p>"
        "<img src='development-accuracy.png' alt='All twelve methods on64 actual updates'>"
        "<h2>Complete results</h2><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + "</pre><h2>Every case</h2>"
        + details
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.v5, args.targets, args.root, args.output)
