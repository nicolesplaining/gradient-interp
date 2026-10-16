"""Post-completion integrity checks and the predeclared two-grid confirmation analysis."""

import argparse
import hashlib
import html
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from .loracle import MODULES
from .loracle_toy_suite import initialize_like
from .loracle_trajectory_toy import fingerprint, read
from .study import atomic_json, file_hash
from .update_loracle_analysis import alive, check_predictions, require
from .update_loracle_collect import tokenize_update, upstream_module
from .update_loracle_confirmation_stats import evaluate
from .update_loracle_data import LABELS, NO_CHANGE, effect_label
from .update_loracle_heldout import SPLITS, checked_data, checked_prediction_lock, checked_preflight
from .update_loracle_inference import verified_selection
from .update_loracle_train import shuffled_directions


def barrier(root):
    controller = read(root / "controller.json")
    require(
        controller["status"] == "complete"
        and not alive(controller["controller_pid"])
        and controller["worker_pid"] is None,
        "Complete dead held-out controller required",
    )
    attempts = controller["attempts"]
    require(
        [a["job"] for a in attempts] == ["preflight", "updates", "predict", "measure"],
        "Expected exactly four successful stages without retries",
    )
    previous = controller["started_unix"]
    for attempt in attempts:
        require(
            not alive(attempt["pid"]) and attempt["returncode"] == 0,
            "A held-out stage remains active or failed",
        )
        require(
            previous
            <= attempt["started_unix"]
            <= attempt["finished_unix"]
            <= controller["deadline_unix"],
            "Stage ordering or original deadline differs",
        )
        previous = attempt["finished_unix"]
    return controller


def archive(root, files):
    path = root / "source_snapshot.tar"
    if not path.exists():
        with tarfile.open(path, "x") as tar:
            for name, expected in files.items():
                require(file_hash(name) == expected, "Execution source changed before archive")
                tar.add(name, arcname=name, recursive=False)
    with tarfile.open(path) as tar:
        require(set(tar.getnames()) == set(files), "Source archive inventory differs")
        for name, expected in files.items():
            require(
                hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected,
                "Archived execution source differs",
            )
    return file_hash(path)


def check_snapshot(value):
    require(
        set(value) == {f"{layer:02d}:{module}" for layer in range(40) for module in MODULES},
        "Expected every target LoRA module",
    )
    require(
        sum(t.numel() for pair in value.values() for t in pair) == 64225280,
        "Target parameter count differs",
    )
    require(
        all(
            len(pair) == 2
            and pair[0].shape[0] == pair[1].shape[1] == 16
            and all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in pair)
            for pair in value.values()
        ),
        "Target rank, dtype or finiteness differs",
    )


def audit(root):
    controller = barrier(root)  # No scientific output parsing before normal completion and death.
    plan = read(root / "plan.json")
    require(file_hash(root / "plan.json") == controller["plan_sha256"], "Held-out plan changed")
    require(
        plan["max_hours"] == 12
        and controller["deadline_unix"] == controller["started_unix"] + 43200,
        "Immutable held-out deadline changed",
    )
    for path, expected in plan["source_files_sha256"].items():
        require(file_hash(path) == expected, "Frozen source changed")
    archive_hash = archive(root, plan["source_files_sha256"])
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, "Held-out model/input changed")
    for name, key in (
        ("data.json", "data_sha256"),
        ("analysis-plan.json", "analysis_plan_sha256"),
        ("selection.json", "selection_sha256"),
    ):
        require(file_hash(root / name) == plan[key], f"Frozen {name} changed")
    selection = verified_selection(root / "selection.json")
    data = read(root / "data.json")
    require(
        data["selection_sha256"] == plan["selection_sha256"]
        and data["original_data_sha256"] == file_hash(plan["original_data_path"]),
        "Held-out data/model binding differs",
    )
    cases = checked_data(data, read(plan["original_data_path"]))
    ids = [c["id"] for c in cases]
    preflight = checked_preflight(root, plan)
    target_preflight = read(root / "target-preflight.json")
    require(
        target_preflight["passed"] and target_preflight["bit_exact_adapter_and_probe_restore"],
        "Target preflight did not pass",
    )
    updates, manifest = read(root / "updates-manifest.json"), read(root / "manifest.json")
    prediction_lock = checked_prediction_lock(root, plan)
    require(
        selection["frozen_unix"]
        < data["created_unix"]
        < plan["created_unix"]
        < controller["started_unix"]
        <= target_preflight["finished_unix"]
        < preflight["finished_unix"]
        < updates["started_unix"]
        < updates["finished_unix"]
        < prediction_lock["finished_unix"]
        < manifest["measurement_started_unix"]
        < manifest["finished_unix"]
        <= controller["deadline_unix"],
        "Model/data/prediction/outcome timeline differs",
    )
    require(
        manifest["status"] == updates["status"] == "complete"
        and manifest["source_commit"] == plan["source_commit"]
        and manifest["cases"] == updates["cases"] == 272
        and manifest["actual_updates"] == updates["actual_updates"] == 256
        and manifest["zero_controls"] == 16
        and manifest["probe_pattern_scores"] == 69632,
        "Incomplete held-out grid",
    )
    for field, name in (
        ("plan_sha256", "plan.json"),
        ("predictions_lock_sha256", "predictions-lock.json"),
        ("updates_manifest_sha256", "updates-manifest.json"),
        ("preflight_sha256", "preflight.json"),
    ):
        require(manifest[field] == file_hash(root / name), "Final manifest hash differs")
    require(
        updates["plan_sha256"] == controller["plan_sha256"]
        and updates["preflight_sha256"] == file_hash(root / "preflight.json")
        and not updates["heldout_probes_measured"],
        "Update stage contract differs",
    )
    require(
        [r["id"] for r in updates["records"]] == ids == [r["id"] for r in manifest["records"]],
        "Missing, duplicated or reordered cases",
    )
    for directory, suffix in (("updates", ".json"), ("measurements", ".json"), ("tokens", ".pt")):
        require(
            {p.stem for p in (root / directory).iterdir() if p.suffix == suffix} == set(ids),
            f"Extra or missing {directory}",
        )
    initials = {}
    for name, expected in updates["initial_files_sha256"].items():
        path = root / "snapshots" / name
        require(file_hash(path) == expected, "Initial target adapter changed")
        seed = int(name.removeprefix("initial-").removesuffix(".pt"))
        value = torch.load(path, map_location="cpu", weights_only=True)
        check_snapshot(value)
        require(
            fingerprint(value) == fingerprint(initialize_like(value, seed)),
            "Wrong target initialization",
        )
        initials[seed] = value
    require(set(initials) == {c["initialization_seed"] for c in cases}, "Target seed grid differs")
    require(
        {p.stem for p in (root / "snapshots").glob("*.pt")}
        == set(ids) | {f"initial-{seed}" for seed in initials},
        "Target snapshot grid differs",
    )
    upstream = upstream_module()
    last_update, last_measure = updates["started_unix"], manifest["measurement_started_unix"]
    for case, update_item, measured_item in zip(cases, updates["records"], manifest["records"]):
        upath, mpath = (
            root / folder / f"{case['id']}.json" for folder in ("updates", "measurements")
        )
        require(
            file_hash(upath) == update_item["sha256"]
            and file_hash(mpath) == measured_item["sha256"],
            "Held-out case output changed",
        )
        update, measured = read(upath), read(mpath)
        require(
            all(update[k] == case[k] for k in ("id", "split", "control", "initialization_seed"))
            and measured["id"] == case["id"],
            "Case metadata differs",
        )
        require(
            last_update < update["finished_unix"] <= updates["finished_unix"]
            and last_measure < measured["finished_unix"] <= manifest["finished_unix"],
            "Held-out case timing differs",
        )
        last_update, last_measure = update["finished_unix"], measured["finished_unix"]
        require(
            update["bit_exact_adapter_restore"]
            and not update["heldout_probes_measured"]
            and measured["bit_exact_adapter_restore"]
            and measured["bit_exact_first_probe_restore"]
            and measured["predictions_lock_sha256"] == file_hash(root / "predictions-lock.json"),
            "Restoration or prediction lock differs",
        )
        before, after = np.asarray(measured["before_ce"]), np.asarray(measured["after_ce"])
        require(
            before.shape == after.shape == (4, 32)
            and effect_label(before, after) == measured["effect"],
            "Held-out measured effect differs",
        )
        start = initials[case["initialization_seed"]]
        require(
            update["initial_sha256"]
            == updates["initial_files_sha256"][f"initial-{case['initialization_seed']}.pt"],
            "Wrong update starting state",
        )
        snapshot, token_path = (
            root / "snapshots" / f"{case['id']}.pt",
            root / "tokens" / f"{case['id']}.pt",
        )
        require(
            file_hash(snapshot) == update["snapshot_sha256"]
            and file_hash(token_path) == update["tokens_sha256"],
            "Held-out native weights or tokens changed",
        )
        updated = torch.load(snapshot, map_location="cpu", weights_only=True)
        check_snapshot(updated)
        stored = torch.load(token_path, map_location="cpu", weights_only=True)
        tokens, norm, token_audit = tokenize_update(start, updated, upstream)
        require(
            torch.equal(tokens, stored)
            and norm == update["physical_update_norm"]
            and token_audit == update["token_audit"],
            "Native encoding or true update norm differs",
        )
        if case["control"]:
            require(
                update["step"] is None
                and norm == 0
                and np.array_equal(before, after)
                and fingerprint(start) == fingerprint(updated),
                "Zero control changed",
            )
        else:
            require(
                update["step"]["optimizer_steps"] == 1
                and norm > 0
                and update["step"]["learning_rate"] == plan["learning_rate"] == 0.0002,
                "Incorrect one-step optimizer update",
            )
        del updated, stored, tokens
    for split in SPLITS:
        subset = [c for c in cases if c["split"] == split]
        for method in ("released", "released-shuffled", "adapted", "adapted-shuffled"):
            check_predictions(read(root / "predictions" / f"{split}-{method}.json"), subset)
        expected = shuffled_directions([{**c, "tokens": c["id"]} for c in subset])
        require(
            read(root / "predictions" / f"{split}-shuffle.json")
            == [{"id": c["id"], "donor": c["direction_donor"]} for c in expected],
            "Held-out shuffled direction mapping differs",
        )
        simple = read(root / "predictions" / f"{split}-simple.json")
        require(
            simple["feature_count"] == 35841
            and simple["majority"] in LABELS
            and [r["id"] for r in simple["development"]] == [c["id"] for c in subset]
            and all(
                r[k] in LABELS
                for r in simple["development"]
                for k in ("ridge", "majority", "shuffled_ridge")
            ),
            "Incomplete simple baseline grid",
        )
    compact = [
        "controller.json",
        "plan.json",
        "data.json",
        "selection.json",
        "analysis-plan.json",
        "preflight.json",
        "target-preflight.json",
        "updates-manifest.json",
        "predictions-lock.json",
        "predictions-manifest.json",
        "manifest.json",
        "source_snapshot.tar",
    ]
    compact += [
        str(p.relative_to(root))
        for folder in ("updates", "measurements", "predictions")
        for p in (root / folder).glob("*.json")
    ]
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "scope": "reserved_and_fresh_confirmation",
            "finished_unix": time.time(),
            "source_commit": plan["source_commit"],
            "source_archive_sha256": archive_hash,
            "all_encodings_and_norms_recomputed": True,
            "predictions_locked_before_measurement": True,
            "cases": 272,
            "actual_updates": 256,
            "zero_controls": 16,
            "files_sha256": {name: file_hash(root / name) for name in compact},
        },
    )


def diagnostics(cases, measurements, predictions):
    targets = [m["effect"]["label"] for m in measurements]
    active = [i for i, c in enumerate(cases) if not c["control"]]
    controls = [i for i, c in enumerate(cases) if c["control"]]

    def scored(indices):
        return {
            "n": len(indices),
            "accuracy": {
                name: float(np.mean([values[i] == targets[i] for i in indices]))
                if indices
                else None
                for name, values in predictions.items()
            },
        }

    groups = {}
    for key in ("source_pattern", "initialization_seed"):
        groups[key] = {
            str(value): scored([i for i in active if cases[i][key] == value])
            for value in sorted({cases[i][key] for i in active})
        }
    groups["measured_target"] = {
        label: scored([i for i in active if targets[i] == label]) for label in LABELS
    }
    axes = {}
    for j, name in enumerate(("format", "case", "order")):
        defined = [i for i in active if targets[i] != NO_CHANGE]
        axes[name] = {
            "n_with_defined_axis": len(defined),
            "weak_updates_excluded_here_only": len(active) - len(defined),
            "accuracy": {
                method: float(
                    np.mean(
                        [
                            values[i] != NO_CHANGE
                            and values[i].split("/")[j] == targets[i].split("/")[j]
                            for i in defined
                        ]
                    )
                )
                if defined
                else None
                for method, values in predictions.items()
            },
        }
    confusion = {}
    for name, values in predictions.items():
        counts = np.zeros((33, 33), dtype=int)
        for i in active:
            counts[LABELS.index(targets[i]), LABELS.index(values[i])] += 1
        confusion[name] = counts.tolist()
    return {
        "actual_updates": scored(active),
        "zero_controls": scored(controls),
        "groups": groups,
        "axes": axes,
        "confusion_rows_true_columns_predicted": confusion,
        "confusion_labels": list(LABELS),
        "span_quantiles_0_25_50_75_100": np.quantile(
            [measurements[i]["effect"]["span"] for i in active], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "top_two_margin_quantiles_0_25_50_75_100": np.quantile(
            [measurements[i]["effect"]["top_two_margin"] for i in active], [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
    }


def analyze(root, output):
    require(not output.exists(), "Preserve earlier held-out analysis outputs")
    integrity = read(root / "integrity.json")
    require(
        integrity["passed"] and integrity["scope"] == "reserved_and_fresh_confirmation",
        "Complete held-out audit required",
    )
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, "Audited held-out output changed")
    data = read(root / "data.json")
    summaries, details, bootstrap = {}, {}, {}
    for split in SPLITS:
        cases = data[split]["cases"]
        measurements = [read(root / "measurements" / f"{c['id']}.json") for c in cases]
        predictions = {
            name.replace("-", "_"): [
                r["prediction"] for r in read(root / "predictions" / f"{split}-{name}.json")
            ]
            for name in ("adapted", "released", "adapted-shuffled", "released-shuffled")
        }
        simple = read(root / "predictions" / f"{split}-simple.json")["development"]
        for name in ("ridge", "majority", "shuffled_ridge"):
            predictions[name] = [r[name] for r in simple]
        targets = [m["effect"]["label"] for m in measurements]
        spans = [m["effect"]["span"] for m in measurements]
        primary = {
            name: predictions[name]
            for name in (
                "adapted",
                "released",
                "adapted_shuffled",
                "majority",
                "ridge",
            )
        }
        summary, draws = evaluate(cases, targets, primary, spans, replicates=4000)
        summary["complete_diagnostics"] = diagnostics(cases, measurements, predictions)
        summaries[split] = summary
        bootstrap.update({f"{split}_{name}": value for name, value in draws.items()})
        details[split] = [
            {**c, "measured": m, "predictions": {n: p[i] for n, p in predictions.items()}}
            for i, (c, m) in enumerate(zip(cases, measurements))
        ]
    output.mkdir(parents=True)
    atomic_json(
        output / "summary.json",
        {
            "scope": "Two separately drawn input sets and eight held-out source compositions of "
            "known operations on one fixed base. Measured reference-loss preferences "
            "after one step.",
            "splits": summaries,
            "both_grids_pass": all(s["all_numerical_gates_pass"] for s in summaries.values()),
            "selection_sha256": file_hash(root / "selection.json"),
            "limitations": [
                "No arbitrary-checkpoint or novel-operation claim",
                "Loss preferences are not free generation",
                "Few composition/seed clusters; bootstrap bounds remain approximate",
                "All weak updates retained",
            ],
        },
    )
    atomic_json(output / "all-cases.json", details)
    np.savez_compressed(root / "confirmation-bootstrap.npz", **bootstrap)
    (output / "executed-analysis.py").write_bytes(Path(__file__).read_bytes())
    from . import update_loracle_confirmation_stats as stats

    (output / "executed-statistics.py").write_bytes(Path(stats.__file__).read_bytes())
    atomic_json(
        output / "analysis-provenance.json",
        {
            "finished_unix": time.time(),
            "integrity_sha256": file_hash(root / "integrity.json"),
            "source_sha256": file_hash(__file__),
            "statistics_sha256": file_hash(stats.__file__),
            "replicates": 4000,
            "replicate_seed": 2026101665,
        },
    )
    render(output)


def render(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary, details = read(output / "summary.json"), read(output / "all-cases.json")
    names = ["released", "majority", "ridge", "adapted_shuffled", "adapted"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split in zip(axes, SPLITS):
        values = [summary["splits"][split]["accuracy"][name] * 100 for name in names]
        bars = ax.bar(range(5), values, color=["#919ba7"] * 4 + ["#176b8c"])
        ax.set(
            xticks=range(5),
            xticklabels=["Released", "Majority", "Ridge", "Shuffled\nadapted", "Adapted"],
            title=split.capitalize(),
            ylim=(0, 105),
            ylabel="Correct measured-effect category (%)",
        )
        ax.axhline(70, linestyle="--", linewidth=1, color="#444")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center")
    fig.tight_layout()
    fig.savefig(output / "heldout-accuracy.png", dpi=180)
    plt.close(fig)
    rows = "".join(
        f"<h2>{split}</h2>"
        + "".join(
            f"<details><summary>{html.escape(c['id'])}: "
            f"{html.escape(c['measured']['effect']['label'])}</summary>"
            f"<pre>{html.escape(json.dumps(c, indent=2))}</pre></details>"
            for c in cases
        )
        for split, cases in details.items()
    )
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>One-update interpreter held-out results</title>"
        "<style>body{font:17px system-ui;max-width:1100px;margin:35px auto;"
        "padding:20px;line-height:1.5}"
        "img{max-width:100%}pre{white-space:pre-wrap;font-size:12px}details{padding:8px}</style>"
        "<h1>Can an adapted LoRAcle interpret a single update?</h1>"
        "<p>All model predictions were locked before scoring any held-out probe. Each test set "
        "contains128 actual updates and eight zero controls. No weak updates were excluded.</p>"
        "<img src='heldout-accuracy.png' alt='All methods on both held-out sets'>"
        "<h2>Fixed criteria and paired uncertainty</h2>"
        f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre><h2>Every case</h2>{rows}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "audit":
        audit(args.root)
    else:
        if args.output is None:
            parser.error("--output is required")
        analyze(args.root, args.output)
