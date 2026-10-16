"""Guarded development comparisons for the fixed small-update interpreter recipe."""

import argparse
import hashlib
import html
import json
import tarfile
import time
from pathlib import Path

import numpy as np

from .loracle_trajectory_toy import read
from .study import atomic_json, file_hash
from .update_loracle_data import LABELS, NO_CHANGE


def require(condition, message):
    if not condition:
        raise ValueError(message)


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return bool(pid and path.exists() and path.read_text().rsplit(")", 1)[-1].split()[0] != "Z")


def check_predictions(rows, cases):
    from .update_loracle_train import OPTION_SYMBOLS

    require([r["id"] for r in rows] == [c["id"] for c in cases], "Incorrect prediction grid")
    for row, case in zip(rows, cases):
        values = np.asarray(row["choice_logits"])
        require(values.shape == (33,) and np.isfinite(values).all(), "Invalid choice scores")
        choice = str(int(values.argmax()))
        require(row["choice_id"] == OPTION_SYMBOLS[int(choice)], "Displayed answer ID differs")
        require(
            row["choice"] == choice and row["prediction"] == case["options"][choice],
            "Stored prediction differs from frozen constrained scoring",
        )
        require(4480 < row["input_tokens"] <= 6144, "Invalid interpreter length")


def audit_training(root):
    controller = read(root / "controller.json")
    require(
        controller["status"] == "complete" and not alive(controller["controller_pid"]),
        "Training must be normally complete and dead before reading predictions",
    )
    attempts = controller["attempts"]
    require(
        len(attempts) == 1
        and not alive(attempts[0]["pid"])
        and attempts[0]["returncode"] == 0
        and controller["worker_pid"] is None,
        "Expected one successfully completed training worker",
    )
    plan, manifest = read(root / "plan.json"), read(root / "manifest.json")
    require(
        file_hash(root / "plan.json") == controller["plan_sha256"] == manifest["plan_sha256"],
        "Training plan changed",
    )
    require(
        controller["deadline_unix"] == controller["started_unix"] + plan["max_hours"] * 3600
        and attempts[0]["finished_unix"] <= controller["deadline_unix"]
        and manifest["finished_unix"] <= controller["deadline_unix"],
        "Deadline changed/exceeded",
    )
    require(
        manifest["status"] == "complete"
        and manifest["epochs"] == 3
        and manifest["optimizer_steps"] == 204
        and manifest["training_cases"] == 272
        and manifest["development_cases"] == 72
        and manifest["reserved_cases_accessed"] == 0,
        "Incomplete training grid",
    )
    require(manifest["source_commit"] == plan["source_commit"], "Scientific source differs")
    from safetensors import safe_open

    for epoch in range(1, 4):
        directory = root / "epochs" / str(epoch)
        required = {
            "adapter_config.json",
            "adapter_model.safetensors",
            "development.json",
            "shuffled-development.json",
        }
        require({p.name for p in directory.iterdir()} == required, "Incomplete epoch artifacts")
        config = read(directory / "adapter_config.json")
        require(
            config["r"] == 256 and config["lora_alpha"] == 32 and config["use_rslora"],
            "Saved adapter architecture differs",
        )
        with safe_open(
            directory / "adapter_model.safetensors", framework="pt", device="cpu"
        ) as weights:
            keys = list(weights.keys())
            require(
                len(keys) == 560 and all("lora_" in key for key in keys),
                "Saved adapter tensor inventory differs",
            )
            count = sum(int(np.prod(weights.get_slice(key).get_shape())) for key in keys)
            require(count == 1027604480, "Saved trainable parameter count differs")
    archive = root / "source_snapshot.tar"
    if not archive.exists():
        with tarfile.open(archive, "x") as tar:
            for path, expected in plan["source_files_sha256"].items():
                require(file_hash(path) == expected, f"Frozen source changed: {path}")
                tar.add(path, arcname=path, recursive=False)
    with tarfile.open(archive) as tar:
        require(
            set(tar.getnames()) == set(plan["source_files_sha256"]), "Source archive grid differs"
        )
        for path, expected in plan["source_files_sha256"].items():
            require(
                hashlib.sha256(tar.extractfile(path).read()).hexdigest() == expected,
                "Archived source hash differs",
            )
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Input changed: {path}")
    for path, expected in manifest["files_sha256"].items():
        require(file_hash(root / path) == expected, f"Output changed: {path}")
    for name, key in [("data.json", "data_sha256"), ("analysis-plan.json", "analysis_plan_sha256")]:
        require(file_hash(root / name) == plan[key], f"Frozen {name} changed")
    collection = Path(plan["collection_root"])
    require(
        file_hash(collection / "integrity.json") == plan["collection_integrity_sha256"],
        "Collection audit changed",
    )
    integrity = read(collection / "integrity.json")
    require(integrity["passed"] and integrity["cases"] == 344, "Incomplete collection audit")
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(collection / name) == expected, "Audited collection output changed")
    data = read(collection / "data.json")
    train = [c for c in data["cases"] if c["split"] == "train"]
    dev = [c for c in data["cases"] if c["split"] == "development"]
    preflight = read(root / "preflight.json")
    require(
        preflight["passed"]
        and preflight["bit_exact_adapter_and_scores_restore"]
        and preflight["hook_calls_with_backward"] >= 2,
        "Training preflight failed",
    )
    history = read(root / "training-history.json")
    require([r["step"] for r in history] == list(range(204)), "Missing optimizer steps")
    from .update_loracle_train import learning_rate

    for epoch in range(3):
        rows = [r for r in history if r["epoch"] == epoch]
        require(
            len(rows) == 68
            and sorted(i for r in rows for i in r["ids"]) == sorted(c["id"] for c in train),
            "Training examples omitted/repeated",
        )
        require(all(len(r["ids"]) == 4 for r in rows), "Incorrect effective batch")
        for row in rows:
            require(
                np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm"]),
                "Nonfinite training values",
            )
            require(
                row["learning_rate"] == learning_rate(row["step"], 204, 11, 0.00003),
                "Learning-rate recipe changed",
            )
        for name in ["development.json", "shuffled-development.json"]:
            check_predictions(read(root / "epochs" / str(epoch + 1) / name), dev)
    for name in ["released-development.json", "released-shuffled-development.json"]:
        check_predictions(read(root / name), dev)
    donors = read(root / "direction-shuffle.json")
    from .update_loracle_train import shuffled_directions

    expected = shuffled_directions([{**c, "tokens": c["id"]} for c in dev])
    require(
        donors == [{"id": c["id"], "donor": c["direction_donor"]} for c in expected],
        "Direction permutation differs",
    )
    hashes = {
        **manifest["files_sha256"],
        **{
            name: file_hash(root / name)
            for name in [
                "plan.json",
                "data.json",
                "controller.json",
                "manifest.json",
                "source_snapshot.tar",
                "analysis-plan.json",
            ]
        },
    }
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "scope": "development_only",
            "finished_unix": time.time(),
            "source_commit": plan["source_commit"],
            "files_sha256": hashes,
            "collection_integrity_sha256": plan["collection_integrity_sha256"],
            "optimizer_steps": 204,
            "epochs": 3,
            "reserved_cases_accessed": 0,
        },
    )


def metrics(predictions, cases, effects):
    require(len(predictions) == len(cases), "Incorrect prediction count")
    require(all(p in LABELS for p in predictions), "Unexpected label")
    actual = [effects[c["id"]]["effect"]["label"] for c in cases]
    exact = np.asarray([p == y for p, y in zip(predictions, actual)])
    mask = np.asarray([not c["control"] for c in cases])
    require(mask.any() and (~mask).any(), "Both actual updates and zero controls are required")
    axes = {}
    for index, axis in enumerate(["format", "case", "order"]):
        eligible = [i for i, y in enumerate(actual) if mask[i] and y != NO_CHANGE]
        axes[axis] = {
            "n": len(eligible),
            "accuracy": float(
                np.mean(
                    [
                        predictions[i] != NO_CHANGE
                        and predictions[i].split("/")[index] == actual[i].split("/")[index]
                        for i in eligible
                    ]
                )
            )
            if eligible
            else None,
        }
    return {
        "actual_update_n": int(mask.sum()),
        "actual_update_correct": int(exact[mask].sum()),
        "actual_update_accuracy": float(exact[mask].mean()),
        "zero_n": int((~mask).sum()),
        "zero_correct": int(exact[~mask].sum()),
        "zero_accuracy": float(exact[~mask].mean()),
        "axes": axes,
        "above_effect_floor_fraction": float(
            np.mean(
                [
                    not effects[c["id"]]["effect"]["below_effect_floor"]
                    for c in cases
                    if not c["control"]
                ]
            )
        ),
    }


def select_epoch(epoch_metrics):
    require(set(epoch_metrics) == {1, 2, 3}, "All three epochs must finish before selection")
    return max(sorted(epoch_metrics), key=lambda e: epoch_metrics[e]["actual_update_accuracy"])


def paired_cluster_intervals(correct, cases, replicates=2000):
    """Crossed source-composition and initialization draws; common indices for every method."""
    active = [i for i, c in enumerate(cases) if not c["control"]]
    patterns = sorted({cases[i]["source_pattern"] for i in active})
    seeds = sorted({cases[i]["initialization_seed"] for i in active})
    cells = {
        (p, s): [
            i
            for i in active
            if cases[i]["source_pattern"] == p and cases[i]["initialization_seed"] == s
        ]
        for p in patterns
        for s in seeds
    }
    require(all(cells.values()), "Incomplete crossed composition/seed grid")
    rng = np.random.default_rng(2026101653)
    samples = {key: [] for key in correct}
    for _ in range(replicates):
        drawn_patterns = rng.choice(patterns, len(patterns))
        drawn_seeds = rng.choice(seeds, len(seeds))
        indices = []
        for pattern in drawn_patterns:
            for seed in drawn_seeds:
                cell = cells[pattern, seed]
                indices.extend(rng.choice(cell, len(cell)).tolist())
        for key, values in correct.items():
            samples[key].append(float(np.asarray(values)[indices].mean()))
    return {key: np.asarray(values) for key, values in samples.items()}


def analyze(root, output):
    require(not output.exists(), "Analysis output already exists; preserve earlier outputs")
    integrity = read(root / "integrity.json")
    require(integrity["passed"] and integrity["scope"] == "development_only", "Audit required")
    for path, expected in integrity["files_sha256"].items():
        require(file_hash(root / path) == expected, "Audited training output changed")
    plan = read(root / "plan.json")
    collection = Path(plan["collection_root"])
    require(
        file_hash(collection / "integrity.json") == integrity["collection_integrity_sha256"],
        "Audited collection changed",
    )
    data = read(collection / "data.json")
    cases = [c for c in data["cases"] if c["split"] == "development"]
    effects = {c["id"]: read(collection / "cases" / f"{c['id']}.json") for c in cases}
    predictions = {}
    for name, filename in [
        ("released", "released-development.json"),
        ("released_shuffled", "released-shuffled-development.json"),
    ]:
        predictions[name] = [r["prediction"] for r in read(root / filename)]
    for epoch in range(1, 4):
        for suffix, name in [("", "development.json"), ("_shuffled", "shuffled-development.json")]:
            predictions[f"epoch{epoch}{suffix}"] = [
                r["prediction"] for r in read(root / "epochs" / str(epoch) / name)
            ]
    simple = read(root / "simple-references.json")
    require(
        [r["id"] for r in simple["development"]] == [c["id"] for c in cases],
        "Simple reference grid differs",
    )
    for name in ["ridge", "majority", "shuffled_ridge"]:
        predictions[name] = [r[name] for r in simple["development"]]
    all_metrics = {name: metrics(values, cases, effects) for name, values in predictions.items()}
    selected = select_epoch({e: all_metrics[f"epoch{e}"] for e in range(1, 4)})
    labels = [effects[c["id"]]["effect"]["label"] for c in cases]
    correct = {
        key: np.asarray([a == b for a, b in zip(values, labels)], dtype=float)
        for key, values in predictions.items()
    }
    boot = paired_cluster_intervals(correct, cases)
    best, shuffled = f"epoch{selected}", f"epoch{selected}_shuffled"
    intervals = {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in boot.items()}
    differences = {
        name: {
            "point": all_metrics[best]["actual_update_accuracy"]
            - all_metrics[name]["actual_update_accuracy"],
            "descriptive_95_interval": np.quantile(
                boot[best] - boot[name], [0.025, 0.975]
            ).tolist(),
        }
        for name in ["released", "majority", "ridge", shuffled]
    }
    m = all_metrics[best]
    gates = {
        "accuracy_70": m["actual_update_accuracy"] >= 0.7,
        "released_advantage_20pp": differences["released"]["point"] >= 0.2,
        "effect_coverage_80": m["above_effect_floor_fraction"] >= 0.8,
        "zero_accuracy_90": m["zero_accuracy"] >= 0.9,
        "majority_advantage_10pp": differences["majority"]["point"] >= 0.1,
        "direction_dependence_10pp": differences[shuffled]["point"] >= 0.1,
    }
    output.mkdir(parents=True)
    summary = {
        "scope": "selected development results, not confirmation",
        "selected_epoch": selected,
        "metrics": all_metrics,
        "descriptive_95_intervals": intervals,
        "paired_differences": differences,
        "development_gates": gates,
        "all_development_gates_pass": all(gates.values()),
        "replicates": 2000,
        "reserved_test_read": False,
        "limitations": [
            "One fixed base checkpoint; new compositions of known operations only",
            "Four adapter initializations and eight development compositions; cluster uncertainty",
            "Development intervals do not remove bias from choosing an epoch on development data",
            "Finite-reference likelihood preference, not generated behavior or every update detail",
        ],
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "all-development-cases.json",
        [
            {
                **c,
                "measured": effects[c["id"]],
                "predictions": {key: values[i] for key, values in predictions.items()},
            }
            for i, c in enumerate(cases)
        ],
    )
    np.savez_compressed(root / "development-bootstrap.npz", **boot)
    # Preserve the exact executed analysis source with its digest.
    (output / "executed-analysis.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(
        output / "analysis-provenance.json",
        {
            "finished_unix": time.time(),
            "training_integrity_sha256": file_hash(root / "integrity.json"),
            "executed_source_sha256": file_hash(__file__),
            "replicate_seed": 2026101653,
        },
    )
    render(output, summary)


def render(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["released", "majority", "ridge", "epoch1", "epoch2", "epoch3"]
    values = [summary["metrics"][name]["actual_update_accuracy"] * 100 for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, values, color=["#858fa0"] * 3 + ["#226b8e"] * 3)
    ax.set(
        ylabel="Correct measured-effect category (%)",
        ylim=(0, 105),
        title="Held-out development compositions: all 64 actual updates",
    )
    ax.axhline(70, color="#555", linestyle="--", linewidth=1, label="Development target")
    for x, value in enumerate(values):
        ax.text(x, value + 1.5, f"{value:.1f}%", ha="center")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "development-accuracy.png", dpi=180)
    plt.close(fig)
    selected = summary["selected_epoch"]
    table = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{m['actual_update_correct']}/64</td>"
        f"<td>{m['zero_correct']}/8</td></tr>"
        for name, m in summary["metrics"].items()
    )
    examples = read(output / "all-development-cases.json")
    details = "".join(
        f"<details><summary>{html.escape(c['id'])}: "
        f"{html.escape(c['measured']['effect']['label'])}</summary><pre>"
        f"{html.escape(json.dumps(c, indent=2))}</pre></details>"
        for c in examples
    )
    body = f"""<!doctype html><meta charset='utf-8'><title>Single-update LoRAcle development</title>
    <style>body{{font:17px system-ui;max-width:1050px;margin:35px auto;
    padding:20px;line-height:1.55}}
    img{{max-width:100%}}table{{border-collapse:collapse}}
    td,th{{padding:8px;border-bottom:1px solid #ddd}}
    pre{{white-space:pre-wrap;font-size:12px}}details{{padding:8px}}</style>
    <h1>Can a trained LoRAcle interpret one optimizer step?</h1>
    <p>This is development evidence. The reserved test remains untouched. The selected checkpoint
    is epoch {selected}, chosen from the three fixed epochs by accuracy
    on all 64 nonzero updates.</p>
    <p>Each input contains the update weights, its physical norm and 33 response-pattern choices.
    The answer is the pattern whose likelihood increased most on four separate prompts, measured
    against all 32 possible reference patterns. No-change remains eligible for weak updates.</p>
    <img src='development-accuracy.png' alt='Accuracy of all comparison methods'>
    <table><tr><th>Method</th><th>Actual updates correct</th>
    <th>Zero controls correct</th></tr>{table}</table>
    <h2>Complete metrics and development criteria</h2>
    <pre>{html.escape(json.dumps(summary, indent=2))}</pre>
    <h2>Every held-out development example</h2>{details}
    <p>These tasks reuse one Qwen3-14B base. Compositions, words, exact prompts and adapter
    initializations are held out; individual operations are familiar. This does not establish
    unrestricted behavior prediction, unseen-operation generalization or a faithful account of
    every change encoded by the update.</p>"""
    (output / "report.html").write_text(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit-training", "analyze-development"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "audit-training":
        audit_training(args.root)
    else:
        if args.output is None:
            parser.error("--output is required")
        analyze(args.root, args.output)


if __name__ == "__main__":
    main()
