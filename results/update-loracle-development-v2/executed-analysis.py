"""Complete-run audit and joint-category development analysis for component questions."""

import argparse
import hashlib
import html
import json
import random
import tarfile
import time
from pathlib import Path

import numpy as np

from .loracle_trajectory_toy import read
from .study import atomic_json, file_hash
from .update_loracle_analysis import alive, metrics, paired_cluster_intervals, require, select_epoch
from .update_loracle_factorized import (
    HEAD_LABELS,
    TRAINING_SEED,
    join_components,
    question_views,
    training_instances,
    training_majorities,
)
from .update_loracle_train import OPTION_SYMBOLS, learning_rate, shuffled_directions


def completion_barrier(controller):
    require(
        controller["status"] == "complete" and not alive(controller["controller_pid"]),
        "Normally complete and dead controller required before reading scientific outputs",
    )
    attempts = controller["attempts"]
    require(
        len(attempts) == 1
        and not alive(attempts[0]["pid"])
        and attempts[0]["returncode"] == 0
        and controller["worker_pid"] is None,
        "Exactly one successful dead training worker required",
    )


def cases_with_effects(collection, split):
    data = read(collection / "data.json")
    cases = [c for c in data["cases"] if c["split"] == split]
    effects = {c["id"]: read(collection / "cases" / f"{c['id']}.json") for c in cases}
    enriched = [
        {
            **c,
            "label": effects[c["id"]]["effect"]["label"],
            "physical_update_norm": effects[c["id"]]["physical_update_norm"],
            "tokens": collection / "tokens" / f"{c['id']}.pt",
        }
        for c in cases
    ]
    return enriched, effects


def check_predictions(rows, cases):
    require([r["id"] for r in rows] == [c["id"] for c in cases], "Prediction case grid differs")
    for row, case in zip(rows, cases, strict=True):
        require(set(row["heads"]) == set(HEAD_LABELS), "All four questions are required")
        for view in question_views(case):
            answer = row["heads"][view["head"]]
            logits = np.asarray(answer["choice_logits"])
            full = np.asarray(answer["full_33_choice_logits"])
            n = len(view["options"])
            require(
                logits.shape == (n,)
                and full.shape == (33,)
                and np.isfinite(full).all()
                and np.array_equal(logits, full[:n]),
                "Invalid head/full choice scores",
            )
            choice = OPTION_SYMBOLS[int(logits.argmax())]
            require(
                answer["choice_id"] == choice and answer["prediction"] == view["options"][choice],
                "Stored component prediction differs from valid-ID scoring",
            )
            require(
                4480 < answer["input_tokens"] <= 6144
                and answer["hook_calls"] >= 1
                and answer["loss"] is None,
                "Invalid evaluation execution metadata",
            )
        require(
            row["prediction"]
            == join_components({h: a["prediction"] for h, a in row["heads"].items()}),
            "Joint prediction differs",
        )


def check_history(history, instances):
    require([r["step"] for r in history] == list(range(780)), "Missing optimizer steps")
    expected = []
    for epoch in range(3):
        ordered = list(instances)
        random.Random(TRAINING_SEED + epoch).shuffle(ordered)
        expected.extend((epoch, ordered[i : i + 4]) for i in range(0, len(ordered), 4))
    require(len(expected) == 780, "Training question count differs")
    for row, (epoch, group) in zip(history, expected, strict=True):
        require(
            row["epoch"] == epoch and row["instances"] == [r["id"] for r in group],
            "Training order or complete epoch inventory differs",
        )
        require(row["weights"] == [r["weight"] for r in group], "Training class weights differ")
        losses = np.asarray(row["unweighted_losses"])
        require(
            losses.shape == (4,)
            and np.isfinite(losses).all()
            and (losses >= 0).all()
            and np.isfinite(row["gradient_norm"])
            and row["gradient_norm"] >= 0,
            "Invalid training loss/gradient values",
        )
        require(
            np.isclose(
                row["weighted_loss"], np.mean(losses * row["weights"]), rtol=1e-12, atol=1e-12
            ),
            "Weighted objective differs",
        )
        require(
            row["learning_rate"] == learning_rate(row["step"], 780, 39, 0.00003),
            "Training learning-rate recipe differs",
        )


def audit(root):
    controller = read(root / "controller.json")
    completion_barrier(controller)
    plan, manifest = read(root / "plan.json"), read(root / "manifest.json")
    require(
        file_hash(root / "plan.json") == controller["plan_sha256"] == manifest["plan_sha256"],
        "Plan hash changed",
    )
    attempt = controller["attempts"][0]
    require(
        plan["recipe"] == "balanced_factorized_v2"
        and plan["max_hours"] == 12
        and controller["deadline_unix"] == controller["started_unix"] + 43200
        and controller["started_unix"]
        <= attempt["started_unix"]
        <= manifest["finished_unix"]
        <= attempt["finished_unix"]
        <= controller["deadline_unix"],
        "Original timeline differs",
    )
    expected_manifest = {
        "status": "complete",
        "optimizer_steps": 780,
        "epochs": 3,
        "training_cases": 272,
        "training_questions": 1040,
        "development_cases": 72,
        "questions_per_evaluation_case": 4,
        "reserved_cases_accessed": 0,
        "source_commit": plan["source_commit"],
    }
    require(all(manifest[k] == v for k, v in expected_manifest.items()), "Incomplete output grid")
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Frozen input changed: {path}")
    for name, key in [("data.json", "data_sha256"), ("analysis-plan.json", "analysis_plan_sha256")]:
        require(file_hash(root / name) == plan[key], f"Frozen {name} changed")
    with tarfile.open(root / "source_snapshot.tar") as tar:
        require(
            set(tar.getnames()) == set(plan["source_files_sha256"]), "Source archive grid differs"
        )
        for path, expected in plan["source_files_sha256"].items():
            require(
                hashlib.sha256(tar.extractfile(path).read()).hexdigest() == expected,
                "Archived source hash differs",
            )
    collection, prior = Path(plan["collection_root"]), Path(plan["prior_training_root"])
    require(
        file_hash(collection / "integrity.json") == plan["collection_integrity_sha256"],
        "Collection integrity changed",
    )
    old = read(collection / "integrity.json")
    require(old["passed"] and old["cases"] == 344, "Complete collection audit required")
    for name, expected in old["files_sha256"].items():
        require(
            file_hash(collection / name) == expected, f"Audited collection output changed: {name}"
        )
    train, _ = cases_with_effects(collection, "train")
    dev, _ = cases_with_effects(collection, "development")
    require(len(train) == 272 and len(dev) == 72, "Case counts differ")
    instances, balance = training_instances(train)
    require(read(root / "balance.json") == balance, "Class balance is not training-only formula")
    require(
        read(root / "training-majorities.json") == training_majorities(train, balance),
        "Majorities differ from original training counts",
    )
    require(
        file_hash(root / "simple-references.json") == file_hash(prior / "simple-references.json"),
        "Existing frozen Ridge reference differs",
    )
    preflight = read(root / "preflight.json")
    require(
        all(
            preflight[k] is True
            for k in [
                "passed",
                "training_only",
                "all_gradients_finite",
                "frozen_base_unchanged",
                "bit_exact_score_restore",
                "bit_exact_adapter_restore",
            ]
        )
        and preflight["trainable_tensors"] == 560
        and preflight["trainable_parameters"] == 1027604480
        and preflight["heads"] == list(HEAD_LABELS)
        and len(preflight["hook_calls"]) == 4
        and min(preflight["hook_calls"]) >= 2
        and attempt["started_unix"] < preflight["finished_unix"] < manifest["finished_unix"],
        "Actual-model preflight did not pass",
    )
    check_history(read(root / "training-history.json"), instances)
    expected_files = {
        "preflight.json",
        "balance.json",
        "training-majorities.json",
        "simple-references.json",
        "released-development.json",
        "released-shuffled-development.json",
        "direction-shuffle.json",
        "training-history.json",
    }
    from safetensors import safe_open

    for epoch in range(1, 4):
        directory = root / "epochs" / str(epoch)
        files = {
            "adapter_config.json",
            "adapter_model.safetensors",
            "development.json",
            "shuffled-development.json",
        }
        require({p.name for p in directory.iterdir()} == files, "Saved epoch inventory differs")
        expected_files.update(f"epochs/{epoch}/{f}" for f in files)
        config = read(directory / "adapter_config.json")
        require(
            config["r"] == 256 and config["lora_alpha"] == 32 and config["use_rslora"],
            "Interpreter adapter architecture differs",
        )
        with safe_open(
            directory / "adapter_model.safetensors", framework="pt", device="cpu"
        ) as weights:
            keys = list(weights.keys())
            require(len(keys) == 560 and all("lora_" in k for k in keys), "Adapter tensors differ")
            require(
                all(weights.get_slice(k).get_dtype() == "F32" for k in keys),
                "Adapter precision differs",
            )
            require(
                sum(int(np.prod(weights.get_slice(k).get_shape())) for k in keys) == 1027604480,
                "Adapter parameter count differs",
            )
        for name in ["development.json", "shuffled-development.json"]:
            check_predictions(read(directory / name), dev)
    require(set(manifest["files_sha256"]) == expected_files, "Manifest omits/unexpected output")
    for name, expected in manifest["files_sha256"].items():
        require(file_hash(root / name) == expected, f"Scientific output changed: {name}")
    for name in ["released-development.json", "released-shuffled-development.json"]:
        check_predictions(read(root / name), dev)
    require(
        read(root / "direction-shuffle.json")
        == [{"id": c["id"], "donor": c["direction_donor"]} for c in shuffled_directions(dev)],
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
                "analysis-plan.json",
                "source_snapshot.tar",
                "preparation.json",
            ]
        },
    }
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "scope": "development_only",
            "recipe": plan["recipe"],
            "finished_unix": time.time(),
            "source_commit": plan["source_commit"],
            "files_sha256": hashes,
            "collection_integrity_sha256": plan["collection_integrity_sha256"],
            "optimizer_steps": 780,
            "epochs": 3,
            "reserved_cases_accessed": 0,
        },
    )


def analyze(root, output):
    require(not output.exists(), "Preserve existing analysis")
    integrity = read(root / "integrity.json")
    require(
        integrity["passed"] and integrity["recipe"] == "balanced_factorized_v2",
        "Complete v2 audit required",
    )
    for name, expected in integrity["files_sha256"].items():
        require(file_hash(root / name) == expected, f"Audited output changed: {name}")
    plan = read(root / "plan.json")
    collection = Path(plan["collection_root"])
    require(
        file_hash(collection / "integrity.json") == integrity["collection_integrity_sha256"],
        "Collection audit changed",
    )
    cases, effects = cases_with_effects(collection, "development")
    predictions, raw = {}, {}
    for name in ["released", "released-shuffled"]:
        raw[name.replace("-", "_")] = read(root / f"{name}-development.json")
    for epoch in range(1, 4):
        for suffix, filename in [
            ("", "development.json"),
            ("_shuffled", "shuffled-development.json"),
        ]:
            raw[f"epoch{epoch}{suffix}"] = read(root / "epochs" / str(epoch) / filename)
    predictions.update({name: [r["prediction"] for r in rows] for name, rows in raw.items()})
    simple = read(root / "simple-references.json")
    require(
        [r["id"] for r in simple["development"]] == [c["id"] for c in cases], "Ridge grid differs"
    )
    for name in ["ridge", "shuffled_ridge"]:
        predictions[name] = [r[name] for r in simple["development"]]
    majority = read(root / "training-majorities.json")
    for name in ["flat", "factorized"]:
        predictions[f"{name}_majority"] = [majority[name]] * len(cases)
    all_metrics = {name: metrics(values, cases, effects) for name, values in predictions.items()}
    selected = select_epoch({e: all_metrics[f"epoch{e}"] for e in range(1, 4)})
    best, shuffled = f"epoch{selected}", f"epoch{selected}_shuffled"
    labels = [effects[c["id"]]["effect"]["label"] for c in cases]
    correct = {k: np.asarray([a == b for a, b in zip(v, labels)]) for k, v in predictions.items()}
    boot = paired_cluster_intervals(correct, cases)
    intervals = {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in boot.items()}
    differences = {
        name: {
            "point": all_metrics[best]["actual_update_accuracy"]
            - all_metrics[name]["actual_update_accuracy"],
            "descriptive_95_interval": np.quantile(
                boot[best] - boot[name], [0.025, 0.975]
            ).tolist(),
        }
        for name in ["released", "flat_majority", "factorized_majority", "ridge", shuffled]
    }
    m = all_metrics[best]
    gates = {
        "accuracy_70": m["actual_update_accuracy"] >= 0.7,
        "released_advantage_20pp": differences["released"]["point"] >= 0.2,
        "effect_coverage_80": m["above_effect_floor_fraction"] >= 0.8,
        "zero_accuracy_90": m["zero_accuracy"] >= 0.9,
        "flat_majority_advantage_10pp": differences["flat_majority"]["point"] >= 0.1,
        "factorized_majority_advantage_10pp": differences["factorized_majority"]["point"] >= 0.1,
        "direction_dependence_10pp": differences[shuffled]["point"] >= 0.1,
    }
    strata = {}
    for field in ["source_pattern", "initialization_seed"]:
        strata[field] = {}
        for value in sorted({c[field] for c in cases if not c["control"]}):
            ids = [i for i, c in enumerate(cases) if not c["control"] and c[field] == value]
            strata[field][str(value)] = {
                name: {"n": len(ids), "correct": int(v[ids].sum())} for name, v in correct.items()
            }
    summary = {
        "scope": "selected development, not confirmation",
        "recipe": "balanced_factorized_v2",
        "selected_epoch": selected,
        "metrics": all_metrics,
        "descriptive_95_intervals": intervals,
        "paired_differences": differences,
        "development_gates": gates,
        "all_development_gates_pass": all(gates.values()),
        "strata": strata,
        "replicates": 2000,
        "reserved_test_read": False,
        "limitations": [
            "Development reuse motivated this recipe; intervals do not correct epoch selection",
            "One fixed Qwen3-14B base and compositions of familiar operations",
            "Finite-reference likelihood winner, not generated behavior or all update details",
            "Matched four-question baseline required; old flat-question score is historical only",
        ],
    }
    output.mkdir(parents=True)
    atomic_json(output / "summary.json", summary)
    records = []
    for i, c in enumerate(cases):
        measured = {k: v for k, v in effects[c["id"]].items() if k != "token_audit"}
        # The full audited per-module token audit stays in the original collection.
        records.append(
            {
                **{k: v for k, v in c.items() if k != "tokens"},
                "measured": measured,
                "questions": [
                    {k: v for k, v in q.items() if k != "tokens"} for q in question_views(c)
                ],
                "predictions": {k: v[i] for k, v in predictions.items()},
                "interpreter_answers": {k: v[i] for k, v in raw.items()},
            }
        )
    atomic_json(output / "all-development-cases.json", records)
    np.savez_compressed(root / "development-bootstrap.npz", **boot)
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
    render(output, summary, records)


def render(output, summary, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [
        "released",
        "flat_majority",
        "factorized_majority",
        "ridge",
        "epoch1",
        "epoch2",
        "epoch3",
    ]
    values = [summary["metrics"][n]["actual_update_accuracy"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), values, color=["#8796a2"] * 4 + ["#21658c"] * 3)
    ax.set(
        xticks=range(len(names)),
        xticklabels=[n.replace("_", "\n") for n in names],
        ylabel="Exact full response pattern correct (%)",
        ylim=(0, 105),
        title="All 64 one-step updates on development compositions",
    )
    ax.axhline(70, color="#555", linestyle="--", label="Fixed development criterion")
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "development-accuracy.png", dpi=180)
    plt.close(fig)
    table = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{m['actual_update_correct']}/64</td>"
        f"<td>{m['zero_correct']}/8</td></tr>"
        for n, m in summary["metrics"].items()
    )
    details = "".join(
        f"<details><summary>{html.escape(c['id'])}: {html.escape(c['label'])}</summary>"
        f"<pre>{html.escape(json.dumps(c, indent=2))}</pre></details>"
        for c in records
    )
    body = f"""<!doctype html><meta charset='utf-8'><title>Component questions for one-update
effects</title>
    <style>body{{font:17px system-ui;max-width:1100px;margin:35px
auto;padding:20px;line-height:1.55}}
    img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid
#ddd}}table{{border-collapse:collapse}}
    pre{{white-space:pre-wrap;font-size:12px}}details{{padding:8px}}</style>
    <h1>Can component questions improve single-update interpretation?</h1>
    <p>The interpreter sees update directions and their physical norm. It answers four questions
    about detectable change, formatting, letter case and order. These answers form one complete
    prediction. That complete prediction is the primary score. Unchanged LoRAcle gets the same
questions.</p>
    <p>Training gives each component and each class equal total weight. Targets come from measured
    relative likelihood changes on separate probe prompts. All actual updates and zero controls
remain.</p>
    <p>This is development, chosen after the first recipe failed. Epoch {summary["selected_epoch"]}
    was selected from all three fixed epochs. Reserved tests remain untouched.</p>
    <img src='development-accuracy.png' alt='All fixed epochs and baseline full-category accuracy'>
    <table><tr><th>Method</th><th>Actual updates</th><th>Zero controls</th></tr>{table}</table>
    <h2>Complete results and unchanged criteria</h2><pre>{
        html.escape(json.dumps(summary, indent=2))
    }</pre>
    <h2>All cases, prompts, measured effects and model answers</h2>{details}"""
    (output / "report.html").write_text(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.root)
    else:
        require(args.output is not None, "--output required")
        analyze(args.root, args.output)
