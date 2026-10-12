"""Read-only analysis of the frozen study outputs; never fits or applies updates."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import read_jsonl
from .study import atomic_json, file_hash
from .suite import crossed_bootstrap


def crossed_ratio(numerator, denominator, square_root=False, repeats=1000):
    """Ratio of sums with paired resampling of crossed seed and batch axes."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    if numerator.ndim != 2 or numerator.shape != denominator.shape:
        raise ValueError("Expected matching seed-by-batch arrays")
    rng = np.random.default_rng(20261011)
    seeds, batches = numerator.shape
    i = rng.integers(seeds, size=(repeats, seeds, 1))
    j = rng.integers(batches, size=(repeats, 1, batches))
    values = numerator[i, j].sum((1, 2)) / np.maximum(denominator[i, j].sum((1, 2)), 1e-30)
    point = numerator.sum() / max(denominator.sum(), 1e-30)
    if square_root:
        point, values = np.sqrt(point), np.sqrt(values)
    return {"value": float(point), "ci95": np.quantile(values, [0.025, 0.975]).tolist()}


def analyze_group(seed_records):
    result = {}
    for rate in sorted({row["learning_rate"] for row in seed_records[0]}):
        rows = [
            sorted((r for r in seed if r["learning_rate"] == rate), key=lambda r: r["batch_id"])
            for seed in seed_records
        ]
        for seed in rows[1:]:
            if [r["batch_id"] for r in seed] != [r["batch_id"] for r in rows[0]]:
                raise ValueError("Seed/batch alignment changed")
        actual = np.array([[r["observed"]["full"] for r in seed] for seed in rows])
        full = np.array([[r["predictions"]["full_gradient"] for r in seed] for seed in rows])
        predicted_atom = np.array([[r["selected_atom_prediction"] for r in s] for s in rows])
        target_indices = np.array([r["groups"].index(r["target_group"]) for r in rows[0]])
        target_mask = np.eye(actual.shape[-1], dtype=bool)[target_indices]
        controls, changes = {}, {}
        for name in ("atom_removed", "random_removed", "rescaled_full"):
            difference = actual - np.array([[r["observed"][name] for r in s] for s in rows])
            target = np.abs(difference[:, target_mask]).reshape(actual.shape[:2])
            off = np.sqrt((difference**2 * ~target_mask).sum(-1) / (~target_mask).sum(-1))
            changes[name] = (target, off, difference)
            controls[name] = {
                "mean_target_absolute_change": float(target.mean()),
                "mean_off_target_rms": float(off.mean()),
                "target_to_off_target_ratio": crossed_ratio(target, off),
            }
        atom_target, atom_off, atom_delta = changes["atom_removed"]
        paired = {}
        for name in ("random_removed", "rescaled_full"):
            target, off, _ = changes[name]
            # Difference between per-batch target shares avoids unstable tiny-effect ratios.
            atom_share = atom_target / np.maximum(atom_target + atom_off, 1e-30)
            control_share = target / np.maximum(target + off, 1e-30)
            paired[name] = {
                "target_effect_ratio": crossed_ratio(atom_target, target),
                "off_target_effect_ratio": crossed_ratio(atom_off, off),
                "difference_in_target_share": crossed_ratio(
                    atom_share - control_share, np.ones_like(atom_share)
                ),
            }
        families = {}
        for family in sorted({r["family_split"] for r in rows[0]}):
            selected = [[r for r in s if r["family_split"] == family] for s in rows]
            families[family] = {
                method: crossed_bootstrap(selected, method)
                for method in ("dictionary", "dictionary_transfer", "full_gradient", "svd")
            }
        error_by_category = ((full - actual) ** 2).sum((0, 1))
        error_by_batch = ((full - actual) ** 2).sum((0, 2))
        result[str(rate)] = {
            "controls": controls,
            "paired_controls": paired,
            "atom_removal_prediction_error": crossed_ratio(
                ((predicted_atom - atom_delta) ** 2).sum(-1),
                (atom_delta**2).sum(-1),
                square_root=True,
            ),
            "family_splits": families,
            "actual_change_rms": float(np.sqrt((actual**2).mean())),
            "full_gradient_error_share_by_category": dict(
                zip(rows[0][0]["groups"], (error_by_category / error_by_category.sum()).tolist())
            ),
            "largest_full_gradient_error_batches": [
                {
                    "batch_id": rows[0][i]["batch_id"],
                    "fraction_of_squared_error": float(error_by_batch[i] / error_by_batch.sum()),
                }
                for i in np.argsort(error_by_batch)[-5:][::-1]
            ],
        }
    return result


def analyze_loracle(root):
    rows = [json.loads(p.read_text()) for p in sorted((root / "answers").glob("*.json"))]
    pure = [r for r in rows if r["kind"] == "pure"]
    mixed = [r for r in rows if r["kind"] != "pure"]
    correct = {
        name: np.array([r["parsed"][name] == r["target_group"] for r in pure], dtype=float)
        for name in pure[0]["parsed"]
    }
    rng = np.random.default_rng(20261011)
    index = rng.integers(len(pure), size=(10000, len(pure)))
    difference = correct["upstream_compatible"] - correct["permuted"]
    return {
        "scope": "Source-category decoding; no effect-size prediction",
        "pure_batch_count": len(pure),
        "correct_counts": {k: int(v.sum()) for k, v in correct.items()},
        "compatible_minus_permuted_accuracy": {
            "value": float(difference.mean()),
            "paired_batch_bootstrap_ci95": np.quantile(
                difference[index].mean(1), [0.025, 0.975]
            ).tolist(),
            "compatible_only_correct": int((difference == 1).sum()),
            "permuted_only_correct": int((difference == -1).sum()),
        },
        "answer_distributions": {
            name: dict(Counter(r["parsed"][name] for r in pure)) for name in correct
        },
        "by_batch_size": {
            str(size): {
                name: float(
                    np.mean(
                        [
                            r["parsed"][name] == r["target_group"]
                            for r in pure
                            if r["batch_size"] == size
                        ]
                    )
                )
                for name in correct
            }
            for size in sorted({r["batch_size"] for r in pure})
        },
        "mixed_batches": {
            "count": len(mixed),
            "fraction_answer_is_present_source_category": {
                name: float(np.mean([r["parsed"][name] in r["source_groups"] for r in mixed]))
                for name in correct
            },
            "warning": "Many categories per mixed batch; membership is an easy descriptive check",
        },
    }


def analyze_study(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    controller = json.loads((root / "controller.json").read_text())
    groups, jobs, generations, atoms, sae = defaultdict(list), {}, {}, {}, {}
    for job in plan["jobs"]:
        directory = root / "jobs" / job["id"]
        manifest = json.loads((directory / "manifest.json").read_text())
        lock = json.loads((directory / "forecast_lock.json").read_text())
        if file_hash(directory / "predictions.jsonl") != lock["sha256"]:
            raise ValueError(f"Forecast integrity failed: {job['id']}")
        if manifest["status"] != "complete" or not manifest["checkpoint_restored"]:
            raise ValueError(f"Job incomplete: {job['id']}")
        records = read_jsonl(directory / "results.jsonl")
        if len(records) != 60 * len(plan["learning_rates"]):
            raise ValueError(f"Missing test records: {job['id']}")
        key = f"{job['model']['name']}/warmup-{job['warmup_steps']}"
        groups[key].append(records)
        checks = json.loads((directory / "finite_difference_checks.json").read_text())
        analytic = np.array([r["analytic"] for r in checks])
        jobs[job["id"]] = {
            "selected_dictionary": manifest["selected_dictionary"],
            "selection": json.loads((directory / "selection.json").read_text()),
            "mean_active_atoms": float(
                np.mean([np.count_nonzero(r["amplitudes"]) for r in records])
            ),
            "mean_raw_gradient_reconstruction_error": float(
                np.mean([r["reconstruction_relative_error"] for r in records])
            ),
            "finite_difference_relative_error": {
                radius: float(
                    np.linalg.norm(np.array([r[radius] for r in checks]) - analytic)
                    / max(np.linalg.norm(analytic), 1e-30)
                )
                for radius in checks[0]
                if radius not in ("atom", "analytic")
            },
            "source_commit": manifest["code_revision"],
        }
        generated = json.loads((directory / "generations.json").read_text())["results"]
        changes = []
        examples = []
        for batch in generated:
            for i, category in enumerate(batch["groups"]):
                base, full, removed = (
                    batch[name][i] for name in ("baseline", "full", "atom_removed")
                )
                changes.append((base != full, full != removed))
                if base != full and len(examples) < 2:
                    examples.append(
                        {
                            "batch_id": batch["batch_id"],
                            "probe_category": category,
                            "prompt": batch["prompts"][i],
                            "baseline": base,
                            "full": full,
                            "atom_removed": removed,
                        }
                    )
        generations[job["id"]] = {
            "count": len(changes),
            "full_changed_count": int(np.array(changes)[:, 0].sum()),
            "atom_removal_changed_count": int(np.array(changes)[:, 1].sum()),
            "examples": examples,
            "scope": "Exact text differences only; shared prompts, not independent accuracy trials",
        }
        dictionary_atoms = read_jsonl(directory / "atoms.jsonl")
        eligible = [r for r in dictionary_atoms if len(r["top_examples"]) >= 10]
        chosen = sorted(eligible, key=lambda r: r["raw_gradient_coherence"], reverse=True)[:3]
        atoms[job["id"]] = [
            {
                "atom": row["atom"],
                "coherence": row["raw_gradient_coherence"],
                "top_example_categories": dict(Counter(x["group"] for x in row["top_examples"])),
                "example_prompts": [x["prompt"] for x in row["top_examples"][:3]],
            }
            for row in chosen
        ]
    for path in sorted((root / "sae").glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        manifest = json.loads((path.parent / "manifest.json").read_text())
        sae[path.parent.name] = {
            "metrics": {
                name: {k: v for k, v in values.items() if k != "train_loss"}
                for name, values in metrics.items()
            },
            "train_chunks": manifest["train_chunks"],
            "validation_chunks": manifest["validation_chunks"],
            "validation_documents": len(manifest["validation_example_ids"]),
            "checkpoint_restored": manifest["checkpoint_restored"],
        }
    result = {
        "status": controller["status"],
        "wall_clock_hours": (controller["updated_unix"] - plan["created_unix"]) / 3600,
        "forecast_jobs": len(jobs),
        "sae_comparisons": len(sae),
        "full_update_cases": sum(len(x) for seeds in groups.values() for x in seeds),
        "analysis_scope": "Post-run descriptive analysis, no protocol selection or further fitting",
        "methods": json.loads((root / "aggregate.json").read_text())["results"],
        "groups": {name: analyze_group(rows) for name, rows in groups.items()},
        "jobs": jobs,
        "generations": generations,
        "highest_coherence_atoms": atoms,
        "sae": sae,
        "loracle": analyze_loracle(root / "loracle"),
    }
    atomic_json(root / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze_study(args.root)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "wall_clock_hours", "forecast_jobs", "sae_comparisons")
            }
        )
    )


if __name__ == "__main__":
    main()
