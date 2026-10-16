"""Exact fixed-calibration effects of native updates, without evaluation-prompt access."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .loracle_toy_suite import initialize_like
from .loracle_trajectory_toy import (
    fingerprint,
    load_target,
    read,
    restore_modules,
    snapshot_modules,
    status,
)
from .study import atomic_json, file_hash
from .update_loracle_analysis import require
from .update_loracle_collect import guard, losses
from .update_loracle_data import PATTERNS
from .update_loracle_physical_profile import profile_label, source_keys
from .update_loracle_sensitivity import calibration_rows, centered_profile, frozen_versions
from .update_loracle_train import shuffled_directions

RECIPE = "exact_calibration_v5"
CHUNK = 4


def validate_case_grid(cases):
    fields = {"id", "split", "control", "initialization_seed", "physical_update_norm"}
    require(
        len(cases) == len({c["id"] for c in cases}) == 344, "Complete unique case grid required"
    )
    require(
        all(set(c) == fields for c in cases),
        "Prediction metadata must exclude examples and effects",
    )
    require({c["split"] for c in cases} == {"train", "development"}, "No test cases allowed")
    seed_sets = []
    for split, total, zeros in (("train", 272, 16), ("development", 72, 8)):
        group = [c for c in cases if c["split"] == split]
        seeds = {c["initialization_seed"] for c in group}
        require(
            len(group) == total and sum(c["control"] for c in group) == zeros and len(seeds) == 4,
            "Wrong split/init/control grid",
        )
        seed_sets.append(seeds)
    require(not seed_sets[0] & seed_sets[1], "Train and development initializations overlap")
    require(
        all(
            np.isfinite(c["physical_update_norm"])
            and c["physical_update_norm"] >= 0
            and (c["physical_update_norm"] == 0) == c["control"]
            for c in cases
        ),
        "Invalid physical norms or controls",
    )


@torch.no_grad()
def calibration_scores(model, tokenizer, calibration):
    """Same document order, completion EOS and chunk size as the frozen linear predictor."""
    result = []
    for pattern in PATTERNS:
        rows = calibration_rows(calibration, pattern)
        values = []
        for start in range(0, 8, CHUNK):
            values.extend(
                losses(model, tokenizer, rows[start : start + CHUNK]).double().cpu().tolist()
            )
        result.append(values)
    scores = np.asarray(result, dtype=np.float64).T
    require(scores.shape == (8, 32) and np.isfinite(scores).all(), "Incomplete calibration scores")
    return scores


def effect_profile(before, after):
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    require(before.shape == after.shape == (8, 32), "Calibration grid changed")
    require(np.isfinite(before).all() and np.isfinite(after).all(), "Nonfinite calibration score")
    raw = (after - before).mean(axis=0)
    profile = centered_profile(raw)
    return {
        "ce_change": raw.tolist(),
        "profile": profile.tolist(),
        "prediction": profile_label(profile),
    }


def donor_assignments(cases):
    """Use the original derangement independently within train and development."""
    result = {}
    for split in ("train", "development"):
        values = [{**c, "tokens": c["id"]} for c in cases if c["split"] == split]
        for row in shuffled_directions(values):
            result[row["id"]] = row["direction_donor"]
    require(set(result) == {c["id"] for c in cases}, "Incomplete donor assignment")
    return result


def scaled_native(initial, updated, scale):
    """Scale the actual donor B in FP32, keeping its own A; never scale an exact loss profile."""
    require(
        set(updated) == set(initial) and np.isfinite(scale) and scale >= 0, "Invalid native input"
    )
    result = {}
    for key, (a0, b0) in initial.items():
        a, b = updated[key]
        require(
            a.dtype == b.dtype == a0.dtype == b0.dtype == torch.float32
            and torch.equal(a, a0)
            and not torch.count_nonzero(b0)
            and b.shape == b0.shape
            and torch.isfinite(a).all()
            and torch.isfinite(b).all(),
            "Expected this initialization's unchanged A and zero initial B",
        )
        scaled = b * float(scale)
        require(torch.isfinite(scaled).all(), "Scaled native factors are nonfinite")
        result[key] = (a, scaled)
    return result


def physical_norm(snapshot):
    total = 0.0
    for a, b in snapshot.values():
        a, b = a.double(), b.double()
        total += float(((a @ a.T) * (b.T @ b)).sum()) * 4.0
    require(total >= -1e-12 and np.isfinite(total), "Invalid native physical norm")
    return float(np.sqrt(max(total, 0.0)))


def score_and_restore(model, tokenizer, calibration, initial, updated, baseline):
    initial_hash = fingerprint(initial)
    versions = frozen_versions(model)
    restore_modules(model, initial)
    try:
        restore_modules(model, updated)
        scores = calibration_scores(model, tokenizer, calibration)
    finally:
        restore_modules(model, initial)
    require(fingerprint(snapshot_modules(model)) == initial_hash, "Adapter restore differs")
    require(frozen_versions(model) == versions, "Frozen base changed")
    with torch.no_grad():
        restored = losses(model, tokenizer, calibration_rows(calibration, PATTERNS[0])[:4])
    require(
        np.array_equal(restored.double().cpu().numpy(), baseline[:4, 0]),
        "Baseline score restore differs",
    )
    return scores


def preflight(model, tokenizer, calibration, output):
    require(not output.exists(), "Preflight already attempted")
    initial = initialize_like(snapshot_modules(model), 2026101601)
    require(set(initial) == set(source_keys()), "Native module grid changed")
    require(
        sum(t.numel() for pair in initial.values() for t in pair) == 64225280, "Native size changed"
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    require(
        len(trainable) == 560 and sum(p.numel() for p in trainable) == 64225280,
        "Trainable target architecture changed",
    )
    restore_modules(model, initial)
    baseline = calibration_scores(model, tokenizer, calibration)
    # Fixed synthetic perturbation independent of all scientific snapshots and outcomes.
    rng = torch.Generator().manual_seed(2026101711)
    synthetic = {
        key: (a, torch.randn(b.shape, generator=rng) * 1e-5) for key, (a, b) in initial.items()
    }
    scores = score_and_restore(model, tokenizer, calibration, initial, synthetic, baseline)
    repeated = calibration_scores(model, tokenizer, calibration)
    require(np.array_equal(repeated, baseline), "Full calibration score restoration failed")
    require(not np.array_equal(scores, baseline), "Synthetic perturbation has no numerical effect")
    norm = physical_norm(synthetic)
    scaled = scaled_native(initial, synthetic, 1.75)
    residual = abs(physical_norm(scaled) / (1.75 * norm) - 1)
    require(residual < 5e-6, "Physical norm scaling failed")
    atomic_json(
        output,
        {
            "passed": True,
            "finished_unix": time.time(),
            "training_only": True,
            "synthetic_only_no_target_updates": True,
            "modules": 280,
            "trainable_tensors": 560,
            "parameters": 64225280,
            "all_256_scores_finite": True,
            "bit_exact_adapter_and_full_score_restore": True,
            "frozen_base_unchanged": True,
            "physical_norm_scale_relative_error": residual,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def run(root):
    plan = guard(root)
    require(
        plan["recipe"] == RECIPE and not (root / "started.json").exists(),
        "Wrong recipe or already attempted; never resume",
    )
    atomic_json(root / "started.json", {"started_unix": time.time()})
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Input changed: {path}")
    data = read(root / "data.json")
    cases, calibration = data["cases"], data["calibration"]
    validate_case_grid(cases)
    require(
        len(cases) == 344 and {c["split"] for c in cases} == {"train", "development"},
        "Only complete original development grid allowed",
    )
    collection = Path(plan["collection_root"])
    initials = {
        seed: torch.load(
            collection / "snapshots" / f"initial-{seed}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for seed in sorted({c["initialization_seed"] for c in cases})
    }
    require(len(initials) == 8, "Wrong initialization grid")
    model, tokenizer = load_target()
    model.eval()
    preflight(model, tokenizer, calibration, root / "preflight.json")
    base_versions = frozen_versions(model)
    baselines = {}
    for seed, initial in initials.items():
        guard(root)
        restore_modules(model, initial)
        baseline = calibration_scores(model, tokenizer, calibration)
        require(
            fingerprint(snapshot_modules(model)) == fingerprint(initial), "Baseline changed adapter"
        )
        require(frozen_versions(model) == base_versions, "Baseline scoring changed frozen base")
        if baselines:
            require(
                np.array_equal(baseline, next(iter(baselines.values()))), "Zero-B baselines differ"
            )
        baselines[seed] = baseline
    atomic_json(
        root / "baselines.json",
        {
            "finished_unix": time.time(),
            "scores": {str(k): v.tolist() for k, v in baselines.items()},
            "all_zero_B_scores_bit_exact": True,
        },
    )
    donors = donor_assignments(cases)
    lookup = {c["id"]: c for c in cases}
    output_dir = root / "scores"
    output_dir.mkdir()
    inventory, predictions = [], []
    for index, case in enumerate(cases):
        profiles = {}
        for kind, donor_id in (("own", case["id"]), ("shuffled", donors[case["id"]])):
            guard(root)
            donor = lookup[donor_id]
            initial = initials[donor["initialization_seed"]]
            path = collection / "snapshots" / f"{donor_id}.pt"
            require(
                file_hash(path) == plan["input_files_sha256"][str(path)], "Native snapshot changed"
            )
            original = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            norm, donor_norm = case["physical_update_norm"], donor["physical_update_norm"]
            require((norm > 0) == (donor_norm > 0), "Zero/actual shuffle changed")
            scale = norm / donor_norm if donor_norm else 1.0
            updated = scaled_native(initial, original, scale)
            actual_norm = physical_norm(updated)
            require(
                np.isclose(actual_norm, norm, rtol=5e-6, atol=1e-10),
                "Scaled physical update norm differs",
            )
            baseline = baselines[donor["initialization_seed"]]
            after = score_and_restore(model, tokenizer, calibration, initial, updated, baseline)
            require(frozen_versions(model) == base_versions, "Frozen base changed across cases")
            if case["control"]:
                require(np.array_equal(after, baseline), "Zero update changed scores")
            profile = effect_profile(baseline, after)
            output = output_dir / f"{case['id']}-{kind}.json"
            atomic_json(
                output,
                {
                    "id": case["id"],
                    "kind": kind,
                    "direction_donor": donor_id,
                    "initialization_seed": donor["initialization_seed"],
                    "donor_scale": scale,
                    "physical_update_norm": norm,
                    "actual_scaled_norm": actual_norm,
                    "snapshot_sha256": file_hash(path),
                    "after_ce": after.tolist(),
                    **profile,
                    "finished_unix": time.time(),
                    "bit_exact_restore": True,
                    "frozen_base_unchanged": True,
                },
            )
            inventory.append({"path": str(output.relative_to(root)), "sha256": file_hash(output)})
            profiles[kind] = profile
            del original, updated
            status(root, "exact_calibration_scores", index * 2 + (1 if kind == "own" else 2), 688)
        predictions.append(
            {
                **case,
                **profiles["own"],
                "shuffled_profile": profiles["shuffled"]["profile"],
                "shuffled_prediction": profiles["shuffled"]["prediction"],
                "direction_donor": donors[case["id"]],
            }
        )
    atomic_json(root / "predictions.json", predictions)
    files = {r["path"]: r["sha256"] for r in inventory}
    files.update(
        {
            name: file_hash(root / name)
            for name in ("started.json", "preflight.json", "baselines.json", "predictions.json")
        }
    )
    atomic_json(
        root / "predictions-lock.json",
        {
            "locked_unix": time.time(),
            "files_sha256": files,
            "cases": 344,
            "scored_adapter_conditions": 688,
            "reserved_cases_accessed": 0,
            "evaluation_probes_accessed_for_prediction": False,
            "target_optimizer_updates": 0,
        },
    )
    files["predictions-lock.json"] = file_hash(root / "predictions-lock.json")
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "recipe": RECIPE,
            "finished_unix": time.time(),
            "source_commit": plan["source_commit"],
            "plan_sha256": file_hash(root / "plan.json"),
            "files_sha256": files,
            "cases": 344,
            "scored_adapter_conditions": 688,
            "calibration_questions": 8,
            "patterns": 32,
            "initializations": 8,
            "target_optimizer_updates": 0,
            "reserved_cases_accessed": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026101711)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args.root)
