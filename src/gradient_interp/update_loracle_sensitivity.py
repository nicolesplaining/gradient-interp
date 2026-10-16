"""Fixed training-only calibration gradients describe native one-step updates."""

import argparse
import re
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
    status,
)
from .study import atomic_json, file_hash
from .update_loracle_analysis import require
from .update_loracle_collect import guard, losses
from .update_loracle_data import PATTERNS, render
from .update_loracle_physical_profile import profile_label, source_keys
from .update_loracle_train import shuffled_directions

CALIBRATION_N = 8
CHUNK = 4


def select_calibration(data):
    """The first eight distinct TRAINING probes, selected without any outcomes."""
    rows, seen = [], set()
    for case in data["cases"]:
        if case["split"] != "train":
            continue
        for probe in case["probes"]:
            if probe["prompt"] in seen:
                continue
            rows.append({"prompt": probe["prompt"], "values": list(probe["values"])})
            seen.add(probe["prompt"])
            if len(rows) == CALIBRATION_N:
                return rows
    raise ValueError("Fewer than eight distinct training probes")


def calibration_rows(calibration, pattern):
    require(len(calibration) == CALIBRATION_N, "Exactly eight calibration probes required")
    require(tuple(pattern) in PATTERNS, "Unknown candidate pattern")
    return [
        {"prompt": row["prompt"], "answer": render(row["values"], pattern)} for row in calibration
    ]


def module_parameters(model):
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            match = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)$", name)
            require(match is not None, "Unexpected native module")
            key = f"{int(match[1]):02d}:{match[2]}"
            result[key] = (module.lora_A["default"].weight, module.lora_B["default"].weight)
    require(set(result) == set(source_keys()), "All280 native modules required")
    return result


def frozen_versions(model):
    return {name: p._version for name, p in model.named_parameters() if not p.requires_grad}


def gradient_b(model, tokenizer, rows):
    """Equal document means, including original EOS, accumulated over all eight probes."""
    require(len(rows) == CALIBRATION_N, "Wrong calibration count")
    model.eval()
    params = module_parameters(model)
    require(
        all(
            a.dtype == b.dtype == torch.float32 and not torch.count_nonzero(b)
            for a, b in params.values()
        ),
        "Calibration requires exact zero-B native adapter",
    )
    model.zero_grad(set_to_none=True)
    scores = []
    for start in range(0, len(rows), CHUNK):
        values = losses(model, tokenizer, rows[start : start + CHUNK])
        require(torch.isfinite(values).all(), "Nonfinite calibration loss")
        (values.sum() / len(rows)).backward()
        scores.extend(values.detach().double().cpu().tolist())
        del values
    result = {}
    for key, (a, b) in params.items():
        require(
            a.grad is not None
            and b.grad is not None
            and torch.isfinite(a.grad).all()
            and torch.isfinite(b.grad).all(),
            "Missing or nonfinite calibration gradients",
        )
        require(not torch.count_nonzero(a.grad), "A gradient must be zero at B=0")
        result[key] = b.grad.detach().cpu().clone()
    model.zero_grad(set_to_none=True)
    return result, scores


def contract(gradients, snapshots, initial, device="cuda"):
    """The adapter scaling is already in dL/dB; never multiply it a second time."""
    require(set(gradients) == set(initial) and bool(snapshots), "Incomplete contraction")
    result = torch.zeros(len(snapshots), device=device, dtype=torch.float64)
    for key in sorted(initial):
        a0, b0 = initial[key]
        g = gradients[key]
        require(
            not torch.count_nonzero(b0)
            and g.shape == b0.shape
            and g.dtype == torch.float32
            and torch.isfinite(g).all(),
            "Wrong initialization or gradient",
        )
        for snap in snapshots:
            require(
                set(snap) == set(initial)
                and torch.equal(snap[key][0], a0)
                and snap[key][1].shape == b0.shape
                and snap[key][1].dtype == torch.float32
                and torch.isfinite(snap[key][1]).all(),
                "Snapshot is not this native one-step update",
            )
        block = torch.stack([s[key][1] for s in snapshots]).to(device=device, dtype=torch.float64)
        result += block.flatten(1) @ g.to(device=device, dtype=torch.float64).flatten()
    return result.cpu().numpy()


def centered_profile(ce_changes):
    changes = np.asarray(ce_changes, dtype=np.float64)
    require(changes.shape[-1] == 32 and np.isfinite(changes).all(), "Need32 finite CE changes")
    return changes.mean(axis=-1, keepdims=True) - changes


def matched_shuffle(profiles, cases):
    """Reuse donor's own-A contraction, then rescale to recipient's physical norm."""
    profiles = np.asarray(profiles, dtype=np.float64)
    require(profiles.shape == (len(cases), 32), "Incomplete profile grid")
    augmented = [{**c, "tokens": c["id"]} for c in cases]
    donors = shuffled_directions(augmented)
    lookup = {c["id"]: i for i, c in enumerate(cases)}
    result, names = [], []
    for case, donor in zip(cases, donors, strict=True):
        j = lookup[donor["direction_donor"]]
        norm, donor_norm = case["physical_update_norm"], cases[j]["physical_update_norm"]
        require((norm > 0) == (donor_norm > 0), "Cannot mix zero and real directions")
        scale = norm / donor_norm if donor_norm else 1.0
        result.append(profiles[j] * scale)
        names.append(cases[j]["id"])
    return np.asarray(result), names


def preflight(model, tokenizer, calibration, initial, output):
    restore_modules(model, initial)
    base = frozen_versions(model)
    before_fingerprint = fingerprint(snapshot_modules(model))
    gradients, scores = gradient_b(model, tokenizer, calibration_rows(calibration, PATTERNS[0]))
    require(fingerprint(snapshot_modules(model)) == before_fingerprint, "Gradient changed adapter")
    params = module_parameters(model)
    require(
        len([p for p in model.parameters() if p.requires_grad]) == 560
        and sum(p.numel() for p in model.parameters() if p.requires_grad) == 64225280,
        "Original native target architecture changed",
    )
    synthetic = {k: (a.clone(), -0.001 * gradients[k]) for k, (a, _) in initial.items()}
    explicit = sum(
        float((gradients[k].double() * synthetic[k][1].double()).sum()) for k in sorted(initial)
    )
    actual = float(contract(gradients, [synthetic], initial)[0])
    require(
        np.isclose(actual, explicit, rtol=1e-7, atol=1e-8) and explicit < 0,
        "Native signed directional contraction failed",
    )
    restore_modules(model, synthetic)
    require(fingerprint(snapshot_modules(model)) != before_fingerprint, "Synthetic change absent")
    with torch.no_grad():
        changed = (
            losses(model, tokenizer, calibration_rows(calibration, PATTERNS[0])[:CHUNK])
            .double()
            .cpu()
            .tolist()
        )
    restore_modules(model, initial)
    with torch.no_grad():
        restored = (
            losses(model, tokenizer, calibration_rows(calibration, PATTERNS[0])[:CHUNK])
            .double()
            .cpu()
            .tolist()
        )
    require(
        restored == scores[:CHUNK] and frozen_versions(model) == base,
        "Adapter/scoring restore or frozen base invariance failed",
    )
    atomic_json(
        output,
        {
            "passed": True,
            "finished_unix": time.time(),
            "training_only": True,
            "modules": len(params),
            "trainable_tensors": 560,
            "trainable_parameters": 64225280,
            "all_gradients_finite": True,
            "all_A_gradients_zero": True,
            "bit_exact_adapter_and_score_restore": True,
            "frozen_base_unchanged": True,
            "directional_contraction": actual,
            "independent_scalar_contraction": explicit,
            "calibration_before": scores,
            "synthetic_changed_first4": changed,
            "synthetic_perturbation_only_no_optimizer": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def run(root):
    plan = guard(root)
    require(
        plan["recipe"] == "base_sensitivity_v4" and not (root / "started.json").exists(),
        "Wrong recipe or already attempted; never resume",
    )
    atomic_json(root / "started.json", {"started_unix": time.time()})
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Input changed: {path}")
    data = read(root / "data.json")
    cases, calibration = data["cases"], data["calibration"]
    require(
        len(cases) == 344 and {c["split"] for c in cases} == {"train", "development"},
        "Only original development collection may be read",
    )
    collection = Path(plan["collection_root"])
    seeds = sorted({c["initialization_seed"] for c in cases})
    require(len(seeds) == 8, "Wrong initialization grid")
    initial_paths = {s: collection / "snapshots" / f"initial-{s}.pt" for s in seeds}
    status(root, "loading_base", 0, 256)
    model, tokenizer = load_target()
    model.eval()
    initial = torch.load(initial_paths[seeds[0]], map_location="cpu", weights_only=True, mmap=True)
    preflight(model, tokenizer, calibration, initial, root / "preflight.json")
    base_versions = frozen_versions(model)
    bank = root / "gradients"
    bank.mkdir()
    raw = np.zeros((len(cases), 32), dtype=np.float64)
    baseline_scores, records, restores = [], [], []
    for seed_index, seed in enumerate(seeds):
        guard(root)
        initial = torch.load(initial_paths[seed], map_location="cpu", weights_only=True, mmap=True)
        indices = [i for i, c in enumerate(cases) if c["initialization_seed"] == seed]
        snapshots = [
            torch.load(
                collection / "snapshots" / f"{cases[i]['id']}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            for i in indices
        ]
        restore_modules(model, initial)
        initial_fingerprint = fingerprint(initial)
        for pattern_index, pattern in enumerate(PATTERNS):
            guard(root)
            gradients, scores = gradient_b(model, tokenizer, calibration_rows(calibration, pattern))
            require(
                fingerprint(snapshot_modules(model)) == initial_fingerprint
                and frozen_versions(model) == base_versions,
                "Calibration mutated model weights",
            )
            path = bank / f"seed{seed}-pattern{pattern_index:02d}.pt"
            torch.save(gradients, path)
            raw[indices, pattern_index] = contract(gradients, snapshots, initial)
            records.append(
                {
                    "seed": seed,
                    "pattern_index": pattern_index,
                    "path": str(path.relative_to(root)),
                    "sha256": file_hash(path),
                    "case_ids": [cases[i]["id"] for i in indices],
                    "calibration_ce": scores,
                    "all_A_gradients_zero": True,
                    "adapter_unchanged": True,
                    "frozen_base_unchanged": True,
                    "finished_unix": time.time(),
                }
            )
            baseline_scores.append(scores)
            status(root, "calibration_gradients", seed_index * 32 + pattern_index + 1, 256)
        restore_modules(model, initial)
        require(
            fingerprint(snapshot_modules(model)) == initial_fingerprint
            and frozen_versions(model) == base_versions,
            "Seed-boundary restore failed",
        )
        restores.append(
            {
                "seed": seed,
                "fingerprint": initial_fingerprint,
                "bit_exact_restore": True,
                "finished_unix": time.time(),
            }
        )
        del snapshots
    # Every seed has the same zero-B function, checked for every calibration answer.
    ce = np.asarray(baseline_scores).reshape(8, 32, CALIBRATION_N)
    require(
        np.array_equal(ce, np.broadcast_to(ce[:1], ce.shape)), "Zero-B function depends on A seed"
    )
    np.save(root / "raw-ce-predictions.npy", raw)
    profile = centered_profile(raw)
    outputs = []
    for split in ("train", "development"):
        indices = [i for i, c in enumerate(cases) if c["split"] == split]
        group = [cases[i] for i in indices]
        shuffled, donors = matched_shuffle(profile[indices], group)
        for j, i in enumerate(indices):
            outputs.append(
                {
                    "id": cases[i]["id"],
                    "split": split,
                    "physical_update_norm": cases[i]["physical_update_norm"],
                    "profile": profile[i].tolist(),
                    "prediction": profile_label(profile[i]),
                    "shuffled_profile": shuffled[j].tolist(),
                    "shuffled_prediction": profile_label(shuffled[j]),
                    "direction_donor": donors[j],
                }
            )
    require([r["id"] for r in outputs] == [c["id"] for c in cases], "Original case order changed")
    atomic_json(root / "predictions.json", outputs)
    atomic_json(
        root / "gradient-manifest.json",
        {
            "records": records,
            "restores": restores,
            "initialization_files_sha256": {str(s): file_hash(p) for s, p in initial_paths.items()},
            "all_zero_B_calibration_scores_bit_exact": True,
        },
    )
    files = [
        "started.json",
        "preflight.json",
        "raw-ce-predictions.npy",
        "predictions.json",
        "gradient-manifest.json",
    ]
    hashes = {name: file_hash(root / name) for name in files}
    hashes.update({r["path"]: r["sha256"] for r in records})
    atomic_json(
        root / "predictions-lock.json",
        {
            "locked_unix": time.time(),
            "files_sha256": hashes,
            "development_effects_used_for_fitting": False,
            "reserved_cases_accessed": 0,
            "predicted_cases": 344,
        },
    )
    hashes["predictions-lock.json"] = file_hash(root / "predictions-lock.json")
    guard(root)
    atomic_json(
        root / "manifest.json",
        {
            "status": "complete",
            "recipe": "base_sensitivity_v4",
            "finished_unix": time.time(),
            "source_commit": plan["source_commit"],
            "plan_sha256": file_hash(root / "plan.json"),
            "cases": 344,
            "gradient_banks": 256,
            "calibration_questions": 8,
            "output_patterns": 32,
            "initializations": 8,
            "target_optimizer_updates": 0,
            "reserved_cases_accessed": 0,
            "files_sha256": hashes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026101701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args.root)
