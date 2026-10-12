"""Exploratory compression comparisons using saved gradients only, on CPU.

All archive outcomes have already been inspected. This sweep cannot supply new
confirmatory evidence. It is intended to choose a small, frozen set of candidates
for a future source-disjoint benchmark, including inexpensive SVD baselines.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Probe, read_jsonl, write_jsonl
from .dictionary import SparseDictionary
from .study import atomic_json, file_hash, group_matrix
from .suite import crossed_bootstrap

RANKS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128)
SPARSITIES = (1, 4, 8, 16, 32)


def compressed_coordinates(coordinates, scale, atoms, seed):
    """Include a rotated-dictionary control with identical atom Gram matrix."""
    generator = torch.Generator().manual_seed(seed + 83000)
    rotation = torch.linalg.qr(torch.randn(atoms.shape[1], atoms.shape[1], generator=generator)).Q
    variants, costs = {}, {}
    for rank in RANKS:
        if rank > coordinates.shape[1]:
            continue
        prediction = torch.zeros_like(coordinates)
        prediction[:, :rank] = coordinates[:, :rank]
        variants[f"svd_{rank}"] = prediction
        costs[f"svd_{rank}"] = 4 * rank
    white = coordinates / scale
    for label, candidate in (
        ("dictionary_omp", atoms),
        ("rotated_dictionary_omp", atoms @ rotation),
    ):
        for sparsity in SPARSITIES:
            if sparsity > min(candidate.shape):
                continue
            dictionary = SparseDictionary(candidate, sparsity, [], "ksvd")
            variants[f"{label}_{sparsity}"] = dictionary.encode(white) @ candidate * scale
            # Fixed-length float32 coefficients and uint16 atom indices.
            costs[f"{label}_{sparsity}"] = 6 * sparsity
    return variants, costs


def sweep_job(root, job, output):
    output.mkdir(parents=True, exist_ok=True)
    source = root / "jobs" / job["id"]
    manifest = json.loads((source / "manifest.json").read_text())
    cached = output / "manifest.json"
    if cached.exists():
        return read_jsonl(output / "results.jsonl")
    archive = read_jsonl(source / "results.jsonl")
    basis = torch.from_numpy(np.load(source / "projection_basis.npy"))
    scale = torch.from_numpy(np.load(source / "projection_scale.npy"))
    selection = manifest["selected_dictionary"]
    with np.load(source / f"dictionary-{selection}.npz", allow_pickle=False) as values:
        atoms = torch.from_numpy(values["atoms"].copy())
    gradients = torch.stack(
        [torch.from_numpy(np.load(source / "batch_gradients" / f"{i}.npy")) for i in range(60)]
    )
    coordinates = gradients @ basis
    probes = [Probe(**r) for r in read_jsonl(root / "data/evaluation_probes.jsonl")]
    _, average = group_matrix(probes)
    jacobian = average @ torch.from_numpy(np.load(source / "evaluation_probe_gradients.npy"))
    projected_jacobian = jacobian @ basis
    print(f"{job['id']}: comparing compression methods on CPU", flush=True)
    variants, costs = compressed_coordinates(coordinates, scale, atoms, job["seed"])
    effects = {name: value @ projected_jacobian.T for name, value in variants.items()}
    results = []
    for row in archive:
        i, rate = row["batch_index"], row["learning_rate"]
        predictions = {name: (-rate * value[i]).tolist() for name, value in effects.items()}
        for name in ("dictionary", "dictionary_debiased", "full_gradient", "zero"):
            predictions[name] = row["predictions"][name]
        results.append(
            {
                "batch_id": row["batch_id"],
                "learning_rate": rate,
                "family_split": row["family_split"],
                "predictions": predictions,
                "observed": {"full": row["observed"]["full"]},
            }
        )
    write_jsonl(output / "results.jsonl", results)
    atomic_json(
        cached,
        {
            "status": "complete",
            "scope": "Exploratory reuse of previously inspected outcomes; no new model evaluations",
            "job": job["id"],
            "source_forecast_hash": manifest["forecast_sha256"],
            "source_results_hash": file_hash(source / "results.jsonl"),
            "analysis_source_hash": file_hash(__file__),
            "bytes_per_update_code": costs,
            "storage_scope": "Float32 coefficients, uint16 sparse indices; shared decoder excluded",
            "compute_scope": "SVD and OMP do not have equal encoding costs; no speedup claim",
        },
    )
    return results


def run_sweep(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())
    torch.set_num_threads(8)
    grouped = {}
    for job in plan["jobs"]:
        rows = sweep_job(root, job, output / job["id"])
        key = f"{job['model']['name']}/warmup-{job['warmup_steps']}"
        grouped.setdefault(key, []).append(rows)
    summaries = {}
    for key, seeds in grouped.items():
        summaries[key] = {}
        for rate in plan["learning_rates"]:
            rows = [[r for r in seed if r["learning_rate"] == rate] for seed in seeds]
            summaries[key][str(rate)] = {
                method: crossed_bootstrap(rows, method) for method in rows[0][0]["predictions"]
            }
    atomic_json(
        output / "summary.json",
        {
            "status": "exploratory_only",
            "additional_model_evaluations": 0,
            "results": summaries,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-exploratory-reuse", action="store_true", required=True)
    args = parser.parse_args()
    run_sweep(args.root, args.output)


if __name__ == "__main__":
    main()
