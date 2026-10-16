"""Freeze exact calibration scoring after the complete failed linear development result."""

import argparse
import hashlib
import importlib.metadata
import subprocess
import tarfile
import time
from pathlib import Path

from gradient_interp.loracle_trajectory_toy import read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_analysis import require
from gradient_interp.update_loracle_exact_calibration import RECIPE
from gradient_interp.update_loracle_factorized_analysis import completion_barrier
from gradient_interp.update_loracle_sensitivity import select_calibration


def prepare(prior, root, expected_commit):
    prior, root = prior.resolve(), root.resolve()
    require(not root.exists(), "New run root already exists; never overwrite")
    completion_barrier(read(prior / "controller.json"))
    audit = read(prior / "integrity.json")
    require(
        audit["passed"] and audit["recipe"] == "base_sensitivity_v4" and audit["cases"] == 344,
        "Complete sensitivity audit required",
    )
    require(
        not read(prior / "analysis/summary.json")["all_development_gates_pass"],
        "Full prior development failure must precede this diagnostic",
    )
    for name, expected in audit["files_sha256"].items():
        require(file_hash(prior / name) == expected, "Prior audited evidence changed")
    previous = read(prior / "plan.json")
    with tarfile.open(prior / "source_snapshot.tar") as archive:
        require(
            set(archive.getnames()) == set(previous["source_files_sha256"]),
            "Prior source archive incomplete",
        )
        for name, expected in previous["source_files_sha256"].items():
            require(
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                "Prior frozen source changed",
            )
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit,
        "Wrong execution source",
    )
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(),
        "Remote checkout must be clean",
    )
    require(
        not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip(),
        "GPU occupied",
    )
    data = read(prior / "data.json")
    collection = Path(previous["collection_root"])
    require(
        data["calibration"] == select_calibration(read(collection / "data.json")),
        "Original training calibration changed",
    )
    require(
        len(data["cases"]) == 344
        and {c["split"] for c in data["cases"]} == {"train", "development"},
        "Wrong original grid",
    )
    inputs = dict(previous["input_files_sha256"])
    for name in (
        "plan.json",
        "data.json",
        "integrity.json",
        "manifest.json",
        "predictions.json",
        "gradient-manifest.json",
        "source_snapshot.tar",
        "analysis/summary.json",
        "analysis/all-development-cases.json",
    ):
        inputs[str(prior / name)] = file_hash(prior / name)
    for path, expected in inputs.items():
        require(file_hash(path) == expected, f"Frozen input changed: {path}")
    paths = subprocess.check_output(
        ["git", "ls-files", "src/gradient_interp"], text=True
    ).splitlines()
    paths.append("scripts/prepare_update_loracle_exact_calibration.py")
    source = {p: file_hash(p) for p in paths if p.endswith(".py")}
    proposal = read(
        Path("results/update-loracle-program/provenance/exact-calibration-proposal.json")
    )
    root.mkdir(parents=True)
    atomic_json(root / "data.json", data)
    atomic_json(
        root / "analysis-plan.json",
        {
            "frozen_unix": time.time(),
            "proposal": proposal,
            "recipe": RECIPE,
            "calibration": (
                "Exact same8 training prompts and32 response patterns asv4; no reselection"
            ),
            "scoring": (
                "FP32 per-document mean completion CE with original EOS; BF16 "
                "pinned base; FP32 native adapters; chunk4; FP64 mean/centering"
            ),
            "cases": 344,
            "scored_adapter_conditions": 688,
            "max_hours": 4,
            "physical_control": (
                "Use donor initialization A and donor B multiplied in FP32 by "
                "recipient/donor physical norm. Score this scaled adapter, never "
                "rescale an already nonlinear response. Verify physical norm at5e-6"
                " relative,1e-10 absolute tolerance."
            ),
            "preflight": (
                "Only fixed synthetic B noise1e-5 on original training "
                "seed2026101601, random seed2026101711. All256 calibration "
                "responses, actual adapter application, exact restoration, nonzero "
                "effect,1.75norm scaling, no optimizer."
            ),
            "audit": (
                "Complete/dead single worker, original4hdeadline, all "
                "source/input/snapshot/score/profile hashes and688record timeline. "
                "Independent recomputation of all means, native physical norms and "
                "shuffle assignments. Re-score own and shuffled first nonzero train"
                " and first nonzero development cases after audit file checks; "
                "restore exactly. Original zero-B v4 calibration CE must match."
            ),
            "analysis": (
                "Original64actual/8zero development cases; all original baselines "
                "plus exact functional predictor and norm-matched shuffle. "
                "Original2000 paired crossed bootstrap seed2026101653. All original"
                " full-category gates unchanged. All272training cases diagnostic "
                "only. No weak-case, margin, family or seed filtering."
            ),
            "error_decomposition": (
                "L-E=(L-C)+(C-E), whereL is old linear calibration profile,C is new"
                " exact calibration profile,E is old measured separate-probe "
                "profile. Report full MSE terms including signed cross term. This "
                "is algebra, not independent causal attribution."
            ),
            "release": (
                "No scientific outcome reading until normal completion/death and "
                "passed complete audit. Preserve all failures, no retries/resume. "
                "Reserved/fresh grids remain uncollected."
            ),
            "scope": (
                "Reused development; finite-reference preference. Additional "
                "base-model and task-library access; not an improved LoRAcle "
                "checkpoint or unrestricted behavioral forecast."
            ),
        },
    )
    plan = {
        "created_unix": time.time(),
        "scope": "development_only",
        "max_hours": 4,
        "recipe": RECIPE,
        "source_commit": expected_commit,
        "source_files_sha256": source,
        "input_files_sha256": inputs,
        "collection_root": str(collection),
        "prior_root": str(prior),
        "baseline_root": previous["baseline_root"],
        "data_sha256": file_hash(root / "data.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "jobs": [
            {
                "id": "predict",
                "module": "gradient_interp.update_loracle_exact_calibration",
                "arguments": ["--root", str(root)],
                "manifest": str(root / "manifest.json"),
            }
        ],
    }
    atomic_json(root / "plan.json", plan)
    with tarfile.open(root / "source_snapshot.tar", "x") as archive:
        for path in source:
            archive.add(path, arcname=path, recursive=False)
    atomic_json(
        root / "preparation.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "source_commit": expected_commit,
            "plan_sha256": file_hash(root / "plan.json"),
            "source_archive_sha256": file_hash(root / "source_snapshot.tar"),
            "source_files": len(source),
            "input_files": len(inputs),
            "cases": 344,
            "conditions": 688,
            "prior_archive_verified": True,
            "no_new_fitting_or_target_training": True,
            "gpu_preflight": "pending training-only first worker step",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "scikit-learn")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    prepare(args.prior, args.root, args.commit)
