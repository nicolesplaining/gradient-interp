"""Freeze one physical-profile development fit, without new target-model work."""

import argparse
import subprocess
import tarfile
import time
from pathlib import Path

from gradient_interp.loracle_trajectory_toy import read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_analysis import alive, require


def prepare(collection, prior, root, expected_commit):
    collection, prior, root = collection.resolve(), prior.resolve(), root.resolve()
    require(not root.exists(), "New decoder requires a new run root")
    controller = read(prior / "controller.json")
    require(
        controller["status"] == "complete"
        and not alive(controller["controller_pid"])
        and all(not alive(a["pid"]) and a["returncode"] == 0 for a in controller["attempts"]),
        "Previous run must be complete and dead",
    )
    audit = read(prior / "integrity.json")
    require(audit["passed"] and audit["optimizer_steps"] == 780, "Complete v2 audit required")
    require(
        not read(prior / "analysis/summary.json")["all_development_gates_pass"],
        "This follow-up is contingent on complete failed v2 development",
    )
    for name, expected in audit["files_sha256"].items():
        require(file_hash(prior / name) == expected, f"Prior audited file changed: {name}")
    old_plan = read(prior / "plan.json")
    with tarfile.open(prior / "source_snapshot.tar") as archive:
        import hashlib

        require(
            set(archive.getnames()) == set(old_plan["source_files_sha256"]),
            "Previous archive inventory differs",
        )
        for name, expected in old_plan["source_files_sha256"].items():
            require(
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                "Previous source was not preserved",
            )
    collection_audit = read(collection / "integrity.json")
    require(
        collection_audit["passed"] and collection_audit["cases"] == 344,
        "Complete original collection required",
    )
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected_commit,
        "Wrong scientific source",
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
    data = read(collection / "data.json")
    cases = [
        {k: c[k] for k in ("id", "split", "control", "initialization_seed")}
        for c in data["cases"]
        if c["split"] in ("train", "development")
    ]
    require(len(cases) == 344, "Wrong development collection grid")
    paths = [
        collection / "integrity.json",
        collection / "data.json",
        prior / "integrity.json",
        prior / "analysis/summary.json",
        prior / "analysis/all-development-cases.json",
        prior / "source_snapshot.tar",
    ]
    for c in cases:
        paths.extend(
            [collection / "cases" / f"{c['id']}.json", collection / "snapshots" / f"{c['id']}.pt"]
        )
    paths.extend(
        collection / "snapshots" / f"initial-{s}.pt"
        for s in sorted({c["initialization_seed"] for c in cases})
    )
    inputs = {str(p): file_hash(p) for p in paths}
    for path, expected in inputs.items():
        p = Path(path)
        if (
            p.is_relative_to(collection)
            and str(p.relative_to(collection)) in collection_audit["files_sha256"]
        ):
            require(
                expected == collection_audit["files_sha256"][str(p.relative_to(collection))],
                "Original collection input changed",
            )
    for case in cases:
        record = read(collection / "cases" / f"{case['id']}.json")
        require(
            inputs[str(collection / "snapshots" / f"{case['id']}.pt")] == record["snapshot_sha256"]
            and inputs[str(collection / "snapshots" / f"initial-{case['initialization_seed']}.pt")]
            == record["initial_sha256"],
            "Consumed native snapshot differs from original completed audit",
        )
    source_paths = subprocess.check_output(
        ["git", "ls-files", "src/gradient_interp"], text=True
    ).splitlines()
    source_paths.append("scripts/prepare_update_loracle_physical_profile.py")
    source = {p: file_hash(p) for p in source_paths if p.endswith(".py")}
    root.mkdir(parents=True)
    atomic_json(
        root / "data.json",
        {"cases": cases, "collection_data_sha256": file_hash(collection / "data.json")},
    )
    analysis = read(
        Path("results/update-loracle-program/provenance/physical-profile-development-proposal.json")
    )
    atomic_json(root / "analysis-plan.json", analysis)
    plan = {
        "created_unix": time.time(),
        "scope": "development_only",
        "max_hours": 2,
        "recipe": "physical_profile_v3",
        "source_commit": expected_commit,
        "source_files_sha256": source,
        "input_files_sha256": inputs,
        "collection_root": str(collection),
        "prior_root": str(prior),
        "data_sha256": file_hash(root / "data.json"),
        "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
        "jobs": [
            {
                "id": "fit",
                "module": "gradient_interp.update_loracle_physical_profile",
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
            "plan_sha256": file_hash(root / "plan.json"),
            "source_commit": expected_commit,
            "source_archive_sha256": file_hash(root / "source_snapshot.tar"),
            "source_files": len(source),
            "input_files": len(inputs),
            "reserved_cases_accessed": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("collection", "prior", "root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    prepare(args.collection, args.prior, args.root, args.commit)
