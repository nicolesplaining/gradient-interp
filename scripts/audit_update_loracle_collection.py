"""Post-completion audit of the fixed one-update development collection."""

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from gradient_interp.loracle_toy_suite import initialize_like
from gradient_interp.loracle_trajectory_toy import fingerprint, read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_collect import tokenize_update, upstream_module
from gradient_interp.update_loracle_data import effect_label, validate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return bool(pid and path.exists() and path.read_text().rsplit(")", 1)[-1].split()[0] != "Z")


def barrier(root):
    controller = read(root / "controller.json")
    require(
        controller["status"] == "complete" and not alive(controller["controller_pid"]),
        "Collection must be normally complete and controller dead",
    )
    attempts = controller["attempts"]
    require(
        len(attempts) == 1 and attempts[0]["returncode"] == 0 and not alive(attempts[0]["pid"]),
        "Expected one completed successful worker",
    )
    require(controller["worker_pid"] is None, "Worker remains assigned")
    require(attempts[0]["finished_unix"] <= controller["deadline_unix"], "Deadline exceeded")
    return controller


def archive(root, hashes):
    path = root / "source_snapshot.tar"
    if not path.exists():
        with tarfile.open(path, "x") as tar:
            for name, expected in hashes.items():
                require(file_hash(name) == expected, f"Frozen source changed: {name}")
                tar.add(name, arcname=name, recursive=False)
    with tarfile.open(path) as tar:
        require(set(tar.getnames()) == set(hashes), "Source archive inventory differs")
        for name, expected in hashes.items():
            require(
                hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected,
                f"Archived source differs: {name}",
            )
    return file_hash(path)


def audit(root):
    controller = barrier(root)  # No case parsing before completion and death.
    plan, manifest = read(root / "plan.json"), read(root / "manifest.json")
    require(file_hash(root / "plan.json") == controller["plan_sha256"], "Plan changed")
    require(
        controller["deadline_unix"] == controller["started_unix"] + plan["max_hours"] * 3600,
        "Original phase deadline was changed",
    )
    for path, expected in plan["source_files_sha256"].items():
        require(file_hash(path) == expected, f"Frozen execution source changed: {path}")
    archive_hash = archive(root, plan["source_files_sha256"])
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Input changed: {path}")
    for name, key in [("data.json", "data_sha256"), ("analysis-plan.json", "analysis_plan_sha256")]:
        require(file_hash(root / name) == plan[key], f"Frozen {name} changed")
    require(
        manifest["status"] == "complete"
        and manifest["cases"] == 344
        and manifest["actual_updates"] == 320
        and manifest["reserved_cases_accessed"] == 0
        and manifest["source_commit"] == plan["source_commit"]
        and manifest["plan_sha256"] == controller["plan_sha256"]
        and manifest["finished_unix"] <= controller["deadline_unix"],
        "Manifest inconsistent",
    )
    data = read(root / "data.json")
    validate(data)
    cases = [c for c in data["cases"] if c["split"] in ("train", "development")]
    expected_ids = [c["id"] for c in cases]
    require([r["id"] for r in manifest["records"]] == expected_ids, "Incorrect case grid")
    for directory, pattern in [("cases", "*.json"), ("tokens", "*.pt")]:
        require(
            {p.stem for p in (root / directory).glob(pattern)} == set(expected_ids),
            f"Extra/missing {directory}",
        )
    preflight = read(root / "preflight.json")
    require(
        preflight["passed"] and file_hash(root / "preflight.json") == manifest["preflight_sha256"],
        "Invalid preflight",
    )
    prior_time = preflight["finished_unix"]
    require(
        controller["attempts"][0]["started_unix"] <= manifest["started_unix"] < prior_time,
        "Preflight did not precede collection",
    )
    initial = {}
    for filename, expected in manifest["initial_files_sha256"].items():
        path = root / "snapshots" / filename
        require(file_hash(path) == expected, "Initial adapter changed")
        seed = int(filename.removeprefix("initial-").removesuffix(".pt"))
        value = torch.load(path, map_location="cpu", weights_only=True)
        require(
            fingerprint(value) == fingerprint(initialize_like(value, seed)),
            "Unexpected initial adapter values",
        )
        initial[seed] = value
    require(set(initial) == {c["initialization_seed"] for c in cases}, "Incorrect seed grid")
    expected_snapshots = set(expected_ids) | {f"initial-{seed}" for seed in initial}
    require(
        {p.stem for p in (root / "snapshots").glob("*.pt")} == expected_snapshots,
        "Extra/missing snapshots",
    )
    upstream = upstream_module()
    results = []
    for case, item in zip(cases, manifest["records"]):
        path = root / "cases" / f"{case['id']}.json"
        require(file_hash(path) == item["sha256"], "Case output changed")
        result = read(path)
        for key in ("id", "split", "control", "source_pattern", "initialization_seed"):
            require(result[key] == case[key], f"Incorrect case metadata: {key}")
        require(
            prior_time < result["finished_unix"] <= manifest["finished_unix"],
            "Case timeline inconsistent",
        )
        prior_time = result["finished_unix"]
        before, after = np.asarray(result["before_ce"]), np.asarray(result["after_ce"])
        require(before.shape == (4, 32) and after.shape == (4, 32), "Incomplete CE grid")
        require(effect_label(before, after) == result["effect"], "Incorrect measured-effect label")
        require(
            result["bit_exact_adapter_restore"] and result["bit_exact_first_probe_restore"],
            "Adapter restoration failed",
        )
        start = initial[case["initialization_seed"]]
        initial_name = f"initial-{case['initialization_seed']}.pt"
        require(
            result["initial_sha256"] == manifest["initial_files_sha256"][initial_name],
            "Incorrect starting state",
        )
        snapshot_path = root / "snapshots" / f"{case['id']}.pt"
        token_path = root / "tokens" / f"{case['id']}.pt"
        require(file_hash(snapshot_path) == result["snapshot_sha256"], "Snapshot changed")
        require(file_hash(token_path) == result["tokens_sha256"], "Direction tokens changed")
        updated = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        stored_tokens = torch.load(token_path, map_location="cpu", weights_only=True)
        tokens, norm, token_audit = tokenize_update(start, updated, upstream)
        require(
            torch.equal(tokens, stored_tokens) and norm == result["physical_update_norm"],
            "Recomputed direction encoding or physical norm differs",
        )
        require(token_audit == result["token_audit"], "Module-wise product audit differs")
        if case["control"]:
            require(
                result["step"] is None
                and norm == 0
                and np.array_equal(before, after)
                and fingerprint(start) == fingerprint(updated),
                "Zero control had an update",
            )
        else:
            require(
                result["step"]["optimizer_steps"] == 1
                and norm > 0
                and result["step"]["learning_rate"] == plan["learning_rate"],
                "Incorrect optimizer update",
            )
        results.append(
            {
                "id": case["id"],
                "snapshot_sha256": result["snapshot_sha256"],
                "tokens_sha256": result["tokens_sha256"],
                "encoding_recomputed": True,
            }
        )
        del stored_tokens, tokens, updated
    files = [
        "plan.json",
        "controller.json",
        "manifest.json",
        "data.json",
        "analysis-plan.json",
        "preflight.json",
        "source_snapshot.tar",
    ] + [f"cases/{c['id']}.json" for c in cases]
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "scope": "development_only",
            "finished_unix": time.time(),
            "source_commit": plan["source_commit"],
            "source_archive_sha256": archive_hash,
            "cases": 344,
            "actual_updates": 320,
            "zero_controls": 24,
            "reserved_cases_accessed": 0,
            "all_native_tokens_and_physical_norms_recomputed": True,
            "records": results,
            "files_sha256": {name: file_hash(root / name) for name in files},
        },
    )
    print(json.dumps({"passed": True, "cases": 344, "actual_updates": 320}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    audit(args.root)
