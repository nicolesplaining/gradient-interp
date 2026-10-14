"""Assemble a complete SVD manifest from frozen, disjoint execution inventories.

Only metadata is parsed. Generated responses and descriptor tensors are hashed
as opaque files; the paired scientific audit remains a separate later step.
"""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

from gradient_interp.paper_svd_replication import condition_grid, validate_preflight
from gradient_interp.study import atomic_json, file_hash

FROZEN_COMMIT = "e12ecc1989531ce48b1352e2f76985c6a7cf1a23"
HOST = "ubuntu@209.20.156.175"
PROVENANCE = (
    "execution-amendment.json",
    "controller-at-amendment.json",
    "prefix-controller.json",
    "prefix-completion-receipt.json",
    "tail-controller.json",
    "tail-manifest.json",
    "tail-completion-receipt.json",
    "tail-preflight.json",
    "tail-source.py",
)


def read(path):
    if path.parent.name == "generations" or path.suffix in {".pt", ".npz"}:
        raise ValueError("Finalization never parses scientific output contents")
    return json.loads(path.read_text())


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def prepare_finalization(root, now, process_alive=alive):
    """Validate metadata and opaque hashes, returning proposed records without writes."""
    require(
        not any((root / n).exists() for n in ("manifest.json", "execution-finalization.json")),
        "Preserve existing completed or partial finalization",
    )
    values = {
        name: read(root / f"{name}.json")
        for name in (
            "plan",
            "specification",
            "controller",
            "selection-lock",
            "execution-amendment",
            "controller-at-amendment",
            "prefix-controller",
            "prefix-completion-receipt",
            "tail-controller",
            "tail-manifest",
            "tail-completion-receipt",
            "tail-preflight",
            "preflight",
            "descriptor-manifest",
        )
    }
    plan, spec, controller, amendment = [
        values[k] for k in ("plan", "specification", "controller", "execution-amendment")
    ]
    require(
        amendment.get("type") == "paper_svd_parallel_execution_amendment"
        and amendment.get("frozen_source_commit") == FROZEN_COMMIT
        and amendment.get("execution_host") == HOST
        and amendment.get("scientific_choices_unchanged") is True
        and amendment.get("scientific_outcomes_read") is False,
        "Unsupported frozen execution amendment",
    )
    require(
        file_hash(Path(__file__)) == amendment.get("finalizer_sha256"),
        "Finalizer changed after amendment freeze",
    )
    assignments = amendment["assignments"]
    require(
        len(assignments) == 2 and {a["seed"] for a in assignments} == {42, 43},
        "Both seed assignments required",
    )
    for entry in assignments:
        prefix, suffix = entry["prefix_indices"], entry["tail_indices"]
        require(
            prefix
            and suffix
            and prefix == list(range(len(prefix)))
            and suffix == list(range(len(prefix), 101)),
            "Ownership overlaps or omits conditions",
        )
    item = next(a for a in assignments if a["seed"] == spec["training_seed"])
    require(str(root.resolve()) == item["original_root"], "Original run root changed")
    start, deadline = item["original_started_unix"], item["original_deadline_unix"]
    require(
        abs(deadline - start - 43200) < 0.01 and start < amendment["frozen_unix"] < now <= deadline,
        "Original deadline or amendment ordering differs",
    )
    for name, key in (
        ("plan.json", "plan_sha256"),
        ("specification.json", "specification_sha256"),
        ("selection-lock.json", "selection_lock_sha256"),
        ("controller-at-amendment.json", "controller_at_amendment_sha256"),
    ):
        require(file_hash(root / name) == item[key], "Frozen original metadata changed")
    require(
        file_hash(root / "preflight.json") == plan["input_hashes"][str(root / "preflight.json")],
        "Original preflight changed",
    )
    require(
        plan["max_hours"] == 12 and plan["scope"] == "development_only",
        "Original scientific plan differs",
    )
    require(
        (spec["chunk"], spec["max_new_tokens"], spec["stop_ids"]) == (8, 300, [1, 106]),
        "Sampler changed",
    )
    require(
        spec["counts"]["conditions"] == 101
        and spec["counts"]["total_responses"] == 14540
        and spec["counts"]["descriptor_gradients"] == 160,
        "Frozen scientific counts differ",
    )
    lock = values["selection-lock"]
    require(
        lock["heldout_outcomes_read"] is False
        and lock["specification_sha256"] == item["specification_sha256"],
        "Selection lock differs",
    )
    require(
        file_hash(root / "directions.pt") == lock["files_sha256"]["directions.pt"]
        and file_hash(root / "data.json") == lock["files_sha256"]["data.json"],
        "Locked direction bank or data changed",
    )
    for name, digest in plan["source_files_sha256"].items():
        require(file_hash(Path(name)) == digest, "Frozen scientific source changed")
    initial, prefix = values["controller-at-amendment"], values["prefix-controller"]
    require(
        controller == prefix
        and initial["status"] == "running"
        and prefix["status"] == "interrupted",
        "Original interrupted controller must be preserved unchanged",
    )
    require(
        controller["started_unix"] == initial["started_unix"] == start
        and controller["deadline_unix"] == initial["deadline_unix"] == deadline,
        "Original controller deadline changed",
    )
    require(controller["plan_sha256"] == item["plan_sha256"], "Original controller plan changed")
    require(
        prefix["attempts"]
        and prefix["attempts"][-1]["returncode"] != 0
        and prefix["worker_pid"] is None,
        "Original model attempt was not interrupted",
    )
    require(
        prefix["attempts"][-1]["pid"] == initial["attempts"][-1]["pid"]
        and prefix["attempts"][-1]["started_unix"] == initial["attempts"][-1]["started_unix"],
        "Model attempt identity changed",
    )
    pids = {controller.get("controller_pid"), controller.get("worker_pid")}
    pids.update(a["pid"] for a in controller["attempts"])
    require(
        not any(process_alive(pid) for pid in pids),
        "An original controller or worker is still alive",
    )
    ah = file_hash(root / "execution-amendment.json")
    pr, tc, tm, tr, pf = [
        values[k]
        for k in (
            "prefix-completion-receipt",
            "tail-controller",
            "tail-manifest",
            "tail-completion-receipt",
            "tail-preflight",
        )
    ]
    require(
        pr.get("type") == "paper_svd_prefix_completion_receipt"
        and pr["seed"] == spec["training_seed"]
        and pr["all_processes_dead"] is True,
        "Genuine prefix death receipt required",
    )
    require(
        pr["controller_sha256"] == file_hash(root / "prefix-controller.json")
        and pr["amendment_sha256"] == ah
        and pr["completed_prefix_indices"] == item["prefix_indices"],
        "Prefix receipt is stale or has different ownership",
    )
    require(
        pr.get("watcher_sha256") == amendment["prefix_watcher_sha256"]
        and pr.get("scientific_output_contents_parsed") is False,
        "Prefix watcher provenance differs",
    )
    require(
        set(pr["checked_pids"]) == pids - {None},
        "Prefix receipt did not check the original processes",
    )
    require(
        amendment["frozen_unix"]
        < prefix["attempts"][-1]["finished_unix"]
        <= pr["observed_unix"]
        <= now,
        "Prefix stop ordering differs",
    )
    require(
        tm.get("type") == "paper_svd_tail_manifest"
        and tr.get("type") == "paper_svd_tail_completion_receipt",
        "Tail metadata types differ",
    )
    require(
        tc["status"] == "complete"
        and tm["seed"] == tr["seed"] == spec["training_seed"]
        and tr["all_processes_dead"] is True,
        "Tail did not complete with dead processes",
    )
    require(
        tc["execution_host"]
        == tm["execution_host"]
        == tr["execution_host"]
        == pf["execution_host"]
        == HOST,
        "Tail execution host differs",
    )
    require(
        tr["manifest_sha256"] == file_hash(root / "tail-manifest.json")
        and tr["controller_sha256"] == file_hash(root / "tail-controller.json"),
        "Tail death receipt is stale",
    )
    require(
        tr["checked_pids"] == [tc["controller_pid"]] and tc["worker_pid"] is None,
        "Tail process identity was not checked",
    )
    require(
        tc["deadline_unix"] == tm["deadline_unix"] == deadline
        and tm["original_started_unix"] == start,
        "Tail deadline changed",
    )
    require(
        amendment["frozen_unix"]
        < pf["completed_unix"]
        < tc["started_unix"]
        <= tm["started_unix"]
        < tm["completed_unix"]
        == tr["completed_unix"]
        <= tr["observed_unix"]
        <= now,
        "Tail completion or preflight ordering differs",
    )
    require(
        tm["completed_unix"] <= deadline and tm["adapter_restored"] is True,
        "Tail missed the deadline or restoration",
    )
    for record in (tc, tm, tr, pf):
        require(
            record["amendment_sha256"] == ah
            and record["helper_sha256"] == amendment["helper_sha256"],
            "Tail helper provenance differs",
        )
    require(
        file_hash(root / "tail-source.py") == amendment["helper_sha256"],
        "Preserved helper source changed",
    )
    validate_preflight(pf, spec)
    require(
        tm["preflight_sha256"] == file_hash(root / "tail-preflight.json"),
        "Tail preflight hash changed",
    )
    for key in ("plan_sha256", "specification_sha256", "selection_lock_sha256"):
        require(tm[key] == item[key], "Tail original-input hash differs")
    require(
        pf["specification_sha256"] == item["specification_sha256"]
        and pf["selection_lock_sha256"] == item["selection_lock_sha256"],
        "Tail preflight input differs",
    )
    require(
        tm["source_files_sha256"] == plan["source_files_sha256"]
        and tm["runtime_asset_hashes"] == item["runtime_asset_hashes"],
        "Tail frozen source or runtime inventory differs",
    )
    require(
        all(
            spec["input_hashes"].get(path) == digest
            for path, digest in item["runtime_asset_hashes"].items()
        ),
        "Tail runtime asset hash differs from original input",
    )
    grid = condition_grid()
    prefix_files = {f"generations/{grid[i]['id']}.json" for i in item["prefix_indices"]}
    suffix_files = {f"generations/{grid[i]['id']}.json" for i in item["tail_indices"]}
    require(
        set(pr["generation_files_sha256"]) == prefix_files
        and set(tm["files_sha256"]) == suffix_files,
        "Prefix or suffix inventory overlaps or is incomplete",
    )
    require(
        tm["assigned_indices"] == tc["assigned_indices"] == item["tail_indices"]
        and tm["assigned_ids"] == [grid[i]["id"] for i in item["tail_indices"]],
        "Tail condition assignment differs",
    )
    require(
        tm["conditions"] == len(suffix_files) and tm["generations"] == 140 * len(suffix_files),
        "Tail counts differ",
    )
    require(
        {str(p.relative_to(root)) for p in (root / "generations").glob("*.json")}
        == prefix_files | suffix_files,
        "Final generation inventory is incomplete or has extra files",
    )
    files = {**pr["generation_files_sha256"], **tm["files_sha256"]}
    for name, digest in files.items():
        require(
            file_hash(root / name) == digest,
            "A committed generation changed or suffix was overwritten",
        )
    descriptor = values["descriptor-manifest"]
    require(
        descriptor["count"] == 160
        and descriptor["adapter_unchanged"] is True
        and descriptor["selection_lock_sha256"] == item["selection_lock_sha256"],
        "Descriptor completion metadata differs",
    )
    require(
        start < descriptor["completed_unix"] < amendment["frozen_unix"],
        "Descriptors were not completed before execution amendment",
    )
    require(
        file_hash(root / "descriptors.pt") == descriptor["sha256"], "Descriptor tensor hash changed"
    )
    files.update({n: file_hash(root / n) for n in ("descriptors.pt", "descriptor-manifest.json")})
    manifest = {
        "started_unix": start,
        "completed_unix": now,
        "plan_sha256": item["plan_sha256"],
        "selection_lock_sha256": item["selection_lock_sha256"],
        "conditions": 101,
        "generations": 14540,
        "descriptor_gradients": 160,
        "adapter_restored": True,
        "peak_gpu_bytes": max(values["preflight"]["peak_gpu_bytes"], tm["peak_gpu_bytes"]),
        "files_sha256": files,
        "execution_amendment_sha256": ah,
        "cost_scope": (
            "Metadata-only assembly. Start is original controller start, including loading. "
            "Completion is assembly time, not summed GPU time. Peak is the maximum of original "
            "training-preflight and tail-worker peaks; original prefix generation peak was not "
            "recorded. Completed conditions are subject to the later paired scientific audit."
        ),
    }
    finalization = {
        "type": "paper_svd_execution_finalization",
        "seed": spec["training_seed"],
        "assembled_unix": now,
        "amendment_sha256": ah,
        "all_processes_dead": True,
        "prefix_indices": item["prefix_indices"],
        "tail_indices": item["tail_indices"],
        "prefix_generation_hashes": pr["generation_files_sha256"],
        "tail_generation_hashes": tm["files_sha256"],
        "provenance_hashes": {name: file_hash(root / name) for name in PROVENANCE},
        "finalizer_pid": os.getpid(),
        "finalizer_sha256": file_hash(Path(__file__)),
        "scientific_output_contents_parsed": False,
        "checked_original_pids": sorted(pids - {None}),
    }
    complete = {
        **controller,
        "status": "complete",
        "checked_unix": now,
        "execution_amendment_sha256": ah,
        "completion_basis": "parallel execution metadata finalization",
        "finalizer_pid": os.getpid(),
        "finalization_unix": now,
    }
    return manifest, finalization, complete


def run(root):
    with (root / "controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, finalization, controller = prepare_finalization(root, time.time())
        atomic_json(root / "execution-finalization.json", finalization)
        atomic_json(root / "manifest.json", manifest)
        controller["execution_finalization_sha256"] = file_hash(
            root / "execution-finalization.json"
        )
        controller["manifest_sha256"] = file_hash(root / "manifest.json")
        atomic_json(root / "controller.json", controller)
    return {
        "status": "complete",
        "conditions": 101,
        "responses": 14540,
        "scientific_output_contents_parsed": False,
    }


def guarded_run(root):
    try:
        return run(root)
    except BaseException as error:
        atomic_json(
            root / f"finalization-failure-{time.time_ns()}.json",
            {
                "status": "failed",
                "observed_unix": time.time(),
                "pid": os.getpid(),
                "error_type": type(error).__name__,
                "error": str(error),
                "finalizer_sha256": file_hash(Path(__file__)),
                "scientific_output_contents_parsed": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(guarded_run(args.root.resolve())))
