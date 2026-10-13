"""Verify the complete direct-likelihood diagnostic on its original host."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gradient_interp.confirmation import verify_files
from gradient_interp.study import atomic_json, file_hash

root = Path(sys.argv[1]).resolve()


def read(p):
    return json.loads(p.read_text())


state, spec, plan, manifest = [
    read(root / n) for n in ("controller.json", "specification.json", "plan.json", "manifest.json")
]
assert state["status"] == "complete" and len(state["attempts"]) == 1
assert state["attempts"][0]["returncode"] == 0
assert (
    state["started_unix"]
    < manifest["started_unix"]
    < manifest["completed_unix"]
    < state["deadline_unix"]
)
assert state["attempts"][0]["finished_unix"] < state["deadline_unix"]
assert state["plan_sha256"] == manifest["plan_sha256"] == file_hash(root / "plan.json")
for name in ("controller_pid", "worker_pid"):
    pid = state.get(name)
    if pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Still active")
for hashes in (spec["input_hashes"], plan["source_files_sha256"], plan["input_hashes"]):
    verify_files(Path.cwd(), hashes)
verify_files(root, manifest["files_sha256"])
expected = {"likelihoods/baseline.npz"} | {
    f"likelihoods/batch-{b:02d}-rate-{r:g}.{ext}"
    for b in range(16)
    for r in spec["rates"]
    for ext in ("npz", "json")
}
assert set(manifest["files_sha256"]) == expected
for name in expected:
    if not name.endswith(".json"):
        continue
    record = read(root / name)
    assert record["plan_sha256"] == state["plan_sha256"]
    assert record["sha256"] == file_hash((root / name).with_suffix(".npz"))
    assert record["adapter_restored"]
    assert (
        plan["frozen_unix"]
        < record["update_started_unix"]
        < record["completed_unix"]
        < manifest["completed_unix"]
    )
    assert name == f"likelihoods/batch-{record['batch_index']:02d}-rate-{record['rate']:g}.json"
assert manifest["adapter_restored"] and manifest["updates"] == 48 and manifest["conditions"] == 49
assert manifest["proposal_trajectories_per_condition"] == 1024
repair = root / "repairs/2026-10-13-scoring-batch"
audit = read(repair / "audit.json")
assert file_hash(repair / "specification.json") == audit["original_specification_sha256"]
assert file_hash(root / "specification.json") == audit["amended_specification_sha256"]
old, amended = read(repair / "specification.json"), dict(spec)
assert old.pop("score_batch_size") == 8 and amended.pop("score_batch_size") == 1 and old == amended
preflight = read(root / "preflight.json")
assert preflight["specification_sha256"] == file_hash(root / "specification.json")
assert preflight["adapter_restored"] and preflight["likelihoods_restored_exactly"]
assert max(preflight["single_vs_batched_abs_error_per_token"]) == 0
assert preflight["parameter_tensors"] == 136
snapshot = root / "source_snapshot.tar"
if not snapshot.exists():
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "-o",
            str(snapshot),
            "67900b80f1d3783b4ef51933cf7b1dbf6197c23c",
        ],
        check=True,
    )
result = {
    "complete": True,
    "verified_unix": time.time(),
    "plan_sha256": file_hash(root / "plan.json"),
    "specification_sha256": file_hash(root / "specification.json"),
    "manifest_sha256": file_hash(root / "manifest.json"),
    "source_snapshot_sha256": file_hash(snapshot),
    "source_input_output_hashes_verified": True,
    "adapter_restored": True,
    "deadline_compliant": True,
    "conditions": 49,
    "updates": 48,
    "elapsed_seconds": manifest["completed_unix"] - state["started_unix"],
}
atomic_json(root / "integrity.json", result)
print(json.dumps(result))
