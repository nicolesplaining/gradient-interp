"""Verify a completed sampled-behavior run on its original host."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gradient_interp.confirmation import verify_files
from gradient_interp.study import atomic_json, file_hash

root = Path(sys.argv[1]).resolve()


def read(path):
    return json.loads(path.read_text())


controller, plan, spec, manifest, lock, data = [
    read(root / name)
    for name in (
        "controller.json",
        "plan.json",
        "specification.json",
        "manifest.json",
        "forecast-lock.json",
        "data.json",
    )
]
assert controller["status"] == "complete"
for key in ("controller_pid", "worker_pid"):
    pid = controller.get(key)
    if pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Process still active")
assert manifest["completed_unix"] <= controller["deadline_unix"]
assert controller["attempts"][-1]["returncode"] == 0
assert all(a["finished_unix"] <= controller["deadline_unix"] for a in controller["attempts"])
repair = root / "repairs/2026-10-13-generation-default"
audit = read(repair / "audit.json")
original = read(repair / "plan.json")
assert file_hash(repair / "plan.json") == audit["original_plan_sha256"]
assert file_hash(root / "plan.json") == audit["amended_plan_sha256"]
assert file_hash(root / "specification.json") == audit["specification_unchanged_sha256"]
assert file_hash(root / "specification.json") == file_hash(repair / "specification.json")
assert file_hash(root / "preflight.json") == file_hash(repair / "preflight.json")
assert controller["started_unix"] == audit["original_started_unix"]
assert controller["deadline_unix"] == audit["immutable_deadline_unix"]
original["source_files_sha256"][audit["changed_source_key"]] = audit["repaired_source_sha256"]
assert original == plan
assert audit["no_forecast_or_update_before_repair"]
assert len(controller["attempts"]) == 2 and controller["attempts"][0]["returncode"] == -15
assert controller["attempts"][0]["finished_unix"] < audit["repaired_unix"] < lock["locked_unix"]
assert (
    controller["plan_sha256"]
    == manifest["plan_sha256"]
    == lock["plan_sha256"]
    == file_hash(root / "plan.json")
)
assert manifest["forecast_lock_sha256"] == file_hash(root / "forecast-lock.json")
assert lock["all_forecasts_before_updates"] and manifest["adapter_restored"]
verify_files(Path.cwd(), plan["source_files_sha256"])
verify_files(Path.cwd(), plan["input_hashes"])
verify_files(Path.cwd(), spec["input_hashes"])
source = Path(spec["source_root"])
verify_files(source, read(source / "factors-manifest.json")["files_sha256"])
verify_files(root, lock["files_sha256"])
verify_files(root, manifest["files_sha256"])
expected = {"forecasts.npz", "forecast-lock.json", "proposals.json", "outcomes/baseline.json"} | {
    f"outcomes/batch-{i:02d}-rate-{rate:g}.json" for i in range(16) for rate in spec["rates"]
}
assert set(manifest["files_sha256"]) == expected
ids = [(q["id"], s) for q in data["probes"] for s in range(spec["evaluation_samples"])]
total = 0
for name in sorted(expected):
    if not name.startswith("outcomes/"):
        continue
    record = read(root / name)
    assert record["plan_sha256"] == manifest["plan_sha256"]
    assert record["forecast_lock_sha256"] == manifest["forecast_lock_sha256"]
    assert record["adapter_restored"]
    assert [(r["id"], r["sample"]) for r in record["rows"]] == ids
    total += len(record["rows"])
    if name.endswith("baseline.json"):
        assert lock["locked_unix"] <= (root / name).stat().st_mtime <= manifest["completed_unix"]
        continue
    assert lock["locked_unix"] < record["update_started_unix"] < record["completed_unix"]
    assert record["completed_unix"] < manifest["completed_unix"]
    assert name == f"outcomes/batch-{record['batch_index']:02d}-rate-{record['rate']:g}.json"
    assert record["style"] == data["batches"][record["batch_index"]]["style"]
proposal = read(root / "proposals.json")
assert proposal["plan_sha256"] == manifest["plan_sha256"]
assert [(r["id"], r["sample"]) for r in proposal["rows"]] == [
    (q["id"], s) for q in data["probes"] for s in range(spec["proposal_samples"])
]
assert total == manifest["evaluation_generations"] == 25088
assert len(proposal["rows"]) == manifest["proposal_generations"] == 1024
assert manifest["updates"] == 48
snapshot = root / "source_snapshot.tar"
if not snapshot.exists():
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(snapshot), audit["repaired_commit"]],
        check=True,
    )
result = {
    "complete": True,
    "verified_unix": time.time(),
    "source_input_output_hashes_verified": True,
    "source_snapshot_sha256": file_hash(snapshot),
    "repair_audit_sha256": file_hash(repair / "audit.json"),
    "plan_sha256": file_hash(root / "plan.json"),
    "manifest_sha256": file_hash(root / "manifest.json"),
    "specification_sha256": file_hash(root / "specification.json"),
    "forecast_lock_sha256": file_hash(root / "forecast-lock.json"),
    "updates": 48,
    "evaluation_generations": total,
    "proposal_generations": 1024,
    "adapter_restored": True,
    "deadline_compliant": True,
    "lock_before_every_update": True,
    "elapsed_seconds_including_repair": manifest["completed_unix"] - controller["started_unix"],
    "successful_attempt_seconds": manifest["completed_unix"]
    - controller["attempts"][-1]["started_unix"],
}
atomic_json(root / "integrity.json", result)
print(json.dumps(result))
