"""Verify a completed temperature-mixture run on its original host."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

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
repair = root / "repairs/2026-10-13-warm-precision"
before, after = [read(repair / n) for n in ("audit-before.json", "audit-after.json")]
assert file_hash(repair / "specification.json") == before["original_specification_sha256"]
assert file_hash(root / "specification.json") == after["amended_specification_sha256"]
assert file_hash(root / "preflight.json") == after["preflight_sha256"]
assert file_hash(root / "data.json") == after["data_unchanged_sha256"]
old = read(repair / "specification.json")
assert (
    sorted(k for k in set(old) | set(spec) if old.get(k) != spec.get(k)) == after["changed_fields"]
)
assert spec["numerics"] == "float32_sdpa_tf32_disabled"
assert (
    before["written_unix"]
    < after["checked_unix"]
    < plan["frozen_unix"]
    < controller["started_unix"]
)
assert len(controller["attempts"]) == 1 and controller["attempts"][0]["returncode"] == 0
assert abs(controller["deadline_unix"] - controller["started_unix"] - 8 * 3600) < 0.01
preflight = read(root / "preflight.json")
assert (
    preflight["parameter_tensors"] == 136
    and preflight["adapter_restored"]
    and preflight["sampled_generation_restored"]
)
assert max(preflight["generation_score_abs_error_per_token"]) < 0.005
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
expected = {
    "forecasts.npz",
    "forecast-lock.json",
    "proposals.json",
    "batch-representations.pt",
    "outcomes/baseline.json",
} | {f"outcomes/batch-{i:02d}-rate-{rate:g}.json" for i in range(16) for rate in spec["rates"]}
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
    (q["id"], s) for q in data["probes"] for s in range(512)
]
assert total == manifest["evaluation_generations"] == 100352
assert len(proposal["rows"]) == manifest["proposal_generations"] == 8192
assert manifest["updates"] == 48
assert all(
    r["component"] == r["sample"] // 256 and r["component_sample"] == r["sample"] % 256
    for r in proposal["rows"]
)
fresh = torch.load(root / "batch-representations.pt", map_location="cpu", weights_only=True)
assert fresh["batch_gradients"].shape == (16, 2228224)
assert (
    fresh["batch_gradients"].dtype == torch.float32
    and torch.isfinite(fresh["batch_gradients"]).all()
)
assert fresh["sparse_codes"].shape == (16, 500)
assert (fresh["batch_gradients"].norm(dim=1) > 0).all()
snapshot = root / "source_snapshot.tar"
if not snapshot.exists():
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(snapshot), after["repaired_commit"]],
        check=True,
    )
result = {
    "complete": True,
    "verified_unix": time.time(),
    "source_input_output_hashes_verified": True,
    "source_snapshot_sha256": file_hash(snapshot),
    "repair_audit_sha256": file_hash(repair / "audit-after.json"),
    "plan_sha256": file_hash(root / "plan.json"),
    "manifest_sha256": file_hash(root / "manifest.json"),
    "specification_sha256": file_hash(root / "specification.json"),
    "forecast_lock_sha256": file_hash(root / "forecast-lock.json"),
    "updates": 48,
    "evaluation_generations": total,
    "proposal_generations": 8192,
    "adapter_restored": True,
    "deadline_compliant": True,
    "lock_before_every_update": True,
    "elapsed_seconds_including_repair": manifest["completed_unix"] - controller["started_unix"],
    "forecast_stage_seconds": lock["locked_unix"] - manifest["started_unix"],
    "post_lock_seconds": manifest["completed_unix"] - lock["locked_unix"],
    "output_bytes": sum((root / n).stat().st_size for n in manifest["files_sha256"]),
    "successful_attempt_seconds": manifest["completed_unix"]
    - controller["attempts"][-1]["started_unix"],
}
atomic_json(root / "integrity.json", result)
print(json.dumps(result))
