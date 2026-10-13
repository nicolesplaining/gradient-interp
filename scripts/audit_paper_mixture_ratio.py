"""Verify the complete direct-likelihood diagnostic on its original host."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
assert manifest["proposal_trajectories_per_condition"] == 8192
preflight = read(root / "preflight.json")
assert preflight["specification_sha256"] == file_hash(root / "specification.json")
assert preflight["adapter_restored"] and preflight["likelihoods_restored_exactly"]
assert max(preflight["single_vs_original_abs_error_per_token"]) == 0
assert preflight["parameter_tensors"] == 136

cache = torch.load(spec["gradient_cache"], map_location="cpu", weights_only=False)
assert cache["batch_gradients"].shape == (16, 2228224)
assert cache["batch_gradients"].dtype == torch.float32
assert torch.isfinite(cache["batch_gradients"]).all()
for name in expected:
    if name.endswith(".npz"):
        with np.load(root / name) as arrays:
            assert arrays["logp"].shape == (16, 512)
            assert np.isfinite(arrays["logp"]).all()
snapshot = root / "source_snapshot.tar"
if not snapshot.exists():
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "-o",
            str(snapshot),
            "e031b89dc2b24f11ed880c36cc44d74103c809b7",
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
    "scores_per_condition": 8192,
    "float32_gradient_cache_verified": True,
    "unique_sequences_per_condition": manifest["unique_sequences_per_condition"],
    "updates": 48,
    "elapsed_seconds": manifest["completed_unix"] - state["started_unix"],
}
atomic_json(root / "integrity.json", result)
print(json.dumps(result))
