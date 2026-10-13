"""Audit a completed batch pilot on its original host and export compact forecasts."""

# ruff: noqa: E501

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1]).resolve()


def read(p):
    return json.loads(p.read_text())


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(base, hashes):
    for name, expected in hashes.items():
        p = Path(name)
        if not p.is_absolute():
            p = base / p
        assert digest(p) == expected, p


controller, plan, spec, manifest, lock, data = [
    read(root / n)
    for n in (
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
assert all(
    a["returncode"] == 0 and a["finished_unix"] <= controller["deadline_unix"]
    for a in controller["attempts"]
)
assert (
    controller["plan_sha256"]
    == manifest["plan_sha256"]
    == lock["plan_sha256"]
    == digest(root / "plan.json")
)
assert manifest["forecast_lock_sha256"] == digest(root / "forecast-lock.json")
assert lock["all_forecasts_before_updates"] and manifest["adapter_restored"]
check(Path.cwd(), plan["source_files_sha256"])
check(Path.cwd(), plan["input_hashes"])
check(Path.cwd(), spec["input_hashes"])
source = Path(spec["source_root"])
check(source, read(source / "factors-manifest.json")["files_sha256"])
check(root, lock["files_sha256"])
check(root, manifest["files_sha256"])
expected = {"forecasts.pt", "forecast-lock.json", "outcomes/baseline.json"} | {
    f"outcomes/batch-{i:02d}-rate-{rate:g}.json" for i in range(16) for rate in spec["rates"]
}
assert set(manifest["files_sha256"]) == expected
expected_ids = [x["id"] for x in data["evaluation"]]
mtime = {}
for name in sorted(expected):
    if not name.startswith("outcomes/"):
        continue
    p = root / name
    row = read(p)
    assert row["plan_sha256"] == manifest["plan_sha256"]
    assert [x["id"] for x in row["generations"]] == expected_ids
    assert lock["locked_unix"] <= p.stat().st_mtime <= manifest["completed_unix"]
    mtime[name] = p.stat().st_mtime
    if name.endswith("baseline.json"):
        continue
    assert row["adapter_restored"] and row["forecast_lock_sha256"] == digest(
        root / "forecast-lock.json"
    )
    assert name == f"outcomes/batch-{row['batch_index']:02d}-rate-{row['rate']:g}.json"
    assert row["style"] == data["batches"][row["batch_index"]]["style"]
    assert np.asarray(row["actual_logp"]).shape == (32, 4)
assert (
    manifest["updates"] == 48
    and manifest["baseline_generations"] == 16
    and manifest["post_update_generations"] == 768
)
snapshot = root / "source_snapshot.tar"
if not snapshot.exists():
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "-o",
            str(snapshot),
            "c9835c0f1d551384f07e6028e0bf81b682ee5d85",
        ],
        check=True,
    )
saved = torch.load(root / "forecasts.pt", map_location="cpu", weights_only=True)
keys = [
    "baseline_logp",
    "completion_tokens",
    "unit_rate_forecasts",
    "sparse_codes",
    "projected_norms",
]
arrays = {k: saved[k].numpy() for k in keys}
arrays["batch_gradient_norms"] = saved["batch_gradients"].norm(dim=1).numpy()
assert arrays["unit_rate_forecasts"].shape == (16, 6, 32, 4)
assert all(np.isfinite(v).all() for v in arrays.values())
np.savez_compressed(root / "compact_forecasts.npz", **arrays)
audit = {
    "complete": True,
    "verified_unix": time.time(),
    "source_input_output_hashes_verified": True,
    "source_snapshot_sha256": digest(snapshot),
    "manifest_sha256": digest(root / "manifest.json"),
    "specification_sha256": digest(root / "specification.json"),
    "plan_sha256": digest(root / "plan.json"),
    "compact_forecasts_sha256": digest(root / "compact_forecasts.npz"),
    "original_forecast_sha256": digest(root / "forecasts.pt"),
    "adapter_restored": True,
    "updates": 48,
    "generations": 784,
    "deadline_compliant": True,
    "runtime_seconds": manifest["completed_unix"] - controller["started_unix"],
    "forecast_locked_unix": lock["locked_unix"],
    "outcome_mtimes": mtime,
    "order_evidence": "Code saves all forecasts before any set_direction call; lock timestamp precedes every outcome file mtime. Individual update times not separately embedded; filesystem times are supporting evidence only.",
}
(root / "integrity.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps({k: v for k, v in audit.items() if k not in ["outcome_mtimes", "order_evidence"]}))
