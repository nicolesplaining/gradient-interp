"""Audit completed tangent runs on the original hosts before checkout changes."""

import ast
import itertools
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import torch

from gradient_interp.confirmation import verify_files
from gradient_interp.study import atomic_json, file_hash

root = Path(sys.argv[1]).resolve()


def read(p):
    return json.loads(p.read_text())


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
            raise ValueError("Still active")
assert abs(controller["deadline_unix"] - controller["started_unix"] - 43200) < 0.01
assert manifest["completed_unix"] < controller["deadline_unix"]
assert len(controller["attempts"]) == 1 and controller["attempts"][0]["returncode"] == 0
assert controller["attempts"][0]["finished_unix"] < controller["deadline_unix"]
ph = file_hash(root / "plan.json")
assert controller["plan_sha256"] == manifest["plan_sha256"] == lock["plan_sha256"] == ph
assert manifest["forecast_lock_sha256"] == file_hash(root / "forecast-lock.json")
assert manifest["adapter_restored"] and lock["adapter_unchanged"]
for hashes in (plan["source_files_sha256"], plan["input_hashes"], spec["input_hashes"]):
    verify_files(Path.cwd(), hashes)
verify_files(root, lock["files_sha256"])
verify_files(root, manifest["files_sha256"])
source = Path(spec["source_root"])
verify_files(source, read(source / "factors-manifest.json")["files_sha256"])
assert lock["gradient_cache_sha256"] == file_hash(Path(spec["gradient_cache"]))
cache = torch.load(spec["gradient_cache"], map_location="cpu", weights_only=True)
assert cache["batch_gradients"].dtype == torch.float32
assert cache["batch_gradients"].shape == (16, 2228224)
assert torch.isfinite(cache["batch_gradients"]).all()
forecasts = {"forecasts/baseline.json"} | {
    f"forecasts/batch-{b:02d}-rate-{rate:g}-{method}.json"
    for b, rate, method in itertools.product(range(4), spec["rates"], spec["methods"][:-1])
}
outcomes = {"outcomes/baseline.json"} | {
    f"outcomes/batch-{b:02d}-rate-{rate:g}.json"
    for b, rate in itertools.product(range(4), spec["rates"])
}
assert set(lock["files_sha256"]) == forecasts
assert set(manifest["files_sha256"]) == forecasts | outcomes | {"forecast-lock.json"}
for names, count, is_forecast in ((forecasts, 32, True), (outcomes, 128, False)):
    expected = [(q["id"], s) for q in data["probes"] for s in range(count)]
    for name in names:
        r = read(root / name)
        assert r["plan_sha256"] == ph
        assert [(x["id"], x["sample"]) for x in r["rows"]] == expected
        assert all(0 < len(x["token_ids"]) <= 96 for x in r["rows"])
        if is_forecast:
            assert r["original_adapter_unchanged"]
            assert r["completed_unix"] <= lock["locked_unix"]
        else:
            assert r["forecast_lock_sha256"] == manifest["forecast_lock_sha256"]
            assert lock["locked_unix"] < r["completed_unix"] <= manifest["completed_unix"]
            if name != "outcomes/baseline.json":
                assert r["adapter_restored"]
                assert lock["locked_unix"] < r["update_started_unix"] < r["completed_unix"]
assert manifest["updates"] == 12
assert manifest["forecast_conditions"] == len(forecasts) == 49
assert manifest["evaluation_conditions"] == len(outcomes) == 13
assert manifest["forecast_generations"] == len(forecasts) * 4 * 32 == 6272
assert manifest["evaluation_generations"] == len(outcomes) * 4 * 128 == 6656
repair = root / "repairs/2026-10-13-jvp-memory"
before, after = [read(repair / n) for n in ("audit-before.json", "audit-after.json")]
old = read(repair / "specification.json")
assert file_hash(repair / "specification.json") == before["original_spec_sha256"]
assert file_hash(root / "specification.json") == after["amended_specification_sha256"]
assert [k for k in spec if spec[k] != old[k]] == ["chunk"]
assert old["chunk"] == 16 and spec["chunk"] == 4
assert file_hash(root / "preflight.json") == after["preflight_sha256"]
assert before["created_unix"] < after["created_unix"] < plan["frozen_unix"]
deployment, launch = [read(root / n) for n in ("deployment-audit.json", "launch-audit.json")]
assert deployment["created_unix"] < plan["frozen_unix"] < controller["started_unix"]
assert launch["created_unix"] < controller["started_unix"]
assert file_hash(root / "preflight_source.py") == deployment["preflight_source_sha256"]
current_source = Path("src/gradient_interp/paper_tangent.py")
assert file_hash(current_source) == deployment["final_source_sha256"]


def functions(path):
    return {
        n.name: ast.dump(n)
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }


tested, final = functions(root / "preflight_source.py"), functions(current_source)
assert tested.keys() == final.keys()
assert [name for name in final if final[name] != tested[name]] == ["freeze"]
snapshot = root / "source_snapshot.tar"
with tarfile.open(snapshot) as tar:
    import hashlib

    for path, digest in plan["source_files_sha256"].items():
        assert hashlib.sha256(tar.extractfile(path).read()).hexdigest() == digest
result = dict(
    complete=True,
    verified_unix=time.time(),
    source_input_output_hashes_verified=True,
    source_snapshot_sha256=file_hash(snapshot),
    plan_sha256=ph,
    manifest_sha256=file_hash(root / "manifest.json"),
    specification_sha256=file_hash(root / "specification.json"),
    forecast_lock_sha256=file_hash(root / "forecast-lock.json"),
    updates=12,
    forecast_generations=6272,
    evaluation_generations=6656,
    original_deadline_verified=True,
    precision_and_memory_amendment_verified=True,
    deployment_and_launcher_audits_verified=True,
)
atomic_json(root / "integrity.json", result)
print(json.dumps(result))
