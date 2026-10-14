"""Audit completed atom-batch runs on their original hosts, without a GPU model load."""

import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from gradient_interp.confirmation import verify_files
from gradient_interp.paper_atom_batches import (
    ENDPOINTS,
    atom_forecasts,
    covariance_effects,
    reward,
)
from gradient_interp.paper_atom_data import checks
from gradient_interp.paper_steering_analysis import quality
from gradient_interp.study import atomic_json, file_hash

root = Path(sys.argv[1]).resolve()


def read(p):
    return json.loads(p.read_text())


c, p, s, m, lock, data, cards = [
    read(root / n)
    for n in (
        "controller.json",
        "plan.json",
        "specification.json",
        "manifest.json",
        "forecast-lock.json",
        "data.json",
        "atom-cards.json",
    )
]
assert c["status"] == "complete" and len(c["attempts"]) == 1
assert c["attempts"][0]["returncode"] == 0
assert abs(c["deadline_unix"] - c["started_unix"] - 43200) < 0.01
assert m["completed_unix"] < c["attempts"][0]["finished_unix"] < c["deadline_unix"]
for pid in (c["controller_pid"], c["attempts"][0]["pid"]):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("Run process remains alive")
ph, lh = file_hash(root / "plan.json"), file_hash(root / "forecast-lock.json")
assert c["plan_sha256"] == m["plan_sha256"] == lock["plan_sha256"] == ph
assert m["forecast_lock_sha256"] == lh
assert m["adapter_restored"] and lock["original_adapter_unchanged"]
for hashes in (p["source_files_sha256"], p["input_hashes"], s["input_hashes"]):
    verify_files(Path.cwd(), hashes)
source = Path(s["source_root"])
verify_files(source, read(source / "factors-manifest.json")["files_sha256"])
verify_files(root, lock["files_sha256"])
verify_files(root, m["files_sha256"])
with tarfile.open(root / "source_snapshot.tar") as tar:
    for name, digest in p["source_files_sha256"].items():
        assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest
pre = read(root / "preflight.json")
assert pre["specification_sha256"] == file_hash(root / "specification.json")
assert pre["tensors"] == 136 and pre["gradient_elements"] == 2228224
assert pre["adapter_restored"] and pre["generation_restored"] and pre["finite_gradient"]
assert pre["max_score_error_per_token"] < 0.005
assert pre["completed_unix"] < p["frozen_unix"] < c["started_unix"]
cache = torch.load(root / "representations.pt", map_location="cpu", weights_only=True)
assert cache["plan_sha256"] == ph
for name, shape in (
    ("batch_gradients", (16, 2228224)),
    ("svd_gradients", (16, 2228224)),
    ("sparse_codes", (16, 500)),
    ("projected", (16, 6800)),
    ("projected_norms", (16, 1)),
):
    assert cache[name].shape == shape and cache[name].dtype == torch.float32
    assert torch.isfinite(cache[name]).all()
z = dict(np.load(root / "forecasts.npz"))
for name, shape in (
    ("units", (24, 128, 532)),
    ("rewards", (24, 128, 8)),
    ("effects", (24, 532, 8)),
    ("coefficients", (16, 500)),
    ("predictions", (16, 6, 24, 8)),
    ("atom_contributions", (16, 24, 500, 8)),
    ("condition_predictions", (72, 2, 24, 8)),
):
    assert z[name].shape == shape and np.isfinite(z[name]).all()
np.testing.assert_array_equal(
    z["coefficients"], (cache["sparse_codes"] * cache["projected_norms"]).numpy()
)
np.testing.assert_allclose(
    covariance_effects(z["units"], z["rewards"]), z["effects"], rtol=1e-10, atol=1e-10
)
np.testing.assert_allclose(
    z["atom_contributions"].sum(2), z["predictions"][:, 0], rtol=1e-7, atol=1e-10
)
np.testing.assert_allclose(
    atom_forecasts(z["coefficients"], z["effects"][:, :500], s["rate"]), z["predictions"][:, 0]
)
np.testing.assert_array_equal(
    z["permutation"], np.random.default_rng(s["control_seed"]).permutation(500)
)
assert np.count_nonzero(z["predictions"][:, -1]) == 0
assay = read(root / "semantic-assay.json")
assert assay["plan_sha256"] == ph and assay["completed_unix"] < lock["locked_unix"]
assert len(assay["rows"]) == 48
assert {(r["id"], r["format"], r["long"]) for r in assay["rows"]} == {
    (q["id"], fmt, long)
    for q in data["semantic"]
    for fmt in ("plain", "numbered", "bullets")
    for long in (False, True)
}
assert np.isfinite([r["sparse_codes"] for r in assay["rows"]]).all()
conditions = read(root / "conditions.json")
assert conditions["plan_sha256"] == ph and conditions["chosen_before_outcomes"]
conditions = conditions["conditions"]
expected = {
    f"batch-{b:02d}-{mode}"
    for b in range(16)
    for mode in ("full", "remove_named", "remove_random", "rescale")
}
expected |= {
    f"atom-{label}-{strength:g}"
    for label in ("numbered", "bullets")
    for strength in (-1.0, -0.5, 0.5, 1.0)
}
assert len(conditions) == 72 and {x["id"] for x in conditions} == expected
assert set(m["files_sha256"]) == {f"outcomes/{name}.json" for name in expected | {"baseline"}} | {
    "forecast-lock.json"
}


def check_rows(rows, probes, count):
    assert [(r["id"], r["sample"]) for r in rows] == [
        (q["id"], i) for q in probes for i in range(count)
    ]
    for q in probes:
        for r in rows:
            if r["id"] != q["id"]:
                continue
            assert 0 < len(r["token_ids"]) <= 96
            assert r["checks"] == checks(r["response"], q["expected"], q["task"])
            assert r["quality"] == quality(r, 96)


proposals = read(root / "proposals.json")
assert proposals["plan_sha256"] == ph and proposals["completed_unix"] < lock["locked_unix"]
check_rows(proposals["rows"], data["probes"], 128)
np.testing.assert_array_equal(
    np.array([reward(r) for r in proposals["rows"]]).reshape(24, 128, len(ENDPOINTS)), z["rewards"]
)
eval_probes = [q for q in data["probes"] if q["split"] == "evaluation"]
assert len(eval_probes) == 16
for condition in [dict(id="baseline", kind="baseline"), *conditions]:
    r = read(root / "outcomes" / (condition["id"] + ".json"))
    assert (
        r["condition"] == condition and r["plan_sha256"] == ph and r["forecast_lock_sha256"] == lh
    )
    assert r["adapter_restored"]
    assert (
        lock["locked_unix"] < r["update_started_unix"] < r["completed_unix"] <= m["completed_unix"]
    )
    check_rows(r["rows"], eval_probes, 64)

# Reconstruct only the physical directions used by the controls. No model/gradients are recomputed.
sys.path.insert(0, "/home/ubuntu/gradient-interp-upstream/gradient_atoms/src")
from gradient_atoms.projection import load_ekfac_eigen, unproject_atom  # noqa: E402

payload = torch.load(source / "atoms.pt", map_location="cpu", weights_only=True)
factors = load_ekfac_eigen(str(source / "factors/paper_ekfac/factors_training"))
indices = {x["target"] for x in conditions if x["kind"] == "batch"} | {
    x["random_atom"] for x in conditions if x["kind"] == "batch"
}
directions = {
    i: unproject_atom(
        payload["dictionary"][i].numpy(), payload["module_info"], factors, 2228224
    ).float()
    for i in indices
}
norm_residuals = []
rng = np.random.default_rng(s["control_seed"])
named = {v["atom_index"] for v in cards.values()}
for b in range(16):
    cc = [x for x in conditions if x.get("batch_index") == b]
    c0 = cc[0]
    target = c0["target"]
    other = c0["random_atom"]
    active = [i for i, x in enumerate(z["coefficients"][b]) if x != 0 and i not in named]
    assert other == int(rng.choice(active or [i for i in range(500) if i not in named]))
    g = cache["batch_gradients"][b]
    removed = g - c0["named_coefficient"] * directions[target]
    np.testing.assert_allclose(
        abs(c0["named_coefficient"]) * directions[target].norm(),
        abs(c0["random_coefficient"]) * directions[other].norm(),
        rtol=2e-6,
        atol=1e-10,
    )
    assert float(removed.norm() / g.norm()) == c0["norm_scale"]
    norm_residuals.append(
        dict(
            batch=b,
            fp64_remaining_norm_relative_error=float(
                (removed.double().norm() - (g * c0["norm_scale"]).double().norm()).abs()
                / removed.double().norm()
            ),
        )
    )
assert m["counts"] == s["counts"]
audit = dict(
    complete=True,
    verified_unix=time.time(),
    source_input_output_hashes_verified=True,
    source_snapshot_sha256=file_hash(root / "source_snapshot.tar"),
    plan_sha256=ph,
    specification_sha256=file_hash(root / "specification.json"),
    manifest_sha256=file_hash(root / "manifest.json"),
    forecast_lock_sha256=lh,
    proposal_responses=3072,
    evaluation_responses=74752,
    semantic_gradients=48,
    updates=72,
    evaluation_conditions=73,
    locks_restores_verified=True,
    physical_control_formula_verified=True,
    physical_norm_residuals=norm_residuals,
    fp32_cache_verified=True,
    original_deadline_verified=True,
    unique_training_gradient_calls=cache["unique_training_gradient_calls"],
)
atomic_json(root / "integrity.json", audit)
print(json.dumps(audit))
