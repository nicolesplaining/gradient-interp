import json
from dataclasses import asdict

import numpy as np
import pytest

from gradient_interp.data import Example, Probe, write_jsonl
from gradient_interp.feature_pilot import METHODS, analyze, run_job
from gradient_interp.feature_run import pilot_budget
from gradient_interp.models import ModelConfig
from gradient_interp.study import atomic_json, file_hash


def test_resume_keeps_the_original_pilot_deadline():
    start, deadline = pilot_budget(None, 1000, 2)
    assert (start, deadline) == (1000, 8200)
    assert pilot_budget({"started_unix": start, "deadline_unix": deadline}, 5000, 12) == (
        start,
        deadline,
    )


def test_offline_pilot_locks_forecasts_restores_weights_and_refuses_active_analysis(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    fit = [Example(str(i), f"Task {i}:", f" result {i % 2}", str(i % 2)) for i in range(8)]
    write_jsonl(data / "fit.jsonl", [asdict(e) for e in fit])
    batches = [
        {"id": "a", "examples": [asdict(Example("b1", "New:", " result 0", "0"))]},
        {"id": "b", "examples": [asdict(Example("b2", "Else:", " result 1", "1"))]},
    ]
    write_jsonl(data / "batches.jsonl", batches)
    for name in ("calibration", "evaluation"):
        probes = [Probe(f"{name}{i}", f"{name}{i}:", f" result {i}", group=str(i)) for i in (0, 1)]
        write_jsonl(data / f"{name}_probes.jsonl", [asdict(p) for p in probes])
    job = {
        "id": "offline",
        "seed": 14,
        "module": "layers.0.linear2",
        "n_features": 16,
        "model": asdict(
            ModelConfig(width=8, max_length=48, trainable_pattern=r"layers\.0\.linear2\.weight$")
        ),
    }
    plan = {
        "jobs": [job],
        "threads": 1,
        "train_documents": 6,
        "sae_k": 4,
        "sae_epochs": 2,
        "sae_learning_rate": 0.001,
        "learning_rates": [0.001],
    }
    atomic_json(tmp_path / "plan.json", plan)
    atomic_json(
        tmp_path / "freeze.json",
        {
            "plan_sha256": file_hash(tmp_path / "plan.json"),
            "data_sha256": {p.name: file_hash(p) for p in data.iterdir()},
            "source_sha256": {},
        },
    )
    run_job(tmp_path, job)
    directory = tmp_path / "jobs/offline"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["checkpoint_restored"]
    locked = json.loads((directory / "forecast_lock.json").read_text())["saved_unix"]
    assert min(p.stat().st_mtime for p in (directory / "results").iterdir()) >= locked
    with np.load(directory / "predictions.npz") as z:
        assert z["predictions"].shape == (len(METHODS), 2, 2, 2)
        assert np.isfinite(z["predictions"]).all()
    atomic_json(tmp_path / "controller.json", {"status": "running"})
    with pytest.raises(ValueError, match="Wait"):
        analyze(tmp_path)
    atomic_json(tmp_path / "controller.json", {"status": "complete"})
    analyze(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["results"]["offline"]["status"] == "complete"
    values = summary["results"]["offline"]["rates"]["0.001"]
    assert values["full_gradient"]["fixed_individual_probes"]["relative_l2_error"] < 0.1
    previous = (directory / "manifest.json").read_bytes()
    run_job(tmp_path, job)
    assert (directory / "manifest.json").read_bytes() == previous
