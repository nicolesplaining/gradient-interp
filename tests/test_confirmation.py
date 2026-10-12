import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from gradient_interp.confirmation import forecast_arrays, removal_vectors, run_confirmation_job
from gradient_interp.confirmation_analysis import aggregate, bootstrap_comparisons, center
from gradient_interp.confirmation_data import partition_fresh
from gradient_interp.confirmation_run import preserved_budget
from gradient_interp.data import Example, make_demo_data, read_jsonl, write_jsonl
from gradient_interp.dictionary import Projection, SparseDictionary
from gradient_interp.models import ModelConfig
from gradient_interp.study import atomic_json, file_hash


def test_fresh_partition_has_disjoint_sources_and_balanced_probes():
    examples = [
        Example(f"{g}-{i}", f"Prompt {g} {i}", "A", g)
        for g in ("a", "b", "c", "d", "e", "f", "g", "h")
        for i in range(300)
    ]
    dev, batches, cal, evaluation = partition_fresh(examples)
    assert len(batches) == 96
    assert len(cal) == len(evaluation) == 256
    ids = [e["id"] for b in dev + batches for e in b["examples"]] + [e.id for e in cal + evaluation]
    assert len(ids) == len(set(ids))
    assert (dev, batches, cal, evaluation) == partition_fresh(examples)


def test_full_rank_predictions_preserve_sign_units_and_probe_order():
    torch.manual_seed(12)
    gradient, cal, evaluation = torch.randn(3, 8), torch.randn(5, 8), torch.randn(6, 8)
    projection = Projection(torch.eye(8), torch.arange(1, 9).float())
    dictionary = SparseDictionary(torch.eye(8), 8, [], "ksvd")
    methods, _, (cp, ep) = forecast_arrays(gradient, projection, dictionary, cal, evaluation, 1)
    for name in ("svd_8", "dictionary_omp_8", "dictionary_refit", "full_gradient"):
        np.testing.assert_allclose(
            ep[methods.index(name)], (-gradient @ evaluation.T).numpy(), atol=3e-6
        )
        np.testing.assert_allclose(cp[methods.index(name)], (-gradient @ cal.T).numpy(), atol=3e-6)
    components = torch.randn(8)
    controls = removal_vectors(gradient[0], components, 3)
    torch.testing.assert_close((gradient[0] - controls["random_removed"]).norm(), components.norm())
    torch.testing.assert_close(controls["rescaled_full"].norm(), controls["atom_removed"].norm())


def test_probe_bootstrap_is_paired_and_constant_shifts_do_not_count():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(3, 6, 3, 5))
    p = y + rng.normal(size=y.shape)
    batches = [{"kind": "pure", "batch_size": 4} for _ in range(6)]
    candidates = [("identical", p, p, "fixed"), ("better", y + 0.1 * (p - y), p, "fixed")]
    stats = bootstrap_comparisons(y, candidates, batches, repeats=96)
    assert stats["identical"]["crossed_95ci"] == [1.0, 1.0]
    np.testing.assert_allclose(stats["better"]["crossed_95ci"], [0.1, 0.1], atol=1e-12)
    np.testing.assert_allclose(center(y.mean(-1) + 100), center(y.mean(-1)), atol=1e-12)


def test_budget_cannot_be_extended_on_resume():
    previous = {"started_unix": 100, "deadline_unix": 200}
    assert preserved_budget(previous, 190, 24) == (100, 200)
    assert preserved_budget(previous, 201, 24) == (100, 200)
    with pytest.raises(ValueError, match="at most"):
        preserved_budget(None, 100, 25)


def test_confirmation_interrupt_resume_lock_and_analysis(tmp_path):
    data = tmp_path / "data"
    make_demo_data(data)
    probes = read_jsonl(data / "probes.jsonl")
    write_jsonl(data / "calibration_probes.jsonl", probes)
    write_jsonl(
        data / "evaluation_probes.jsonl",
        [{**p, "id": "new-" + p["id"], "prompt": "New " + p["prompt"]} for p in probes],
    )
    batches = read_jsonl(data / "batches.jsonl")[:2]
    for b in batches:
        b.update(kind="pure", batch_size=len(b["examples"]), target_group="identity")
    write_jsonl(data / "batches.jsonl", batches)
    atomic_json(
        data / "manifest.json",
        {"files_sha256": {p.name: file_hash(p) for p in data.glob("*.jsonl")}},
    )
    job = {
        "id": "tiny",
        "seed": 10,
        "warmup_steps": 1,
        "model": asdict(ModelConfig(width=8, score_batch_size=4)),
    }
    plan = {
        "benchmark": str(data),
        "jobs": [job],
        "threads": 1,
        "rank": 16,
        "atoms": 16,
        "alpha": 0.1,
        "dictionary_iterations": 2,
        "control_sparsity": 8,
        "learning_rates": [0.0001, 0.001],
        "control_batch_indices": [0],
        "generation_tokens": 0,
        "protocol": {
            "primary_rate": 0.0001,
            "primary_comparisons_per_model": [
                ["fixed", "svd_12"],
                ["transfer", "svd_12"],
                ["fixed", "rotated_dictionary_omp_8"],
            ],
        },
    }
    output = tmp_path / "jobs/tiny"

    def interrupt(message):
        if message == "evaluating fixed case 2/4":
            assert (output / "forecast_lock.json").exists()
            assert (output / "results/batch-000-rate-0.npz").exists()
            raise InterruptedError("simulated interruption")

    with pytest.raises(InterruptedError):
        run_confirmation_job(plan, job, output, interrupt)
    lock_hash = file_hash(output / "forecast_lock.json")
    outcome_hash = file_hash(output / "results/batch-000-rate-0.npz")
    run_confirmation_job(plan, job, output, lambda _: None)
    assert file_hash(output / "forecast_lock.json") == lock_hash
    assert file_hash(output / "results/batch-000-rate-0.npz") == outcome_hash
    assert len(list((output / "results").glob("*.npz"))) == 4
    atomic_json(tmp_path / "plan.json", plan)
    atomic_json(tmp_path / "controller.json", {"status": "running"})
    with pytest.raises(ValueError, match="Do not inspect"):
        aggregate(tmp_path, repeats=64)
    atomic_json(tmp_path / "controller.json", {"status": "complete"})
    result = aggregate(tmp_path, repeats=64)
    assert not any(v["passes"] for v in result["results"][""]["primary"].values())
    assert result["unfinished_jobs"] == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["checkpoint_restored"]
    with (output / "predictions.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="Frozen file changed"):
        run_confirmation_job(plan, job, output, lambda _: None)
