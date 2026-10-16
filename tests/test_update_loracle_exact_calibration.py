import copy
import time

import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_exact_calibration as worker
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import NO_CHANGE, PATTERN_IDS, PATTERNS


def test_exact_profile_equal_document_weight_sign_common_shift_and_floor():
    before = np.arange(256).reshape(8, 32) / 100 + 1
    delta = np.tile(np.linspace(-0.25, 0.25, 32), (8, 1))
    result = worker.effect_profile(before, before + delta)
    np.testing.assert_allclose(result["ce_change"], delta.mean(axis=0), atol=1e-15)
    assert result["prediction"] == PATTERN_IDS[0]
    np.testing.assert_allclose(
        result["profile"], worker.effect_profile(before, before + delta + 3)["profile"], atol=1e-15
    )
    assert worker.effect_profile(before, before + 0.01 * delta)["prediction"] == NO_CHANGE
    with pytest.raises(ValueError, match="grid"):
        worker.effect_profile(before[:4], before[:4])


def test_native_scaling_preserves_own_initial_A_and_physical_direction():
    rng = torch.Generator().manual_seed(7)
    a, b = torch.randn(3, 5, generator=rng), torch.randn(4, 3, generator=rng)
    initial, updated = {"m": (a, torch.zeros_like(b))}, {"m": (a, b)}
    original = copy.deepcopy(updated)
    for scale in (0.0, 0.25, 1.0, 2.75):
        result = worker.scaled_native(initial, updated, scale)
        assert torch.equal(result["m"][0], a)
        dense = 2 * result["m"][1].double() @ a.double()
        assert worker.physical_norm(result) == pytest.approx(float(dense.norm()), rel=1e-14)
        assert worker.physical_norm(result) == pytest.approx(
            worker.physical_norm(updated) * scale, rel=2e-7
        )
    assert all(torch.equal(x, y) for x, y in zip(updated["m"], original["m"], strict=True))
    for bad, scale in (({"m": (a + 1, b)}, 1), (updated, float("nan")), (updated, -1)):
        with pytest.raises(ValueError):
            worker.scaled_native(initial, bad, scale)
    with pytest.raises(ValueError):
        worker.scaled_native(updated, updated, 1)


def model_fixture(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.ones(2, 3))
            self.b = torch.nn.Parameter(torch.zeros(4, 2))
            self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    model = Model()

    def snapshot(m):
        return {"m": (m.a.detach().clone(), m.b.detach().clone())}

    def restore(m, value):
        with torch.no_grad():
            m.a.copy_(value["m"][0])
            m.b.copy_(value["m"][1])

    monkeypatch.setattr(worker, "snapshot_modules", snapshot)
    monkeypatch.setattr(worker, "restore_modules", restore)
    calibration = [{"prompt": str(i), "values": ["abc", "d", "ef", "ghij"]} for i in range(8)]
    rows_seen = []

    def loss(m, tok, rows):
        assert tok is None and len(rows) == 4
        rows_seen.append(rows)
        z = m.b.sum()
        return torch.stack(
            [
                z * z + z * (1 + len(row["answer"]) / 10) + 2 + int(row["prompt"]) / 100
                for row in rows
            ]
        )

    monkeypatch.setattr(worker, "losses", loss)
    return model, calibration, snapshot(model), rows_seen


def test_scores_use_every_original_pattern_document_and_fixed_chunk(monkeypatch):
    model, calibration, initial, calls = model_fixture(monkeypatch)
    scores = worker.calibration_scores(model, None, calibration)
    assert scores.shape == (8, 32) and len(calls) == 64
    for i, pattern in enumerate(PATTERNS):
        assert calls[2 * i] + calls[2 * i + 1] == worker.calibration_rows(calibration, pattern)
    assert torch.equal(model.a, initial["m"][0]) and not torch.count_nonzero(model.b)


def test_exact_shuffled_adapter_is_scored_nonlinearly_and_restored(monkeypatch):
    model, calibration, initial, _ = model_fixture(monkeypatch)
    baseline = worker.calibration_scores(model, None, calibration)
    updated = {"m": (initial["m"][0], torch.ones_like(initial["m"][1]) * 0.125)}
    own = worker.score_and_restore(model, None, calibration, initial, updated, baseline)
    shuffled = worker.score_and_restore(
        model, None, calibration, initial, worker.scaled_native(initial, updated, 2), baseline
    )
    assert not np.allclose(shuffled - baseline, 2 * (own - baseline))
    np.testing.assert_array_equal(worker.calibration_scores(model, None, calibration), baseline)
    assert not torch.count_nonzero(model.b)


def test_scoring_failure_restores_every_native_tensor(monkeypatch):
    model, calibration, initial, _ = model_fixture(monkeypatch)
    baseline = worker.calibration_scores(model, None, calibration)

    def fail(*args):
        raise RuntimeError("synthetic scoring failure")

    monkeypatch.setattr(worker, "calibration_scores", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        worker.score_and_restore(
            model, None, calibration, initial, {"m": (initial["m"][0], torch.ones(4, 2))}, baseline
        )
    assert torch.equal(model.a, initial["m"][0]) and not torch.count_nonzero(model.b)


def test_donor_mapping_stays_within_split_and_keeps_zero_controls():
    cases = [
        {"id": f"{split}-{i}", "split": split, "control": i == 2}
        for split in ("train", "development")
        for i in range(3)
    ]
    result = worker.donor_assignments(cases)
    assert result == {
        "train-0": "train-1",
        "train-1": "train-0",
        "train-2": "train-2",
        "development-0": "development-1",
        "development-1": "development-0",
        "development-2": "development-2",
    }


def test_complete_worker_locks_all688_conditions_without_retraining_or_test_inputs(
    tmp_path, monkeypatch
):
    model, calibration, _, _ = model_fixture(monkeypatch)
    root, collection = tmp_path / "run", tmp_path / "collection"
    root.mkdir()
    (collection / "snapshots").mkdir(parents=True)
    initials = {
        seed: {"m": (torch.ones(2, 3) * (seed + 1), torch.zeros(4, 2))} for seed in range(8)
    }
    for seed, value in initials.items():
        torch.save(value, collection / "snapshots" / f"initial-{seed}.pt")
    cases = []
    for i in range(344):
        train = i < 272
        zero = (256 <= i < 272) or i >= 336
        seed = i % 4 + (0 if train else 4)
        value = {
            "m": (
                initials[seed]["m"][0],
                torch.full((4, 2), 0 if zero else (i + 1) * 1e-5 / (seed + 1), dtype=torch.float32),
            )
        }
        name = f"case-{i}"
        torch.save(value, collection / "snapshots" / f"{name}.pt")
        cases.append(
            {
                "id": name,
                "split": "train" if train else "development",
                "control": zero,
                "initialization_seed": seed,
                "physical_update_norm": worker.physical_norm(value),
            }
        )
    worker.validate_case_grid(cases)
    bad = copy.deepcopy(cases)
    bad[0]["probes"] = "forbidden evaluation input"
    with pytest.raises(ValueError, match="exclude"):
        worker.validate_case_grid(bad)
    bad = copy.deepcopy(cases)
    bad[-1] = bad[0]
    with pytest.raises(ValueError, match="unique"):
        worker.validate_case_grid(bad)
    atomic_json(root / "data.json", {"cases": cases, "calibration": calibration})
    atomic_json(root / "analysis-plan.json", {"synthetic_fixture": True})
    original_hashes = {str(p): file_hash(p) for p in (collection / "snapshots").iterdir()}
    atomic_json(
        root / "plan.json",
        {
            "recipe": worker.RECIPE,
            "source_commit": "fixture",
            "source_files_sha256": {},
            "input_files_sha256": original_hashes,
            "data_sha256": file_hash(root / "data.json"),
            "analysis_plan_sha256": file_hash(root / "analysis-plan.json"),
            "collection_root": str(collection),
        },
    )
    atomic_json(root / "controller.json", {"deadline_unix": time.time() + 3600})
    monkeypatch.setattr(worker, "load_target", lambda: (model, None))
    monkeypatch.setattr(
        worker, "preflight", lambda *args: atomic_json(args[-1], {"passed": True, "fixture": True})
    )

    # Nonlinear exact scoring fixture; zero baseline is exactly3 across all initializations.
    def scores(m, tok, cal):
        assert cal == calibration and tok is None
        z = float(m.b.detach().sum())
        p = np.arange(32, dtype=float)
        return np.tile(3 + z * p / 100 + z * z * p * p / 1000, (8, 1))

    monkeypatch.setattr(worker, "calibration_scores", scores)
    monkeypatch.setattr(worker, "losses", lambda m, t, rows: torch.full((len(rows),), 3.0))
    worker.run(root)
    manifest = worker.read(root / "manifest.json")
    lock = worker.read(root / "predictions-lock.json")
    assert (
        manifest["scored_adapter_conditions"] == 688 and manifest["target_optimizer_updates"] == 0
    )
    assert len(list((root / "scores").glob("*.json"))) == 688
    assert lock["evaluation_probes_accessed_for_prediction"] is False
    assert all(file_hash(path) == digest for path, digest in original_hashes.items())
    rows = worker.read(root / "predictions.json")
    assert len(rows) == 344 and all(r["prediction"] == NO_CHANGE for r in rows if r["control"])
    for case in cases:
        for kind in ("own", "shuffled"):
            record = worker.read(root / "scores" / f"{case['id']}-{kind}.json")
            assert record["bit_exact_restore"] and record["frozen_base_unchanged"]
            assert record["actual_scaled_norm"] == pytest.approx(
                case["physical_update_norm"], rel=5e-6, abs=1e-10
            )
    with pytest.raises(ValueError, match="already attempted"):
        worker.run(root)
