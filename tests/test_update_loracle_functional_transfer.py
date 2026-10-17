import copy

import numpy as np
import pytest

from gradient_interp import update_loracle_functional_transfer as worker
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import PATTERN_IDS


def records(n=48):
    rng = np.random.default_rng(192)
    x = rng.normal(size=(n, 32))
    x -= x.mean(axis=1, keepdims=True)
    x[-2:] = 0
    y = np.roll(x, 3, axis=1) * 1.7
    return [
        {
            "id": str(i),
            "split": "train",
            "control": i >= n - 2,
            "profile": a.tolist(),
            "target": b.tolist(),
        }
        for i, (a, b) in enumerate(zip(x, y, strict=True))
    ]


def test_fixed_mapping_recovers_signed_profiles_and_preserves_zero_and_scale():
    data = records()
    original = copy.deepcopy(data)
    model = worker.fit_training(data)
    rng = np.random.default_rng(89)
    x = rng.normal(size=(20, 32))
    x -= x.mean(axis=1, keepdims=True)
    predicted, labels = worker.infer(model, x)
    assert np.mean((predicted - np.roll(x, 3, axis=1) * 1.7) ** 2) < 0.01
    assert len(labels) == 20 and any(np.min(p) < 0 for p in predicted)
    scaled, _ = worker.infer(model, 2 * x)
    np.testing.assert_allclose(scaled, 2 * predicted, atol=1e-12)
    zero, zlabels = worker.infer(model, np.zeros((2, 32)))
    assert not zero.any() and zlabels == ["no_change", "no_change"]
    expected = 0.01 * np.mean(
        [np.sum(np.array(r["profile"]) ** 2) for r in data if not r["control"]]
    )
    assert float(model["regularization"]) == pytest.approx(expected)
    assert data == original


@pytest.mark.parametrize("change", ["split", "duplicate", "nan", "shape", "uncentered", "zero"])
def test_invalid_or_nontraining_data_rejected(change):
    data = records()
    if change == "split":
        data[0]["split"] = "development"
    elif change == "duplicate":
        data[1]["id"] = data[0]["id"]
    elif change == "nan":
        data[0]["target"][0] = float("nan")
    elif change == "shape":
        for row in data:
            row["profile"] = row["profile"][:31]
    elif change == "uncentered":
        data[0]["profile"][0] += 1
    else:
        data[-1]["profile"][0:2] = [1, -1]
    with pytest.raises(ValueError):
        worker.fit_training(data)


def test_models_with_same_training_data_ignore_external_development_targets():
    data = records()
    first = worker.fit_training(data)
    unrelated_development = [{**r, "split": "development", "target": [999.0] * 32} for r in data]
    assert unrelated_development[0]["target"][0] == 999
    second = worker.fit_training(data)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    with pytest.raises(ValueError):
        worker.infer(first, np.full((1, 32), float("nan")))


def completed_fixture(tmp_path):
    v5, targets = tmp_path / "v5", tmp_path / "targets"
    v5.mkdir()
    targets.mkdir()
    (v5 / "analysis").mkdir()
    rows, cases, hashes = [], [], {}
    methods = (
        "released",
        "flat_majority",
        "factorized_majority",
        "ridge",
        "epoch1",
        "physical_profile",
        "sensitivity",
        "sensitivity_shuffled",
        "functional",
        "functional_shuffled",
    )
    for split, n, zeros in (("train", 256, 16), ("development", 64, 8)):
        for i in range(n + zeros):
            control = i >= n
            profile = np.zeros(32)
            if not control:
                profile[i % 32] = 1
                profile -= profile.mean()
            target = np.roll(profile, 3)
            effect = {
                "centered_improvements": target.tolist(),
                "mean_ce_changes": (-target).tolist(),
                "label": worker.profile_label(target),
                "below_effect_floor": control,
                "span": float(np.ptp(target)),
            }
            row = {
                "id": f"{split}-{i:04d}",
                "split": split,
                "control": control,
                "profile": profile.tolist(),
                "shuffled_profile": np.roll(profile, 10).tolist(),
            }
            rows.append(row)
            if split == "train":
                p = targets / f"{row['id']}.json"
                atomic_json(p, {**row, "effect": effect})
                hashes[f"/frozen/cases/{p.name}"] = file_hash(p)
            else:
                cases.append(
                    {
                        **row,
                        "source_pattern": PATTERN_IDS[(i // 8) % 8],
                        "initialization_seed": 10 + (i // 2) % 4,
                        "batch": [],
                        "probes": [],
                        "label": effect["label"],
                        "measured": {"effect": effect},
                        "exact_comparison_predictions": {k: "no_change" for k in methods},
                    }
                )
    atomic_json(v5 / "predictions.json", rows)
    atomic_json(v5 / "data.json", {"cases": rows})
    atomic_json(v5 / "analysis/all-development-cases.json", cases)
    atomic_json(v5 / "integrity.json", {"synthetic_fixture": True})
    atomic_json(v5 / "local-copy-receipt.json", {"synthetic_fixture": True})
    return v5, targets, {"collection_root": "/frozen", "input_files_sha256": hashes}


def test_all272_target_hashes_and_mean_loss_derivation_required(tmp_path):
    v5, targets, plan = completed_fixture(tmp_path)
    rows = worker.read(v5 / "predictions.json")
    data, hashes = worker.collect_training(rows, targets, plan)
    assert len(data) == len(hashes) == 272
    p = targets / "train-0000.json"
    t = worker.read(p)
    t["effect"]["centered_improvements"][0] += 1
    atomic_json(p, t)
    with pytest.raises(ValueError, match="target changed"):
        worker.collect_training(rows, targets, plan)
    plan["input_files_sha256"]["/frozen/cases/train-0000.json"] = file_hash(p)
    with pytest.raises(ValueError, match="derivation"):
        worker.collect_training(rows, targets, plan)


def test_full_cpu_fit_predict_lock_then_all_method_analysis(tmp_path, monkeypatch):
    v5, targets, plan = completed_fixture(tmp_path)
    monkeypatch.setattr(worker, "checked_v5", lambda _: plan)
    root, output = tmp_path / "run", tmp_path / "output"
    original_read = worker.read

    def check_target_join(path):
        if path == v5 / "analysis/all-development-cases.json":
            assert (root / "model-lock.json").is_file()
            assert (root / "predictions-lock.json").is_file()
        return original_read(path)

    monkeypatch.setattr(worker, "read", check_target_join)
    worker.run(v5, targets, root, output)
    summary = original_read(output / "summary.json")
    assert len(summary["metrics"]) == 12 and summary["replicates"] == 2000
    assert summary["metrics"]["transfer"]["actual_update_correct"] == 64
    assert summary["metrics"]["transfer_shuffled"]["actual_update_correct"] == 0
    assert summary["metrics"]["transfer"]["zero_correct"] == 8
    assert len(original_read(output / "all-development-cases.json")) == 72
    assert len(original_read(root / "predictions.json")) == 344
    assert len(original_read(root / "model-lock.json")["training_ids"]) == 272
    assert original_read(output / "validation.json")["passed"]
    assert (output / "report.html").is_file() and (root / "source_snapshot.tar").is_file()
    with pytest.raises(ValueError, match="overwrite"):
        worker.run(v5, targets, root, output)


def test_active_prior_and_incomplete_audit_rejected(tmp_path):
    atomic_json(tmp_path / "controller.json", {"status": "running"})
    with pytest.raises(ValueError):
        worker.checked_v5(tmp_path)


def test_corrupt_model_is_detected_at_inference():
    model = worker.fit_training(records())
    model["beta"][0, 0] += 1
    with pytest.raises(ValueError, match="contraction"):
        worker.infer(model, np.ones((2, 32)))


def test_prior_audit_requires_complete_rescoring_and_unchanged_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "completion_barrier", lambda c: None)
    atomic_json(tmp_path / "controller.json", {"status": "complete"})
    atomic_json(tmp_path / "plan.json", {"original": True})
    evidence = tmp_path / "evidence.json"
    atomic_json(evidence, {"fixed": 1})
    hashes = {"evidence.json": file_hash(evidence)}
    audit = {
        "passed": True,
        "recipe": "exact_calibration_v5",
        "cases": 344,
        "scored_adapter_conditions": 688,
        "all_profiles_and_native_norms_verified": True,
        "reserved_cases_accessed": 0,
        "files_sha256": hashes,
        "fixed_scoring_rechecks": [{"passed": True} for _ in range(4)],
    }
    atomic_json(tmp_path / "integrity.json", audit)
    atomic_json(
        tmp_path / "local-copy-receipt.json",
        {"files_sha256": hashes, "handoff": {"stage": "complete"}},
    )
    assert worker.checked_v5(tmp_path) == {"original": True}
    atomic_json(evidence, {"fixed": 2})
    with pytest.raises(ValueError, match="evidence changed"):
        worker.checked_v5(tmp_path)
    atomic_json(evidence, {"fixed": 1})
    audit["fixed_scoring_rechecks"].pop()
    atomic_json(tmp_path / "integrity.json", audit)
    with pytest.raises(ValueError, match="Rechecks"):
        worker.checked_v5(tmp_path)
