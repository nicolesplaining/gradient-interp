import copy

import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_sensitivity_analysis as analysis
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import NO_CHANGE, PATTERN_IDS
from gradient_interp.update_loracle_sensitivity import centered_profile, contract, matched_shuffle


def test_independent_contraction_matches_signed_dense_physical_derivative():
    rng = torch.Generator().manual_seed(73)
    a = torch.randn(3, 5, generator=rng)
    g = torch.randn(7, 5, generator=rng)
    b = torch.randn(7, 3, generator=rng)
    initial = {"x": (a, torch.zeros_like(b))}
    updates = [{"x": (a, b)}, {"x": (a, -2 * b)}, initial]
    gradients = {"x": 2 * g @ a.T}
    expected = float((g.double() * (2 * b.double() @ a.double())).sum())
    result = analysis.independent_contract(gradients, updates, initial, "cpu")
    np.testing.assert_allclose(result, [expected, -2 * expected, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result, contract(gradients, updates, initial, "cpu"), atol=1e-12)
    with pytest.raises(ValueError, match="initialization"):
        analysis.independent_contract(gradients, [{"x": (-a, b)}], initial, "cpu")


def test_prediction_verification_rejects_unthresholded_or_rescaled_answers():
    cases = [
        {
            "id": str(i),
            "split": split,
            "control": i % 3 == 2,
            "physical_update_norm": 0 if i % 3 == 2 else i + 1,
        }
        for i, split in enumerate(["train"] * 3 + ["development"] * 3)
    ]
    raw = np.zeros((6, 32))
    raw[[0, 3], 0] = -0.01
    raw[[1, 4], 1] = -0.3
    profiles = centered_profile(raw)
    rows = []
    for ids in ([0, 1, 2], [3, 4, 5]):
        shuffled, donors = matched_shuffle(profiles[ids], [cases[i] for i in ids])
        for j, i in enumerate(ids):
            rows.append(
                {
                    **cases[i],
                    "profile": profiles[i].tolist(),
                    "prediction": analysis.profile_label(profiles[i]),
                    "shuffled_profile": shuffled[j].tolist(),
                    "shuffled_prediction": analysis.profile_label(shuffled[j]),
                    "direction_donor": donors[j],
                }
            )
    analysis.verify_outputs(raw, rows, cases)
    changed = copy.deepcopy(rows)
    changed[0]["prediction"] = PATTERN_IDS[0]
    with pytest.raises(ValueError, match="category"):
        analysis.verify_outputs(raw, changed, cases)
    changed = copy.deepcopy(rows)
    changed[1]["profile"][0] += 0.1
    with pytest.raises(AssertionError):
        analysis.verify_outputs(raw, changed, cases)


def test_gates_do_not_promote_axis_or_zero_success():
    metrics = {
        n: {"actual_update_accuracy": 0.1}
        for n in ("released", "flat_majority", "factorized_majority", "sensitivity_shuffled")
    }
    metrics["sensitivity"] = {
        "actual_update_accuracy": 0.69,
        "zero_accuracy": 1,
        "above_effect_floor_fraction": 1,
    }
    assert not all(analysis.development_gates(metrics).values())
    metrics["sensitivity"]["actual_update_accuracy"] = 0.75
    assert all(analysis.development_gates(metrics).values())
    metrics["sensitivity_shuffled"]["actual_update_accuracy"] = 0.7
    assert not all(analysis.development_gates(metrics).values())


def test_audit_and_analysis_block_incomplete_before_reading_scientific_outputs(tmp_path):
    for controller in [
        {"status": "running", "controller_pid": 0},
        {
            "status": "complete",
            "controller_pid": 0,
            "worker_pid": None,
            "attempts": [{"pid": 0, "returncode": 1}],
        },
    ]:
        atomic_json(tmp_path / "controller.json", controller)
        with pytest.raises(ValueError):
            analysis.audit(tmp_path)
        with pytest.raises(ValueError):
            analysis.analyze(tmp_path, tmp_path / "out")


def test_undefined_profile_ratio_is_explicit():
    result = analysis.profile_metrics(np.ones((2, 32)), np.zeros((2, 32)))
    assert result["nrmse_vs_zero"] is None
    assert result["rmse"] == 1


def test_complete_analysis_keeps_weak_cases_all_controls_and_historical_methods(tmp_path):
    root, baseline, prior, collection, output = [
        tmp_path / p for p in ("root", "baseline", "prior", "collection", "output")
    ]
    for p in (root, baseline / "analysis", prior, collection / "cases"):
        p.mkdir(parents=True)
    cases, rows, old = [], [], []
    for i in range(72):
        zero, weak = i >= 64, i == 63
        label = NO_CHANGE if zero or weak else PATTERN_IDS[i % 32]
        profile = np.zeros(32)
        if not zero:
            profile[i % 32] = 0.01 if weak else 0.5
            profile -= profile.mean()
        cases.append(
            {
                "id": str(i),
                "control": zero,
                "source_pattern": str(i // 8),
                "initialization_seed": (i % 8) // 2,
                "label": label,
                "measured": {
                    "effect": {
                        "label": label,
                        "below_effect_floor": zero or weak,
                        "centered_improvements": profile.tolist(),
                    }
                },
                "predictions": {
                    k: NO_CHANGE
                    for k in ("released", "flat_majority", "factorized_majority", "ridge", "epoch1")
                },
            }
        )
        rows.append(
            {
                "id": str(i),
                "split": "development",
                "profile": profile.tolist(),
                "prediction": label,
                "shuffled_prediction": NO_CHANGE,
            }
        )
        old.append({"id": str(i), "prediction": NO_CHANGE})
    # Minimal independent training diagnostic fixture; no fitting or selection occurs here.
    train = [{**rows[0], "id": "train", "split": "train"}]
    atomic_json(collection / "cases/train.json", cases[0]["measured"])
    baseline_path = baseline / "analysis/all-development-cases.json"
    old_path = prior / "predictions.json"
    atomic_json(baseline_path, cases)
    atomic_json(old_path, old)
    atomic_json(root / "predictions.json", train + rows)
    atomic_json(
        root / "controller.json",
        {
            "status": "complete",
            "controller_pid": 0,
            "worker_pid": None,
            "attempts": [{"pid": 0, "returncode": 0}],
        },
    )
    atomic_json(
        root / "plan.json",
        {
            "baseline_root": str(baseline),
            "prior_root": str(prior),
            "collection_root": str(collection),
            "input_files_sha256": {str(p): file_hash(p) for p in (baseline_path, old_path)},
        },
    )
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "recipe": "base_sensitivity_v4",
            "files_sha256": {
                n: file_hash(root / n) for n in ("predictions.json", "controller.json", "plan.json")
            },
        },
    )
    analysis.analyze(root, output)
    summary = analysis.read(output / "summary.json")
    assert summary["metrics"]["sensitivity"]["actual_update_n"] == 64
    assert summary["metrics"]["sensitivity"]["actual_update_correct"] == 64
    assert summary["metrics"]["sensitivity"]["zero_n"] == 8
    assert summary["metrics"]["sensitivity"]["above_effect_floor_fraction"] == 63 / 64
    assert summary["all_development_gates_pass"]
    assert summary["profile_metrics"]["rmse"] == 0
    assert len(analysis.read(output / "all-development-cases.json")) == 72
    assert len(summary["metrics"]) == 8
    assert (output / "profile-predictions.png").is_file()
    with pytest.raises(ValueError, match="Preserve"):
        analysis.analyze(root, output)
