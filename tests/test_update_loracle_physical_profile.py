import numpy as np
import pytest
import torch

from gradient_interp.update_loracle_data import NO_CHANGE, PATTERN_IDS
from gradient_interp.update_loracle_physical_profile import (
    fit_profile,
    module_gram,
    predict_profiles,
    profile_label,
    shuffled_cross_gram,
)


def factors():
    g = torch.Generator().manual_seed(9)
    return (
        torch.randn(5, 3, 11, generator=g, dtype=torch.float64),
        torch.randn(5, 7, 3, generator=g, dtype=torch.float64),
    )


def test_gram_matches_dense_signed_updates():
    a, b = factors()
    b[1] = -b[0]
    a[1] = a[0]
    b[4].zero_()
    dense = (2 * b @ a).flatten(1)
    gram = module_gram(a, b)
    torch.testing.assert_close(gram, dense @ dense.T, rtol=1e-12, atol=1e-10)
    assert gram[0, 1] < 0
    assert not torch.count_nonzero(gram[4])


def test_gram_is_invariant_to_independent_low_rank_basis_changes():
    a, b = factors()
    g = torch.Generator().manual_seed(8)
    rotations = torch.randn(5, 3, 3, generator=g, dtype=torch.float64)
    rotations += 5 * torch.eye(3, dtype=torch.float64)
    transformed_a = rotations @ a
    transformed_b = b @ torch.linalg.inv(rotations)
    torch.testing.assert_close(
        module_gram(a, b), module_gram(transformed_a, transformed_b), rtol=1e-12, atol=1e-10
    )


def test_gram_retains_physical_magnitude():
    a, b = factors()
    scale = torch.tensor([1.0, 3.0, 0.2, -2.0, 0.0], dtype=torch.float64)
    expected = module_gram(a, b) * scale[:, None] * scale[None, :]
    torch.testing.assert_close(
        module_gram(a, b * scale[:, None, None]), expected, rtol=1e-12, atol=1e-10
    )


def test_wrong_precision_and_nonfinite_factors_fail():
    a, b = factors()
    with pytest.raises(ValueError):
        module_gram(a.float(), b.float())
    a[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        module_gram(a, b)


def test_full_profile_regression_generalizes_linear_combination_and_preserves_zero():
    x = np.vstack([np.eye(3), -np.eye(3), np.zeros(3)])
    rng = np.random.default_rng(8)
    weights = rng.normal(size=(3, 32))
    weights -= weights.mean(axis=1, keepdims=True)
    targets = x @ weights
    alpha, penalty = fit_profile(x @ x.T, targets, [True] * 6 + [False])
    assert penalty == 0.01
    unseen = np.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]])
    predicted, labels = predict_profiles(unseen @ x.T, alpha)
    np.testing.assert_allclose(predicted[0], (unseen @ weights)[0] * 2 / 2.01, atol=1e-12)
    assert np.all(predicted[1] == 0) and labels[1] == NO_CHANGE


def test_prediction_uses_all_magnitudes_and_original_floor():
    p = np.zeros(32)
    p[2] = 0.019
    assert profile_label(p) == NO_CHANGE
    p[2] = 0.02
    assert profile_label(p) == PATTERN_IDS[2]
    p[1] = 0.02
    assert profile_label(p) == PATTERN_IDS[1]
    assert profile_label(-p) == PATTERN_IDS[0]


def test_ridge_rejects_invalid_targets_and_zero_controls():
    x = np.eye(3)
    y = np.zeros((3, 32))
    with pytest.raises(ValueError):
        fit_profile(x, y, [True, True, False])
    with pytest.raises(ValueError):
        fit_profile(x, np.ones((3, 32)), [True] * 3)
    x[0, 1] = 0.1
    with pytest.raises(ValueError):
        fit_profile(x, y, [True] * 3)


def test_shuffle_retains_recipient_magnitude_and_zero_control():
    cross = np.array([[1.0, 2.0], [6.0, 12.0], [0.0, 0.0]])
    result = shuffled_cross_gram(cross, [2.0, 6.0, 0.0], [1, 0, 2])
    np.testing.assert_array_equal(result, [[2.0, 4.0], [3.0, 6.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        shuffled_cross_gram(cross, [2.0, 6.0, 0.0], [2, 0, 1])


def test_analysis_keeps_weak_updates_all_profiles_and_gates(tmp_path):
    from gradient_interp import update_loracle_physical_profile_analysis as analysis
    from gradient_interp.study import atomic_json, file_hash

    root, prior, output = tmp_path / "run", tmp_path / "prior", tmp_path / "report"
    root.mkdir()
    (prior / "analysis").mkdir(parents=True)
    cases, predictions = [], []
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
                    for k in ["released", "flat_majority", "factorized_majority", "ridge", "epoch1"]
                },
            }
        )
        predictions.append(
            {
                "id": str(i),
                "profile": profile.tolist(),
                "prediction": label,
                "shuffled_prediction": NO_CHANGE,
            }
        )
    baseline = prior / "analysis/all-development-cases.json"
    atomic_json(baseline, cases)
    atomic_json(root / "predictions.json", predictions)
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
        {"prior_root": str(prior), "input_files_sha256": {str(baseline): file_hash(baseline)}},
    )
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "recipe": "physical_profile_v3",
            "files_sha256": {
                n: file_hash(root / n) for n in ["predictions.json", "controller.json", "plan.json"]
            },
        },
    )
    analysis.analyze(root, output)
    summary = analysis.read(output / "summary.json")
    assert summary["metrics"]["physical_profile"]["actual_update_n"] == 64
    assert summary["metrics"]["physical_profile"]["actual_update_correct"] == 64
    assert summary["metrics"]["physical_profile"]["above_effect_floor_fraction"] == 63 / 64
    assert summary["all_development_gates_pass"]
    assert summary["profile_metrics"]["rmse"] == 0
    assert len(analysis.read(output / "all-development-cases.json")) == 72
    assert (output / "development-accuracy.png").is_file()
    with pytest.raises(ValueError, match="Preserve"):
        analysis.analyze(root, output)


def test_physical_audit_blocks_live_or_failed_runs_before_scientific_reads(tmp_path):
    from gradient_interp import update_loracle_physical_profile_analysis as analysis
    from gradient_interp.study import atomic_json

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
