import copy

import numpy as np
import pytest

from gradient_interp import update_loracle_exact_calibration_analysis as analysis
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import NO_CHANGE, PATTERN_IDS
from gradient_interp.update_loracle_exact_calibration import effect_profile


def test_audit_and_analysis_refuse_live_failed_or_partial_runs(tmp_path):
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


def test_independent_profile_check_rejects_posthoc_scaling_floor_and_dropped_rows():
    baseline = np.ones((8, 32))
    after = baseline.copy()
    after[:, 0] -= 0.1
    record = {"after_ce": after.tolist(), **effect_profile(baseline, after)}
    analysis.checked_profile(record, baseline)
    bad = copy.deepcopy(record)
    bad["profile"] = (np.array(bad["profile"]) * 2).tolist()
    with pytest.raises(AssertionError):
        analysis.checked_profile(bad, baseline)
    bad = copy.deepcopy(record)
    bad["prediction"] = NO_CHANGE
    with pytest.raises(ValueError, match="category"):
        analysis.checked_profile(bad, baseline)
    bad = copy.deepcopy(record)
    bad["after_ce"] = bad["after_ce"][:4]
    with pytest.raises(ValueError, match="grid"):
        analysis.checked_profile(bad, baseline)


def test_error_decomposition_keeps_covariance_instead_of_claiming_additive_causes():
    rng = np.random.default_rng(8)
    target, exact, linear = [rng.normal(size=(5, 32)) for _ in range(3)]
    result = analysis.error_decomposition(linear, exact, target)
    assert result["total_linear_to_evaluation_mse"] == pytest.approx(
        result["linear_to_exact_calibration_mse"]
        + result["exact_calibration_to_evaluation_mse"]
        + result["signed_cross_term"]
    )
    assert abs(result["identity_residual"]) < 1e-14
    assert result["signed_cross_term"] != pytest.approx(0)


def test_new_method_cannot_change_original_full_category_gate():
    values = {
        n: {"actual_update_accuracy": 0.1}
        for n in ("released", "flat_majority", "factorized_majority", "functional_shuffled")
    }
    values["functional"] = {
        "actual_update_accuracy": 0.69,
        "above_effect_floor_fraction": 1,
        "zero_accuracy": 1,
    }
    assert not analysis.gates(values)["accuracy_70"]
    values["functional"]["actual_update_accuracy"] = 0.75
    assert all(analysis.gates(values).values())
    values["functional_shuffled"]["actual_update_accuracy"] = 0.7
    assert not analysis.gates(values)["functional_shuffled_advantage_10pp"]


def test_full_analysis_preserves_all_cases_controls_comparisons_and_weak_updates(tmp_path):
    root, prior, collection, out = [tmp_path / p for p in ("root", "prior", "collection", "output")]
    for path in (root, prior / "analysis", collection / "cases"):
        path.mkdir(parents=True)
    cases, rows = [], []
    names = (
        "released",
        "flat_majority",
        "factorized_majority",
        "ridge",
        "epoch1",
        "physical_profile",
        "sensitivity",
        "sensitivity_shuffled",
    )
    for i in range(72):
        zero, weak = i >= 64, i == 63
        profile = np.zeros(32)
        if not zero:
            profile[i % 32] = 0.01 if weak else 0.5
        profile -= profile.mean()
        label = NO_CHANGE if zero or weak else PATTERN_IDS[i % 32]
        cases.append(
            {
                "id": str(i),
                "control": zero,
                "label": label,
                "source_pattern": str(i // 8),
                "initialization_seed": i % 8 // 2,
                "measured": {
                    "effect": {
                        "label": label,
                        "below_effect_floor": zero or weak,
                        "centered_improvements": profile.tolist(),
                    }
                },
                "comparison_predictions": {n: NO_CHANGE for n in names},
                "sensitivity_decoder": {"profile": (profile * 2).tolist()},
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
    train = []
    for i in range(272):
        row = {**rows[i % 72], "id": f"train-{i}", "split": "train"}
        train.append(row)
        atomic_json(collection / "cases" / f"train-{i}.json", cases[i % 72]["measured"])
    original = prior / "analysis/all-development-cases.json"
    atomic_json(original, cases)
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
            "prior_root": str(prior),
            "collection_root": str(collection),
            "input_files_sha256": {str(original): file_hash(original)},
        },
    )
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "recipe": analysis.RECIPE,
            "cases": 344,
            "scored_adapter_conditions": 688,
            "all_profiles_and_native_norms_verified": True,
            "fixed_scoring_rechecks": [{"passed": True}] * 4,
            "files_sha256": {
                n: file_hash(root / n) for n in ("predictions.json", "controller.json", "plan.json")
            },
        },
    )
    analysis.analyze(root, out)
    summary = analysis.read(out / "summary.json")
    assert summary["metrics"]["functional"]["actual_update_n"] == 64
    assert summary["metrics"]["functional"]["actual_update_correct"] == 64
    assert summary["metrics"]["functional"]["zero_n"] == 8
    assert summary["metrics"]["functional"]["above_effect_floor_fraction"] == 63 / 64
    assert len(summary["metrics"]) == 10 and summary["training_diagnostic"]["cases"] == 272
    assert len(analysis.read(out / "all-development-cases.json")) == 72
    assert summary["profile_metrics"]["rmse"] == 0 and (out / "profile-predictions.png").is_file()
    with pytest.raises(ValueError, match="Preserve"):
        analysis.analyze(root, out)
