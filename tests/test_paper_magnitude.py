import copy
import json
import sys
import time
import types

import numpy as np
import pytest
import torch

from gradient_interp.paper_magnitude import (
    ARMS,
    assert_restored,
    bootstrap_errors,
    bootstrap_interval,
    encode,
    error_metrics,
    learning_rows,
    loss_forecasts,
    validate_cache,
    validate_data,
    validate_dedup_audit,
    validate_staged_data,
    validate_tokenization,
    verify_run,
)
from gradient_interp.study import atomic_json, file_hash


def test_training_geometry_keeps_relative_magnitudes_in_only_one_arm():
    projected = np.array([[3.0, 4.0], [6.0, 8.0], [0.0, 0.0]], dtype=np.float32)
    normalized, nscale = learning_rows(projected, ARMS[0])
    magnitude, mscale = learning_rows(projected, ARMS[1])
    np.testing.assert_allclose(np.linalg.norm(normalized[:2], axis=1), [1, 1])
    np.testing.assert_allclose(magnitude[1], 2 * magnitude[0])
    np.testing.assert_array_equal(mscale, np.full(3, 7.5))
    np.testing.assert_allclose(magnitude * mscale[:, None], projected)
    np.testing.assert_allclose(normalized * nscale[:, None], projected)
    # A common physical change of units must not alter either training geometry.
    for arm in ARMS:
        np.testing.assert_allclose(
            learning_rows(projected * 11, arm)[0], learning_rows(projected, arm)[0]
        )
    with pytest.raises(ValueError, match="All fitting"):
        learning_rows(np.zeros((3, 2)), ARMS[1])


def test_common_encoder_restores_batch_magnitude_and_lasso_shrinkage():
    dictionary = np.eye(2)
    projected = np.array([[3.0, 4.0], [0.0, 0.0], [-3.0, 4.0]])
    code, reconstructed = encode(projected, dictionary)
    # For an orthonormal dictionary, Lasso soft-thresholds each unit coefficient.
    np.testing.assert_allclose(code, [[2.5, 3.5], [0.0, 0.0], [-2.5, 3.5]], atol=1e-12)
    np.testing.assert_allclose(reconstructed, code)
    code2, reconstructed2 = encode(projected * 19, dictionary)
    np.testing.assert_allclose(code2, code * 19)
    np.testing.assert_allclose(reconstructed2, reconstructed * 19)
    np.testing.assert_allclose(encode(-projected, dictionary)[1], -reconstructed)
    with pytest.raises(ValueError):
        encode(projected, np.eye(3))


def test_loss_forecast_matches_linear_target_and_quadratic_curvature():
    probe_gradient = np.array([[1.0, -2.0], [-3.0, 0.5]])
    directions = np.array(
        [[[2.0, 3.0], [0.7, -1.0], [0.0, 0.0]], [[-2.0, 0.3], [1.7, -2.0], [0.0, 0.0]]]
    )
    rates = np.array([0.001, 0.01])
    prediction = loss_forecasts(probe_gradient, directions, rates)
    assert prediction.shape == (2, 2, 3, 2)
    theta = np.array([0.4, -0.8])
    for b in range(2):
        for r, eta in enumerate(rates):
            for m in range(3):
                delta = -eta * directions[b, m]
                for q in range(2):

                    def linear(point):
                        return probe_gradient[q] @ point

                    np.testing.assert_allclose(
                        prediction[b, r, m, q], linear(theta + delta) - linear(theta), atol=1e-14
                    )

                    # A quadratic loss has a known, explicitly omitted curvature term.
                    def quadratic(step):
                        return probe_gradient[q] @ step + 0.5 * step @ step

                    np.testing.assert_allclose(
                        quadratic(delta) - prediction[b, r, m, q], 0.5 * delta @ delta, atol=1e-14
                    )
    np.testing.assert_array_equal(prediction[:, :, 2], 0)
    with pytest.raises(ValueError):
        loss_forecasts(probe_gradient, directions, [-0.01])


def test_metrics_do_not_make_zero_or_weak_effects_a_success():
    result = error_metrics(np.zeros((2, 3)), np.zeros((2, 3)))
    assert result["nrmse_vs_zero"] is None
    assert result["actual_on_predicted_slope_through_origin"] is None
    assert not result["meaningful_effect_gate"]
    result = error_metrics([0.001, -0.001], [0.001, -0.001])
    assert result["nrmse_vs_zero"] == 0
    assert not result["meaningful_effect_gate"]
    result = error_metrics([0.02, -0.02], [0.03, -0.03])
    assert result["nrmse_vs_zero"] == pytest.approx(1 / 3)
    assert result["actual_on_predicted_slope_through_origin"] == pytest.approx(1.5)
    assert result["meaningful_effect_gate"]
    with pytest.raises(ValueError):
        error_metrics([np.nan], [1])


def toy_data():
    def row(i, split):
        return dict(
            id=str(i),
            prompt=f"question {i}",
            answer=f"answer {i}",
            source_id=f"source-{i}",
            cluster_id=f"cluster-{i}",
            template_id=split,
            full_tokens=10,
            completion_tokens=3,
        )

    return dict(
        batches=[dict(id="batch", documents=[row(0, "batch"), row(1, "batch")])],
        calibration=[row(2, "cal")],
        evaluation=[row(3, "eval"), row(4, "eval")],
    )


def test_split_checks_reject_source_cluster_template_and_prompt_leakage():
    expected = dict(batches=1, batch_size=2, calibration=1, evaluation=2)
    data = toy_data()
    assert len(validate_data(data, expected)) == 5
    for key in ("source_id", "cluster_id", "template_id", "prompt", "id"):
        broken = copy.deepcopy(data)
        broken["evaluation"][0][key] = broken["calibration"][0][key]
        with pytest.raises(ValueError):
            validate_data(broken, expected)
    # Template IDs are optional for natural sources only; the audit must explain policy.
    natural = copy.deepcopy(data)
    for row in natural["calibration"] + natural["evaluation"]:
        row.pop("template_id")
    validate_data(natural, expected)


def test_historical_dedup_guard_requires_evidence_and_matching_data(tmp_path):
    data_path = tmp_path / "data.json"
    atomic_json(data_path, toy_data())
    audit = dict(
        data_sha256=file_hash(data_path),
        prior_input_files_sha256={"old": "abc"},
        source_deduplication="Exclude fitting and prior source IDs",
        near_duplicate_rule="Fixed normalized token Jaccard >=.8",
        template_split_policy="Natural untemplated documents, source clusters split",
        scope="development",
        exact_prompt_collisions=[],
        source_collisions=[],
        near_duplicate_collisions=[],
    )
    validate_dedup_audit(audit, data_path)
    for name in ("data_sha256", "near_duplicate_rule", "template_split_policy"):
        broken = {**audit, name: ""}
        with pytest.raises(ValueError):
            validate_dedup_audit(broken, data_path)
    with pytest.raises(ValueError):
        validate_dedup_audit({**audit, "source_collisions": ["fitting-1"]}, data_path)


def test_run_guard_requires_controller_deadline_and_pinned_inputs(tmp_path):
    frozen = tmp_path / "input.json"
    frozen.write_text("{}")
    atomic_json(
        tmp_path / "plan.json",
        dict(
            scope="development_only",
            source_files_sha256={},
            input_hashes={str(frozen): file_hash(frozen)},
        ),
    )
    controller = dict(
        status="running",
        plan_sha256=file_hash(tmp_path / "plan.json"),
        deadline_unix=time.time() + 60,
    )
    atomic_json(tmp_path / "controller.json", controller)
    assert verify_run(tmp_path) == controller["deadline_unix"]
    atomic_json(tmp_path / "controller.json", {**controller, "deadline_unix": time.time() - 1})
    with pytest.raises(ValueError, match="deadline"):
        verify_run(tmp_path)
    atomic_json(tmp_path / "controller.json", controller)
    atomic_json(tmp_path / "manifest.json", {"complete": True})
    with pytest.raises(ValueError, match="never resume"):
        verify_run(tmp_path)
    (tmp_path / "manifest.json").unlink()
    frozen.write_text(json.dumps({"changed": True}))
    with pytest.raises((ValueError, AssertionError)):
        verify_run(tmp_path)


def test_adapter_assertion_is_bit_exact():
    original = {"a": torch.tensor([1.0, 2.0])}
    params = [("a", original["a"].clone())]
    assert_restored(params, original)
    params[0][1][0] += 1e-6
    with pytest.raises(ValueError, match="exactly"):
        assert_restored(params, original)


def test_cache_staged_data_and_scoring_token_guards(tmp_path):
    validate_cache({"gradients": torch.ones(2, 3), "forecasts": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="forecasts"):
        validate_cache({"forecasts": torch.tensor([float("inf")])})
    atomic_json(tmp_path / "data.json", toy_data())
    atomic_json(
        tmp_path / "data-audit.json", {"staged_data_sha256": file_hash(tmp_path / "data.json")}
    )
    validate_staged_data(tmp_path)
    atomic_json(tmp_path / "data.json", {"changed": True})
    with pytest.raises(ValueError, match="Staged scientific"):
        validate_staged_data(tmp_path)

    def tokenize(chat, tokenizer, max_length):
        return {"input_ids": list(range(10)), "labels": [-100] * 7 + [1, 2, 3]}

    assert validate_tokenization(toy_data(), None, tokenize)["all_targets_untruncated"]
    broken = toy_data()
    broken["evaluation"][0]["completion_tokens"] = 4
    with pytest.raises(ValueError, match="tokenization differs"):
        validate_tokenization(broken, None, tokenize)


def test_bootstrap_preserves_method_rate_pairing_and_undefined_draws():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(8, 1, 9))
    predicted = truth[:, :, None] * 0.7 + rng.normal(scale=0.1, size=(8, 1, 1, 9))
    predicted = np.repeat(predicted, 2, axis=2)
    truth = np.concatenate([truth, truth * 10], axis=1)
    predicted = np.concatenate([predicted, predicted * 10], axis=1)
    draws = bootstrap_errors(predicted, truth, 50, 77)
    np.testing.assert_array_equal(draws["mse_difference_magnitude_minus_normalized"], 0)
    np.testing.assert_allclose(draws["nrmse"][:, 0], draws["nrmse"][:, 1])
    np.testing.assert_allclose(draws["mae"][:, 1], 10 * draws["mae"][:, 0])
    assert bootstrap_interval(draws["mae"][:, 0, 0], 4)["marginal_coverage"] == 0.9875
    zero = bootstrap_errors(np.zeros((2, 2, 2, 3)), np.zeros((2, 2, 3)), 10)
    interval = bootstrap_interval(zero["nrmse"][:, 0, 0])
    assert interval["interval"] is None and interval["undefined_replicates"] == 10


def test_worker_locks_before_updates_and_restores_on_failure_and_resume(tmp_path, monkeypatch):
    import gradient_interp.paper_magnitude as module

    data = toy_data()
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    original = parameter.detach().clone()
    params = [("weight", parameter)]
    spec = {
        "source_root": str(tmp_path),
        "source_config": {},
        "rates": [0.0003, 0.003],
        "counts": {"actual_updates": 2},
    }
    for name, value in (("data.json", data), ("specification.json", spec), ("plan.json", {})):
        atomic_json(tmp_path / name, value)
    atomic_json(
        tmp_path / "data-audit.json", {"staged_data_sha256": file_hash(tmp_path / "data.json")}
    )
    np.savez(tmp_path / "dictionaries.npz", normalized_fit=np.eye(2), magnitude_fit=np.eye(2))
    monkeypatch.setattr(module, "verify_run", lambda _: time.time() + 60)
    monkeypatch.setattr(module, "validate_data", lambda _: None)
    monkeypatch.setattr(module, "validate_architecture", lambda *args: None)
    monkeypatch.setattr(module, "validate_tokenization", lambda *args: None)
    monkeypatch.setattr(
        module,
        "stable_setup",
        lambda _: (None, None, None, None, {"lora_names": ["weight"]}, params),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    state = {"updates": 0, "fail": True}

    def fake_score(model, tokenizer, tokenize, prompt, answer):
        if state["updates"] == 2 and state["fail"]:
            raise RuntimeError("deliberate scoring failure")
        feature = torch.tensor([float(prompt.split()[-1]) + 1, -0.7])
        return -(parameter @ feature + 20) * 3, 3

    def checked_update(parameters, originals, direction, scale):
        locked = read_json_for_test(tmp_path / "forecast-lock.json")
        assert locked["all_forecasts_before_updates"] and locked["original_adapter_unchanged"]
        assert locked["files_sha256"]["forecast-cache.pt"] == file_hash(
            tmp_path / "forecast-cache.pt"
        )
        state["updates"] += 1
        from gradient_interp.paper_steering import set_direction

        set_direction(parameters, originals, direction, scale)

    monkeypatch.setattr(module, "scalar_logp", fake_score)
    monkeypatch.setattr(module, "set_direction", checked_update)
    fake_projection = types.ModuleType("gradient_atoms.projection")
    fake_projection.load_ekfac_eigen = lambda _: {}
    fake_projection.project_gradients_ekfac = lambda g, names, factors, k: (g, {})
    fake_projection.unproject_atom = lambda x, info, factors, size: torch.from_numpy(x).float()
    monkeypatch.setitem(sys.modules, "gradient_atoms", types.ModuleType("gradient_atoms"))
    monkeypatch.setitem(sys.modules, "gradient_atoms.projection", fake_projection)
    with pytest.raises(RuntimeError, match="deliberate"):
        module.run(tmp_path)
    torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    assert not (tmp_path / "manifest.json").exists()
    state["fail"] = False
    module.run(tmp_path)
    # First completed condition is reused; only the failed second update is repeated.
    assert state["updates"] == 3
    torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    manifest = read_json_for_test(tmp_path / "manifest.json")
    assert manifest["actual_updates"] == 2 and manifest["adapter_restored"]
    cache = torch.load(tmp_path / "forecast-cache.pt", weights_only=True)
    for ri, rate in enumerate(spec["rates"]):
        outcome = read_json_for_test(tmp_path / "outcomes" / f"batch-00-rate-{rate:g}.json")
        actual = np.array(outcome["mean_completion_ce"]) - cache["baseline_mean_ce"].numpy()
        np.testing.assert_allclose(actual, cache["predicted_mean_ce_change"][0, ri, 2], atol=3e-6)


def read_json_for_test(path):
    return json.loads(path.read_text())
