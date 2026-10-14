import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradient_interp import paper_magnitude_analysis as analysis
from gradient_interp.paper_magnitude import ARMS, METHODS, RATES, bootstrap_errors
from gradient_interp.study import atomic_json, file_hash

AUDITOR_PATH = Path(__file__).parents[1] / "scripts/audit_paper_magnitude.py"
AUDITOR_SPEC = importlib.util.spec_from_file_location("magnitude_audit_test", AUDITOR_PATH)
auditor = importlib.util.module_from_spec(AUDITOR_SPEC)
AUDITOR_SPEC.loader.exec_module(auditor)


def completion_receipt(seed=42):
    return dict(
        type="paper_magnitude_completion_receipt",
        seed=seed,
        root=f"/synthetic/seed{seed}",
        started_unix=1000,
        deadline_unix=44200,
        observed_unix=50000,
        completed_unix=2000,
        all_processes_dead=True,
        status="complete",
        release_reason="complete",
        controller_sha256="a" * 64,
        plan_sha256="b" * 64,
        specification_sha256="c" * 64,
        manifest_sha256="d" * 64,
    )


def analysis_plan(path):
    atomic_json(
        path,
        dict(
            source_commit=analysis.FROZEN_COMMIT,
            written_unix=100,
            scientific_outcomes_read=False,
        ),
    )
    return path


def synthetic_arrays():
    # Deliberately heterogeneous batches and probes: independent crossed resampling matters.
    b = np.linspace(0.012, 0.05, 32)[:, None, None]
    q = np.linspace(0.7, 1.3, 80)[None, None, :]
    rates = np.array([1, 10])[None, :, None]
    actual_change = -b * q * rates
    predictions = actual_change[:, :, None] * np.array([0.8, 0.9, 1, 0])[None, None, :, None]
    return dict(
        predicted_change=predictions,
        actual_mean_ce=1 + actual_change,
        baseline_mean_ce=np.ones(80),
        completion_tokens=np.full(80, 10),
        coefficients=np.ones((32, 2, 500)),
        batch_gradient_norm=np.ones(32),
        reconstructed_gradient_norm=np.broadcast_to([0.8, 0.9, 1, 0], (32, 4)).copy(),
        physical_relative_error=np.broadcast_to([0.2, 0.1, 0, 1], (32, 4)).copy(),
        physical_cosine=np.broadcast_to([1, 1, 1, np.nan], (32, 4)).copy(),
        projected_norm=np.ones(32),
        projected_relative_error=np.broadcast_to([0.2, 0.1], (32, 2)).copy(),
        projected_target_gain=np.broadcast_to([0.8, 0.9], (32, 2)).copy(),
        coefficient_l0=np.full((32, 2), 500),
    )


def synthetic_data():
    def probe(index):
        return dict(
            id=f"probe-{index}",
            source_id=f"source-{index}",
            cluster_id=f"cluster-{index}",
            completion_tokens=10,
            prompt=f"Synthetic question {index}: <untrusted markup>",
            answer=f"Synthetic reference {index}",
        )

    return dict(
        batches=[dict(id=f"batch-{index}") for index in range(32)],
        calibration=[probe(index) for index in range(16)],
        evaluation=[probe(index) for index in range(16, 80)],
    )


def compact_export(parent, seed, plan):
    root = parent / f"seed{seed}"
    root.mkdir()
    np.savez_compressed(root / "arrays.npz", **synthetic_arrays())
    atomic_json(root / "data.json", synthetic_data())
    integrity = dict(
        complete=True,
        seed=seed,
        source_commit=analysis.FROZEN_COMMIT,
        source_input_output_hashes_verified=True,
        full_grid_verified=True,
        locks_restores_verified=True,
        original_deadline_verified=True,
        tokenization_verified=True,
        cache_recomputed=True,
        completion_receipts=[completion_receipt(42), completion_receipt(43)],
        analysis_plan_sha256=file_hash(plan),
        updates=64,
        probes=80,
        fit=dict(
            started_unix=200,
            completed_unix=300,
            original_dictionary_reused=True,
            magnitude_global_scale=2.5,
        ),
        worker_started_unix=1000,
        worker_completed_unix=2000,
        peak_memory_bytes=10**9,
    )
    atomic_json(root / "integrity.json", integrity)
    atomic_json(
        root / "export.json",
        dict(
            type="paper_magnitude_compact_export",
            seed=seed,
            methods=list(METHODS),
            rates=list(RATES),
            analysis_plan_sha256=file_hash(plan),
            files_sha256={
                n: file_hash(root / n) for n in ("arrays.npz", "data.json", "integrity.json")
            },
        ),
    )
    return root


def test_receipts_block_live_and_early_failed_runs_and_keep_deadline_release_distinct(tmp_path):
    receipt = completion_receipt()
    assert auditor.validate_receipt(receipt) is receipt
    for change, message in (
        ({"all_processes_dead": False}, "dead processes"),
        ({"deadline_unix": 99999}, "original deadline"),
        ({"completed_unix": 45000}, "completed-run"),
        ({"plan_sha256": ""}, "Missing receipt hash"),
        (
            {"release_reason": "original_deadline", "status": "failed", "observed_unix": 3000},
            "early failure",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            auditor.validate_receipt({**receipt, **change})
    deadline = {**receipt, "release_reason": "original_deadline", "status": "deadline"}
    auditor.validate_receipt(deadline)
    paths = [tmp_path / f"receipt{s}.json" for s in (42, 43)]
    for path, seed in zip(paths, (42, 43), strict=True):
        atomic_json(path, completion_receipt(seed))
    plan = analysis_plan(tmp_path / "plan.json")
    assert len(auditor.require_barrier(paths, plan)) == 2
    with pytest.raises(ValueError, match="both seeds"):
        auditor.require_barrier(paths[:1], plan)
    with pytest.raises(ValueError, match="distinct seeds"):
        auditor.require_barrier([paths[0], paths[0]], plan)
    atomic_json(plan, {**json.loads(plan.read_text()), "written_unix": 60000})
    with pytest.raises(ValueError, match="too late"):
        auditor.require_barrier(paths, plan)


def test_auditor_runs_barrier_before_any_scientific_file_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Scientific cache opened without both completion receipts")

    monkeypatch.setattr(auditor.torch, "load", forbidden)
    with pytest.raises(ValueError, match="both seeds"):
        auditor.audit(tmp_path / "not-created", [], tmp_path / "plan", tmp_path / "export")


def test_metadata_receipt_preserves_failed_attempt_until_original_deadline(tmp_path, monkeypatch):
    atomic_json(tmp_path / "plan.json", dict(max_hours=12))
    atomic_json(tmp_path / "specification.json", dict(training_seed=42))
    atomic_json(
        tmp_path / "controller.json",
        dict(
            status="failed",
            started_unix=1000,
            deadline_unix=44200,
            plan_sha256=file_hash(tmp_path / "plan.json"),
            controller_pid=123,
            worker_pid=124,
            attempts=[dict(pid=124)],
        ),
    )
    monkeypatch.setattr(auditor, "process_dead", lambda pid: True)
    monkeypatch.setattr(auditor.time, "time", lambda: 3000)
    destination = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="early failure"):
        auditor.receipt(tmp_path, destination)
    assert not destination.exists()
    monkeypatch.setattr(auditor.time, "time", lambda: 50000)
    result = auditor.receipt(tmp_path, destination)
    assert result["release_reason"] == "original_deadline"
    assert result["status"] == "failed"
    assert result["checked_pids"] == [123, 124]
    with pytest.raises(ValueError, match="Preserve"):
        auditor.receipt(tmp_path, destination)


def test_compact_export_barrier_precedes_numeric_hash_or_load(tmp_path, monkeypatch):
    plan = analysis_plan(tmp_path / "plan.json")
    exports = [compact_export(tmp_path, seed, plan) for seed in (42, 43)]
    bad = json.loads((exports[1] / "integrity.json").read_text())
    bad["completion_receipts"][1]["release_reason"] = "original_deadline"
    atomic_json(exports[1] / "integrity.json", bad)

    def guarded_hash(path):
        assert Path(path).name != "arrays.npz", "A numeric file was accessed before both barriers"
        return file_hash(path)

    monkeypatch.setattr(analysis, "file_hash", guarded_hash)
    with pytest.raises(ValueError, match="complete result grid"):
        analysis.require_complete_exports(exports, plan)


def test_deadline_release_records_failure_without_opening_partial_scientific_grid(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    root.mkdir()
    atomic_json(root / "specification.json", dict(training_seed=42))
    atomic_json(root / "controller.json", dict(status="deadline"))
    atomic_json(root / "plan.json", dict(max_hours=12))
    own = {**completion_receipt(), "release_reason": "original_deadline", "status": "deadline"}
    for filename, key in (
        ("controller.json", "controller_sha256"),
        ("plan.json", "plan_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        own[key] = file_hash(root / filename)
    paths = [tmp_path / "receipt42.json", tmp_path / "receipt43.json"]
    atomic_json(paths[0], own)
    atomic_json(paths[1], completion_receipt(43))
    plan = analysis_plan(tmp_path / "analysis-plan.json")

    def forbidden(*args, **kwargs):
        pytest.fail("Partial cache must not be interpreted as a completed scientific grid")

    monkeypatch.setattr(auditor.torch, "load", forbidden)
    output = tmp_path / "export"
    result = auditor.audit(root, paths, plan, output)
    assert result["complete"] is False and result["scientific_outputs_read"] is False
    assert (root / "audit-incomplete.json").exists()
    assert not output.exists()
    with pytest.raises(ValueError, match="Preserve the existing incomplete"):
        auditor.audit(root, paths, plan, output)


def test_export_integrity_and_exact_schema_are_required(tmp_path):
    plan = analysis_plan(tmp_path / "plan.json")
    exports = [compact_export(tmp_path, seed, plan) for seed in (42, 43)]
    assert [row[1]["seed"] for row in analysis.require_complete_exports(exports[::-1], plan)] == [
        42,
        43,
    ]
    with pytest.raises(ValueError, match="distinct seeds"):
        analysis.require_complete_exports([exports[0], exports[0]], plan)
    arrays = exports[0] / "arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Compact export changed"):
        analysis.require_complete_exports(exports, plan)


def test_norm_diagnostics_preserve_loss_and_zero_denominators():
    original = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    reconstructed = original[:, None] * torch.tensor([0.8, 0.9, 1, 0])[None, :, None]
    cache = dict(
        batch_gradients=original,
        reconstructed_gradients=reconstructed,
        projected=original,
        coefficients=reconstructed[:, :2],
    )
    result = auditor.norm_diagnostics(cache, {arm: np.eye(2) for arm in ARMS})
    np.testing.assert_allclose(result["batch_gradient_norm"], [5, 0])
    np.testing.assert_allclose(result["physical_relative_error"][0], [0.2, 0.1, 0, 1], atol=3e-8)
    np.testing.assert_allclose(result["projected_target_gain"][0], [0.8, 0.9], atol=3e-8)
    assert np.isnan(result["physical_cosine"][0, 3])
    assert np.isnan(result["physical_relative_error"][1]).all()
    assert np.isnan(result["projected_relative_error"][1]).all()
    np.testing.assert_array_equal(result["coefficient_l0"], [[2, 2], [0, 0]])


def test_metrics_use_both_strict_direction_thresholds_and_report_impossible_ce():
    p = np.array([0.001, 0.003, -0.004, -2, 0])
    y = np.array([0.002, -0.003, -0.004, -0.01, 0.5])
    metrics = analysis.detailed_metrics(p, y, np.ones(5))
    assert metrics["total_pairs"] == 5
    assert metrics["direction_eligible_pairs"] == 3
    assert metrics["direction_matched_pairs"] == 2
    assert metrics["direction_agreement"] == pytest.approx(2 / 3)
    assert metrics["predicted_negative_ce_pairs"] == 1
    assert metrics["error_p95_nats_per_token"] == pytest.approx(np.quantile(abs(p - y), 0.95))
    undefined = analysis.detailed_metrics(np.zeros(2), np.zeros(2), np.ones(2))
    assert undefined["nrmse_vs_zero"] is None
    assert undefined["direction_agreement"] is None
    assert undefined["direction_eligible_pairs"] == 0


def test_interval_gates_require_all_seeds_rates_and_keep_weak_effects_inconclusive():
    record = dict(
        simultaneous={
            "actual_rms": {"interval": [0.01, 0.03]},
            "nrmse": {"interval": [0.1, 0.5]},
            "mae": {"interval": [0.002, 0.01]},
        }
    )
    passed = analysis.gate_status(record)
    assert passed["cleared"]
    weak = copy.deepcopy(record)
    weak["simultaneous"]["actual_rms"]["interval"][0] = 0.009
    assert analysis.gate_status(weak)["status"] == "inconclusive_signal"
    unknown = copy.deepcopy(record)
    unknown["simultaneous"]["nrmse"]["interval"] = None
    assert analysis.gate_status(unknown)["status"] == "failed_accuracy"
    rows = [
        dict(seed=s, rate=r, method=arm, scope="evaluation", gate=passed)
        for s in (42, 43)
        for r in RATES
        for arm in ARMS
    ]
    assert all(v["cleared_every_seed_and_rate"] for v in analysis.combined_gates(rows).values())
    rows[0]["gate"] = analysis.gate_status(weak)
    assert not analysis.combined_gates(rows)[ARMS[0]]["cleared_every_seed_and_rate"]
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.combined_gates(rows[1:])
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.combined_gates(rows + rows[:1])


def test_crossed_bootstrap_is_paired_across_methods_rates_and_retains_undefined():
    arrays = synthetic_arrays()
    p = arrays["predicted_change"][..., 16:]
    y = arrays["actual_mean_ce"][..., 16:] - 1
    draws = bootstrap_errors(p, y, replicates=40)
    np.testing.assert_allclose(
        draws["nrmse"], np.broadcast_to([0.2, 0.1, 0, 1], (40, 2, 4)), atol=1e-14
    )
    np.testing.assert_allclose(draws["actual_rms"][:, 1], draws["actual_rms"][:, 0] * 10)
    np.testing.assert_allclose(draws["mae"][:, 1], draws["mae"][:, 0] * 10, atol=1e-14)
    # Reproduce a draw manually: two independent index vectors, shared by all arms and rates.
    rng = np.random.default_rng(2026101409)
    bi, qi = rng.integers(32, size=32), rng.integers(64, size=64)
    mse = np.mean((p[bi][..., qi] - y[bi][..., qi][:, :, None]) ** 2, axis=(0, 3))
    np.testing.assert_allclose(
        draws["mse_difference_magnitude_minus_normalized"][0], mse[:, 1] - mse[:, 0]
    )
    zero_draws = bootstrap_errors(np.zeros_like(p), np.zeros_like(y), replicates=10)
    assert np.isnan(zero_draws["nrmse"]).all()
    interval = analysis.bootstrap_interval(zero_draws["nrmse"][:, 0, 0], family_size=4)
    assert interval["interval"] is None and interval["undefined_replicates"] == 10
    assert interval["marginal_coverage"] == 0.9875


def test_summary_retains_complete_grid_fixed_examples_and_separate_calibration(tmp_path):
    arrays, data = synthetic_arrays(), synthetic_data()
    first = tmp_path / "first"
    first.mkdir()
    summary = analysis.summarize(42, arrays, data, first, replicates=20)
    assert {k: len(v) for k, v in summary.items()} == dict(
        metrics=16,
        comparisons=4,
        batches=512,
        probes=640,
        pairs=20480,
        examples=512,
        reconstruction=128,
    )
    assert {r["batch"] for r in summary["examples"]} == set(range(8))
    assert {r["probe_index"] for r in summary["examples"]} == set(range(16, 24))
    assert {r["method"] for r in summary["examples"]} == set(METHODS)
    assert {r["rate"] for r in summary["examples"]} == set(RATES)
    modified = copy.deepcopy(arrays)
    modified["actual_mean_ce"][..., :16] = 0.9
    modified["predicted_change"][..., :16] = 0
    second = tmp_path / "second"
    second.mkdir()
    changed = analysis.summarize(42, modified, data, second, replicates=20)

    def evaluation(rows):
        return [row for row in rows if row["scope"] == "evaluation"]

    assert evaluation(summary["metrics"]) == evaluation(changed["metrics"])
    assert summary["metrics"] != changed["metrics"]


def test_array_validation_rejects_nonfinite_targets_but_keeps_undefined_cosines():
    arrays = synthetic_arrays()
    analysis.validate_arrays(arrays)
    for field, index in (("actual_mean_ce", (0, 0, 0)), ("coefficients", (0, 0, 0))):
        bad = copy.deepcopy(arrays)
        bad[field][index] = np.nan
        with pytest.raises(ValueError, match="Nonfinite"):
            analysis.validate_arrays(bad)
    bad = copy.deepcopy(arrays)
    bad["physical_cosine"][0, 0] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        analysis.validate_arrays(bad)
    bad = copy.deepcopy(arrays)
    bad["predicted_change"][0, 0, 3, 0] = 0.001
    with pytest.raises(ValueError, match="exactly zero"):
        analysis.validate_arrays(bad)


def test_full_synthetic_analysis_writes_all_outputs_and_refuses_overwrite(tmp_path):
    plan = analysis_plan(tmp_path / "plan.json")
    exports = [compact_export(tmp_path, seed, plan) for seed in (42, 43)]
    report, work = tmp_path / "report", tmp_path / "work"
    analysis.run(exports, plan, report, work)
    summary = json.loads((report / "summary.json").read_text())
    assert summary["counts"]["pairs"] == 40960
    assert summary["counts"]["examples"] == 1024
    assert summary["bootstrap_replicates"] == 1000
    assert (work / "first-reading.json").exists()
    assert (work / "analysis-complete.json").exists()
    text = (report / "report.html").read_text()
    assert "teacher-forced" in text
    assert "&lt;untrusted markup&gt;" in text
    assert "<untrusted markup>" not in text
    for figure in ("forecast-errors", "effect-size-scatter", "reconstruction"):
        assert (report / f"{figure}.png").stat().st_size > 1000
        assert (report / f"{figure}.pdf").stat().st_size > 1000
    for seed in (42, 43):
        for scope in ("evaluation", "calibration"):
            draws = np.load(work / f"seed{seed}-{scope}-bootstrap.npz")
            assert draws["nrmse"].shape == (1000, 2, 4)
    with pytest.raises(ValueError, match="already exists"):
        analysis.run(exports, plan, report, work)


def preliminary_authorization(path, plan):
    atomic_json(
        path,
        dict(
            type="paper_magnitude_preliminary_authorization",
            scope="preliminary_one_seed",
            seed=43,
            user_quote=analysis.PRELIMINARY_QUOTE,
            written_unix=2500,
            analysis_plan_sha256=file_hash(plan),
            source_commit=analysis.FROZEN_COMMIT,
            scientific_outcomes_read_before_amendment=False,
            criteria_unchanged=True,
            workers_unchanged=True,
            remaining_seed_required_for_full_conclusion=True,
        ),
    )
    return path


def preliminary_export(parent, plan, authorization):
    root = compact_export(parent, 43, plan)
    integrity = json.loads((root / "integrity.json").read_text())
    integrity.update(
        preliminary=True,
        complete_two_seed_claim=False,
        preliminary_amendment_sha256=file_hash(authorization),
        preliminary_amendment=json.loads(authorization.read_text()),
        missing_seeds=[42],
        completion_receipts=[completion_receipt(43)],
    )
    atomic_json(root / "integrity.json", integrity)
    manifest = json.loads((root / "export.json").read_text())
    manifest.update(
        type="paper_magnitude_preliminary_compact_export",
        preliminary_amendment_sha256=file_hash(authorization),
    )
    manifest["files_sha256"]["integrity.json"] = file_hash(root / "integrity.json")
    atomic_json(root / "export.json", manifest)
    return root


def test_preliminary_permission_is_separate_exact_and_does_not_open_normal_barrier(tmp_path):
    plan = analysis_plan(tmp_path / "plan.json")
    amendment = preliminary_authorization(tmp_path / "amendment.json", plan)
    receipt = tmp_path / "receipt43.json"
    atomic_json(receipt, completion_receipt(43))
    assert len(auditor.require_preliminary_receipt(receipt, plan, amendment)) == 1
    with pytest.raises(ValueError, match="both seeds"):
        auditor.require_barrier([receipt], plan)
    original = json.loads(amendment.read_text())
    for change in (
        {"seed": 42},
        {"user_quote": "inferred permission"},
        {"criteria_unchanged": False},
        {"scientific_outcomes_read_before_amendment": True},
        {"written_unix": 50},
        {"analysis_plan_sha256": "different"},
    ):
        atomic_json(amendment, {**original, **change})
        with pytest.raises(ValueError, match="Explicit dated"):
            auditor.require_preliminary_receipt(receipt, plan, amendment)
    atomic_json(amendment, original)
    atomic_json(receipt, {**completion_receipt(43), "all_processes_dead": False})
    with pytest.raises(ValueError, match="dead processes"):
        auditor.require_preliminary_receipt(receipt, plan, amendment)


def test_preliminary_analysis_is_labeled_and_never_claims_two_seed_success(tmp_path):
    plan = analysis_plan(tmp_path / "plan.json")
    amendment = preliminary_authorization(tmp_path / "amendment.json", plan)
    export = preliminary_export(tmp_path, plan, amendment)
    with pytest.raises(ValueError, match="Both compact exports"):
        analysis.require_complete_exports([export], plan)
    with pytest.raises(ValueError, match="not a complete audit"):
        analysis.require_complete_exports([export, export], plan)
    metadata = analysis.require_preliminary_export([export], plan, amendment)
    assert len(metadata) == 1 and metadata[0][1]["seed"] == 43
    report, work = tmp_path / "preliminary-report", tmp_path / "preliminary-work"
    summary = analysis.run_preliminary([export], plan, report, work, amendment)
    assert summary["preliminary"] and not summary["complete_two_seed_claim"]
    assert summary["available_seeds"] == [43] and summary["missing_seeds"] == [42]
    assert summary["counts"]["pairs"] == 20480 and summary["counts"]["examples"] == 512
    assert all(row["seed"] == 43 for row in summary["metrics"])
    assert {row["method"] for row in summary["metrics"]} == set(METHODS)
    for gate in summary["gates"].values():
        assert gate["cleared_every_seed_and_rate"] is None
        assert not gate["full_two_seed_conclusion_available"]
    text = (report / "report.html").read_text()
    assert "PRELIMINARY seed43" in text and "Seed42 is still required" in text
    assert "Both original source/input/cache/output audits passed" not in text
    first = json.loads((work / "first-reading.json").read_text())
    assert not first["both_complete_exports_verified"] and first["preliminary_user_authorized"]
    assert first["preliminary_amendment_sha256"] == file_hash(amendment)
    # The same guard used at run time rejects a later authorization-file mutation.
    original = json.loads(amendment.read_text())
    atomic_json(amendment, {**original, "written_unix": 2600})
    with pytest.raises(ValueError, match="authorization mismatch"):
        analysis.require_preliminary_export([export], plan, amendment)
