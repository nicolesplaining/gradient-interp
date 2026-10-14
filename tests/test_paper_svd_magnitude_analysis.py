import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradient_interp import paper_svd_magnitude_analysis as analysis
from gradient_interp.paper_svd_magnitude import METHODS, RATES, component_contributions
from gradient_interp.study import atomic_json, file_hash

AUDITOR_PATH = Path(__file__).parents[1] / "scripts/audit_paper_svd_magnitude.py"
AUDITOR_SPEC = importlib.util.spec_from_file_location("magnitude_audit_test", AUDITOR_PATH)
auditor = importlib.util.module_from_spec(AUDITOR_SPEC)
AUDITOR_SPEC.loader.exec_module(auditor)


def completion_receipt(seed=42):
    return dict(
        type="paper_svd_magnitude_completion_receipt",
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
    rates = np.array([1, 0.1])[None, :, None]
    actual_change = -b * q * rates
    predictions = actual_change[:, :, None] * np.array([0.8, 0.5, 1, 0])[None, None, :, None]
    coefficients = np.broadcast_to(
        b[:, 0, 0, None, None] * np.array([0.8, 0.5])[None, :, None] / (RATES[0] * 500),
        (32, 2, 500),
    ).copy()
    profiles = np.broadcast_to(q[0, 0], (2, 500, 80)).copy()
    return dict(
        coefficient_active_above_1e_minus6=np.full((32, 2), 500),
        component_loss_profiles=profiles,
        component_loss_contributions=component_contributions(coefficients, profiles),
        component_physical_norms=np.ones((2, 500)),
        predicted_change=predictions,
        actual_mean_ce=1 + actual_change,
        baseline_mean_ce=np.ones(80),
        completion_tokens=np.full(80, 10),
        coefficients=coefficients,
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
    integrity["unscaled_components_recomputed"] = True
    atomic_json(root / "named-selections.json", selections())
    atomic_json(root / "integrity.json", integrity)
    atomic_json(
        root / "export.json",
        dict(
            type="paper_svd_magnitude_compact_export",
            seed=seed,
            methods=list(METHODS),
            rates=list(RATES),
            analysis_plan_sha256=file_hash(plan),
            files_sha256={
                n: file_hash(root / n)
                for n in ("arrays.npz", "data.json", "integrity.json", "named-selections.json")
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
    result = auditor.norm_diagnostics(cache, {arm: np.eye(2) for arm in METHODS[:2]})
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


def selections():
    from gradient_interp.paper_svd_replication import FAMILIES

    return {
        method: {
            label: dict(component=i, orientation=1, candidate_only=True)
            for i, label in enumerate(FAMILIES)
        }
        for method in METHODS[:2]
    }


def test_component_shares_keep_cancellation_zero_denominators_and_all_components():
    arrays = synthetic_arrays()
    arrays["component_loss_contributions"][:] = 0
    arrays["component_loss_contributions"][0, 0, 0, 0, 16] = 2
    arrays["component_loss_contributions"][0, 0, 0, 9, 16] = -2
    rows, named = analysis.component_tables(42, arrays, selections())
    assert len(rows) == 2 * 2 * 2 * 500
    assert len(named) == 2 * 2 * 32 * 80
    row = next(
        r
        for r in named
        if r["method"] == "svd500"
        and r["batch"] == 0
        and r["probe_index"] == 16
        and r["rate"] == RATES[0]
    )
    assert row["named_absolute_share"] == 0.5
    assert row["named_signed_share"] is None
    assert row["cancellation_fraction"] == 1
    assert row["zero_signed_denominator"] and not row["zero_absolute_denominator"]
    zero = next(r for r in named if r["batch"] == 1)
    assert zero["named_absolute_share"] is None and zero["zero_absolute_denominator"]


def test_compact_arrays_enforce_component_sum_and_exact_activity():
    arrays = synthetic_arrays()
    analysis.validate_arrays(arrays)
    arrays["component_loss_contributions"][0, 0, 0, 0, 0] += 1
    with pytest.raises(AssertionError):
        analysis.validate_arrays(arrays)
    arrays = synthetic_arrays()
    arrays["coefficient_l0"][0, 0] -= 1
    with pytest.raises(AssertionError):
        analysis.validate_arrays(arrays)


def test_summarize_preserves_all_splits_pairs_fixed_examples_and_primary_gate(tmp_path):
    result = analysis.summarize(42, synthetic_arrays(), synthetic_data(), tmp_path, replicates=40)
    assert result["gate"]["cleared"]
    assert len(result["metrics"]) == 16
    assert len(result["pairs"]) == 32 * 2 * 4 * 80
    assert len(result["examples"]) == 8 * 2 * 4 * 8
    assert {r["scope"] for r in result["metrics"]} == {"calibration", "evaluation"}
    for r in result["comparisons"]:
        assert r["dictionary_minus_svd_nrmse"] == pytest.approx(0.3)
    arrays = synthetic_arrays()
    arrays["predicted_change"][:, 0, 0] = 0
    result = analysis.summarize(42, arrays, synthetic_data(), tmp_path, replicates=40)
    assert not result["gate"]["cleared"]
    assert "svd_nrmse" in result["gate"]["failed_quantities"]


def test_full_analysis_requires_both_audits_and_preserves_1000_replicates(tmp_path):
    plan = analysis_plan(tmp_path / "analysis-plan.json")
    exports = [compact_export(tmp_path, seed, plan) for seed in (42, 43)]
    report, work = tmp_path / "report", tmp_path / "work"
    summary = analysis.run(exports, plan, report, work)
    assert summary["gates"]["cleared_both_seeds"]
    assert summary["counts"]["pairs"] == 40960
    assert summary["counts"]["components"] == 8000
    assert summary["counts"]["coefficients"] == 64000
    assert summary["counts"]["profiles"] == 160000
    draws = np.load(work / "seed42-evaluation-bootstrap.npz")
    assert draws["nrmse"].shape == (1000, 2, 4)
    body = (report / "report.html").read_text()
    assert "&lt;untrusted markup&gt;" in body
    assert "99.375%" in body and "cannot rescue" in body
    with pytest.raises(ValueError, match="Preserve"):
        analysis.run(exports, plan, report, work)


def test_export_pair_cannot_mix_completion_barriers(tmp_path):
    plan = analysis_plan(tmp_path / "analysis-plan.json")
    exports = [compact_export(tmp_path, seed, plan) for seed in (42, 43)]
    integrity = json.loads((exports[1] / "integrity.json").read_text())
    integrity["completion_receipts"][0]["manifest_sha256"] = "e" * 64
    atomic_json(exports[1] / "integrity.json", integrity)
    with pytest.raises(ValueError, match="receipt barrier"):
        analysis.require_complete_exports(exports, plan)
