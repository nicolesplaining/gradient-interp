import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from gradient_interp.paper_svd_analysis import (
    METRICS,
    bootstrap,
    controlled_statistics,
    descriptor_differences,
    development_gates,
    guard,
    interval,
    paired_weights,
    plots,
    render_report,
    signed_response,
    summarize,
)
from gradient_interp.paper_svd_replication import (
    FAMILIES,
    METHODS,
    condition_grid,
    descriptor_data,
)
from gradient_interp.study import atomic_json, file_hash

SPEC = importlib.util.spec_from_file_location(
    "audit_paper_svd", Path(__file__).parents[1] / "scripts/audit_paper_svd.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def completion_receipt(seed):
    return {
        "type": "paper_svd_completion_receipt",
        "seed": seed,
        "status": "complete",
        "release_reason": "complete",
        "started_unix": 110.0,
        "completed_unix": 120.0,
        "deadline_unix": 43310.0,
        "observed_unix": 130.0,
        "all_processes_dead": True,
        "manifest_sha256": "a" * 64,
        "controller_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
        "specification_sha256": "d" * 64,
    }


def test_release_barrier_requires_both_dead_runs_and_no_early_failed_release():
    receipts = [completion_receipt(seed) for seed in (42, 43)]
    AUDIT.validate_receipts(receipts, 140.0)
    for replacement in (
        {"all_processes_dead": False},
        {"status": "failed"},
        {"release_reason": "original_deadline", "status": "failed"},
        {"observed_unix": 119.0},
        {"deadline_unix": 50000.0},
    ):
        changed = copy.deepcopy(receipts)
        changed[1].update(replacement)
        with pytest.raises(ValueError):
            AUDIT.validate_receipts(changed, 140.0)
    receipts[1].update(
        release_reason="original_deadline", status="budget_exhausted", observed_unix=43311.0
    )
    AUDIT.validate_receipts(receipts, 43312.0)
    with pytest.raises(ValueError, match="Both seed"):
        AUDIT.validate_receipts(receipts[:1], 43312.0)


def test_receipt_inspects_metadata_only_and_refuses_live_worker(tmp_path, monkeypatch):
    atomic_json(tmp_path / "plan.json", {"max_hours": 12})
    atomic_json(tmp_path / "specification.json", {"training_seed": 42})
    controller = {
        "status": "complete",
        "started_unix": 110,
        "deadline_unix": 43310,
        "controller_pid": 1,
        "worker_pid": None,
        "attempts": [{"pid": 2}],
        "plan_sha256": file_hash(tmp_path / "plan.json"),
    }
    atomic_json(tmp_path / "controller.json", controller)
    atomic_json(tmp_path / "manifest.json", {"completed_unix": 120})
    monkeypatch.setattr(AUDIT.time, "time", lambda: 140)
    monkeypatch.setattr(AUDIT, "process_alive", lambda pid: pid == 2)
    with pytest.raises(ValueError, match="processes must be dead"):
        AUDIT.receipt(tmp_path, tmp_path / "receipt.json")
    monkeypatch.setattr(AUDIT, "process_alive", lambda pid: False)
    record = AUDIT.receipt(tmp_path, tmp_path / "receipt.json")
    assert record["all_processes_dead"] and record["release_reason"] == "complete"
    assert not (tmp_path / "descriptors.pt").exists()


def test_full_analysis_rejects_partial_deadline_release_before_result_access(tmp_path):
    plan = tmp_path / "analysis-plan.json"
    atomic_json(
        plan,
        {
            "scientific_outcomes_read": False,
            "written_unix": 100,
            "source_commit": AUDIT.FROZEN_COMMIT,
        },
    )
    receipts = [completion_receipt(seed) for seed in (42, 43)]
    receipts[1].update(status="budget_exhausted", release_reason="original_deadline")
    barrier = tmp_path / "barrier.json"
    atomic_json(
        barrier,
        {
            "type": "paper_svd_pair_barrier",
            "created_unix": 140,
            "analysis_plan_sha256": file_hash(plan),
            "receipts": receipts,
        },
    )
    with pytest.raises(ValueError, match="Complete dead-worker"):
        guard([tmp_path / "absent42", tmp_path / "absent43"], plan, barrier)


def test_paired_weights_preserve_strata_and_controlled_item_groups():
    published, items = paired_weights(64)
    np.testing.assert_allclose(published[:, :, :60].sum(axis=-1), 0.6)
    np.testing.assert_allclose(published[:, :, 60:].sum(axis=-1), 0.4)
    np.testing.assert_allclose(items.sum(axis=-1), 1)
    np.testing.assert_array_equal(published, paired_weights(64)[0])
    assert np.any(published[0] != published[1])


def synthetic_arrays():
    n = len(METRICS)
    pub = np.zeros((2, 5, 5, 2, 100, n))
    ctrl = np.zeros((2, 5, 5, 2, 40, n))
    baseline = np.zeros((40, n))
    ctrl[..., METRICS.index("controlled_content_check")] = 1
    baseline[:, METRICS.index("controlled_content_check")] = 1
    descriptor = np.ones((2, 5, 5, 2, 8, 2))
    return pub, ctrl, baseline, descriptor


def test_primary_averages_all_strengths_instead_of_best_strength():
    pub, _, _, _ = synthetic_arrays()
    pub[:, :, 0, 0, :, 0] = 1
    np.testing.assert_allclose(signed_response(pub)[..., 0], 0.1)
    with pytest.raises(ValueError, match="five strengths"):
        signed_response(pub[:, :, :4])


def test_bootstrap_preserves_method_pairing_and_shared_descriptor_labels():
    pub, ctrl, baseline, descriptor = synthetic_arrays()
    rng = np.random.default_rng(9)
    pub[0] = rng.random(pub[0].shape)
    pub[1] = pub[0]
    descriptor[0] = rng.random(descriptor[0].shape)
    descriptor[1] = descriptor[0]
    result = bootstrap(pub, ctrl, baseline, descriptor, 32)
    np.testing.assert_array_equal(result["aggregate_advantage"], 0)
    np.testing.assert_array_equal(result["family_advantage"], 0)
    np.testing.assert_array_equal(result["descriptor_mean"][:, 0], result["descriptor_mean"][:, 1])
    _, iw = paired_weights(32)
    expected = sum(iw[0, 3, i] * descriptor[0, 2, 3, 1, i, 0] for i in range(8))
    assert result["descriptor_mean"][0, 0, 2, 3, 1, 0] == pytest.approx(expected)


def test_positive_part_quality_prevents_cancellation_and_reports_worst():
    _, ctrl, baseline, _ = synthetic_arrays()
    content = METRICS.index("controlled_content_check")
    baseline[:, content] = 0.5
    ctrl[..., content] = 0.5
    ctrl[:, :, :, 0, :, content] = 0
    ctrl[:, :, :, 1, :, content] = 1
    values = controlled_statistics(ctrl, baseline, np.full((1, 40), 1 / 40))
    np.testing.assert_allclose(values["mean_harm"][..., 0], 0.25)
    np.testing.assert_allclose(values["worst_harm"][..., 0], 0.5)
    np.testing.assert_allclose(values["mean_harm"][..., 1:], 0)


def test_descriptor_pairs_keep_family_item_and_length_and_zeros():
    data = descriptor_data()
    codes, cosines = np.zeros((160, 10)), np.zeros((160, 10))
    for i, row in enumerate(data["descriptor_pairs"]):
        if row["label"] == "target":
            value = 2 if row["verbosity"] == "long" else 1
            codes[i] = value
            cosines[i] = value / 10
    diff, rows = descriptor_differences(data, codes, cosines)
    assert diff.shape == (2, 5, 5, 2, 8, 2) and len(rows) == 1600
    np.testing.assert_array_equal(diff[..., 0, :, 0], 1)
    np.testing.assert_array_equal(diff[..., 1, :, 0], 2)
    assert sum(row["oriented_coefficient"] == 0 for row in rows) == 800
    with pytest.raises(ValueError, match="incomplete"):
        descriptor_differences({"descriptor_pairs": data["descriptor_pairs"][:-1]}, codes, cosines)


def test_undefined_intervals_are_not_silently_dropped_and_familywise_is_wider():
    assert interval([0, np.nan, 1])["undefined_replicates"] == 1
    assert interval([0, np.nan, 1])["lo"] is None
    normal, corrected = interval(np.arange(1000)), interval(np.arange(1000), 5)
    assert corrected["coverage"] == 0.99 and normal["coverage"] == 0.95
    assert corrected["lo"] < normal["lo"] < normal["hi"] < corrected["hi"]


def test_strong_numerical_signal_cannot_override_descriptor_or_quality_failure():
    pub, ctrl, baseline, descriptor = synthetic_arrays()
    for fi in range(5):
        pub[1, fi, :, 0, :, fi] = 0.6
        ctrl[1, fi, :, 0, :, fi] = 0.6
    boot = bootstrap(pub, ctrl, baseline, descriptor, 16)
    controls = controlled_statistics(ctrl, baseline, np.full((1, 40), 1 / 40))
    family, aggregate = np.full((2, 5), 0.3), np.array([0, 0.3])
    good = development_gates(family, aggregate, boot, descriptor, controls)
    assert good["all_svd_requirements_pass"]
    descriptor[1, 0, 0] = 0
    bad = development_gates(family, aggregate, boot, descriptor, controls)
    assert bad["numerical_thresholds_pass"] and not bad["all_svd_requirements_pass"]
    descriptor[1, 0, 0] = 1
    controls["mean_harm"][0, 1, 0, 0] = 0.06
    assert not development_gates(family, aggregate, boot, descriptor, controls)[
        "all_svd_requirements_pass"
    ]


def test_compact_export_and_summary_cover_full_synthetic_grid(tmp_path):
    data = descriptor_data()
    data["published_questions"] = [
        {
            "id": f"published-{family}-{i:03d}",
            "family": family,
            "split": "inviting" if i < 60 else "control",
            "prompt": "Synthetic prompt",
        }
        for family in FAMILIES
        for i in range(100)
    ]
    records = []
    for condition in condition_grid():
        subset = [
            q
            for q in data["published_questions"]
            if not condition["method"] or q["family"] == condition["family"]
        ] + data["questions"]
        rows = []
        for q in subset:
            checks = {name: False for name in METRICS}
            checks.update(tokens=2, words=1)
            if q["split"] == "controlled":
                checks.update(controlled_content_check=True, required_content_present=True)
            else:
                del checks["controlled_content_check"], checks["required_content_present"]
            rows.append(
                {**q, "checks": checks, "response": "Synthetic answer", "token_ids": [7, 1]}
            )
        records.append({"rows": rows})
    descriptors = {
        "coefficients": {method: torch.zeros(160, 500) for method in METHODS},
        "selected_physical_cosines": torch.zeros(160, 10),
        "completion_lengths": torch.ones(160, dtype=torch.int64),
    }
    selection = {
        method: {f: {"component": i, "orientation": 1} for i, f in enumerate(FAMILIES)}
        for method in METHODS
    }
    candidates = {
        method: [{"component": i, "coherence": 0, "top_doc_indices": []} for i in range(500)]
        for method in METHODS
    }
    hashes = AUDIT.compact_exports(
        tmp_path, records, data, descriptors, selection, candidates, [], {}
    )
    assert len(hashes) == 3
    for name, value in (
        ("data", data),
        ("specification", {"training_seed": 42}),
        ("fit", {}),
        ("selection", selection),
    ):
        atomic_json(tmp_path / f"{name}.json", value)
    summary = summarize(tmp_path, tmp_path, 16)
    assert len(summary["primary"]) == 18
    assert len(summary["candidates"]) == 1000
    assert len(summary["descriptor_values"]) == 1600
    assert len(summary["examples"]) == 5090
    assert not summary["gates"]["all_svd_requirements_pass"]
    assert all(
        r["positive_pairs"] == 0 and r["zero_pairs"] == 8 for r in summary["descriptor_summary"]
    )
    second = {**summary, "seed": 43}
    plots([summary, second], tmp_path)
    render_report([summary, second], tmp_path)
    assert (tmp_path / "report.html").exists()
    assert len(list(tmp_path.glob("*.png"))) == 5


def test_audited_export_hash_guard_refuses_stale_or_incomplete_provenance(tmp_path):
    plan_path = tmp_path / "analysis-plan.json"
    atomic_json(
        plan_path,
        {
            "source_commit": AUDIT.FROZEN_COMMIT,
            "scientific_outcomes_read": False,
            "written_unix": 100,
        },
    )
    receipts, roots = [], []
    names = (
        "plan",
        "specification",
        "controller",
        "manifest",
        "selection-lock",
        "descriptor-manifest",
        "selection",
        "candidates",
        "data",
        "fit",
        "preflight",
        "data-overlap-audit",
    )
    for seed in (42, 43):
        root = tmp_path / str(seed)
        root.mkdir()
        roots.append(root)
        record = completion_receipt(seed)
        for name in names:
            atomic_json(root / f"{name}.json", {"synthetic": True, "seed": seed})
            if name in ("plan", "specification", "controller", "manifest"):
                record[name + "_sha256"] = file_hash(root / f"{name}.json")
        receipts.append(record)
        for name in ("analysis-export.npz", "fixed-examples.json", "analysis-candidates.json"):
            (root / name).write_text("synthetic bytes only")
    barrier = tmp_path / "barrier.json"
    atomic_json(
        barrier,
        {
            "type": "paper_svd_pair_barrier",
            "created_unix": 140,
            "analysis_plan_sha256": file_hash(plan_path),
            "receipts": receipts,
        },
    )
    for seed, root in zip((42, 43), roots, strict=True):
        flags = {
            key: True
            for key in (
                "complete",
                "source_input_output_hashes_verified",
                "selection_and_orientations_verified",
                "physical_norm_matching_verified",
                "descriptor_coefficients_recomputed",
                "all_saved_generation_metrics_recomputed",
                "original_deadline_verified",
                "all_adapter_restores_verified",
            )
        }
        atomic_json(
            root / "integrity.json",
            {
                **flags,
                "seed": seed,
                "barrier_sha256": file_hash(barrier),
                "analysis_plan_sha256": file_hash(plan_path),
                "counts": {"conditions": 101, "responses": 14540, "descriptor_gradients": 160},
                "metadata_hashes": {name: file_hash(root / f"{name}.json") for name in names},
                "analysis_export_hashes": {
                    name: file_hash(root / name)
                    for name in (
                        "analysis-export.npz",
                        "fixed-examples.json",
                        "analysis-candidates.json",
                    )
                },
            },
        )
    guard(roots, plan_path, barrier)
    changed = roots[1] / "analysis-export.npz"
    changed.write_text("changed outcome")
    with pytest.raises(ValueError, match="compact outcome changed"):
        guard(roots, plan_path, barrier)
    changed.write_text("synthetic bytes only")
    audit = AUDIT.read(roots[1] / "integrity.json")
    del audit["analysis_export_hashes"]["analysis-export.npz"]
    atomic_json(roots[1] / "integrity.json", audit)
    with pytest.raises(ValueError, match="export grid incomplete"):
        guard(roots, plan_path, barrier)


def test_barrier_refuses_late_plan_and_preserves_existing_receipt(tmp_path):
    paths = []
    for seed in (42, 43):
        path = tmp_path / f"receipt{seed}.json"
        atomic_json(path, completion_receipt(seed))
        paths.append(path)
    plan_path = tmp_path / "analysis-plan.json"
    atomic_json(
        plan_path,
        {
            "source_commit": AUDIT.FROZEN_COMMIT,
            "scientific_outcomes_read": False,
            "written_unix": 131,
        },
    )
    with pytest.raises(ValueError, match="plan must precede"):
        AUDIT.barrier(paths, plan_path, tmp_path / "barrier.json")
    with pytest.raises(ValueError, match="Preserve"):
        AUDIT.receipt(tmp_path / "nonexistent", paths[0])
