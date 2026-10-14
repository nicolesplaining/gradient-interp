import json

import numpy as np
import pytest
import torch

from gradient_interp.paper_atom_batches import intervention_conditions
from gradient_interp.paper_atom_batches_analysis import (
    ENDPOINTS,
    combined_units,
    contrasts,
    grouped_weights,
    guard,
    metrics,
    scopes,
    stratified_questions,
    weighted_covariance,
)
from gradient_interp.paper_tangent_analysis import interval_fields
from gradient_interp.study import file_hash


def test_weighted_covariance_matches_explicit_paired_resampling():
    rng = np.random.default_rng(72)
    units = rng.normal(size=(3, 11, 4))
    rewards = rng.integers(0, 2, size=(3, 11, 2)).astype(float)
    indices = rng.integers(0, 11, size=(3, 11))
    weights = np.array([np.bincount(i, minlength=11) for i in indices]) / 11
    actual = weighted_covariance(units, rewards, weights)
    expected = np.empty((4, 3, 2))
    for q in range(3):
        for d in range(4):
            for e in range(2):
                expected[d, q, e] = np.cov(
                    units[q, indices[q], d], rewards[q, indices[q], e], ddof=1
                )[0, 1]
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    # A behavior with no observed events retains a zero derivative for every draw.
    np.testing.assert_array_equal(weighted_covariance(units, np.zeros_like(rewards), weights), 0)


def test_atom_combination_and_control_signs_preserve_linear_effects():
    rng = np.random.default_rng(62)
    scores = rng.normal(size=(24, 128, 532))
    coeff = np.zeros((16, 500))
    coeff[:, :4] = rng.normal(size=(16, 4))
    coeff[0, 0] = 0
    cards = dict(
        numbered=dict(atom_index=0, orientation=1), bullets=dict(atom_index=1, orientation=-1)
    )
    directions = torch.tensor(rng.normal(size=(500, 19)), dtype=torch.float32)
    gradients = torch.tensor(rng.normal(size=(16, 19)), dtype=torch.float32)
    batches = [dict(target="bullets" if i % 4 == 2 else "numbered") for i in range(16)]
    conditions = intervention_conditions(gradients, coeff, directions, batches, cards, 14)
    z = dict(units=scores, coefficients=coeff, permutation=np.arange(499, -1, -1))
    combined = combined_units(z, cards, conditions)
    assert combined.shape == (24, 128, 240)
    np.testing.assert_array_equal(combined[:, :, 5::6][:, :, :16], 0)
    for b in (0, 3, 15):
        expected = sum(coeff[b, a] * scores[:, :, a] for a in range(4))
        np.testing.assert_allclose(combined[:, :, b * 6], expected)
        np.testing.assert_array_equal(combined[:, :, b * 6 + 3], scores[:, :, 500 + b])
        named = conditions[b * 4 + 1]
        np.testing.assert_allclose(
            combined[:, :, 96 + 2 * (b * 4 + 1)],
            expected - named["named_coefficient"] * scores[:, :, named["target"]],
        )
        random = conditions[b * 4 + 2]
        np.testing.assert_allclose(
            combined[:, :, 96 + 2 * (b * 4 + 2) + 1],
            scores[:, :, 500 + b]
            - random["random_coefficient"] * scores[:, :, random["random_atom"]],
        )
    for ci in range(64, 72):
        c = conditions[ci]
        expected = c["coefficient"] * scores[:, :, c["atom_index"]]
        np.testing.assert_array_equal(combined[:, :, 96 + 2 * ci], expected)
        np.testing.assert_array_equal(combined[:, :, 96 + 2 * ci + 1], expected)


def test_resampling_keeps_task_strata_and_content_blocks_paired_across_styles():
    rng = np.random.default_rng(12)
    for per_task in (2, 4):
        for _ in range(20):
            qw = stratified_questions(rng, per_task)
            np.testing.assert_allclose([qw[t::4].sum() for t in range(4)], 0.25)
            np.testing.assert_allclose(qw * (4 * per_task), np.round(qw * (4 * per_task)))
    block = np.array([0.5, 0, 0.25, 0.25])
    bw, qw = grouped_weights(np.repeat(block / 4, 4), stratified_questions(rng, 4))
    for s in range(4):
        np.testing.assert_array_equal(bw[s + 1, s::4], block)
        assert np.count_nonzero(bw[s + 1, np.arange(16) % 4 != s]) == 0
        assert np.count_nonzero(qw[s + 5, np.arange(16) % 4 != s]) == 0
    np.testing.assert_allclose(bw.sum(1), 1)
    np.testing.assert_allclose(qw.sum(1), 1)


def test_primary_transfer_uses_only_separate_calibration_questions():
    p = np.zeros((16, 6, 24, 8))
    p[:, :, :8] = np.arange(8)[None, None, :, None]
    p[:, :, 8:] = 100
    weights = np.zeros(8)
    weights[3] = 1
    transferred, same = scopes(p, weights)
    np.testing.assert_array_equal(transferred, 3)
    np.testing.assert_array_equal(same, 100)


def test_zero_changes_and_unclipped_probability_violations_are_reported():
    pred = np.zeros((16, 6, 16, 8))
    pred[:, 0] = 0.2
    target = np.zeros((16, 16, 8))
    bw, qw = grouped_weights(np.full(16, 1 / 16), np.full(16, 1 / 16))
    result = metrics(pred, target, bw, qw, np.full((16, 8), 0.9))
    np.testing.assert_allclose(result["rmse"][:, 0], 0.2)
    np.testing.assert_allclose(result["bias"][:, 0], 0.2)
    assert np.isnan(result["nrmse"]).all()
    for i, endpoint in enumerate(ENDPOINTS):
        if endpoint in ("tokens", "words"):
            assert np.isnan(result["outside_probability_range_fraction"][:, :, i]).all()
        else:
            np.testing.assert_array_equal(result["outside_probability_range_fraction"][:, 0, i], 1)
            np.testing.assert_array_equal(result["outside_probability_range_fraction"][:, 5, i], 0)
    interval = interval_fields(np.array([1.0, np.nan, 2.0]), "ratio")
    assert interval == {"ratio_lo": None, "ratio_hi": None, "ratio_defined_replicates": 2}


def test_selective_contrasts_cancel_shared_full_update_and_keep_zero_edits():
    rng = np.random.default_rng(52)
    values = rng.normal(size=(72, 16, 8))
    base = values[:64:4].copy()
    values[1:64:4] = base
    values[2:64:4] = base + 0.25
    values[3:64:4] = base - 0.5
    c = contrasts(values)
    np.testing.assert_array_equal(c[0], 0)
    np.testing.assert_allclose(c[1], 0.25)
    np.testing.assert_allclose(c[2], -0.5)
    np.testing.assert_allclose(c[3], -0.25)
    np.testing.assert_allclose(c[4], 0.5)


def audited_roots(tmp_path):
    roots = []
    for seed in (42, 43):
        root = tmp_path / str(seed)
        root.mkdir()
        (root / "result.json").write_text("frozen data")
        files = {"result.json": file_hash(root / "result.json")}
        values = {
            "controller": dict(status="complete", deadline_unix=20),
            "plan": {},
            "specification": {},
            "manifest": dict(completed_unix=19, adapter_restored=True, files_sha256=files),
            "forecast-lock": dict(original_adapter_unchanged=True, files_sha256=files),
        }
        for name, value in values.items():
            (root / f"{name}.json").write_text(json.dumps(value))
        integrity = dict(
            complete=True,
            source_input_output_hashes_verified=True,
            physical_control_formula_verified=True,
        )
        for name in ("plan", "specification", "manifest", "forecast-lock"):
            integrity[name.replace("-", "_") + "_sha256"] = file_hash(root / f"{name}.json")
        (root / "integrity.json").write_text(json.dumps(integrity))
        roots.append(root)
    plan = tmp_path / "analysis-plan.json"
    plan.write_text(json.dumps(dict(outcomes_read_before_plan=False)))
    return roots, plan


def test_guard_requires_both_complete_audits_and_rechecks_result_hashes(tmp_path):
    roots, plan = audited_roots(tmp_path)
    guard(roots, plan)
    with pytest.raises(AssertionError):
        guard(roots[:1], plan)
    controller = roots[1] / "controller.json"
    controller.write_text(json.dumps(dict(status="running", deadline_unix=20)))
    with pytest.raises(AssertionError):
        guard(roots, plan)
    controller.write_text(json.dumps(dict(status="complete", deadline_unix=20)))
    (roots[0] / "result.json").write_text("modified after audit")
    with pytest.raises(AssertionError):
        guard(roots, plan)


def test_guard_rejects_plan_written_after_outcome_reading(tmp_path):
    roots, plan = audited_roots(tmp_path)
    plan.write_text(json.dumps(dict(outcomes_read_before_plan=True)))
    with pytest.raises(AssertionError):
        guard(roots, plan)
