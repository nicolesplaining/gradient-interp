import copy

import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_exact_calibration_inference as inference
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import NO_CHANGE, PATTERN_IDS
from gradient_interp.update_loracle_physical_profile import profile_label
from gradient_interp.update_loracle_sensitivity import matched_shuffle


def evidence(correct=64):
    cases, rows = [], []
    for i in range(72):
        zero = i >= 64
        target = NO_CHANGE if zero else PATTERN_IDS[i % 32]
        cases.append(
            {
                "id": str(i),
                "control": zero,
                "label": target,
                "measured": {"effect": {"label": target, "below_effect_floor": zero}},
                "predictions": {
                    n: NO_CHANGE for n in ("released", "flat_majority", "factorized_majority")
                },
            }
        )
        rows.append(
            {
                "id": str(i),
                "prediction": target if i < correct or zero else NO_CHANGE,
                "shuffled_prediction": NO_CHANGE,
            }
        )
    m = inference.recomputed_metrics(cases, rows)
    g = inference.development_gates(m)
    return (
        cases,
        rows,
        {
            "recipe": inference.RECIPE,
            "reserved_test_read": False,
            "metrics": m,
            "development_gates": g,
            "all_development_gates_pass": all(g.values()),
        },
    )


def test_gate_recomputes_full_description_and_cannot_accept_edited_summary():
    cases, rows, summary = evidence()
    m, gates = inference.checked_development(summary, cases, rows)
    assert m["functional"]["actual_update_correct"] == 64 and all(gates.values())
    failed_cases, failed_rows, failed = evidence(44)
    with pytest.raises(ValueError, match="criteria failed"):
        inference.checked_development(failed, failed_cases, failed_rows)
    # The successful report cannot be attached to failing predictions.
    with pytest.raises(ValueError, match="criteria failed"):
        inference.checked_development(summary, failed_cases, failed_rows)


def test_gate_keeps_weak_updates_and_requires_all_cases_and_controls():
    cases, rows, _ = evidence()
    cases[0]["label"] = NO_CHANGE
    cases[0]["measured"]["effect"] = {"label": NO_CHANGE, "below_effect_floor": True}
    m = inference.recomputed_metrics(cases, rows)
    assert m["functional"]["actual_update_n"] == 64
    assert m["functional"]["actual_update_correct"] == 63
    assert m["functional"]["above_effect_floor_fraction"] == 63 / 64
    with pytest.raises(ValueError, match="grid"):
        inference.recomputed_metrics(cases[1:], rows[1:])
    bad = copy.deepcopy(cases)
    bad[-1]["id"] = bad[0]["id"]
    with pytest.raises(ValueError, match="grid"):
        inference.recomputed_metrics(bad, rows)


def test_partial_or_failed_run_cannot_open_selection(tmp_path):
    for c in [
        {"status": "running", "controller_pid": 0},
        {
            "status": "complete",
            "controller_pid": 0,
            "worker_pid": None,
            "attempts": [{"pid": 0, "returncode": 1}],
        },
    ]:
        atomic_json(tmp_path / "controller.json", c)
        with pytest.raises(ValueError):
            inference.freeze_selection(tmp_path, tmp_path / "analysis", tmp_path / "selection.json")
        assert not (tmp_path / "selection.json").exists()


def test_new_schema_cannot_be_used_by_old_adapter_loader(tmp_path):
    from gradient_interp.update_loracle_inference import verified_selection

    p = tmp_path / "selection.json"
    atomic_json(p, {"schema": inference.SCHEMA})
    with pytest.raises(ValueError, match="Unknown selection"):
        verified_selection(p)


def selection_fixture(tmp_path, monkeypatch):
    from gradient_interp import update_loracle_exact_calibration as worker

    root, collection, baseline, analysis = [
        tmp_path / p for p in ("run", "collection", "baseline", "analysis")
    ]
    for p in (root, collection / "cases", baseline / "analysis", analysis):
        p.mkdir(parents=True)
    model = tmp_path / "base/model.safetensors"
    model.parent.mkdir()
    model.write_bytes(b"synthetic pinned base fixture")
    monkeypatch.setattr(inference, "BASE_PATH", str(model.parent))
    cases, original, targets, raw = [], [], [], []
    for split, n, zeros, seed in (("train", 272, 16, 1), ("development", 72, 8, 11)):
        for i in range(n):
            control = i >= n - zeros
            case = {
                "id": f"{split}-{i}",
                "split": split,
                "control": control,
                "initialization_seed": seed + i % 4,
                "physical_update_norm": 0.0 if control else 1.0,
            }
            cases.append(case)
            original.append(
                {
                    **case,
                    "probes": [
                        {"prompt": f"{split} prompt {i}", "values": ["abc", "d", "efgh", "ij"]}
                    ],
                }
            )
            profile = np.zeros(32)
            if not control:
                profile[i % 32] = 0.5
                profile -= profile.mean()
            raw.append(-profile)
            label = NO_CHANGE if control else PATTERN_IDS[i % 32]
            effect = {
                "effect": {
                    "label": label,
                    "below_effect_floor": control,
                    "centered_improvements": profile.tolist(),
                }
            }
            if split == "development":
                targets.append(
                    {
                        **case,
                        "label": label,
                        "measured": effect,
                        "predictions": {
                            name: NO_CHANGE
                            for name in ("released", "flat_majority", "factorized_majority")
                        },
                    }
                )
                atomic_json(collection / "cases" / f"{case['id']}.json", effect)
    raw = np.array(raw)
    profiles = -raw
    rows = []
    for split in ("train", "development"):
        ids = [i for i, c in enumerate(cases) if c["split"] == split]
        shuffled, donors = matched_shuffle(profiles[ids], [cases[i] for i in ids])
        for j, i in enumerate(ids):
            rows.append(
                {
                    **cases[i],
                    "profile": profiles[i].tolist(),
                    "prediction": profile_label(profiles[i]),
                    "shuffled_profile": shuffled[j].tolist(),
                    "shuffled_prediction": profile_label(shuffled[j]),
                    "direction_donor": donors[j],
                }
            )
    atomic_json(collection / "data.json", {"cases": original})
    atomic_json(
        root / "data.json",
        {"cases": cases, "calibration": inference.select_calibration({"cases": original})},
    )
    atomic_json(root / "predictions.json", rows)
    baseline_scores = np.ones((8, 32)) * 2
    atomic_json(
        root / "baselines.json",
        {"scores": {str(c["initialization_seed"]): baseline_scores.tolist() for c in cases}},
    )
    (root / "scores").mkdir()
    for row in rows:
        for kind, profile in (("own", row["profile"]), ("shuffled", row["shuffled_profile"])):
            atomic_json(
                root / "scores" / f"{row['id']}-{kind}.json",
                {
                    "initialization_seed": row["initialization_seed"],
                    "after_ce": (baseline_scores - np.asarray(profile)).tolist(),
                },
            )
    bp = baseline / "analysis/all-development-cases.json"
    atomic_json(bp, targets)
    inputs = [collection / "data.json", bp, model] + list((collection / "cases").iterdir())
    atomic_json(
        root / "plan.json",
        {
            "collection_root": str(collection),
            "baseline_root": str(baseline),
            "source_commit": "synthetic-only",
            "input_files_sha256": {str(p): file_hash(p) for p in inputs},
            "source_files_sha256": {
                "src/gradient_interp/update_loracle_exact_calibration.py": file_hash(
                    worker.__file__
                )
            },
        },
    )
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
        root / "integrity.json",
        {
            "passed": True,
            "recipe": inference.RECIPE,
            "cases": 344,
            "scored_adapter_conditions": 688,
            "all_profiles_and_native_norms_verified": True,
            "reserved_cases_accessed": 0,
            "fixed_scoring_rechecks": [{"passed": True}] * 4,
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
            },
        },
    )
    selected_metrics = inference.recomputed_metrics(targets, rows[272:])
    gates = inference.development_gates(selected_metrics)
    assert all(gates.values())
    atomic_json(
        analysis / "summary.json",
        {
            "recipe": inference.RECIPE,
            "reserved_test_read": False,
            "metrics": selected_metrics,
            "development_gates": gates,
            "all_development_gates_pass": True,
        },
    )
    (analysis / "executed-analysis.py").write_text("# synthetic fixture only\n")
    atomic_json(
        analysis / "analysis-provenance.json",
        {
            "integrity_sha256": file_hash(root / "integrity.json"),
            "executed_source_sha256": file_hash(analysis / "executed-analysis.py"),
        },
    )
    return root, analysis


def test_complete_selection_binds_evidence_calibration_and_new_method_contract(
    tmp_path, monkeypatch
):
    root, analysis = selection_fixture(tmp_path, monkeypatch)
    path = tmp_path / "selection.json"
    selection = inference.freeze_selection(root, analysis, path)
    assert inference.verified_selection(path) == selection
    assert selection["schema"] == inference.SCHEMA
    assert selection["reserved_test_accessed"] is False
    assert selection["predictor"]["candidate_patterns"] == [list(p) for p in inference.PATTERNS]
    with pytest.raises(ValueError, match="overwrite"):
        inference.freeze_selection(root, analysis, path)
    changed = copy.deepcopy(selection)
    changed["calibration"][0]["prompt"] = "evaluation information"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="calibration changed"):
        inference.verified_selection(path)
    atomic_json(path, selection)
    (analysis / "summary.json").write_text('{"edited":true}')
    with pytest.raises(ValueError, match="evidence changed"):
        inference.verified_selection(path)


def test_exact_native_inference_matches_frozen_worker_and_restores(monkeypatch):
    from gradient_interp import update_loracle_exact_calibration as worker

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.ones(2, 3))
            self.b = torch.nn.Parameter(torch.zeros(4, 2))
            self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    model = Model()
    initial = {"m": (model.a.detach().clone(), model.b.detach().clone())}
    calibration = [{"prompt": str(i), "values": ["abc", "d", "ef", "ghij"]} for i in range(8)]
    lock = {
        "schema": inference.SCHEMA,
        "predictor": inference.predictor_contract(),
        "calibration": calibration,
    }
    monkeypatch.setattr(inference, "verified_selection", lambda p: lock)
    monkeypatch.setattr(inference, "source_keys", lambda: ["m"])

    def restore(m, snapshot):
        with torch.no_grad():
            m.a.copy_(snapshot["m"][0])
            m.b.copy_(snapshot["m"][1])

    def snapshot(m):
        return {"m": (m.a.detach().clone(), m.b.detach().clone())}

    def losses(m, tok, rows):
        z = m.b.sum()
        return torch.stack([2 + z * z + z * len(r["answer"]) / 100 for r in rows])

    monkeypatch.setattr(inference, "restore_modules", restore)
    monkeypatch.setattr(worker, "restore_modules", restore)
    monkeypatch.setattr(worker, "snapshot_modules", snapshot)
    monkeypatch.setattr(worker, "losses", losses)
    updates = [
        {"m": (initial["m"][0], torch.full_like(initial["m"][1], scale))}
        for scale in (0.125, 0.25, 0.0)
    ]
    outputs = inference.predict_with_model("synthetic-selection", model, None, initial, updates)
    assert len(outputs) == 3 and outputs[-1]["prediction"] == NO_CHANGE
    assert not np.allclose(np.asarray(outputs[0]["ce_change"]) * 2, outputs[1]["ce_change"])
    for updated, output in zip(updates, outputs, strict=True):
        baseline = worker.calibration_scores(model, None, calibration)
        after = worker.score_and_restore(model, None, calibration, initial, updated, baseline)
        expected = worker.effect_profile(baseline, after)
        assert output["prediction"] == expected["prediction"]
        assert output["profile"] == expected["profile"]
    assert not model.b.count_nonzero() and torch.equal(model.a, initial["m"][0])
    with pytest.raises(ValueError, match="unchanged A"):
        inference.predict_with_model(
            "synthetic-selection",
            model,
            None,
            initial,
            [{"m": (initial["m"][0] + 1, initial["m"][1])}],
        )


def test_modified_base_rejected_before_model_load(tmp_path, monkeypatch):
    base = tmp_path / "model.safetensors"
    base.write_bytes(b"original")
    expected = file_hash(base)
    monkeypatch.setattr(
        inference, "verified_selection", lambda _: {"base_files_sha256": {str(base): expected}}
    )
    monkeypatch.setattr(inference, "load_target", lambda: pytest.fail("Modified base loaded"))
    base.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pinned base changed"):
        inference.prepare_model(tmp_path / "selection.json")


def test_contract_preserves_scope_and_comparator_interface():
    contract = inference.predictor_contract()
    assert contract["no_change_span"] == 0.02 and len(contract["candidate_patterns"]) == 32
    assert "new evaluation prompts" in contract["excluded_inputs"]
    assert "measured effects" in contract["excluded_inputs"]
    assert "beyond ordinary LoRAcle" in contract["comparison_limitation"]
