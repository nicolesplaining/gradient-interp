import copy

import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_sensitivity_inference as inference
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import NO_CHANGE, PATTERN_IDS
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
    assert m["sensitivity"]["actual_update_correct"] == 64 and all(gates.values())
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
    assert m["sensitivity"]["actual_update_n"] == 64
    assert m["sensitivity"]["actual_update_correct"] == 63
    assert m["sensitivity"]["above_effect_floor_fraction"] == 63 / 64
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


def setup_bank(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    atomic_json(selection, {"synthetic_fixture_only": True})
    initial = {"m": (torch.arange(12, dtype=torch.float32).reshape(3, 4), torch.zeros(5, 3))}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(initial["m"][0].clone())
            self.b = torch.nn.Parameter(initial["m"][1].clone())
            self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    model = Model()
    calibration = [{"prompt": str(i), "values": ["abcd", "ef", "z", "ghijk"]} for i in range(8)]
    monkeypatch.setattr(
        inference,
        "verified_selection",
        lambda _: {"calibration": calibration, "base_files_sha256": {}},
    )
    monkeypatch.setattr(inference, "source_keys", lambda: ["m"])
    monkeypatch.setattr(inference, "load_target", lambda: (model, None))

    def restore(m, initial):
        with torch.no_grad():
            m.a.copy_(initial["m"][0])
            m.b.copy_(initial["m"][1])

    monkeypatch.setattr(inference, "restore_modules", restore)
    monkeypatch.setattr(
        inference, "snapshot_modules", lambda m: {"m": (m.a.detach().clone(), m.b.detach().clone())}
    )
    calls = []

    def gradients(m, tok, rows):
        assert m is model and len(rows) == 8
        assert [r["prompt"] for r in rows] == [str(i) for i in range(8)]
        i = len(calls)
        calls.append(rows)
        return {"m": torch.full_like(m.b, -i * 0.001)}, [1.0] * 8

    monkeypatch.setattr(inference, "gradient_b", gradients)
    directory = tmp_path / "bank"
    manifest = inference.build_bank(selection, initial, directory)
    return selection, initial, directory, manifest, calls, model


def test_bank_to_native_prediction_preserves_sign_magnitude_zero_and_scope(tmp_path, monkeypatch):
    selection, initial, directory, manifest, calls, model = setup_bank(tmp_path, monkeypatch)
    assert len(calls) == len(manifest["records"]) == 32
    assert torch.equal(model.a, initial["m"][0]) and not torch.count_nonzero(model.b)
    b = torch.ones(5, 3)
    snapshots = [{"m": (initial["m"][0], factor * b)} for factor in (1.0, -1.0, 0.001, 0.0)]
    result = inference.predict_native(selection, initial, snapshots, directory, "cpu")
    assert [r["prediction"] for r in result] == [
        PATTERN_IDS[31],
        PATTERN_IDS[0],
        NO_CHANGE,
        NO_CHANGE,
    ]
    np.testing.assert_allclose(
        result[1]["ce_change"], -np.array(result[0]["ce_change"]), atol=1e-12
    )
    assert result[-1]["ce_change"] == [0] * 32
    assert result[0]["gradient_bank_sha256"] == file_hash(directory / "manifest.json")
    with pytest.raises(ValueError, match="exists"):
        inference.build_bank(selection, initial, directory)


def test_native_prediction_rejects_other_initializations_and_modified_gradient_files(
    tmp_path, monkeypatch
):
    selection, initial, directory, _, _, _ = setup_bank(tmp_path, monkeypatch)
    other = {"m": (initial["m"][0] + 1, initial["m"][1])}
    with pytest.raises(ValueError, match="bound"):
        inference.predict_native(selection, other, [other], directory, "cpu")
    with pytest.raises(ValueError, match="native one-step"):
        inference.predict_native(selection, initial, [other], directory, "cpu")
    p = directory / "pattern00.pt"
    torch.save({"m": torch.ones(5, 3)}, p)
    with pytest.raises(ValueError, match="bank changed"):
        inference.predict_native(selection, initial, [initial], directory, "cpu")


def test_contract_discloses_extra_access_and_original_target():
    c = inference.predictor_contract()
    assert c["no_change_span"] == 0.02
    assert len(c["labels"]) == 33 and len(c["candidate_patterns"]) == 32
    assert "measured effects" in c["excluded_inputs"]
    assert "new evaluation prompts" in c["excluded_inputs"]
    assert "beyond ordinary LoRAcle" in c["comparison_limitation"]


def test_modified_base_is_rejected_before_loading_and_calibration(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    atomic_json(selection, {"fixture": True})
    base = tmp_path / "model.safetensors"
    base.write_bytes(b"original pinned model")
    expected = file_hash(base)
    base.write_bytes(b"wrong model")
    monkeypatch.setattr(
        inference, "verified_selection", lambda _: {"base_files_sha256": {str(base): expected}}
    )
    monkeypatch.setattr(inference, "source_keys", lambda: ["m"])
    monkeypatch.setattr(inference, "load_target", lambda: pytest.fail("Wrong base was loaded"))
    initial = {"m": (torch.ones(3, 4), torch.zeros(5, 3))}
    with pytest.raises(ValueError, match="Pinned base model changed"):
        inference.build_bank(selection, initial, tmp_path / "bank")
    assert not (tmp_path / "bank").exists()


def selection_fixture(tmp_path, monkeypatch):
    from gradient_interp import update_loracle_sensitivity as worker

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
                    "prediction": inference.profile_label(profiles[i]),
                    "shuffled_profile": shuffled[j].tolist(),
                    "shuffled_prediction": inference.profile_label(shuffled[j]),
                    "direction_donor": donors[j],
                }
            )
    atomic_json(collection / "data.json", {"cases": original})
    atomic_json(
        root / "data.json",
        {"cases": cases, "calibration": inference.select_calibration({"cases": original})},
    )
    atomic_json(root / "predictions.json", rows)
    np.save(root / "raw-ce-predictions.npy", raw)
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
                "src/gradient_interp/update_loracle_sensitivity.py": file_hash(worker.__file__)
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
            "gradient_banks": 256,
            "all_native_contractions_verified": True,
            "reserved_cases_accessed": 0,
            "derivative_recomputations": [{"passed": True}, {"passed": True}],
            "files_sha256": {p.name: file_hash(p) for p in root.iterdir() if p.is_file()},
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
