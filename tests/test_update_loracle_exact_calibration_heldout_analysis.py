import copy

import numpy as np
import pytest
import torch

from gradient_interp import update_loracle_exact_calibration_heldout_analysis as analysis
from gradient_interp.loracle_trajectory_toy import fingerprint, read
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE, description, effect_label
from gradient_interp.update_loracle_exact_calibration import (
    effect_profile,
    physical_norm,
    scaled_native,
)
from gradient_interp.update_loracle_inference import COMPONENT_INTERFACE, interface_contract


def controller():
    return {
        "status": "complete",
        "controller_pid": 1,
        "worker_pid": None,
        "started_unix": 1,
        "deadline_unix": 100,
        "attempts": [
            {
                "job": name,
                "pid": i + 2,
                "started_unix": 2 + 2 * i,
                "finished_unix": 3 + 2 * i,
                "returncode": 0,
            }
            for i, name in enumerate(("preflight", "updates", "predict", "measure"))
        ],
    }


@pytest.mark.parametrize(
    "defect", ["running", "failed", "alive", "partial", "retry", "late", "ordering"]
)
def test_barrier_rejects_before_any_scientific_read(tmp_path, monkeypatch, defect):
    value = controller()
    monkeypatch.setattr(analysis, "alive", lambda pid: False)
    if defect in ("running", "failed"):
        value["status"] = defect
    elif defect == "alive":
        monkeypatch.setattr(analysis, "alive", lambda pid: pid == 5)
    elif defect == "partial":
        value["attempts"].pop()
    elif defect == "retry":
        value["attempts"].append(copy.deepcopy(value["attempts"][-1]))
    elif defect == "late":
        value["attempts"][-1]["finished_unix"] = 101
    else:
        value["attempts"][-1]["started_unix"] = 2
    atomic_json(tmp_path / "controller.json", value)
    for operation in (
        lambda: analysis.audit(tmp_path),
        lambda: analysis.analyze(tmp_path, tmp_path / "output"),
    ):
        with pytest.raises(ValueError):
            operation()


def scored_record(before, after):
    result = effect_profile(before, after)
    return {
        **result,
        "before_ce": before.tolist(),
        "after_ce": after.tolist(),
        "description": description(result["prediction"]),
        "bit_exact_restore": True,
    }


def test_independent_profile_preserves_negative_changes_and_stable_ties():
    before = np.full((8, 32), 2.0)
    after = before + np.arange(8)[:, None] / 100
    after[:, 3] -= 0.2
    after[:, 8] -= 0.2
    record = scored_record(before, after)
    actual = analysis.checked_exact_profile(record, before)
    assert record["prediction"] == LABELS[3]
    assert record["ce_change"][3] < 0
    np.testing.assert_allclose(actual, -np.mean(after - before, axis=0) + np.mean(after - before))
    weak = scored_record(before, before + np.linspace(0, 0.0199, 32))
    assert weak["prediction"] == NO_CHANGE
    analysis.checked_exact_profile(weak, before)


@pytest.mark.parametrize("defect", ["baseline", "clip", "category", "profile", "finite", "restore"])
def test_profile_audit_rejects_changed_scientific_values(defect):
    before = np.ones((8, 32))
    after = before.copy()
    after[:, 5] -= 0.3
    record = scored_record(before, after)
    if defect == "baseline":
        record["before_ce"][0][0] += 0.01
    elif defect == "clip":
        record["ce_change"][5] = 0.0
    elif defect == "category":
        record["prediction"] = NO_CHANGE
    elif defect == "profile":
        record["profile"] = [-v for v in record["profile"]]
    elif defect == "finite":
        record["after_ce"][0][0] = float("nan")
    else:
        record["bit_exact_restore"] = False
    with pytest.raises((ValueError, AssertionError)):
        analysis.checked_exact_profile(record, before)


def native_fixture():
    initial = {"m": (torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]), torch.zeros(3, 2))}
    native = {"m": (initial["m"][0], torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.1, -0.1]]))}
    donor = {
        "id": "donor",
        "initialization_seed": 3,
        "snapshot_sha256": "original-file",
        "physical_update_norm": physical_norm(native),
    }
    recipient = {"id": "recipient", "physical_update_norm": donor["physical_update_norm"] * 1.75}
    scale = recipient["physical_update_norm"] / donor["physical_update_norm"]
    scaled = scaled_native(initial, native, scale)
    record = {
        "id": "recipient",
        "kind": "shuffled",
        "direction_donor": "donor",
        "initialization_seed": 3,
        "donor_scale": scale,
        "physical_update_norm": recipient["physical_update_norm"],
        "actual_scaled_norm": physical_norm(scaled),
        "snapshot_sha256": "original-file",
        "initial_native_sha256": fingerprint(initial),
        "updated_native_sha256": fingerprint(scaled),
        "finished_unix": 3,
    }
    return record, recipient, donor, initial, native


def test_native_shuffle_preserves_donor_A_and_matches_recipient_physical_norm():
    args = native_fixture()
    assert analysis.check_native_score(*args, "shuffled", (2, 4)) == pytest.approx(1.75)


@pytest.mark.parametrize("defect", ["own_A", "scale", "fingerprint", "norm", "late", "donor"])
def test_native_shuffle_detects_wrong_direction_or_lock_timing(defect):
    record, recipient, donor, initial, native = native_fixture()
    if defect == "own_A":
        initial["m"] = (initial["m"][0] + 1, initial["m"][1])
    elif defect == "scale":
        record["donor_scale"] = 1.0
    elif defect == "fingerprint":
        record["updated_native_sha256"] = fingerprint(native)
    elif defect == "norm":
        record["actual_scaled_norm"] *= 2
    elif defect == "late":
        record["finished_unix"] = 4
    else:
        record["direction_donor"] = "recipient"
    with pytest.raises(ValueError):
        analysis.check_native_score(record, recipient, donor, initial, native, "shuffled", (2, 4))


def test_original_statistics_keep_weak_updates_and_require_both_majorities():
    cases = [
        {
            "id": f"{p}-{s}-{i}",
            "control": False,
            "source_pattern": f"pattern-{p}",
            "initialization_seed": s,
        }
        for p in range(8)
        for s in range(4)
        for i in range(4)
    ]
    cases += [{"id": f"zero-{i}", "control": True} for i in range(8)]
    targets = [NO_CHANGE] * 64 + [LABELS[0]] * 64 + [NO_CHANGE] * 8
    predictions = {
        name: [LABELS[0]] * 128 + [NO_CHANGE] * 8
        for name in (
            "functional",
            "released",
            "functional_shuffled",
            "majority",
            "ridge",
            "factorized_majority",
        )
    }
    result, draws = analysis.evaluate(
        cases, targets, predictions, [0.0] * 64 + [0.2] * 64 + [0.0] * 8, 1000
    )
    assert result["n_actual_updates"] == 128
    assert result["accuracy"]["functional"] == 0.5
    assert result["measurable_effect_fraction"] == 0.5
    assert not result["all_numerical_gates_pass"]
    assert result["paired_exact_reference"]["assignments"] == 256
    assert "adapted" not in result["accuracy"] and "functional" in draws
    np.testing.assert_array_equal(draws["functional"], draws["ridge"])
    del predictions["factorized_majority"]
    with pytest.raises(ValueError, match="comparator"):
        analysis.evaluate(cases, targets, predictions, [0.0] * 64 + [0.2] * 64 + [0.0] * 8, 1000)


def synthetic_analysis_root(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "measurements").mkdir()
    (root / "predictions").mkdir()
    atomic_json(root / "controller.json", controller())
    atomic_json(root / "selection.json", {"synthetic_only": True})
    data = {}
    for split in analysis.SPLITS:
        cases = [
            {
                "id": f"{split}-{p}-{s}-{i}",
                "control": False,
                "source_pattern": f"pattern-{p}",
                "initialization_seed": s,
                "batch": [{"prompt": "synthetic train", "answer": "a"}],
                "probes": [{"prompt": "synthetic probe"}],
            }
            for p in range(8)
            for s in range(4)
            for i in range(4)
        ]
        cases += [
            {
                "id": f"{split}-zero-{i}",
                "control": True,
                "source_pattern": None,
                "initialization_seed": i % 4,
                "batch": [],
                "probes": [{"prompt": "synthetic control"}],
            }
            for i in range(8)
        ]
        data[split] = {"cases": cases}
        methods = {
            name: []
            for name in ("functional", "functional-shuffled", "released", "released-shuffled")
        }
        simple = []
        for i, case in enumerate(cases):
            before = np.ones((4, 32))
            after = before.copy()
            if not case["control"] and not (split == "confirmation" and i < 64):
                after[:, 0] -= 0.2
            effect = effect_label(before, after)
            atomic_json(
                root / "measurements" / f"{case['id']}.json",
                {
                    "id": case["id"],
                    "before_ce": before.tolist(),
                    "after_ce": after.tolist(),
                    "effect": effect,
                },
            )
            for name in methods:
                label = (
                    NO_CHANGE
                    if case["control"]
                    else LABELS[0]
                    if name == "functional"
                    else LABELS[1]
                )
                methods[name].append(
                    {
                        "id": case["id"],
                        "prediction": label,
                        "profile": effect["centered_improvements"]
                        if name == "functional"
                        else [0.0] * 32,
                    }
                )
            simple.append(
                {
                    "id": case["id"],
                    **{
                        n: LABELS[1]
                        for n in ("majority", "factorized_majority", "ridge", "shuffled_ridge")
                    },
                }
            )
        for name, rows in methods.items():
            atomic_json(root / "predictions" / f"{split}-{name}.json", rows)
        atomic_json(root / "predictions" / f"{split}-simple.json", {"development": simple})
    atomic_json(root / "data.json", data)
    files = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*.json")}
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "recipe": analysis.RECIPE,
            "scope": "reserved_and_fresh_confirmation",
            "cases": 272,
            "actual_updates": 256,
            "zero_controls": 16,
            "exact_calibration_conditions": 544,
            "fixed_scoring_rechecks": [{"passed": True}] * 4,
            "question_interface": interface_contract(COMPONENT_INTERFACE),
            "files_sha256": files,
        },
    )


def test_full_analysis_requires_both_grids_keeps_all_cases_and_methods(tmp_path, monkeypatch):
    root, output = tmp_path / "run", tmp_path / "report"
    synthetic_analysis_root(root)
    monkeypatch.setattr(analysis, "alive", lambda pid: False)
    analysis.analyze(root, output)
    summary, cases = read(output / "summary.json"), read(output / "all-cases.json")
    assert summary["splits"]["reserved"]["all_numerical_gates_pass"]
    assert not summary["splits"]["confirmation"]["all_numerical_gates_pass"]
    assert not summary["both_grids_pass"]
    assert {k: len(v) for k, v in cases.items()} == {"reserved": 136, "confirmation": 136}
    assert summary["splits"]["confirmation"]["accuracy"]["functional"] == 0.5
    assert len(cases["reserved"][0]["predictions"]) == 8
    assert (output / "executed-analysis.py").exists()
    assert (output / "report.html").exists()
    samples = np.load(root / "confirmation-bootstrap.npz")
    assert samples["reserved_functional"].shape == (4000,)
    assert samples["confirmation_functional"].shape == (4000,)
    with pytest.raises(ValueError, match="Preserve"):
        analysis.analyze(root, output)


def test_analysis_rejects_changed_audited_output(tmp_path, monkeypatch):
    synthetic_analysis_root(tmp_path)
    monkeypatch.setattr(analysis, "alive", lambda pid: False)
    path = tmp_path / "predictions" / "reserved-functional.json"
    rows = read(path)
    rows[0]["prediction"] = NO_CHANGE
    atomic_json(path, rows)
    with pytest.raises(ValueError, match="Audited"):
        analysis.analyze(tmp_path, tmp_path / "report")


def test_complete_calibration_audit_checks_all544_conditions_and_fixed_four(tmp_path, monkeypatch):
    root, old = tmp_path / "test", tmp_path / "development"
    for path in (
        root / "snapshots",
        root / "updates",
        root / "calibration",
        root / "predictions",
        old,
    ):
        path.mkdir(parents=True)
    baseline = np.ones((8, 32))
    atomic_json(old / "baselines.json", {"scores": {str(i): baseline.tolist() for i in range(8)}})
    atomic_json(
        old / "integrity.json",
        {"passed": True, "files_sha256": {"baselines.json": file_hash(old / "baselines.json")}},
    )
    initial, cases, lookup, metadata = {}, [], {}, {}
    for split_no, split in enumerate(analysis.SPLITS):
        for i in range(136):
            seed = 10 + split_no * 4 + i % 4
            initial[seed] = {"m": (torch.tensor([[1.0, seed / 10.0]]), torch.zeros(2, 1))}
            control = i >= 128
            case = {
                "id": f"{split}-{i}",
                "split": split,
                "control": control,
                "initialization_seed": seed,
            }
            cases.append(case)
            native = {
                "m": (
                    initial[seed]["m"][0],
                    torch.zeros(2, 1) if control else torch.tensor([[0.01 * (i + 1)], [0.02]]),
                )
            }
            lookup[case["id"]] = native
            path = root / "snapshots" / f"{case['id']}.pt"
            torch.save(native, path)
            row = {
                **case,
                "snapshot_sha256": file_hash(path),
                "physical_update_norm": physical_norm(native),
            }
            atomic_json(root / "updates" / f"{case['id']}.json", row)
            metadata[case["id"]] = row
    donors = analysis.donor_map(cases)
    for split in analysis.SPLITS:
        for method, kind in (("functional", "own"), ("functional-shuffled", "shuffled")):
            rows = []
            for case in [c for c in cases if c["split"] == split]:
                recipient = metadata[case["id"]]
                donor = metadata[case["id"] if kind == "own" else donors[case["id"]]]
                scale = (
                    recipient["physical_update_norm"] / donor["physical_update_norm"]
                    if donor["physical_update_norm"]
                    else 1.0
                )
                native = scaled_native(
                    initial[donor["initialization_seed"]], lookup[donor["id"]], scale
                )
                after = baseline.copy()
                after[:, 0] -= physical_norm(native) ** 2 / 100
                record = {
                    **scored_record(baseline, after),
                    "id": case["id"],
                    "kind": kind,
                    "direction_donor": donor["id"],
                    "initialization_seed": donor["initialization_seed"],
                    "donor_scale": scale,
                    "physical_update_norm": recipient["physical_update_norm"],
                    "actual_scaled_norm": physical_norm(native),
                    "snapshot_sha256": donor["snapshot_sha256"],
                    "initial_native_sha256": fingerprint(initial[donor["initialization_seed"]]),
                    "updated_native_sha256": fingerprint(native),
                    "finished_unix": 3,
                }
                atomic_json(root / "calibration" / f"{case['id']}-{kind}.json", record)
                rows.append(record)
            atomic_json(root / "predictions" / f"{split}-{method}.json", rows)
    calls = []
    original = analysis.checked_exact_profile

    def checked(*args):
        calls.append(args[0]["id"])
        return original(*args)

    monkeypatch.setattr(analysis, "checked_exact_profile", checked)

    def recheck(root, selected, initials, requests):
        assert {(r[0], r[1]) for r in requests} == {
            (f"{s}-0", kind) for s in analysis.SPLITS for kind in ("own", "shuffled")
        }
        return [{"id": r[0], "kind": r[1], "passed": True} for r in requests]

    monkeypatch.setattr(analysis, "fixed_rescore", recheck)
    checked_rows = analysis.audit_functional_scores(
        root,
        cases,
        initial,
        {"development_root": str(old)},
        {"finished_unix": 2},
        {"finished_unix": 4},
    )
    assert len(checked_rows) == 4 and len(calls) == 544
    path = root / "calibration" / "reserved-0-shuffled.json"
    record = read(path)
    record["updated_native_sha256"] = "wrong-direction"
    atomic_json(path, record)
    path = root / "predictions" / "reserved-functional-shuffled.json"
    rows = read(path)
    rows[0] = record
    atomic_json(path, rows)
    with pytest.raises(ValueError, match="direction"):
        analysis.audit_functional_scores(
            root,
            cases,
            initial,
            {"development_root": str(old)},
            {"finished_unix": 2},
            {"finished_unix": 4},
        )


def test_fixed_rescore_compares_fresh_baseline_and_exact_after(tmp_path, monkeypatch):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "calibration").mkdir()
    initial = {"m": (torch.ones(1, 2), torch.zeros(2, 1))}
    native = {"m": (initial["m"][0], torch.ones(2, 1))}
    torch.save(native, tmp_path / "snapshots" / "donor.pt")
    donor = {"id": "donor", "initialization_seed": 3}
    before, after = np.ones((8, 32)), np.ones((8, 32)) * 2
    requests = [(f"case-{i}", "own" if i % 2 == 0 else "shuffled", donor, 1.0) for i in range(4)]
    for case, kind, _, _ in requests:
        atomic_json(
            tmp_path / "calibration" / f"{case}-{kind}.json",
            {"before_ce": before.tolist(), "after_ce": after.tolist()},
        )
    monkeypatch.setattr(analysis, "prepare_model", lambda _: ({}, object(), object()))
    monkeypatch.setattr(analysis, "restore_modules", lambda *args: None)
    monkeypatch.setattr(analysis, "calibration_scores", lambda *args: before.copy())
    monkeypatch.setattr(analysis, "score_and_restore", lambda *args: after.copy())
    assert len(analysis.fixed_rescore(tmp_path, {"calibration": []}, {3: initial}, requests)) == 4
    monkeypatch.setattr(analysis, "score_and_restore", lambda *args: after + 0.001)
    with pytest.raises(ValueError, match="exact scoring recheck"):
        analysis.fixed_rescore(tmp_path, {"calibration": []}, {3: initial}, requests)
    monkeypatch.setattr(analysis, "calibration_scores", lambda *args: before + 0.001)
    with pytest.raises(ValueError, match="baseline recheck"):
        analysis.fixed_rescore(tmp_path, {"calibration": []}, {3: initial}, requests)
