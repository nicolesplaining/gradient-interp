"""Complete-epoch scoring, unchanged success gates and metadata-only release barriers."""

import pytest

from gradient_interp import update_loracle_contextual_analysis as a
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE


@pytest.mark.parametrize(
    "mutation", ["active", "controller_alive", "worker_alive", "retry", "fail"]
)
def test_audit_and_analysis_do_not_read_results_before_release(tmp_path, monkeypatch, mutation):
    c = {
        "status": "complete",
        "controller_pid": 100,
        "worker_pid": None,
        "attempts": [{"pid": 101, "returncode": 0}],
    }
    live = set()
    if mutation == "active":
        c["status"] = "running"
    elif mutation == "controller_alive":
        live.add(100)
    elif mutation == "worker_alive":
        live.add(101)
    elif mutation == "retry":
        c["attempts"] *= 2
    else:
        c["attempts"][0]["returncode"] = 1
    monkeypatch.setattr(a, "alive", lambda pid: pid in live)

    def read(path):
        assert path.name == "controller.json", "Scientific output opened before release"
        return c

    monkeypatch.setattr(a, "read", read)
    with pytest.raises(ValueError):
        a.audit(tmp_path)
    with pytest.raises(ValueError):
        a.analyze(tmp_path, tmp_path / "output")


def test_analysis_uses_new_matched_baseline_and_retains_all_epochs_and_controls(tmp_path):
    root, collection, output = tmp_path / "run", tmp_path / "collection", tmp_path / "report"
    root.mkdir()
    (collection / "cases").mkdir(parents=True)
    labels = [label for label in LABELS if label != NO_CHANGE]
    cases = []
    for i in range(72):
        case = {
            "id": str(i),
            "split": "development",
            "control": i >= 64,
            "source_pattern": str(i // 8),
            "initialization_seed": (i % 8) // 2,
            "probes": [{"prompt": f"Synthetic query {i}/{j}"} for j in range(4)],
        }
        cases.append(case)
        atomic_json(
            collection / "cases" / f"{i}.json",
            {
                "physical_update_norm": 0.0 if i >= 64 else 1.0,
                "effect": {
                    "label": NO_CHANGE if i >= 64 else labels[i % 32],
                    "below_effect_floor": i >= 64,
                },
            },
        )
    atomic_json(collection / "data.json", {"cases": cases})
    atomic_json(collection / "integrity.json", {"passed": True})
    # Dead negative PIDs are metadata-only fixture identities.
    atomic_json(
        root / "controller.json",
        {
            "status": "complete",
            "controller_pid": -1,
            "worker_pid": None,
            "attempts": [{"pid": -2, "returncode": 0}],
        },
    )
    atomic_json(
        root / "integrity.json",
        {
            "passed": True,
            "recipe": "contextual_factorized_v7",
            "files_sha256": {},
            "collection_integrity_sha256": file_hash(collection / "integrity.json"),
        },
    )
    atomic_json(root / "plan.json", {"collection_root": str(collection)})

    def predictions(n):
        return [
            {
                "id": str(i),
                "prediction": NO_CHANGE
                if i >= 64
                else labels[i % 32]
                if i < n
                else labels[(i + 1) % 32],
            }
            for i in range(72)
        ]

    for name, n in [("released", 16), ("released-shuffled", 8)]:
        atomic_json(root / f"{name}-development.json", predictions(n))
    for epoch, n in [(1, 16), (2, 48), (3, 48)]:
        directory = root / "epochs" / str(epoch)
        directory.mkdir(parents=True)
        atomic_json(directory / "development.json", predictions(n))
        atomic_json(directory / "shuffled-development.json", predictions(8))
    atomic_json(
        root / "simple-references.json",
        {
            "development": [
                {"id": str(i), "ridge": NO_CHANGE, "shuffled_ridge": NO_CHANGE} for i in range(72)
            ]
        },
    )
    atomic_json(root / "training-majorities.json", {"flat": labels[0], "factorized": labels[1]})
    a.analyze(root, output)
    result = a.read(output / "summary.json")
    assert result["selected_epoch"] == 2  # Earliest complete-accuracy tie.
    assert result["paired_differences"]["released"]["point"] == 0.5
    assert result["metrics"]["released"]["actual_update_correct"] == 16
    assert result["metrics"]["epoch3"]["actual_update_correct"] == 48
    assert result["metrics"]["epoch2"]["zero_correct"] == 8
    assert result["all_development_gates_pass"]
    assert result["reserved_test_read"] is False
    assert len(result["metrics"]) == 12
    records = a.read(output / "all-development-cases.json")
    assert len(records) == 72
    assert all(len(row["questions"]) == 4 for row in records)
    assert all(
        "Evaluation prompts as a JSON list:" in row["questions"][0]["text"] for row in records
    )
    assert (output / "development-accuracy.png").is_file()
    assert (output / "executed-analysis.py").is_file()
    with pytest.raises(ValueError, match="Preserve"):
        a.analyze(root, output)
