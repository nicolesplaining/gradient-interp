import hashlib
import json
from dataclasses import asdict

import pytest

from gradient_interp.benchmark import partition_examples, prompt_key, validate_partitions
from gradient_interp.data import Example, make_demo_data, read_jsonl, write_jsonl
from gradient_interp.models import ModelConfig
from gradient_interp.study import run_study_job
from gradient_interp.suite import crossed_bootstrap


def test_benchmark_is_source_disjoint_and_reserves_task_families():
    train, test = [], []
    for group in ("Coding", "Classify", "Generation", "Extract"):
        train.extend(
            Example(f"{group}-{i}", f"Prompt {group} {i}", "answer", group) for i in range(300)
        )
        test.extend(
            Example(f"probe-{group}-{i}", f"Probe {group} {i}", "answer", group) for i in range(20)
        )
    partitions = partition_examples(train, test, fit_count=128, seed=42)
    counts = validate_partitions(*partitions)
    assert counts["fit"] == 128
    assert not {"Coding", "Classify"} & {e.group for e in partitions[0]}
    assert {b["batch_size"] for b in partitions[2]} == {1, 4, 16}
    assert {b["family_split"] for b in partitions[2]} == {"seen", "unseen_family", "mixed"}
    assert prompt_key("A   question") == prompt_key("a question")
    copied = partitions[0] + [partitions[3][0]]
    with pytest.raises(ValueError, match="overlap"):
        validate_partitions(copied, *partitions[1:])


def test_study_locks_forecasts_and_resumes_completed_job(tmp_path):
    data = tmp_path / "data"
    make_demo_data(data)
    probes = read_jsonl(data / "probes.jsonl")
    write_jsonl(data / "calibration_probes.jsonl", probes)
    evaluation = [{**p, "id": "test-" + p["id"], "prompt": "New " + p["prompt"]} for p in probes]
    write_jsonl(data / "evaluation_probes.jsonl", evaluation)
    batches = read_jsonl(data / "batches.jsonl")[:2]
    for batch in batches:
        batch.update(
            target_group="identity",
            family_split="seen",
            kind="mixed",
            batch_size=len(batch["examples"]),
        )
    write_jsonl(data / "batches.jsonl", batches)
    write_jsonl(data / "dev_batches.jsonl", batches)
    (data / "manifest.json").write_text('{"test_fixture": true}')
    plan = {
        "benchmark": str(data),
        "threads": 1,
        "rank": 8,
        "dictionary_sizes": [8],
        "alphas": [0.1],
        "dictionary_iterations": 5,
        "finite_difference_atoms": 2,
        "calibration_radius": 0.001,
        "learning_rates": [0.001, 0.01],
        "generation_batches": 0,
    }
    job = {"seed": 42, "warmup_steps": 0, "model": asdict(ModelConfig(width=8, score_batch_size=4))}
    output = tmp_path / "study"

    def progress(message):
        if message.startswith("measuring intervention"):
            predictions = read_jsonl(output / "predictions.jsonl")
            assert len(predictions) == 4
            assert all("observed" not in row for row in predictions)
            lock = json.loads((output / "forecast_lock.json").read_text())
            assert (
                lock["sha256"]
                == hashlib.sha256((output / "predictions.jsonl").read_bytes()).hexdigest()
            )

    run_study_job(plan, job, output, progress)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["checkpoint_restored"]
    assert len(read_jsonl(output / "results.jsonl")) == 4
    # A second invocation does not recompute or overwrite completed outcomes.
    before = (output / "results.jsonl").stat().st_mtime_ns
    run_study_job(plan, job, output, lambda _: None)
    assert before == (output / "results.jsonl").stat().st_mtime_ns
    with pytest.raises(ValueError, match="exactly the same"):
        run_study_job({**plan, "rank": 4}, job, output)


def test_crossed_bootstrap_preserves_paired_baseline_comparison():
    seeds = []
    for seed in range(3):
        seeds.append(
            [
                {
                    "batch_id": str(batch),
                    "predictions": {
                        "dictionary": [float(batch + seed)],
                        "full_gradient": [float(batch + seed)],
                    },
                    "observed": {"full": [float(batch + seed + 1)]},
                }
                for batch in range(5)
            ]
        )
    result = crossed_bootstrap(seeds, "dictionary", repeats=100)
    assert result["error_ratio_vs_full_gradient"] == 1.0
    assert result["ratio_95ci"] == [1.0, 1.0]
    assert result["seed_count"] == 3


def test_loracle_category_parser_rejects_ambiguous_answers():
    from gradient_interp.loracle_study import parse_category

    categories = ["Open QA", "Closed QA", "Coding"]
    assert parse_category("Coding", categories) == "Coding"
    assert parse_category("The category is Open QA.", categories) == "Open QA"
    assert parse_category("Coding or Open QA", categories) is None
