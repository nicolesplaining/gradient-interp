import hashlib
import json

import numpy as np
import pytest

from gradient_interp.config import DictionaryConfig, ExperimentConfig
from gradient_interp.data import load_data, make_demo_data, read_jsonl, write_jsonl
from gradient_interp.experiment import run_experiment
from gradient_interp.models import ModelConfig
from gradient_interp.predict import predict_batches


def test_end_to_end_forecasts_precede_outcomes(tmp_path):
    data = tmp_path / "data"
    make_demo_data(data)
    fit = read_jsonl(data / "fit.jsonl")[:9]
    batches = read_jsonl(data / "batches.jsonl")[:2]
    probes = read_jsonl(data / "probes.jsonl")[:2]
    write_jsonl(data / "fit.jsonl", fit)
    write_jsonl(data / "batches.jsonl", batches)
    write_jsonl(data / "probes.jsonl", probes)
    config = ExperimentConfig(
        fit_path=str(data / "fit.jsonl"),
        batches_path=str(data / "batches.jsonl"),
        probes_path=str(data / "probes.jsonl"),
        threads=1,
        learning_rates=[0.001, 0.01],
        model=ModelConfig(width=8),
        dictionary=DictionaryConfig(rank=4, n_atoms=4, sparsity=2, iterations=5),
    )
    output = tmp_path / "run"
    stages = []

    def progress(message):
        stages.append(message)
        if message.startswith("Measuring"):
            predictions = read_jsonl(output / "predictions.jsonl")
            assert len(predictions) == 4
            assert all("observed" not in row for row in predictions)
            assert not (output / "results.jsonl").exists()

    summary = run_experiment(config, output, progress=progress)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["checkpoint_restored"]
    assert manifest["scope"] == "numerical smoke test"
    assert (
        manifest["prediction_file_sha256_before_evaluation"]
        == hashlib.sha256((output / "predictions.jsonl").read_bytes()).hexdigest()
    )
    results = read_jsonl(output / "results.jsonl")
    assert len(results) == 4
    assert set(results[0]["observed"]) == {
        "full",
        "reconstructed",
        "atom_only",
        "atom_removed",
        "random_removed",
    }
    assert summary["0.001"]["full_gradient"]["rmse"] < summary["0.001"]["zero"]["rmse"]
    assert np.isfinite(np.load(output / "dictionary.npz")["atoms"]).all()
    predictions = predict_batches(output, data / "batches.jsonl", tmp_path / "new.jsonl")
    assert all(not row["update_evaluated"] for row in predictions)
    for forecast, result in zip(predictions, results, strict=True):
        np.testing.assert_allclose(
            forecast["prediction"], result["predictions"]["dictionary"], atol=1e-7, rtol=1e-5
        )
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(config, output)


def test_rejects_dictionary_evaluation_leakage(tmp_path):
    make_demo_data(tmp_path)
    fitting_example = read_jsonl(tmp_path / "fit.jsonl")[0]
    write_jsonl(tmp_path / "batches.jsonl", [{"id": "leaked", "examples": [fitting_example]}])
    with pytest.raises(ValueError, match="overlap"):
        load_data(tmp_path / "fit.jsonl", tmp_path / "batches.jsonl", tmp_path / "probes.jsonl")
