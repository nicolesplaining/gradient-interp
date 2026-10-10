"""Use a saved calibration to forecast new batches without evaluating their updates."""

import json
from pathlib import Path

import numpy as np
import torch

from .data import Example, read_jsonl, write_jsonl
from .forecast import AtomForecaster
from .models import LanguageModel, ModelConfig


def predict_batches(run_directory, batches_path, output, learning_rate=None, device=None):
    directory, output = Path(run_directory), Path(output)
    if output.exists():
        raise ValueError(f"Prediction output already exists: {output}")
    manifest = json.loads((directory / "manifest.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("A completed calibration run is required")
    model_config = ModelConfig(**config["model"])
    if manifest.get("resolved_model_revision"):
        model_config.revision = manifest["resolved_model_revision"]
    if device:
        model_config.device = device
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    backend = LanguageModel(model_config)
    layout = json.loads((directory / "parameter_layout.json").read_text())
    if backend.space.layout() != layout:
        raise ValueError("Model parameter layout differs from the calibrated checkpoint")
    backend.space.assign(
        torch.from_numpy(np.load(directory / "parameters.npy", allow_pickle=False))
    )
    if backend.space.fingerprint() != manifest["parameter_fingerprint"]:
        raise ValueError("Saved checkpoint fingerprint does not match the calibration")
    forecaster = AtomForecaster.load(directory)
    rates = [learning_rate] if learning_rate is not None else config["learning_rates"]
    if any(not np.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Learning rates must be finite and positive")
    results = []
    for batch in read_jsonl(batches_path):
        examples = [Example(**example) for example in batch["examples"]]
        if not examples:
            raise ValueError("Cannot predict an empty batch")
        gradient = backend.gradient(examples)
        for rate in rates:
            forecast = forecaster.predict(-rate * gradient)
            results.append(
                {
                    "batch_id": batch["id"],
                    "learning_rate": rate,
                    "probe_ids": manifest["probe_ids"],
                    "prediction": forecast.predicted.tolist(),
                    "coefficients": forecast.coefficients.tolist(),
                    "atom_contributions": forecast.contributions.tolist(),
                    "update_evaluated": False,
                    "parameter_fingerprint": manifest["parameter_fingerprint"],
                }
            )
    if backend.space.fingerprint() != manifest["parameter_fingerprint"]:
        raise RuntimeError("Prediction unexpectedly changed model weights")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, results)
    return results
