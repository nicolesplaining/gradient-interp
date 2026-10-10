import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import load_data, write_jsonl
from .forecast import AtomForecaster, prediction_metrics
from .models import LanguageModel

REFERENCES = {
    "gradient_atoms_paper": "https://arxiv.org/abs/2603.14665",
    "gradient_atoms_code": "https://github.com/jrosseruk/gradient_atoms/tree/"
    "a83858557b411249097861f25d57eb13020eb334",
    "dictionary_estimator": "https://scikit-learn.org/stable/modules/generated/"
    "sklearn.decomposition.MiniBatchDictionaryLearning.html",
    "online_dictionary_learning": "https://www.di.ens.fr/~fbach/mairal_icml09.pdf",
    "ksvd": "https://doi.org/10.1109/TSP.2006.881199",
    "tracin": "https://arxiv.org/abs/2002.08484",
    "lora": "https://arxiv.org/abs/2106.09685",
    "loracles_code": "https://github.com/ceselder/loracles/tree/"
    "be71fa947a037f90d3d179a16cbd2ebc5d273a66",
}


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _code_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _atom_records(forecaster, gradients, examples, probes, top_k):
    codes = forecaster.dictionary.encode(forecaster.projection.encode(gradients))
    records = []
    for j in range(len(forecaster.directions)):
        active = torch.where(codes[:, j].abs() > 1e-8)[0]
        order = active[torch.argsort(codes[active, j].abs(), descending=True)[:top_k]]
        raw = gradients[order]
        raw = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-20)
        cosine = raw @ raw.T
        coherence = (
            float(cosine[~torch.eye(len(order), dtype=torch.bool)].mean())
            if len(order) > 1
            else None
        )
        records.append(
            {
                "atom": j,
                "active_examples": len(active),
                "raw_gradient_coherence": coherence,
                "top_examples": [
                    {**asdict(examples[i]), "coefficient": float(codes[i, j])}
                    for i in order.tolist()
                ],
                "effect_per_unit_parameter_step": {
                    probe.id: float(forecaster.effects[j, k]) for k, probe in enumerate(probes)
                },
            }
        )
    return records


def run_experiment(config, output, progress=print):
    config.validate()
    code_revision = _code_revision()
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.glob("*.py")
    }
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    fit, batches, probes = load_data(config.fit_path, config.batches_path, config.probes_path)
    probe_ids = [probe.id for probe in probes]
    target = probe_ids.index(config.target_probe) if config.target_probe else None
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", asdict(config))
    backend = LanguageModel(config.model)
    space = backend.space
    estimated_gradient_bytes = 4 * space.size * (len(fit) + len(probes))
    if estimated_gradient_bytes > config.max_gradient_memory_gb * 1024**3:
        raise ValueError(
            "Gradient matrix exceeds max_gradient_memory_gb. Use LoRA, select fewer "
            "trainable parameters/examples, or explicitly raise the limit."
        )
    original_fingerprint = space.fingerprint()
    np.save(output / "parameters.npy", space.vector().numpy())
    baseline = backend.scores(probes)
    progress(f"Collecting {len(fit)} fitting gradients over {space.size:,} parameters")
    gradients = torch.empty(len(fit), space.size)
    for i, example in enumerate(fit):
        gradients[i] = backend.gradient([example])
    progress("Learning dictionary on fitting examples only")
    forecaster = AtomForecaster.fit(gradients, **asdict(config.dictionary), seed=config.seed)
    progress(f"Calibrating {len(forecaster.directions)} atoms on {len(probes)} fixed probes")
    forecaster.calibrate(
        lambda delta: backend.updated_scores(delta, probes), config.calibration_radius
    )
    probe_gradients = backend.probe_gradients(probes)
    generator = torch.Generator().manual_seed(config.seed + 1)
    random_basis = torch.linalg.qr(
        torch.randn(space.size, forecaster.projection.basis.shape[1], generator=generator)
    ).Q
    write_jsonl(
        output / "atoms.jsonl",
        _atom_records(forecaster, gradients, fit, probes, config.top_examples),
    )
    np.savez_compressed(
        output / "dictionary.npz",
        **{key: value.numpy() for key, value in forecaster.state().items()},
    )
    save_json(
        output / "dictionary_meta.json",
        {
            "algorithm": forecaster.dictionary.algorithm,
            "alpha": forecaster.dictionary.alpha,
            "sparsity": forecaster.dictionary.sparsity,
            "training_reconstruction_mse": forecaster.dictionary.history,
        },
    )
    save_json(output / "parameter_layout.json", space.layout())
    save_json(output / "probes.json", [asdict(probe) for probe in probes])
    # Every held-out forecast is persisted before any held-out update is evaluated.
    progress(f"Forecasting {len(batches)} held-out batches before measuring their updates")
    forecasts = []
    cache = output / "batch_gradients"
    cache.mkdir()
    for batch_index, batch in enumerate(batches):
        gradient = backend.gradient(batch.examples)
        np.save(cache / f"{batch_index}.npy", gradient.numpy())
        for rate in config.learning_rates:
            update = -rate * gradient
            forecast = forecaster.predict(update, target)
            component = (
                forecast.amplitudes[forecast.selected_atom]
                * forecaster.directions[forecast.selected_atom]
            )
            projected = forecaster.projection.decode(forecaster.projection.encode(update))
            predictions = {
                "dictionary": forecast.predicted,
                "full_gradient": probe_gradients @ update,
                "svd": probe_gradients @ projected,
                "random_subspace": (probe_gradients @ random_basis) @ (update @ random_basis),
                "zero": torch.zeros(len(probes)),
            }
            record = {
                "batch_id": batch.id,
                "batch_index": batch_index,
                "control_seed": config.seed + 1000 + batch_index,
                "learning_rate": rate,
                "probe_ids": probe_ids,
                "predictions": {key: value.tolist() for key, value in predictions.items()},
                "coefficients": forecast.coefficients.tolist(),
                "amplitudes": forecast.amplitudes.tolist(),
                "atom_contributions": forecast.contributions.tolist(),
                "selected_atom": forecast.selected_atom,
                "selected_atom_prediction": forecast.contributions[forecast.selected_atom].tolist(),
                "update_norm": float(update.norm()),
                "removed_component_norm": float(component.norm()),
                "reconstruction_relative_error": float(
                    (update - forecast.reconstructed_update).norm() / update.norm()
                )
                if update.norm() > 0
                else 0.0,
                "projected_relative_error": float((update - projected).norm() / update.norm())
                if update.norm() > 0
                else 0.0,
            }
            forecasts.append(record)
    write_jsonl(output / "predictions.jsonl", forecasts)
    predictions_hash = hashlib.sha256((output / "predictions.jsonl").read_bytes()).hexdigest()
    results = []
    for index, forecast in enumerate(forecasts):
        progress(f"Measuring batch update {index + 1}/{len(forecasts)}")
        gradient = torch.from_numpy(np.load(cache / f"{forecast['batch_index']}.npy"))
        update = -forecast["learning_rate"] * gradient
        amplitudes = torch.tensor(forecast["amplitudes"])
        atom = forecast["selected_atom"]
        component = amplitudes[atom] * forecaster.directions[atom]
        control = torch.randn(
            space.size, generator=torch.Generator().manual_seed(forecast["control_seed"])
        )
        control = control / control.norm() * component.norm()
        interventions = {
            "full": update,
            "reconstructed": amplitudes @ forecaster.directions,
            "atom_only": component,
            "atom_removed": update - component,
            "random_removed": update - control,
            "rescaled_full": update * ((update - component).norm() / update.norm())
            if update.norm() > 0
            else update,
        }
        observed = {
            name: (backend.updated_scores(delta, probes) - baseline).tolist()
            for name, delta in interventions.items()
        }
        results.append({**forecast, "observed": observed})
    if space.fingerprint() != original_fingerprint:
        raise RuntimeError("Evaluation changed the reference checkpoint")
    write_jsonl(output / "results.jsonl", results)
    summary = {}
    for rate in config.learning_rates:
        subset = [row for row in results if row["learning_rate"] == rate]
        summary[str(rate)] = {
            method: prediction_metrics(
                [row["predictions"][method] for row in subset],
                [row["observed"]["full"] for row in subset],
            )
            for method in forecasts[0]["predictions"]
        }
        summary[str(rate)]["atom_removal"] = prediction_metrics(
            [row["selected_atom_prediction"] for row in subset],
            [
                np.subtract(row["observed"]["full"], row["observed"]["atom_removed"]).tolist()
                for row in subset
            ],
        )
    save_json(output / "summary.json", summary)
    grouped = {}
    for group in sorted({probe.group for probe in probes}):
        indices = [i for i, probe in enumerate(probes) if probe.group == group]
        grouped[group] = {}
        for rate in config.learning_rates:
            subset = [row for row in results if row["learning_rate"] == rate]
            grouped[group][str(rate)] = {
                "forecast": prediction_metrics(
                    [[row["predictions"]["dictionary"][i] for i in indices] for row in subset],
                    [[row["observed"]["full"][i] for i in indices] for row in subset],
                ),
                "mean_effect_by_intervention": {
                    intervention: float(
                        np.mean(
                            [[row["observed"][intervention][i] for i in indices] for row in subset]
                        )
                    )
                    for intervention in results[0]["observed"]
                },
            }
    save_json(output / "groups.json", grouped)
    save_json(
        output / "manifest.json",
        {
            "status": "complete",
            "seed": config.seed,
            "code_revision": code_revision,
            "source_sha256": source_hashes,
            "model": asdict(config.model),
            "resolved_model_revision": backend.resolved_revision,
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "numpy", "scikit-learn"]
            },
            "parameter_fingerprint": original_fingerprint,
            "checkpoint_restored": True,
            "trainable_parameters": space.size,
            "baseline_scores": baseline.tolist(),
            "fit_examples": len(fit),
            "held_out_batches": len(batches),
            "probe_count": len(probes),
            "probe_ids": probe_ids,
            "prediction_file_sha256_before_evaluation": predictions_hash,
            "data_sha256": {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in [
                    ("fit", config.fit_path),
                    ("batches", config.batches_path),
                    ("probes", config.probes_path),
                ]
            },
            "loss_reduction": "mean of per-example mean completion-token negative log likelihood",
            "score_units": "per-probe metric; probability with a negative is pair-normalized",
            "optimizer": "stateless SGD; no momentum/decay; one fixed checkpoint",
            "projection": "uncentered gradient SVD with optional RMS whitening, not EKFAC",
            "scope": "numerical smoke test"
            if config.model.kind == "tiny"
            else "pretrained model experiment",
            "calibration_max_half_radius_discrepancy": float(
                forecaster.calibration_discrepancy.max()
            ),
            "baseline_note": "full_gradient, svd, random_subspace use supervised probe derivatives",
            "references": REFERENCES,
        },
    )
    progress(f"Saved results to {output}")
    return summary
