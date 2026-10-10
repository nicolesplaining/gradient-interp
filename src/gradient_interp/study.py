"""Resumable large-data forecasting and causal intervention experiment.

The dictionary follows Gradient Atoms' normalized sklearn estimator, using an
explicit SVD projection rather than EKFAC. Analytic atom calibration uses the
same local derivative as the full-gradient baseline. Independent finite steps
check selected atom derivatives; all held-out forecasts precede interventions.
See REFERENCES in experiment.py for papers and exact upstream revisions.
"""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import Example, Probe, read_jsonl, write_jsonl
from .dictionary import Projection, SparseDictionary
from .experiment import REFERENCES, _atom_records, _code_revision
from .forecast import AtomForecaster, prediction_metrics
from .models import LanguageModel, ModelConfig


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def atomic_array(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def atomic_archive(path, **values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    temporary.replace(path)


def group_matrix(probes):
    groups = sorted({probe.group for probe in probes})
    matrix = torch.zeros(len(groups), len(probes))
    for i, group in enumerate(groups):
        indices = [j for j, probe in enumerate(probes) if probe.group == group]
        matrix[i, indices] = 1 / len(indices)
    return groups, matrix


def cached_gradients(backend, examples, directory, progress):
    """Flush progress in blocks; an interrupted block is safely recomputed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path, cursor_path = directory / "gradients.npy", directory / "cursor.json"
    done = json.loads(cursor_path.read_text())["completed"] if cursor_path.exists() else 0
    if done and not path.exists():
        raise ValueError("Gradient progress exists without its cached matrix")
    if path.exists():
        matrix = np.load(path, mmap_mode="r+")
        if matrix.shape != (len(examples), backend.space.size):
            raise ValueError("Cached gradient dimensions changed")
    else:
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=(len(examples), backend.space.size)
        )
    for i in range(done, len(examples)):
        matrix[i] = backend.gradient([examples[i]]).numpy()
        if (i + 1) % 25 == 0 or i + 1 == len(examples):
            matrix.flush()
            atomic_json(cursor_path, {"completed": i + 1, "total": len(examples)})
            progress(f"fitting gradients {i + 1}/{len(examples)}")
    return torch.from_numpy(matrix)


def _load_or_collect_probe_gradients(backend, probes, path):
    if path.exists():
        return torch.from_numpy(np.load(path, allow_pickle=False))
    gradients = backend.probe_gradients(probes)
    atomic_array(path, gradients.numpy())
    return gradients


def _load_or_fit_dictionary(values, path, atoms, alpha, iterations, seed):
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            return SparseDictionary(
                torch.from_numpy(archive["atoms"].copy()),
                8,
                archive["history"].tolist(),
                "minibatch",
                alpha,
            )
    dictionary = SparseDictionary.fit(values, atoms, 8, iterations, seed, "minibatch", alpha)
    atomic_archive(path, atoms=dictionary.atoms.numpy(), history=dictionary.history)
    return dictionary


def warm_checkpoint(backend, fit, steps, path, seed, progress):
    if path.exists():
        backend.space.assign(torch.from_numpy(np.load(path, allow_pickle=False)))
        return
    if steps:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(fit))
        optimizer = torch.optim.AdamW(
            [p for _, p in backend.space.named], lr=1e-4, weight_decay=0.0
        )
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            chosen = [fit[int(indices[(step * 4 + j) % len(indices)])] for j in range(4)]
            for example in chosen:
                (backend.loss(example) / len(chosen)).backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in backend.space.named], 1.0)
            optimizer.step()
            if (step + 1) % 16 == 0:
                progress(f"warmup {step + 1}/{steps}")
        backend.model.zero_grad(set_to_none=True)
    atomic_array(path, backend.space.vector().numpy())


def bootstrap_errors(records, method, repeats=1000, seed=42):
    """Resample whole disjoint-source batches, never individual probe coordinates."""
    predicted = np.array([row["predictions"][method] for row in records], dtype=float)
    observed = np.array([row["observed"]["full"] for row in records], dtype=float)
    error = ((predicted - observed) ** 2).sum(1)
    signal = (observed**2).sum(1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(records), size=(repeats, len(records)))
    values = np.sqrt(error[indices].sum(1) / np.maximum(signal[indices].sum(1), 1e-30))
    return np.quantile(values, [0.025, 0.975]).tolist()


def summarize_study(records, groups, seed):
    summary = {}
    for rate in sorted({row["learning_rate"] for row in records}):
        subset = [row for row in records if row["learning_rate"] == rate]
        methods = {}
        for method in subset[0]["predictions"]:
            methods[method] = {
                **prediction_metrics(
                    [row["predictions"][method] for row in subset],
                    [row["observed"]["full"] for row in subset],
                ),
                "batch_bootstrap_95ci_relative_error": bootstrap_errors(subset, method, seed=seed),
            }
        removals, predicted_removals, selectivity = [], [], []
        for row in subset:
            difference = np.array(row["observed"]["full"]) - row["observed"]["atom_removed"]
            removals.append(difference)
            predicted_removals.append(row["selected_atom_prediction"])
            target = groups.index(row["target_group"])
            off = np.arange(len(groups)) != target
            entry = {"batch_id": row["batch_id"], "target_group": row["target_group"]}
            for control in ("atom_removed", "random_removed", "rescaled_full"):
                change = np.array(row["observed"]["full"]) - row["observed"][control]
                entry[control] = {
                    "target_effect": float(change[target]),
                    "target_absolute_effect": float(abs(change[target])),
                    "off_target_rms": float(np.sqrt(np.mean(change[off] ** 2))),
                }
            selectivity.append(entry)
        splits = {}
        for split in sorted({row["family_split"] for row in subset}):
            rows = [row for row in subset if row["family_split"] == split]
            splits[split] = {
                method: prediction_metrics(
                    [row["predictions"][method] for row in rows],
                    [row["observed"]["full"] for row in rows],
                )
                for method in ("dictionary", "dictionary_transfer", "full_gradient", "svd")
            }
        summary[str(rate)] = {
            "methods": methods,
            "atom_removal": prediction_metrics(predicted_removals, np.asarray(removals)),
            "family_splits": splits,
            "selectivity": selectivity,
            "batch_count": len(subset),
        }
    return summary


def run_study_job(plan, job, output, progress=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    code_revision = _code_revision()
    source_hashes = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    benchmark = Path(plan["benchmark"])
    data_manifest = json.loads((benchmark / "manifest.json").read_text())
    for name, expected_hash in data_manifest.get("files_sha256", {}).items():
        if file_hash(benchmark / name) != expected_hash:
            raise ValueError(f"Frozen benchmark file changed: {name}")
    specification = {
        "plan": plan,
        "job": job,
        "data_manifest_sha256": file_hash(benchmark / "manifest.json"),
    }
    specification = json.loads(json.dumps(specification))
    spec_path = output / "specification.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
        raise ValueError("A resumed job must use exactly the same configuration and data")
    atomic_json(spec_path, specification)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete":
        progress("already complete")
        return

    def update(message):
        progress(message)
        atomic_json(output / "status.json", {"stage": message, "updated_unix": time.time()})

    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    torch.set_num_threads(plan.get("threads", 8))
    fit = [Example(**row) for row in read_jsonl(benchmark / "fit.jsonl")]
    batches = read_jsonl(benchmark / "batches.jsonl")
    dev_batches = read_jsonl(benchmark / "dev_batches.jsonl")
    calibration = [Probe(**row) for row in read_jsonl(benchmark / "calibration_probes.jsonl")]
    evaluation = [Probe(**row) for row in read_jsonl(benchmark / "evaluation_probes.jsonl")]
    groups, cal_matrix = group_matrix(calibration)
    eval_groups, eval_matrix = group_matrix(evaluation)
    if groups != eval_groups:
        raise ValueError("Calibration and evaluation task groups differ")
    model_config = ModelConfig(**job["model"])
    backend = LanguageModel(model_config)
    warm_checkpoint(
        backend, fit, job["warmup_steps"], output / "parameters.npy", job["seed"], update
    )
    fingerprint = backend.space.fingerprint()
    atomic_json(output / "parameter_layout.json", backend.space.layout())
    update("collecting fitting gradients")
    gradients = cached_gradients(backend, fit, output / "fitting", update)
    projection_path = output / "projection_basis.npy"
    if projection_path.exists() and (output / "projection_scale.npy").exists():
        projection = Projection(
            torch.from_numpy(np.load(projection_path)),
            torch.from_numpy(np.load(output / "projection_scale.npy")),
        )
    else:
        update("fitting uncentered gradient SVD")
        torch.manual_seed(job["seed"])
        projection = Projection.fit(gradients, plan["rank"], True)
        atomic_array(output / "projection_scale.npy", projection.scale.numpy())
        atomic_array(projection_path, projection.basis.numpy())
    values = projection.encode(gradients)
    update("collecting independent calibration and evaluation probe derivatives")
    cal_jacobian = _load_or_collect_probe_gradients(
        backend, calibration, output / "calibration_probe_gradients.npy"
    )
    eval_jacobian = _load_or_collect_probe_gradients(
        backend, evaluation, output / "evaluation_probe_gradients.npy"
    )
    cal_jacobian = cal_matrix @ cal_jacobian
    eval_jacobian = eval_matrix @ eval_jacobian
    cal_projected = (cal_jacobian @ projection.basis) * projection.scale
    eval_projected = (eval_jacobian @ projection.basis) * projection.scale

    update("collecting development batch gradients")
    dev_path = output / "development_gradients.npy"
    if dev_path.exists():
        dev_gradients = torch.from_numpy(np.load(dev_path))
    else:
        dev_gradients = torch.stack(
            [
                backend.gradient([Example(**row) for row in batch["examples"]])
                for batch in dev_batches
            ]
        )
        atomic_array(dev_path, dev_gradients.numpy())
    dev_values = projection.encode(dev_gradients)
    dev_truth = dev_gradients @ cal_jacobian.T
    candidates = {}
    selection = []
    for atoms in plan["dictionary_sizes"]:
        for alpha in plan["alphas"]:
            name = f"k{atoms}-a{alpha}"
            update(f"fitting dictionary {name}")
            dictionary = _load_or_fit_dictionary(
                values,
                output / f"dictionary-{name}.npz",
                atoms,
                alpha,
                plan["dictionary_iterations"],
                job["seed"],
            )
            codes = dictionary.encode(dev_values)
            dev_prediction = codes @ dictionary.atoms @ cal_projected.T
            error = float((dev_prediction - dev_truth).norm() / dev_truth.norm().clamp_min(1e-20))
            selection.append(
                {
                    "name": name,
                    "validation_relative_error": error,
                    "mean_l0": float((codes != 0).sum(1).float().mean()),
                }
            )
            candidates[name] = dictionary
    best = min(selection, key=lambda row: (row["validation_relative_error"], row["name"]))["name"]
    atomic_json(
        output / "selection.json",
        {
            "selected": best,
            "candidates": selection,
            "criterion": "Development-batch derivative reconstruction on calibration probes only",
            "test_outcomes_used": False,
        },
    )
    update(f"selected {best}; preparing atom effects")
    forecaster = AtomForecaster(projection, candidates[best])
    forecaster.effects = forecaster.directions @ cal_jacobian.T
    eval_effects = forecaster.directions @ eval_jacobian.T
    baseline = eval_matrix @ backend.scores(evaluation)
    cal_baseline = cal_matrix @ backend.scores(calibration)
    atomic_array(output / "baseline_evaluation.npy", baseline.numpy())
    write_jsonl(
        output / "atoms.jsonl",
        _atom_records(forecaster, gradients, fit, [Probe(g, "group", "score") for g in groups], 20),
    )

    fd_path = output / "finite_difference_checks.json"
    if not fd_path.exists():
        update("checking analytic calibration against finite atom steps")
        rng = np.random.default_rng(job["seed"])
        selected = rng.choice(
            len(forecaster.directions),
            min(plan["finite_difference_atoms"], len(forecaster.directions)),
            replace=False,
        )
        checks = []
        for atom in selected:
            direction = forecaster.directions[int(atom)]
            row = {"atom": int(atom), "analytic": forecaster.effects[int(atom)].tolist()}
            for radius in (plan["calibration_radius"], plan["calibration_radius"] / 2):
                plus = cal_matrix @ backend.updated_scores(radius * direction, calibration)
                minus = cal_matrix @ backend.updated_scores(-radius * direction, calibration)
                row[str(radius)] = ((plus - minus) / (2 * radius)).tolist()
            checks.append(row)
        atomic_json(fd_path, checks)

    batch_directory = output / "batch_gradients"
    batch_directory.mkdir(exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    if not prediction_path.exists():
        update("forecasting all test batches before measuring any update")
        generator = torch.Generator().manual_seed(job["seed"] + 100)
        random_basis = torch.linalg.qr(
            torch.randn(backend.space.size, plan["rank"], generator=generator)
        ).Q
        random_jacobian = eval_jacobian @ random_basis
        forecasts = []
        for index, batch in enumerate(batches):
            path = batch_directory / f"{index}.npy"
            if path.exists():
                gradient = torch.from_numpy(np.load(path))
            else:
                gradient = backend.gradient([Example(**row) for row in batch["examples"]])
                atomic_array(path, gradient.numpy())
            target = groups.index(batch["target_group"])
            for rate in plan["learning_rates"]:
                update_vector = -rate * gradient
                coordinates = projection.encode(update_vector)
                forecast = forecaster.predict(update_vector, target)
                predicted = {
                    "dictionary": forecast.amplitudes @ eval_effects,
                    "dictionary_transfer": forecast.predicted,
                    "full_gradient": eval_jacobian @ update_vector,
                    "full_gradient_transfer": cal_jacobian @ update_vector,
                    "svd": coordinates @ eval_projected.T,
                    "random_subspace": random_jacobian @ (update_vector @ random_basis),
                    "zero": torch.zeros(len(groups)),
                }
                support = torch.where(forecast.coefficients.abs() > 0)[0]
                if len(support):
                    atoms = candidates[best].atoms[support]
                    unshrunk = torch.linalg.lstsq(atoms.T, coordinates).solution
                    predicted["dictionary_debiased"] = unshrunk @ atoms @ eval_projected.T
                else:
                    predicted["dictionary_debiased"] = torch.zeros(len(groups))
                for name, dictionary in candidates.items():
                    predicted[f"dictionary_{name}"] = (
                        dictionary.encode(coordinates) @ dictionary.atoms @ eval_projected.T
                    )
                forecasts.append(
                    {
                        "batch_id": batch["id"],
                        "batch_index": index,
                        "batch_size": batch["batch_size"],
                        "kind": batch["kind"],
                        "family_split": batch["family_split"],
                        "target_group": batch["target_group"],
                        "learning_rate": rate,
                        "groups": groups,
                        "predictions": {name: value.tolist() for name, value in predicted.items()},
                        "amplitudes": forecast.amplitudes.tolist(),
                        "selected_atom": forecast.selected_atom,
                        "selected_atom_prediction": (
                            forecast.amplitudes[forecast.selected_atom]
                            * eval_effects[forecast.selected_atom]
                        ).tolist(),
                        "selected_atom_transfer_prediction": forecast.contributions[
                            forecast.selected_atom
                        ].tolist(),
                        "update_norm": float(update_vector.norm()),
                        "reconstruction_relative_error": float(
                            (update_vector - forecast.reconstructed_update).norm()
                            / update_vector.norm().clamp_min(1e-20)
                        ),
                        "control_seed": job["seed"] + 10000 + index,
                    }
                )
            update(f"test forecasts {index + 1}/{len(batches)}")
        temporary = prediction_path.with_suffix(".jsonl.tmp")
        write_jsonl(temporary, forecasts)
        temporary.replace(prediction_path)
        atomic_json(
            output / "forecast_lock.json",
            {"sha256": file_hash(prediction_path), "saved_unix": time.time()},
        )
    forecasts = read_jsonl(prediction_path)
    lock_path = output / "forecast_lock.json"
    if not lock_path.exists():
        if (output / "results").exists():
            raise ValueError("Cannot recover a missing forecast lock after evaluation started")
        atomic_json(lock_path, {"sha256": file_hash(prediction_path), "saved_unix": time.time()})
    lock = json.loads(lock_path.read_text())
    if lock["sha256"] != file_hash(prediction_path):
        raise ValueError("Forecasts changed after being locked")

    result_directory = output / "results"
    result_directory.mkdir(exist_ok=True)
    for index, forecast in enumerate(forecasts):
        path = result_directory / f"{index:05d}.json"
        if path.exists():
            continue
        update(f"measuring intervention {index + 1}/{len(forecasts)}")
        gradient = torch.from_numpy(np.load(batch_directory / f"{forecast['batch_index']}.npy"))
        delta = -forecast["learning_rate"] * gradient
        amplitudes = torch.tensor(forecast["amplitudes"])
        atom = forecast["selected_atom"]
        component = amplitudes[atom] * forecaster.directions[atom]
        control = torch.randn(
            backend.space.size, generator=torch.Generator().manual_seed(forecast["control_seed"])
        )
        control = control / control.norm() * component.norm()
        interventions = {
            "full": delta,
            "reconstructed": amplitudes @ forecaster.directions,
            "atom_removed": delta - component,
            "random_removed": delta - control,
            "rescaled_full": delta * ((delta - component).norm() / delta.norm().clamp_min(1e-20)),
        }
        observed = {
            name: (eval_matrix @ backend.updated_scores(vector, evaluation) - baseline).tolist()
            for name, vector in interventions.items()
        }
        atomic_json(path, {**forecast, "observed": observed})
    results = [
        json.loads((result_directory / f"{i:05d}.json").read_text()) for i in range(len(forecasts))
    ]
    write_jsonl(output / "results.jsonl", results)
    atomic_json(output / "summary.json", summarize_study(results, groups, job["seed"]))

    # Prespecified batches, maximum prespecified step size; no outcome-based selection.
    generation_path = output / "generations.json"
    if plan.get("generation_batches", 0) and not generation_path.exists():
        update("checking generated continuations on prespecified pure batches")
        prompts = [next(p.prompt for p in evaluation if p.group == group) for group in groups]
        clean = backend.generate(prompts, plan["generation_tokens"])
        generations = []
        selected = [
            r
            for r in forecasts
            if r["kind"] == "pure"
            and r["batch_size"] == 16
            and r["learning_rate"] == max(plan["learning_rates"])
        ]
        for row in selected[: plan["generation_batches"]]:
            delta = -row["learning_rate"] * torch.from_numpy(
                np.load(batch_directory / f"{row['batch_index']}.npy")
            )
            component = (
                row["amplitudes"][row["selected_atom"]]
                * forecaster.directions[row["selected_atom"]]
            )
            with backend.space.temporary_update(delta):
                full = backend.generate(prompts, plan["generation_tokens"])
            with backend.space.temporary_update(delta - component):
                removed = backend.generate(prompts, plan["generation_tokens"])
            generations.append(
                {
                    "batch_id": row["batch_id"],
                    "groups": groups,
                    "prompts": prompts,
                    "baseline": clean,
                    "full": full,
                    "atom_removed": removed,
                }
            )
        atomic_json(
            generation_path,
            {
                "scope": "Greedy qualitative checks; not semantic accuracy judgments",
                "results": generations,
            },
        )
    if backend.space.fingerprint() != fingerprint:
        raise RuntimeError("Study changed the reference checkpoint")
    atomic_json(
        manifest_path,
        {
            "status": "complete",
            "code_revision": code_revision,
            "source_sha256": source_hashes,
            "model": asdict(model_config),
            "resolved_model_revision": backend.resolved_revision,
            "seed": job["seed"],
            "warmup_steps": job["warmup_steps"],
            "warmup_optimizer": "AdamW lr=1e-4, decay=0, batch=4, clip=1; fitting data only",
            "update_optimizer": "One stateless SGD step; mean of per-example mean completion loss",
            "parameter_fingerprint": fingerprint,
            "checkpoint_restored": True,
            "data_manifest": data_manifest,
            "groups": groups,
            "selected_dictionary": best,
            "forecast_sha256": lock["sha256"],
            "forecast_saved_unix": lock["saved_unix"],
            "calibration": (
                "Analytic directional derivatives; independent finite checks on preset atoms"
            ),
            "fixed_probe_scope": (
                "dictionary/full_gradient/svd use evaluation-probe derivatives before updates"
            ),
            "transfer_scope": (
                "*_transfer use calibration-document derivatives to predict separate "
                "evaluation-document mean scores by category"
            ),
            "calibration_baseline": cal_baseline.tolist(),
            "evaluation_baseline": baseline.tolist(),
            "elapsed_this_invocation_seconds": time.time() - started,
            "references": REFERENCES,
        },
    )
    update("complete")
