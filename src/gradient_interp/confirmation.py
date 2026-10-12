"""Frozen fresh-source experiment; save per-document forecasts before any updates."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .cached_sweep import compressed_coordinates
from .data import Example, Probe, read_jsonl, write_jsonl
from .dictionary import Projection, sparse_code
from .experiment import REFERENCES, _atom_records, _code_revision
from .forecast import AtomForecaster
from .models import LanguageModel, ModelConfig
from .study import (
    _load_or_collect_probe_gradients,
    _load_or_fit_dictionary,
    atomic_archive,
    atomic_array,
    atomic_json,
    cached_gradients,
    file_hash,
    group_matrix,
    warm_checkpoint,
)


def verify_files(directory, hashes):
    for name, expected in hashes.items():
        if file_hash(Path(directory) / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")


def removal_vectors(delta, component, seed):
    control = torch.randn(delta.shape, generator=torch.Generator().manual_seed(seed))
    control = control / control.norm().clamp_min(1e-20) * component.norm()
    return {
        "atom_removed": delta - component,
        "random_removed": delta - control,
        "rescaled_full": delta * ((delta - component).norm() / delta.norm().clamp_min(1e-20)),
    }


def forecast_arrays(gradients, projection, dictionary, cal_jacobian, eval_jacobian, seed):
    """Unit-learning-rate predictions, preserving individual probe documents."""
    raw = -gradients @ projection.basis
    variants, costs = compressed_coordinates(raw, projection.scale, dictionary.atoms, seed)
    white = raw / projection.scale
    codes = dictionary.encode(white)
    variants["dictionary_lars"] = codes @ dictionary.atoms * projection.scale
    refit = torch.zeros_like(raw)
    for i, code in enumerate(codes):
        support = torch.where(code.abs() > 0)[0]
        if len(support):
            atoms = dictionary.atoms[support]
            refit[i] = torch.linalg.lstsq(atoms.T, white[i]).solution @ atoms * projection.scale
    variants["dictionary_refit"] = refit
    variants["zero"] = torch.zeros_like(raw)
    methods = list(variants) + ["full_gradient"]
    outputs = []
    for jacobian in (cal_jacobian, eval_jacobian):
        projected = jacobian @ projection.basis
        effects = [coordinates @ projected.T for coordinates in variants.values()]
        effects.append(-gradients @ jacobian.T)
        outputs.append(torch.stack(effects).numpy())
    return methods, costs, outputs


def run_confirmation_job(plan, job, output, progress=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = Path(plan["benchmark"])
    manifest = json.loads((data / "manifest.json").read_text())
    verify_files(data, manifest["files_sha256"])
    specification = {
        "plan": plan,
        "job": job,
        "data_manifest_sha256": file_hash(data / "manifest.json"),
    }
    specification = json.loads(json.dumps(specification))
    spec = output / "specification.json"
    if spec.exists() and json.loads(spec.read_text()) != specification:
        raise ValueError("Resumes require exactly the same frozen specification")
    atomic_json(spec, specification)
    lock_path = output / "forecast_lock.json"
    if lock_path.exists():
        verify_files(output, json.loads(lock_path.read_text())["files_sha256"])
    if (output / "manifest.json").exists():
        progress("already complete")
        return
    started = time.time()

    def update(message):
        progress(message, flush=True) if progress is print else progress(message)
        atomic_json(output / "status.json", {"stage": message, "updated_unix": time.time()})

    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    torch.set_num_threads(plan.get("threads", 8))
    fit = [Example(**r) for r in read_jsonl(data / "fit.jsonl")]
    batches = read_jsonl(data / "batches.jsonl")
    cal = [Probe(**r) for r in read_jsonl(data / "calibration_probes.jsonl")]
    evaluation = [Probe(**r) for r in read_jsonl(data / "evaluation_probes.jsonl")]
    groups, cal_average = group_matrix(cal)
    if groups != group_matrix(evaluation)[0]:
        raise ValueError("Probe groups disagree")
    update("loading model")
    backend = LanguageModel(ModelConfig(**job["model"]))
    warm_checkpoint(
        backend, fit, job["warmup_steps"], output / "parameters.npy", job["seed"], update
    )
    fingerprint = backend.space.fingerprint()
    atomic_json(output / "parameter_layout.json", backend.space.layout())
    update("collecting fitting gradients")
    gradients = cached_gradients(backend, fit, output / "fitting", update)
    basis_path, scale_path = output / "projection_basis.npy", output / "projection_scale.npy"
    if basis_path.exists() and scale_path.exists():
        projection = Projection(
            torch.from_numpy(np.load(basis_path)), torch.from_numpy(np.load(scale_path))
        )
    else:
        update("fitting gradient SVD")
        torch.manual_seed(job["seed"])
        projection = Projection.fit(gradients, plan["rank"], True)
        atomic_array(basis_path, projection.basis.numpy())
        atomic_array(scale_path, projection.scale.numpy())
    update("fitting dictionary")
    dictionary = _load_or_fit_dictionary(
        projection.encode(gradients),
        output / "dictionary.npz",
        plan["atoms"],
        plan["alpha"],
        plan["dictionary_iterations"],
        job["seed"],
    )
    cal_path, eval_path = (
        output / "calibration_probe_gradients.npy",
        output / "evaluation_probe_gradients.npy",
    )
    update("collecting calibration derivatives")
    cal_jacobian = _load_or_collect_probe_gradients(backend, cal, cal_path)
    update("collecting evaluation derivatives")
    eval_jacobian = _load_or_collect_probe_gradients(backend, evaluation, eval_path)
    forecaster = AtomForecaster(projection, dictionary)
    forecaster.effects = forecaster.directions @ (cal_average @ cal_jacobian).T
    atoms_path = output / "atoms.jsonl"
    if not atoms_path.exists():
        write_jsonl(
            atoms_path,
            _atom_records(
                forecaster, gradients, fit, [Probe(g, "category", "score") for g in groups], 20
            ),
        )
    del gradients

    predictions_path = output / "predictions.npz"
    metadata_path = output / "forecast_metadata.json"
    component_path = output / "components.npy"
    batch_path = output / "batch_gradients.npy"
    if not lock_path.exists():
        if (output / "results").exists():
            raise ValueError("Outcomes exist without a forecast lock")
        if batch_path.exists():
            batch_gradients = torch.from_numpy(np.load(batch_path))
        else:
            collected = []
            for i, batch in enumerate(batches):
                update(f"batch gradients {i + 1}/{len(batches)}")
                collected.append(backend.gradient([Example(**e) for e in batch["examples"]]))
            batch_gradients = torch.stack(collected)
            atomic_array(batch_path, batch_gradients.numpy())
        update("forecasting every method and batch before interventions")
        methods, costs, (cal_predictions, eval_predictions) = forecast_arrays(
            batch_gradients, projection, dictionary, cal_jacobian, eval_jacobian, job["seed"]
        )
        codes = sparse_code(
            projection.encode(-batch_gradients), dictionary.atoms, plan["control_sparsity"]
        )
        cal_atom_effects = projection.decode(dictionary.atoms) @ (cal_average @ cal_jacobian).T
        # Removing a common category shift cannot make an atom appear selective.
        cal_atom_effects -= cal_atom_effects.mean(1, keepdim=True)
        components, selections = [], []
        for i, batch in enumerate(batches):
            target = groups.index(batch["target_group"])
            selected = int((codes[i] * cal_atom_effects[:, target]).abs().argmax())
            components.append(projection.decode(codes[i, selected] * dictionary.atoms[selected]))
            selections.append(
                {
                    "atom": selected,
                    "coefficient": float(codes[i, selected]),
                    "target_group": batch["target_group"],
                }
            )
        components = torch.stack(components)
        atomic_array(component_path, components.numpy())
        update("measuring reference scores and numerical repeatability")
        baseline = backend.scores(evaluation).double().numpy()
        repeat = (
            backend.updated_scores(torch.zeros(backend.space.size), evaluation).double().numpy()
        )
        atomic_archive(
            predictions_path,
            calibration=cal_predictions,
            evaluation=eval_predictions,
            atom_calibration=(components @ cal_jacobian.T).numpy(),
            atom_evaluation=(components @ eval_jacobian.T).numpy(),
            baseline=baseline,
            zero_update_noise=repeat - baseline,
        )
        atomic_json(
            metadata_path,
            {
                "methods": methods,
                "bytes_per_update_code": costs,
                "groups": groups,
                "batch_ids": [b["id"] for b in batches],
                "selections": selections,
                "calibration_probe_ids": [p.id for p in cal],
                "evaluation_probe_ids": [p.id for p in evaluation],
                "prediction_units": "Effect per unit learning rate; multiply by frozen rate",
            },
        )
        atomic_json(
            lock_path,
            {
                "saved_unix": time.time(),
                "parameter_fingerprint": fingerprint,
                "files_sha256": {
                    p.name: file_hash(p)
                    for p in (
                        predictions_path,
                        metadata_path,
                        component_path,
                        batch_path,
                        spec,
                        basis_path,
                        scale_path,
                        output / "dictionary.npz",
                        output / "parameters.npy",
                    )
                },
            },
        )
    lock = json.loads(lock_path.read_text())
    verify_files(output, lock["files_sha256"])
    if fingerprint != lock["parameter_fingerprint"]:
        raise ValueError("Checkpoint changed since forecasts were locked")
    with np.load(predictions_path) as values:
        baseline = values["baseline"].copy()
    batch_gradients = torch.from_numpy(np.load(batch_path))
    components = torch.from_numpy(np.load(component_path))
    del cal_jacobian, eval_jacobian, forecaster
    results = output / "results"
    results.mkdir(exist_ok=True)
    total = len(batches) * len(plan["learning_rates"])
    for i, batch in enumerate(batches):
        for ri, rate in enumerate(plan["learning_rates"]):
            path = results / f"batch-{i:03d}-rate-{ri}.npz"
            if path.exists():
                continue
            update(f"evaluating fixed case {i * len(plan['learning_rates']) + ri + 1}/{total}")
            delta = -rate * batch_gradients[i]
            vectors = {"full": delta}
            if i in plan["control_batch_indices"] and ri == 0:
                vectors.update(
                    removal_vectors(delta, rate * components[i], job["seed"] + i + 90000)
                )
            observed = {
                name: backend.updated_scores(vector, evaluation).double().numpy() - baseline
                for name, vector in vectors.items()
            }
            atomic_archive(path, **observed)

    gen_path = output / "generations.json"
    if plan.get("generation_tokens", 0) and not gen_path.exists():
        update("generating preset qualitative examples")
        prompts = [next(p.prompt for p in evaluation if p.group == group) for group in groups]
        clean = backend.generate(prompts, plan["generation_tokens"])
        generated = []
        selected = [
            i for i, b in enumerate(batches) if b["kind"] == "pure" and b["batch_size"] == 16
        ][:2]
        for i in selected:
            rate = max(plan["learning_rates"])
            delta = -rate * batch_gradients[i]
            with backend.space.temporary_update(delta):
                full = backend.generate(prompts, plan["generation_tokens"])
            with backend.space.temporary_update(delta - rate * components[i]):
                removed = backend.generate(prompts, plan["generation_tokens"])
            generated.append({"batch_id": batches[i]["id"], "full": full, "atom_removed": removed})
        atomic_json(
            gen_path,
            {
                "prompts": prompts,
                "baseline": clean,
                "results": generated,
                "scope": "Prespecified qualitative examples; no accuracy score",
            },
        )
    if fingerprint != backend.space.fingerprint():
        raise RuntimeError("Reference checkpoint was changed")
    verify_files(output, lock["files_sha256"])
    atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "job": job,
            "code_revision": _code_revision(),
            "resolved_model_revision": backend.resolved_revision,
            "source_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
            "forecast_lock_sha256": file_hash(lock_path),
            "forecast_saved_unix": lock["saved_unix"],
            "result_files_sha256": {p.name: file_hash(p) for p in results.glob("*.npz")},
            "data_manifest_sha256": file_hash(data / "manifest.json"),
            "checkpoint_restored": True,
            "completed_unix": time.time(),
            "elapsed_this_invocation_seconds": time.time() - started,
            "references": REFERENCES,
        },
    )
    update("complete")
