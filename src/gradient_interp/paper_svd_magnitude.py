"""Prospective SVD500 versus dictionary forecasts of reference completion loss.

This new development phase reuses fixed fitting directions, never selected
steering-vector multipliers. It makes no generated-behavior or sparse-explanation
claim. New held-out natural data must be approved and staged before preparation.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .confirmation import verify_files
from .paper_batch import flat_grad, scalar_logp
from .paper_magnitude import (
    COUNTS,
    assert_restored,
    bootstrap_interval,
    encode,
    loss_forecasts,
    matrix,
    training_example,
    validate_architecture,
    validate_cache,
    validate_data,
    validate_dedup_audit,
    validate_staged_data,
    validate_tokenization,
)
from .paper_mixture import stable_setup
from .paper_setup import read_json
from .paper_steering import restore, set_direction
from .paper_svd_replication import FAMILIES, assert_selection_lock
from .study import atomic_json, file_hash

REPRESENTATIONS = ("svd500", "dictionary")
METHODS = (*REPRESENTATIONS, "full_gradient", "zero")
RATES = (0.003, 0.0003)
PRIMARY_RATE = 0.003
BOOTSTRAP_SEED = 2026101453
SIMULTANEOUS_FAMILY = 8
PRIOR_DATA_SHA256 = "88d48369df1b95bb346c65ac8e9111b196946a478f33e732559a870216642933"


def protocol():
    return {
        "scope": "development_only",
        "methods": list(METHODS),
        "rates": list(RATES),
        "primary_rate": PRIMARY_RATE,
        "secondary_rate": 0.0003,
        "counts": {**COUNTS, "actual_updates": 64, "components_per_representation": 500},
        "max_hours": 12,
        "basis": "Reuse the exact SVD500 basis in the existing locked directions.pt, and the "
        "original normalized-row 500-atom dictionary. No fitting or new candidate selection.",
        "svd_encoding": "For projected batch gradient x and orthonormal rows V, coefficients "
        "are x @ V.T and reconstruction is coefficients @ V. Preserve x's magnitude. "
        "No unit normalization, regression, sparse penalty, or steering-norm multiplier.",
        "dictionary_encoding": "Original alpha .1 lasso_lars on the normalized batch "
        "projection, followed by restoration of its original projected norm.",
        "physical_geometry": "Unproject unscaled basis rows through the original EKFAC "
        "inverse transform. Never use directions.pt selected matched-norm steering vectors. "
        "All 500 signed component contributions are retained, including dense SVD codes.",
        "objective": "Teacher-forced mean completion CE with original training EOS. "
        "Batch gradient is the mean of eight per-document mean-token CE gradients. Each "
        "probe gradient uses its own mean-token completion CE. All rows are untruncated.",
        "forecast": "For each batch, rate, representation and probe, forecast -rate times "
        "the original probe-loss gradient dotted with the reconstructed batch gradient. "
        "Lock every forecast, code and contribution before any actual update.",
        "target": "The same actual full-gradient SGD update supplies both representations' "
        "targets. Score all 80 probes after each of 64 updates and restore all 136 tensors.",
        "calibration": "Sixteen separate descriptive probes. No fitted scale or tuning. "
        "The 64 evaluation probes determine the primary fixed accuracy criteria.",
        "criteria": {
            "rate": PRIMARY_RATE,
            "minimum_actual_rms": 0.01,
            "maximum_svd_nrmse": 0.5,
            "maximum_svd_mae": 0.01,
            "minimum_dictionary_minus_svd_nrmse": 0.10,
            "required_seeds": [42, 43],
            "rule": "All four bounds must clear in each seed at .003. A signal below "
            "the floor is inconclusive. The .0003 result is descriptive and cannot rescue "
            "a failed primary result. No selection of methods, rates or seeds.",
        },
        "uncertainty": {
            "replicates": 1000,
            "seed": BOOTSTRAP_SEED,
            "resampling": "Independently resample 32 batches and 64 evaluation probe IDs, "
            "with the same indices for every method and rate. Report calibration separately. "
            "Each source/lexical cluster occurs in only one selected probe.",
            "pointwise_coverage": 0.95,
            "simultaneous_family": SIMULTANEOUS_FAMILY,
            "simultaneous_marginal_coverage": 0.99375,
            "family": "Two seeds times four primary quantities. Bonferroni 99.375% "
            "marginal percentile intervals target at least 95% family coverage. "
            "Undefined bootstrap ratios remain undefined and prevent success.",
        },
        "reporting": "All methods/rates/seeds and all per-batch/probe predictions and actual "
        "losses; RMSE, MAE, nRMSE, signal RMS, signed means, slope, max/p90/p95 errors; "
        "direction agreement only when both magnitudes exceed .001; all norms and "
        "reconstruction errors. First eight batches and first eight evaluation probes "
        "are fixed examples, with every method/rate. Report negative predicted CE unchanged.",
        "component_reporting": "Retain all 500 coefficients and all component loss "
        "contributions per representation, rate and probe. Copy the existing training-only "
        "named selections and signs without new labels. Report dense active counts and "
        "named contribution shares without claiming a small interpretable set. Count both "
        "exact nonzero and absolute coefficient greater than 1e-6 on magnitude-bearing codes.",
        "limits": "New natural rows and locked forecasts, but reused adapters and fitting "
        "data: prospective development, not confirmation. Reference CE accuracy does not "
        "establish unrestricted generated behavior or component interpretability. "
        "Reference methods do not replace held-out description/intervention evidence.",
        "reading_barrier": "Do not read held-out forecast/outcome content until both seeds "
        "finish normally with processes dead or reach original deadlines with processes dead. "
        "Early technical failures do not release the reading barrier. Partial grids remain "
        "failures and are not analyzed as complete.",
    }


def encode_svd(projected, basis):
    x, v = matrix(projected, "batch projections"), matrix(basis, "fixed SVD basis")
    if x.shape[1] != v.shape[1]:
        raise ValueError("SVD dimensions differ")
    if not np.allclose(v @ v.T, np.eye(len(v)), rtol=0, atol=1e-4):
        raise ValueError("Frozen SVD rows are not orthonormal")
    coefficients = x.astype(np.float64) @ v.astype(np.float64).T
    return coefficients, coefficients @ v.astype(np.float64)


def unproject_reconstructions(projected, unproject):
    """Hand projected rows to the original float32 EKFAC inverse without rescaling."""
    rows = matrix(projected, "reconstructed projections").astype(np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("Reconstruction is outside the float32 inverse range")
    return torch.stack([unproject(row) for row in rows])


def component_contributions(coefficients, profiles, rates=RATES):
    """B,M,K coefficients and M,K,Q unscaled physical loss profiles -> B,R,M,K,Q."""
    c, p = np.asarray(coefficients), np.asarray(profiles)
    rates = np.asarray(rates)
    if (
        c.ndim != 3
        or p.ndim != 3
        or c.shape[1:] != p.shape[:2]
        or not np.isfinite(c).all()
        or not np.isfinite(p).all()
        or rates.ndim != 1
        or not len(rates)
        or not np.isfinite(rates).all()
        or np.any(rates <= 0)
    ):
        raise ValueError("Invalid component contribution geometry")
    return -np.einsum("bmk,mkq,r->brmkq", c, p, rates, optimize=True)


def validate_selection(selection, components=500):
    if set(selection) != set(REPRESENTATIONS):
        raise ValueError("Both frozen training selections are required")
    for method in REPRESENTATIONS:
        if set(selection[method]) != set(FAMILIES):
            raise ValueError("Frozen named families changed")
        for card in selection[method].values():
            if (
                not 0 <= card["component"] < components
                or card["orientation"] not in (-1, 1)
                or card.get("candidate_only") is not True
            ):
                raise ValueError("Invalid frozen candidate card")


def load_bases(spec):
    root = Path(spec["basis_root"])
    packet = torch.load(root / "directions.pt", map_location="cpu", weights_only=True)
    bases = {method: packet["bases"][method].numpy() for method in REPRESENTATIONS}
    if any(x.shape != (500, 6800) or not np.isfinite(x).all() for x in bases.values()):
        raise ValueError("The frozen 500 by 6800 basis geometry differs")
    encode_svd(np.zeros((1, 6800)), bases["svd500"])
    original = torch.load(
        Path(spec["source_root"]) / "atoms.pt", map_location="cpu", weights_only=True
    )
    np.testing.assert_array_equal(bases["dictionary"], original["dictionary"].numpy())
    if packet["lora_names"] != original["lora_names"]:
        raise ValueError("Fixed basis uses another LoRA ordering")
    return bases, {key: packet[key] for key in ("lora_names", "module_info")}


def validate_projection_info(current, original):
    if len(current) != len(original):
        raise ValueError("Frozen projection module count changed")
    for new, old in zip(current, original, strict=True):
        for key in ("name", "ekfac_key", "offset", "n_params", "d_in", "d_out", "k"):
            if new[key] != old[key]:
                raise ValueError(f"Frozen projection layout changed: {key}")
        for key in ("topk_idx", "topk_evals"):
            if not torch.equal(torch.as_tensor(new[key]), torch.as_tensor(old[key])):
                raise ValueError(f"Frozen projection eigenbasis changed: {key}")


def physical_component_profiles(bases, probe_gradients, unproject, status=None, device="cuda"):
    """Stream unscaled rows so no matched steering multiplier or whole bank is needed."""
    q = torch.as_tensor(probe_gradients, dtype=torch.float64, device=device)
    if q.ndim != 2 or not torch.isfinite(q).all():
        raise ValueError("Invalid physical probe-loss gradients")
    profiles, norms = [], []
    for method in REPRESENTATIONS:
        rows, lengths = [], []
        for start in range(0, len(bases[method]), 16):
            bank = torch.stack([unproject(row) for row in bases[method][start : start + 16]])
            if bank.ndim != 2 or bank.shape[1] != q.shape[1] or not torch.isfinite(bank).all():
                raise ValueError("Invalid unscaled physical basis row")
            lengths.extend(bank.double().norm(dim=1).tolist())
            rows.append((bank.to(device=device, dtype=torch.float64) @ q.T).cpu().numpy())
            if status:
                status(
                    f"profiles_{method}", min(start + 16, len(bases[method])), len(bases[method])
                )
        profiles.append(np.concatenate(rows))
        norms.append(lengths)
    return np.stack(profiles), np.asarray(norms)


def bootstrap_metrics(predictions, actual, replicates=1000, seed=BOOTSTRAP_SEED):
    p, y = np.asarray(predictions), np.asarray(actual)
    if p.ndim != 4 or p.shape[2] != 4 or y.shape != (p.shape[0], p.shape[1], p.shape[3]):
        raise ValueError("Expected B,R,4,Q forecasts and B,R,Q targets")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or replicates < 2:
        raise ValueError("Expected finite paired forecasts and targets")
    rng = np.random.default_rng(seed)
    result = {key: [] for key in ("nrmse", "mae", "actual_rms", "dictionary_minus_svd_nrmse")}
    for _ in range(replicates):
        bi, qi = (
            rng.integers(p.shape[0], size=p.shape[0]),
            rng.integers(p.shape[3], size=p.shape[3]),
        )
        pp, yy = p[bi][..., qi], y[bi][..., qi]
        mse, zero = np.mean((pp - yy[:, :, None]) ** 2, (0, 3)), np.mean(yy**2, (0, 2))
        relative = np.full_like(mse, np.nan, dtype=np.float64)
        np.divide(mse, zero[:, None], out=relative, where=zero[:, None] > 0)
        relative = np.sqrt(relative)
        result["nrmse"].append(relative)
        result["mae"].append(np.mean(abs(pp - yy[:, :, None]), (0, 3)))
        result["actual_rms"].append(np.sqrt(zero))
        result["dictionary_minus_svd_nrmse"].append(relative[:, 1] - relative[:, 0])
    return {key: np.asarray(value) for key, value in result.items()}


def primary_gate(draws, rates=RATES):
    ri = list(rates).index(PRIMARY_RATE)
    quantities = {
        "actual_rms": draws["actual_rms"][:, ri],
        "svd_nrmse": draws["nrmse"][:, ri, 0],
        "svd_mae": draws["mae"][:, ri, 0],
        "dictionary_minus_svd_nrmse": draws["dictionary_minus_svd_nrmse"][:, ri],
    }
    intervals = {
        key: bootstrap_interval(value, family_size=SIMULTANEOUS_FAMILY)
        for key, value in quantities.items()
    }
    checks = {
        "actual_rms": (0, 0.01, True),
        "svd_nrmse": (1, 0.5, False),
        "svd_mae": (1, 0.01, False),
        "dictionary_minus_svd_nrmse": (0, 0.10, True),
    }
    failed = []
    for key, (end, limit, lower) in checks.items():
        bound = intervals[key]["interval"]
        if bound is None or (bound[end] < limit if lower else bound[end] > limit):
            failed.append(key)
    return {
        "cleared": not failed,
        "failed_quantities": failed,
        "intervals": intervals,
        "signal_inconclusive": "actual_rms" in failed,
        "primary_rate": PRIMARY_RATE,
    }


def combined_primary_gates(by_seed):
    if set(by_seed) != {42, 43}:
        raise ValueError("Both seed-specific primary gates are required")
    return {
        "cleared_both_seeds": all(row["cleared"] for row in by_seed.values()),
        "seeds": by_seed,
        "primary_rate": PRIMARY_RATE,
    }


def prepare(root, basis_root, data_path, audit_path):
    root, basis_root = Path(root).resolve(), Path(basis_root).resolve()
    if (root / "specification.json").exists():
        raise ValueError("Already prepared; preserve the original specification")
    basis_spec = assert_selection_lock(basis_root)
    source = Path(basis_spec["source_root"])
    config, integrity = (
        read_json(source / "configuration.json"),
        read_json(source / "integrity.json"),
    )
    if not integrity["complete"] or config["seed"] not in (42, 43):
        raise ValueError("Completed original seed42/43 reconstruction required")
    data, audit = read_json(data_path), read_json(audit_path)
    validate_data(data)
    validate_dedup_audit(audit, Path(data_path))
    if audit.get("excluded_previous_magnitude_data_sha256") != PRIOR_DATA_SHA256:
        raise ValueError("New data must explicitly exclude the prior 336 magnitude documents")
    for split in ("calibration", "evaluation"):
        for field in ("cluster_id", "source_id"):
            if len({row[field] for row in data[split]}) != len(data[split]):
                raise ValueError("Probe bootstrap requires unique source and lexical clusters")
    hashes = {}
    for stage in ("train", "factors", "atoms"):
        path = source / f"{stage}-manifest.json"
        if file_hash(path) != integrity["stages"][stage]["manifest_sha256"]:
            raise ValueError("An original source manifest changed")
        verify_files(source, read_json(path)["files_sha256"])
        hashes.update({str(source / n): h for n, h in read_json(path)["files_sha256"].items()})
        hashes[str(path)] = file_hash(path)
    if (
        file_hash(source / "configuration.json")
        != read_json(source / "plan.json")["configuration_sha256"]
    ):
        raise ValueError("Original source configuration changed")
    for key, value in (
        ("n_atoms", 500),
        ("dictionary_alpha", 0.1),
        ("lora_tensors", 136),
        ("lora_parameters", 2228224),
    ):
        if config[key] != value:
            raise ValueError(f"Original paper architecture differs: {key}")
    paths = [
        source / n for n in ("configuration.json", "integrity.json", "plan.json", "training.jsonl")
    ]
    paths += [
        basis_root / n
        for n in ("specification.json", "selection-lock.json", "selection.json", "directions.pt")
    ]
    paths += [Path(data_path).resolve(), Path(audit_path).resolve()]
    base = Path(config["base_directory"])
    verify_files(base, config["model_asset_hashes"])
    hashes.update(
        {str(base / name): digest for name, digest in config["model_asset_hashes"].items()}
    )
    hashes.update({str(path): file_hash(path) for path in paths})
    spec = {
        **protocol(),
        "source_root": str(source),
        "source_config": config,
        "basis_root": str(basis_root),
        "training_seed": config["seed"],
        "prepared_unix": time.time(),
        "input_hashes": hashes,
    }
    bases, _ = load_bases(spec)
    del bases
    selection = read_json(basis_root / "selection.json")
    validate_selection(selection)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "data.json", data)
    audit["staged_data_sha256"] = file_hash(root / "data.json")
    atomic_json(root / "data-audit.json", audit)
    atomic_json(root / "named-selections.json", selection)
    atomic_json(root / "specification.json", spec)
    atomic_json(
        root / "preparation-lock.json",
        {
            "locked_unix": time.time(),
            "new_gradients_or_outcomes_computed": False,
            "source_files_sha256": {
                str(p): file_hash(p) for p in Path("src/gradient_interp").glob("*.py")
            },
            "input_hashes": {
                **hashes,
                **{
                    str(root / n): file_hash(root / n)
                    for n in (
                        "data.json",
                        "data-audit.json",
                        "named-selections.json",
                        "specification.json",
                    )
                },
            },
        },
    )


def verify_preparation(root):
    lock = read_json(root / "preparation-lock.json")
    if lock["new_gradients_or_outcomes_computed"]:
        raise ValueError("Preparation lock must precede new scientific gradients")
    verify_files(Path.cwd(), lock["source_files_sha256"])
    verify_files(Path.cwd(), lock["input_hashes"])
    validate_staged_data(root)
    return read_json(root / "specification.json")


def preflight(root):
    spec = verify_preparation(root)
    if (root / "preflight.json").exists():
        raise ValueError("Preflight already exists; preserve it")
    model, scoring, _, tokenize, payload, parameters = stable_setup(spec)
    layout = validate_architecture(model, parameters, spec["source_config"])
    tokenization = validate_tokenization(read_json(root / "data.json"), scoring, tokenize)
    original = {name: p.detach().clone() for name, p in parameters}
    row = training_example(Path(spec["source_root"]))
    value, count = scalar_logp(model, scoring, tokenize, row["prompt"], row["answer"])
    gradient, baseline = flat_grad(-value / count, parameters), float(-value.detach() / count)
    del value
    value, long_count = scalar_logp(
        model, scoring, tokenize, row["prompt"], (row["answer"] + " ") * 64
    )
    long_gradient = flat_grad(-value / long_count, parameters)
    del value, long_gradient
    set_direction(parameters, original, gradient, PRIMARY_RATE)
    try:
        if all(torch.equal(p, original[name]) for name, p in parameters):
            raise ValueError("Training preflight did not change any adapter tensor")
        with torch.no_grad():
            changed, changed_n = scalar_logp(model, scoring, tokenize, row["prompt"], row["answer"])
        if changed_n != count or not torch.isfinite(changed):
            raise ValueError("Invalid actual training preflight score")
    finally:
        restore(parameters, original)
    assert_restored(parameters, original)
    with torch.no_grad():
        restored, restored_n = scalar_logp(model, scoring, tokenize, row["prompt"], row["answer"])
    if restored_n != count or float(-restored / count) != baseline:
        raise ValueError("Training score was not restored exactly")
    from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac, unproject_atom

    factors = load_ekfac_eigen(
        str(Path(spec["source_root"]) / "factors/paper_ekfac/factors_training")
    )
    projected, info = project_gradients_ekfac(gradient[None], payload["lora_names"], factors, 50)
    bases, basis_info = load_bases(spec)
    validate_projection_info(info, basis_info["module_info"])
    coefficients, reconstructions = [], []
    for method in REPRESENTATIONS:
        c, x = (
            encode_svd(projected.numpy(), bases[method])
            if method == "svd500"
            else encode(projected.numpy(), bases[method])
        )
        coefficients.append(c)
        reconstructions.append(
            unproject_reconstructions(
                x, lambda row: unproject_atom(row, info, factors, gradient.numel())
            )[0]
        )
    profiles, norms = physical_component_profiles(
        bases, gradient[None], lambda x: unproject_atom(x, info, factors, gradient.numel())
    )
    contributions = component_contributions(np.stack(coefficients, 1), profiles)
    forecasts = loss_forecasts(
        gradient[None].numpy(), torch.stack(reconstructions)[None].numpy(), RATES
    )
    np.testing.assert_allclose(contributions.sum(3), forecasts, rtol=1e-4, atol=1e-7)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Nonfinite or zero physical component norm")
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "preparation_lock_sha256": file_hash(root / "preparation-lock.json"),
            "layout_verified": True,
            "layout": layout,
            "tensors": 136,
            "parameter_count": 2228224,
            "training_input_only": True,
            "long_training_completion_checked": True,
            "finite_gradient": True,
            "adapter_restored": True,
            "scoring_restored": True,
            "unscaled_component_profiles_verified": True,
            "tokenization": tokenization,
            "component_sum_max_abs_error": float(abs(contributions.sum(3) - forecasts).max()),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def freeze(root):
    spec = verify_preparation(root)
    if (root / "plan.json").exists():
        raise ValueError("Original plan already frozen")
    pre = read_json(root / "preflight.json")
    required = (
        "layout_verified",
        "training_input_only",
        "long_training_completion_checked",
        "finite_gradient",
        "adapter_restored",
        "scoring_restored",
        "unscaled_component_profiles_verified",
    )
    if any(pre.get(key) is not True for key in required):
        raise ValueError("All training-only numerical preflight checks must pass")
    if (
        pre["tensors"] != 136
        or pre["parameter_count"] != 2228224
        or pre["specification_sha256"] != file_hash(root / "specification.json")
        or pre["preparation_lock_sha256"] != file_hash(root / "preparation-lock.json")
        or pre["tokenization"]["documents"] != 336
        or not pre["tokenization"]["all_targets_untruncated"]
        or not pre["tokenization"]["original_eos_convention"]
    ):
        raise ValueError("Preflight architecture, data or source provenance differs")
    preparation = read_json(root / "preparation-lock.json")
    atomic_json(
        root / "plan.json",
        {
            "scope": "development_only",
            "frozen_unix": time.time(),
            "max_hours": spec["max_hours"],
            "source_files_sha256": preparation["source_files_sha256"],
            "input_hashes": {
                **preparation["input_hashes"],
                **{
                    str(root / n): file_hash(root / n)
                    for n in ("preparation-lock.json", "preflight.json")
                },
            },
            "jobs": [
                {
                    "id": "svd-magnitude",
                    "module": "gradient_interp.paper_svd_magnitude",
                    "arguments": ["run", "--root", str(root)],
                    "manifest": str(root / "manifest.json"),
                }
            ],
        },
    )


def verify_run(root):
    plan, controller = read_json(root / "plan.json"), read_json(root / "controller.json")
    if plan["scope"] != "development_only" or controller["status"] != "running":
        raise ValueError("An active development controller is required")
    if controller["plan_sha256"] != file_hash(root / "plan.json"):
        raise ValueError("Controller plan changed")
    if (root / "manifest.json").exists():
        raise ValueError("Completed workers must never resume")
    if abs(controller["deadline_unix"] - controller["started_unix"] - 43200) > 1e-5:
        raise ValueError("The original twelve-hour deadline changed")
    if time.time() >= controller["deadline_unix"]:
        raise ValueError("Original deadline reached")
    verify_files(Path.cwd(), plan["source_files_sha256"])
    verify_files(Path.cwd(), plan["input_hashes"])
    return controller["deadline_unix"]


def run(root):
    deadline, started = verify_run(root), time.time()
    spec, data = read_json(root / "specification.json"), read_json(root / "data.json")
    validate_staged_data(root)
    validate_data(data)
    model, scoring, _, tokenize, payload, parameters = stable_setup(spec)
    validate_architecture(model, parameters, spec["source_config"])
    validate_tokenization(data, scoring, tokenize)
    original = {name: p.detach().clone() for name, p in parameters}
    probes = data["calibration"] + data["evaluation"]
    cache_path, lock_path = root / "forecast-cache.pt", root / "forecast-lock.json"

    def status(stage, completed, total):
        if time.time() >= deadline:
            raise TimeoutError("Original controller deadline reached")
        atomic_json(
            root / "status.json",
            {"stage": stage, "completed": completed, "total": total, "updated_unix": time.time()},
        )

    def loss(row):
        value, count = scalar_logp(model, scoring, tokenize, row["prompt"], row["answer"])
        if count != row["completion_tokens"]:
            raise ValueError(f"Completion count changed: {row['id']}")
        return -value / count, count

    if not lock_path.exists():
        if (root / "outcomes").exists():
            raise ValueError("Actual outcomes exist before the forecast lock")
        gradients = []
        for index, batch in enumerate(data["batches"]):
            per_doc = []
            for row in batch["documents"]:
                value, _ = loss(row)
                per_doc.append(flat_grad(value, parameters))
            gradients.append(torch.stack(per_doc).mean(0))
            status("batch_gradients", index + 1, 32)
        gradients = torch.stack(gradients)
        from gradient_atoms.projection import (
            load_ekfac_eigen,
            project_gradients_ekfac,
            unproject_atom,
        )

        factors = load_ekfac_eigen(
            str(Path(spec["source_root"]) / "factors/paper_ekfac/factors_training")
        )
        projected, info = project_gradients_ekfac(gradients, payload["lora_names"], factors, 50)
        bases, basis_info = load_bases(spec)
        validate_projection_info(info, basis_info["module_info"])
        coefficients, reconstructions = [], []
        for method in REPRESENTATIONS:
            c, x = (
                encode_svd(projected.numpy(), bases[method])
                if method == "svd500"
                else encode(projected.numpy(), bases[method])
            )
            coefficients.append(torch.from_numpy(c))
            reconstructions.append(
                unproject_reconstructions(
                    x, lambda row: unproject_atom(row, info, factors, gradients.shape[1])
                )
            )
        reconstructions = torch.stack((*reconstructions, gradients, torch.zeros_like(gradients)), 1)
        baseline, counts, probe_gradients = [], [], []
        for index, row in enumerate(probes):
            value, count = loss(row)
            baseline.append(float(value.detach()))
            counts.append(count)
            probe_gradients.append(flat_grad(value, parameters))
            status("probe_gradients", index + 1, 80)
        probe_gradients = torch.stack(probe_gradients)
        profiles, component_norms = physical_component_profiles(
            bases,
            probe_gradients,
            lambda x: unproject_atom(x, info, factors, gradients.shape[1]),
            status,
        )
        codes = torch.stack(coefficients, 1)
        contributions = component_contributions(codes.numpy(), profiles)
        forecasts = loss_forecasts(probe_gradients.numpy(), reconstructions.numpy(), RATES)
        np.testing.assert_allclose(contributions.sum(3), forecasts[:, :, :2], rtol=1e-4, atol=1e-7)
        assert_restored(parameters, original)
        saved = {
            "batch_gradients": gradients,
            "projected": projected,
            "coefficients": codes,
            "coefficient_exact_nonzero": torch.count_nonzero(codes, dim=2),
            "coefficient_active_above_1e_minus6": torch.count_nonzero(abs(codes) > 1e-6, dim=2),
            "reconstructed_gradients": reconstructions,
            "probe_gradients": probe_gradients,
            "baseline_mean_ce": torch.tensor(baseline, dtype=torch.float64),
            "completion_tokens": torch.tensor(counts),
            "component_loss_profiles": torch.from_numpy(profiles),
            "component_loss_contributions": torch.from_numpy(contributions),
            "component_physical_norms": torch.from_numpy(component_norms),
            "predicted_mean_ce_change": torch.from_numpy(forecasts),
        }
        validate_cache(saved)
        torch.save(saved, cache_path)
        atomic_json(
            lock_path,
            {
                "locked_unix": time.time(),
                "plan_sha256": file_hash(root / "plan.json"),
                "files_sha256": {cache_path.name: file_hash(cache_path)},
                "methods": list(METHODS),
                "rates": list(RATES),
                "all_forecasts_before_updates": True,
                "original_adapter_unchanged": True,
                "unscaled_basis_used": True,
                "basis_directions_sha256": file_hash(Path(spec["basis_root"]) / "directions.pt"),
                "component_sum_max_abs_error": float(
                    abs(contributions.sum(3) - forecasts[:, :, :2]).max()
                ),
            },
        )
        del factors, reconstructions, probe_gradients, profiles, contributions, bases
    else:
        lock = read_json(lock_path)
        if lock["plan_sha256"] != file_hash(root / "plan.json"):
            raise ValueError("Existing forecast lock changed")
        verify_files(root, lock["files_sha256"])
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        validate_cache(saved)
    outcomes = root / "outcomes"
    outcomes.mkdir(exist_ok=True)
    completed = []
    for b, batch in enumerate(data["batches"]):
        for rate in RATES:
            path = outcomes / f"batch-{b:02d}-rate-{rate:g}.json"
            if path.exists():
                old = read_json(path)
                if (
                    old["forecast_lock_sha256"] != file_hash(lock_path)
                    or not old["adapter_restored"]
                ):
                    raise ValueError("Existing outcome does not match the frozen forecast")
                completed.append(path)
                continue
            status("actual_updates", len(completed), 64)
            update_started = time.time()
            restore(parameters, original)
            set_direction(parameters, original, saved["batch_gradients"][b], rate)
            try:
                values, counts = [], []
                with torch.no_grad():
                    for row in probes:
                        value, count = loss(row)
                        values.append(float(value))
                        counts.append(count)
            finally:
                restore(parameters, original)
            assert_restored(parameters, original)
            if counts != saved["completion_tokens"].tolist() or not np.isfinite(values).all():
                raise ValueError("Actual score changed tokenization or was nonfinite")
            atomic_json(
                path,
                {
                    "plan_sha256": file_hash(root / "plan.json"),
                    "forecast_lock_sha256": file_hash(lock_path),
                    "batch_index": b,
                    "batch_id": batch["id"],
                    "rate": rate,
                    "update_started_unix": update_started,
                    "completed_unix": time.time(),
                    "mean_completion_ce": values,
                    "completion_tokens": counts,
                    "adapter_restored": True,
                },
            )
            completed.append(path)
    if len(completed) != 64:
        raise ValueError("Incomplete actual update grid")
    assert_restored(parameters, original)
    status("complete", 64, 64)
    atomic_json(
        root / "manifest.json",
        {
            "started_unix": started,
            "completed_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "forecast_lock_sha256": file_hash(lock_path),
            "actual_updates": 64,
            "probes": 80,
            "adapter_restored": True,
            "original_deadline_unix": deadline,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "claim_scope": spec["limits"],
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p) for p in [cache_path, lock_path, *completed]
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "freeze", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--basis-root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-audit", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if any(v is None for v in (args.basis_root, args.data, args.data_audit)):
            parser.error("prepare requires --basis-root, --data and --data-audit")
        prepare(args.root, args.basis_root, args.data, args.data_audit)
    else:
        {"preflight": preflight, "freeze": freeze, "run": run}[args.command](args.root)


if __name__ == "__main__":
    main()
