"""Prospective loss-effect prediction with two dictionary training geometries.

This experiment predicts teacher-forced mean completion CE. It does not infer
unrestricted generation behavior from a few fixed answers or proposal samples.
"""

import argparse
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode

from .confirmation import verify_files
from .paper_batch import flat_grad, scalar_logp
from .paper_mixture import stable_setup
from .paper_setup import read_docs, read_json, validate_layout
from .paper_steering import restore, set_direction
from .study import atomic_json, file_hash

ARMS = ("normalized_fit", "magnitude_fit")
METHODS = (*ARMS, "full_gradient", "zero")
COUNTS = {"batches": 32, "batch_size": 8, "calibration": 16, "evaluation": 64}
RATES = (0.0003, 0.003)
LEARNER = {"n_components": 500, "alpha": 0.1, "batch_size": 256, "max_iter": 100}


def matrix(value, name):
    x = np.asarray(value)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError(f"Invalid finite matrix: {name}")
    return x


def learning_rows(projected, arm):
    """Return fit rows and their divisor; only one arm removes row magnitudes."""
    x = matrix(projected, "fitting projections")
    norms = np.linalg.norm(x, axis=1)
    if arm == "normalized_fit":
        divisors = np.maximum(norms, 1e-8)
    elif arm == "magnitude_fit":
        positive = norms[norms > 0]
        if not len(positive):
            raise ValueError("All fitting gradients are zero")
        # The fitting-only global median fixes units for alpha=.1. It does not
        # remove the relative magnitude of any pair of nonzero fitting rows.
        divisors = np.full(len(x), np.median(positive), dtype=x.dtype)
    else:
        raise ValueError(f"Unknown fitting arm: {arm}")
    return x / divisors[:, None], divisors


def encode(projected, dictionary, alpha=0.1):
    """Both arms use the same unit-row Lasso, then restore each batch norm.

    The contrast is dictionary *learning* geometry. A common encoder avoids mixing
    that contrast with an inference penalty change or with the difference between
    summed fitting CE and mean-token batch CE. Returned coefficients carry scale.
    """
    x, d = matrix(projected, "batch projections"), matrix(dictionary, "dictionary")
    if x.shape[1] != d.shape[1] or alpha <= 0:
        raise ValueError("Dictionary dimension or alpha mismatch")
    norms = np.linalg.norm(x, axis=1)
    direction = x / np.maximum(norms[:, None], 1e-12)
    codes = sparse_encode(direction, d, algorithm="lasso_lars", alpha=alpha)
    scaled = codes * norms[:, None]
    return scaled, scaled @ d


def loss_forecasts(probe_gradients, reconstructed_gradients, rates=RATES):
    """Grad L dot negative SGD update; return B,R,M,Q signed CE changes."""
    q = matrix(probe_gradients, "probe loss gradients").astype(np.float64)
    g = np.asarray(reconstructed_gradients, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    if g.ndim != 3 or g.shape[2] != q.shape[1] or not np.isfinite(g).all():
        raise ValueError("Expected finite B,M,P reconstructed gradients")
    if rates.ndim != 1 or not len(rates) or np.any(rates <= 0) or not np.isfinite(rates).all():
        raise ValueError("Rates must be finite and positive")
    return -np.einsum("qp,bmp,r->brmq", q, g, rates, optimize=True)


def error_metrics(predicted, actual):
    """Keep weak effects and undefined relative errors visible."""
    p, y = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if p.shape != y.shape or p.size == 0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Expected matching finite forecast and target arrays")
    residual = p - y
    mse, zero_mse = float(np.mean(residual**2)), float(np.mean(y**2))
    actual_rms = np.sqrt(zero_mse)
    slope = float(np.sum(p * y) / np.sum(p * p)) if np.any(p) else None
    return {
        "rmse_nats_per_token": float(np.sqrt(mse)),
        "mae_nats_per_token": float(np.mean(abs(residual))),
        "actual_rms_nats_per_token": float(actual_rms),
        "nrmse_vs_zero": float(np.sqrt(mse / zero_mse)) if zero_mse else None,
        "mse_difference_vs_zero": mse - zero_mse,
        "actual_on_predicted_slope_through_origin": slope,
        "mean_predicted_change": float(np.mean(p)),
        "mean_actual_change": float(np.mean(y)),
        "worst_absolute_error": float(np.max(abs(residual))),
        "meaningful_effect_gate": bool(actual_rms >= 0.01),
    }


def bootstrap_errors(predictions, actual, replicates=1000, seed=2026101409):
    """Paired B,R,M,Q forecasts and B,R,Q targets, with independent B/Q draws.

    The same batch and probe indices apply to every method and rate. These
    intervals describe this development sample, not a fresh-seed population.
    """
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if p.ndim != 4 or p.shape[2] < 2 or y.shape != (p.shape[0], p.shape[1], p.shape[3]):
        raise ValueError("Expected B,R,M,Q forecasts and B,R,Q targets")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or replicates < 2:
        raise ValueError("Expected finite inputs and at least two bootstrap replicates")
    rng = np.random.default_rng(seed)
    result = {
        key: []
        for key in (
            "nrmse",
            "mae",
            "actual_rms",
            "mse_difference_vs_zero",
            "mse_difference_magnitude_minus_normalized",
        )
    }
    for _ in range(replicates):
        bi = rng.integers(p.shape[0], size=p.shape[0])
        qi = rng.integers(p.shape[3], size=p.shape[3])
        draw_p, draw_y = p[bi][..., qi], y[bi][..., qi]
        mse = np.mean((draw_p - draw_y[:, :, None]) ** 2, axis=(0, 3))
        zero = np.mean(draw_y**2, axis=(0, 2))
        relative = np.full_like(mse, np.nan)
        np.divide(mse, zero[:, None], out=relative, where=zero[:, None] > 0)
        result["nrmse"].append(np.sqrt(relative))
        result["mae"].append(np.mean(abs(draw_p - draw_y[:, :, None]), axis=(0, 3)))
        result["actual_rms"].append(np.sqrt(zero))
        result["mse_difference_vs_zero"].append(mse - zero[:, None])
        result["mse_difference_magnitude_minus_normalized"].append(mse[:, 1] - mse[:, 0])
    return {key: np.asarray(value) for key, value in result.items()}


def bootstrap_interval(draws, family_size=1):
    """Bonferroni pointwise or family intervals; never discard undefined draws."""
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim != 1 or family_size < 1:
        raise ValueError("An individual bootstrap estimand and positive family size are required")
    undefined = int(np.count_nonzero(~np.isfinite(x)))
    tail = 0.025 / family_size
    bounds = None if undefined else np.quantile(x, [tail, 1 - tail]).tolist()
    return {
        "interval": bounds,
        "undefined_replicates": undefined,
        "family_size": family_size,
        "marginal_coverage": 1 - 2 * tail,
    }


def canonical(text):
    return " ".join(text.casefold().split())


def validate_data(data, expected=COUNTS):
    if len(data["batches"]) != expected["batches"]:
        raise ValueError("Wrong batch count")
    rows = []
    for batch in data["batches"]:
        if len(batch["documents"]) != expected["batch_size"]:
            raise ValueError("Wrong batch size")
        rows.extend(("batch", row) for row in batch["documents"])
    for split in ("calibration", "evaluation"):
        if len(data[split]) != expected[split]:
            raise ValueError(f"Wrong {split} count")
        rows.extend((split, row) for row in data[split])
    ids, prompts, groups = set(), set(), {}
    for split, row in rows:
        for key in ("id", "prompt", "answer", "source_id", "cluster_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing nonempty document field: {key}")
        if (
            not isinstance(row.get("full_tokens"), int)
            or not isinstance(row.get("completion_tokens"), int)
            or not 0 < row["completion_tokens"] < row["full_tokens"] <= 500
        ):
            raise ValueError("Missing or invalid audited untruncated token counts")
        if row["id"] in ids or canonical(row["prompt"]) in prompts:
            raise ValueError("Duplicate document id or normalized prompt")
        ids.add(row["id"])
        prompts.add(canonical(row["prompt"]))
        for field in ("source_id", "cluster_id", "template_id"):
            if field == "template_id" and not row.get(field):
                continue
            key = field, row[field]
            if key in groups and groups[key] != split:
                raise ValueError(f"Cross-split {field} overlap")
            groups[key] = split
    batch_ids = [b["id"] for b in data["batches"]]
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("Duplicate batch id")
    return rows


def validate_dedup_audit(audit, data_path):
    if audit.get("data_sha256") != file_hash(data_path):
        raise ValueError("Dedup audit does not match the data")
    required = (
        "prior_input_files_sha256",
        "source_deduplication",
        "near_duplicate_rule",
        "template_split_policy",
        "scope",
    )
    if any(not audit.get(k) for k in required):
        raise ValueError("Incomplete historical source/template dedup audit")
    for field in ("exact_prompt_collisions", "source_collisions", "near_duplicate_collisions"):
        if field not in audit or audit[field]:
            raise ValueError(f"Missing or failed dedup check: {field}")


def prepare(root, source, data_path, audit_path):
    """Freeze the scientific contrast before fitting the magnitude dictionary."""
    root, source = Path(root).resolve(), Path(source).resolve()
    if (root / "specification.json").exists():
        raise ValueError("This root is already prepared")
    integrity = read_json(source / "integrity.json")
    if not integrity["complete"]:
        raise ValueError("A completed source audit is required")
    data, audit = read_json(data_path), read_json(audit_path)
    validate_data(data)
    validate_dedup_audit(audit, Path(data_path))
    config = read_json(source / "configuration.json")
    for key, value in (
        ("n_atoms", 500),
        ("dictionary_alpha", 0.1),
        ("dictionary_batch_size", 256),
        ("dictionary_max_iter", 100),
        ("lora_tensors", 136),
        ("lora_parameters", 2228224),
    ):
        if config[key] != value:
            raise ValueError(f"The paper source configuration differs: {key}")
    # Verify against prior audited hashes, rather than accepting any current
    # dictionary/factor/checkpoint content merely because its file exists.
    original_hashes = {}
    for stage in ("train", "factors", "atoms"):
        manifest_path = source / f"{stage}-manifest.json"
        if file_hash(manifest_path) != integrity["stages"][stage]["manifest_sha256"]:
            raise ValueError(f"Original audited {stage} manifest changed")
        hashes = read_json(manifest_path)["files_sha256"]
        verify_files(source, hashes)
        original_hashes.update({str(source / n): h for n, h in hashes.items()})
    if (
        file_hash(source / "configuration.json")
        != read_json(source / "plan.json")["configuration_sha256"]
    ):
        raise ValueError("Original source configuration changed")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "data.json", data)
    audit["staged_data_sha256"] = file_hash(root / "data.json")
    atomic_json(root / "data-audit.json", audit)
    inputs = [
        source / p
        for p in (
            "configuration.json",
            "integrity.json",
            "atoms.pt",
            "training.jsonl",
            "factors-manifest.json",
            "atoms-manifest.json",
            "train-manifest.json",
            "plan.json",
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
        )
    ]
    spec = {
        "scope": "development_only",
        "source_root": str(source),
        "source_config": config,
        "training_seed": config["seed"],
        "max_hours": 12,
        "rates": list(RATES),
        "methods": list(METHODS),
        "counts": {**COUNTS, "actual_updates": 64},
        "learner": {
            **LEARNER,
            "random_state": config["seed"],
            "transform_algorithm": "lasso_lars",
            "n_jobs": 8,
        },
        "fitting_gradient_convention": "Original per-document summed completion CE gradients.",
        "batch_gradient_convention": "Mean of eight per-document mean-token completion CE "
        "gradients.",
        "probe_loss": "Per-document mean-token completion CE with the original training EOS "
        "convention.",
        "normalized_fit": "Reuse the audited original 500-atom normalized fitting dictionary.",
        "magnitude_fit": "Fit the same 500-atom learner on projected fitting rows divided by one "
        "positive-row median norm, computed from fitting rows only. Relative row "
        "magnitudes remain. Alpha, seed and all other learning defaults are fixed.",
        "encoding": "Both arms encode unit projected batch directions with alpha .1 and restore "
        "each projected batch norm. This isolates dictionary learning geometry and "
        "does not confuse summed fitting CE with mean-token batch CE scale.",
        "primary": "Prospective per-batch/per-probe teacher-forced mean completion-CE change on "
        "64 evaluation documents. Exact pre-update probe gradients contracted with "
        "each reconstructed batch gradient. No outcome regression or fitted scale.",
        "calibration": "Sixteen separate probe documents. Report predictions and actual changes "
        "descriptively, but never use them to select methods, rates or rescale predictions.",
        "evaluation": "All forecasts and original-adapter assertions locked before any of the "
        "64 actual full-gradient SGD updates. The same actual update targets both "
        "dictionary arms. Restore all 136 LoRA tensors after every condition.",
        "criteria": {
            "minimum_actual_rms_nats_per_token": 0.01,
            "maximum_nrmse": 0.5,
            "maximum_mae_nats_per_token": 0.01,
            "rule": "Report both arms and both rates in both reused seeds. A strong "
            "phase-one loss-prediction result requires the meaningful-effect "
            "gate and both error thresholds in every seed/rate, using simultaneous "
            "intervals across two arms and two rates per seed. Weak effects are "
            "inconclusive. No winner promotion.",
        },
        "analysis": "Wait for both runs. Audit full grids, source and output hashes, locks and "
        "restores. Report all per-probe/per-batch errors and tails. Independently "
        "bootstrap batches and probe source clusters, preserving pairing across "
        "arms and rates, 1000 replicates, seed 2026101409. Pointwise 95% intervals and "
        "Bonferroni simultaneous intervals across two arms and two rates per seed. "
        "No sampling-MC uncertainty for exact CE. "
        "Only development evidence on reused checkpoints. Do not infer interpretability "
        "from forecast accuracy or reinterpret CE prediction as generated behavior.",
        "limits": "This focused phase makes no unrestricted sampled-behavior prediction claim. "
        "It does not repeat failed proposal reweighting or expand proposal banks. "
        "A separate frozen study is required before any generation-behavior claim.",
        "input_hashes": {**original_hashes, **{str(p): file_hash(p) for p in inputs}},
    }
    atomic_json(root / "specification.json", spec)
    atomic_json(
        root / "fit-plan.json",
        {
            "written_unix": time.time(),
            "source_files_sha256": {
                str(p): file_hash(p) for p in Path("src/gradient_interp").glob("*.py")
            },
            "input_hashes": {
                **spec["input_hashes"],
                **{
                    str(root / n): file_hash(root / n)
                    for n in ("specification.json", "data.json", "data-audit.json")
                },
            },
            "new_magnitude_fit_already_computed": False,
        },
    )


def verify_fit_plan(root):
    plan = read_json(root / "fit-plan.json")
    if plan["new_magnitude_fit_already_computed"]:
        raise ValueError("Magnitude fitting must follow its frozen plan")
    verify_files(Path.cwd(), plan["source_files_sha256"])
    verify_files(Path.cwd(), plan["input_hashes"])
    return read_json(root / "specification.json")


def fit(root):
    spec = verify_fit_plan(root)
    if (root / "dictionaries.npz").exists() or (root / "fit.json").exists():
        raise ValueError("Dictionary fit already exists; preserve it and never overwrite")
    started = time.time()
    payload = torch.load(
        Path(spec["source_root"]) / "atoms.pt", map_location="cpu", weights_only=True
    )
    projected = payload["projected_gradients"].numpy()
    if projected.shape != (5000, 6800) or payload["dictionary"].shape != (500, 6800):
        raise ValueError("The audited fitting geometry differs from the paper")
    rows, scales = learning_rows(projected, "magnitude_fit")
    learner = MiniBatchDictionaryLearning(**spec["learner"], verbose=1)
    learner.fit(rows)
    dictionary = learner.components_
    if not np.isfinite(dictionary).all():
        raise ValueError("Nonfinite magnitude dictionary")
    output = root / "dictionaries.npz"
    np.savez_compressed(
        output,
        normalized_fit=payload["dictionary"].numpy(),
        magnitude_fit=dictionary,
        fitting_global_scale=scales[0],
    )
    atomic_json(
        root / "fit.json",
        {
            "started_unix": started,
            "completed_unix": time.time(),
            "fit_plan_sha256": file_hash(root / "fit-plan.json"),
            "dictionaries_sha256": file_hash(output),
            "original_dictionary_reused": True,
            "fitting_rows": len(rows),
            "fitting_columns": rows.shape[1],
            "magnitude_global_scale": float(scales[0]),
            "row_norm_quantiles_before_global_scaling": np.quantile(
                np.linalg.norm(projected, axis=1), [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
            "learner_parameters": learner.get_params(),
            "iterations": int(learner.n_iter_),
            "sklearn_version": importlib.metadata.version("scikit-learn"),
            "only_fitting_rows_used": True,
            "new_batch_outcomes_read": False,
        },
    )


def fit_verified(root):
    spec = verify_fit_plan(root)
    audit = read_json(root / "fit.json")
    if audit["fit_plan_sha256"] != file_hash(root / "fit-plan.json"):
        raise ValueError("Fit plan changed")
    if audit["dictionaries_sha256"] != file_hash(root / "dictionaries.npz"):
        raise ValueError("Dictionary file changed")
    return spec


def training_example(source):
    # Preflight uses an already-seen fitting item, not a held-out evaluation row.
    for doc in read_docs(source / "training.jsonl"):
        messages = doc["messages"]
        for user, assistant in zip(messages, messages[1:]):
            if user["role"] == "user" and assistant["role"] == "assistant":
                return {"prompt": user["content"], "answer": assistant["content"]}
    raise ValueError("No training pair available for the numerical preflight")


def assert_restored(parameters, original):
    if any(not torch.equal(p, original[name]) for name, p in parameters):
        raise ValueError("Original adapter was not restored exactly")


def validate_architecture(model, parameters, config):
    layout = validate_layout(model.named_parameters(), config)
    if [row["name"] for row in layout] != [name for name, _ in parameters]:
        raise ValueError("Cached LoRA ordering differs from the all-layer model layout")
    if len(parameters) != 136 or sum(p.numel() for _, p in parameters) != 2228224:
        raise ValueError("Incorrect all-layer q/v LoRA parameter layout")
    return layout


def validate_cache(saved):
    for name, value in saved.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite forecast cache tensor: {name}")


def validate_staged_data(root):
    audit = read_json(root / "data-audit.json")
    if audit.get("staged_data_sha256") != file_hash(root / "data.json"):
        raise ValueError("Staged scientific data differ from the dedup audit")


def validate_tokenization(data, tokenizer, tokenize):
    rows = [row for batch in data["batches"] for row in batch["documents"]]
    rows += data["calibration"] + data["evaluation"]
    for row in rows:
        chat = {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["answer"]},
            ]
        }
        full = tokenize(chat, tokenizer, max_length=None)
        capped = tokenize(chat, tokenizer, max_length=500)
        count = sum(label != -100 for label in full["labels"][1:])
        if (
            full != capped
            or len(full["input_ids"]) != row["full_tokens"]
            or count != row["completion_tokens"]
        ):
            raise ValueError(
                f"Scoring tokenization differs from the untruncated audit: {row['id']}"
            )
    return {
        "documents": len(rows),
        "maximum_full_tokens": max(r["full_tokens"] for r in rows),
        "all_targets_untruncated": True,
        "original_eos_convention": True,
    }


def preflight(root):
    spec = fit_verified(root)
    if (root / "preflight.json").exists():
        raise ValueError("Preflight already exists")
    model, scoring, _, tokenize, payload, parameters = stable_setup(spec)
    layout = validate_architecture(model, parameters, spec["source_config"])
    tokenization = validate_tokenization(read_json(root / "data.json"), scoring, tokenize)
    original = {name: p.detach().clone() for name, p in parameters}
    row = training_example(Path(spec["source_root"]))
    value, n = scalar_logp(model, scoring, tokenize, row["prompt"], row["answer"])
    gradient = flat_grad(-value / n, parameters)
    baseline = float(-value.detach() / n)
    del value
    # Exercise the full scoring cap using only an already-seen training answer.
    long_answer = (row["answer"] + " ") * 64
    long_value, long_count = scalar_logp(model, scoring, tokenize, row["prompt"], long_answer)
    long_gradient = flat_grad(-long_value / long_count, parameters)
    if not torch.isfinite(long_gradient).all():
        raise ValueError("Nonfinite long training-completion gradient")
    del long_value, long_gradient
    set_direction(parameters, original, gradient, spec["rates"][0])
    try:
        if all(torch.equal(p, original[name]) for name, p in parameters):
            raise ValueError("The training preflight update did not change any adapter tensor")
        with torch.no_grad():
            changed, count = scalar_logp(model, scoring, tokenize, row["prompt"], row["answer"])
        if count != n or not torch.isfinite(changed):
            raise ValueError("Invalid preflight update scoring")
    finally:
        restore(parameters, original)
    assert_restored(parameters, original)
    with torch.no_grad():
        restored, restored_n = scalar_logp(model, scoring, tokenize, row["prompt"], row["answer"])
    if restored_n != n or float(-restored / n) != baseline:
        raise ValueError("Teacher-forced scoring was not restored exactly")
    from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac, unproject_atom

    factors = load_ekfac_eigen(
        str(Path(spec["source_root"]) / "factors/paper_ekfac/factors_training")
    )
    projected, info = project_gradients_ekfac(gradient[None], payload["lora_names"], factors, 50)
    dictionaries = np.load(root / "dictionaries.npz")
    for arm in ARMS:
        _, reconstruction = encode(projected.numpy(), dictionaries[arm])
        lifted = unproject_atom(reconstruction[0], info, factors, gradient.numel())
        if lifted.shape != gradient.shape or not torch.isfinite(lifted).all():
            raise ValueError("Invalid reconstructed training gradient")
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "fit_sha256": file_hash(root / "fit.json"),
            "finite_gradient": True,
            "adapter_restored": True,
            "scoring_restored": True,
            "tensors": len(parameters),
            "parameter_count": gradient.numel(),
            "layout_verified": True,
            "layout": layout,
            "tokenization": tokenization,
            "training_input_only": True,
            "long_training_completion_checked": True,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def freeze(root):
    spec = fit_verified(root)
    if (root / "plan.json").exists():
        raise ValueError("The controller plan is already frozen")
    pre = read_json(root / "preflight.json")
    if pre["specification_sha256"] != file_hash(root / "specification.json"):
        raise ValueError("Specification changed since preflight")
    if pre["fit_sha256"] != file_hash(root / "fit.json"):
        raise ValueError("Dictionary fit changed since preflight")
    if not all(
        pre[k]
        for k in (
            "adapter_restored",
            "scoring_restored",
            "finite_gradient",
            "layout_verified",
            "training_input_only",
            "long_training_completion_checked",
        )
    ):
        raise ValueError("Preflight did not pass")
    if pre["tensors"] != 136 or pre["parameter_count"] != 2228224:
        raise ValueError("Preflight did not use the full paper LoRA architecture")
    if (
        pre["tokenization"]["documents"] != 336
        or not pre["tokenization"]["all_targets_untruncated"]
        or not pre["tokenization"]["original_eos_convention"]
    ):
        raise ValueError("Preflight did not verify the full untruncated data tokenization")
    validate_staged_data(root)
    atomic_json(
        root / "plan.json",
        {
            "scope": "development_only",
            "frozen_unix": time.time(),
            "max_hours": spec["max_hours"],
            "source_files_sha256": {
                str(p): file_hash(p) for p in Path("src/gradient_interp").glob("*.py")
            },
            "input_hashes": {
                **spec["input_hashes"],
                **{
                    str(root / n): file_hash(root / n)
                    for n in (
                        "specification.json",
                        "data.json",
                        "data-audit.json",
                        "fit-plan.json",
                        "fit.json",
                        "dictionaries.npz",
                        "preflight.json",
                    )
                },
            },
            "jobs": [
                {
                    "id": "magnitude",
                    "module": "gradient_interp.paper_magnitude",
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
        raise ValueError("Controller plan mismatch")
    if (root / "manifest.json").exists():
        raise ValueError("Completed scientific workers must never resume")
    verify_files(Path.cwd(), plan["source_files_sha256"])
    verify_files(Path.cwd(), plan["input_hashes"])
    if time.time() >= controller["deadline_unix"]:
        raise ValueError("The immutable deadline has passed")
    return controller["deadline_unix"]


def run(root):
    deadline = verify_run(root)
    started = time.time()
    spec, data = read_json(root / "specification.json"), read_json(root / "data.json")
    validate_staged_data(root)
    validate_data(data)
    model, scoring, _, tokenize, payload, parameters = stable_setup(spec)
    validate_architecture(model, parameters, spec["source_config"])
    validate_tokenization(data, scoring, tokenize)
    original = {name: p.detach().clone() for name, p in parameters}
    probes = data["calibration"] + data["evaluation"]
    cache, lock = root / "forecast-cache.pt", root / "forecast-lock.json"

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
            raise ValueError(f"Actual score token count changed: {row['id']}")
        return -value / count, count

    if not lock.exists():
        if (root / "outcomes").exists():
            raise ValueError("Outcomes exist before the forecast lock")
        gradients = []
        for index, batch in enumerate(data["batches"]):
            per_doc = []
            for row in batch["documents"]:
                value, _ = loss(row)
                per_doc.append(flat_grad(value, parameters))
            gradients.append(torch.stack(per_doc).mean(0))
            status("batch_gradients", index + 1, len(data["batches"]))
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
        dictionaries = np.load(root / "dictionaries.npz")
        reconstructions, coefficients = [], []
        for arm in ARMS:
            code, reconstructed = encode(projected.numpy(), dictionaries[arm])
            coefficients.append(torch.from_numpy(code))
            reconstructions.append(
                torch.stack(
                    [
                        unproject_atom(x, info, factors, gradients.shape[1]).float()
                        for x in reconstructed
                    ]
                )
            )
        reconstructions = torch.stack((*reconstructions, gradients, torch.zeros_like(gradients)), 1)
        baseline, token_counts, jacobians = [], [], []
        for index, row in enumerate(probes):
            value, count = loss(row)
            baseline.append(float(value.detach()))
            token_counts.append(count)
            jacobians.append(flat_grad(value, parameters))
            status("probe_gradients", index + 1, len(probes))
        jacobians = torch.stack(jacobians)
        forecasts = loss_forecasts(jacobians.numpy(), reconstructions.numpy(), spec["rates"])
        assert_restored(parameters, original)
        saved = {
            "batch_gradients": gradients,
            "projected": projected,
            "coefficients": torch.stack(coefficients, 1),
            "reconstructed_gradients": reconstructions,
            "probe_gradients": jacobians,
            "baseline_mean_ce": torch.tensor(baseline, dtype=torch.float64),
            "completion_tokens": torch.tensor(token_counts),
            "predicted_mean_ce_change": torch.from_numpy(forecasts),
        }
        validate_cache(saved)
        torch.save(saved, cache)
        atomic_json(
            lock,
            {
                "locked_unix": time.time(),
                "plan_sha256": file_hash(root / "plan.json"),
                "files_sha256": {cache.name: file_hash(cache)},
                "methods": list(METHODS),
                "all_forecasts_before_updates": True,
                "original_adapter_unchanged": True,
                "dictionaries_sha256": file_hash(root / "dictionaries.npz"),
            },
        )
        del factors, reconstructions, jacobians
    else:
        recorded = read_json(lock)
        if recorded["plan_sha256"] != file_hash(root / "plan.json"):
            raise ValueError("Forecast lock plan changed")
        verify_files(root, recorded["files_sha256"])
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        validate_cache(saved)
    outcomes = root / "outcomes"
    outcomes.mkdir(exist_ok=True)
    completed = []
    for batch_index, batch in enumerate(data["batches"]):
        for rate in spec["rates"]:
            path = outcomes / f"batch-{batch_index:02d}-rate-{rate:g}.json"
            if path.exists():
                old = read_json(path)
                if old["forecast_lock_sha256"] != file_hash(lock) or not old["adapter_restored"]:
                    raise ValueError("Existing outcome does not match the frozen forecast")
                completed.append(path)
                continue
            status("actual_updates", len(completed), spec["counts"]["actual_updates"])
            update_started = time.time()
            restore(parameters, original)
            set_direction(parameters, original, saved["batch_gradients"][batch_index], rate)
            try:
                scores, counts = [], []
                with torch.no_grad():
                    for row in probes:
                        value, count = loss(row)
                        scores.append(float(value))
                        counts.append(count)
            finally:
                restore(parameters, original)
            assert_restored(parameters, original)
            if counts != saved["completion_tokens"].tolist() or not np.isfinite(scores).all():
                raise ValueError("Actual scoring changed tokenization or produced nonfinite values")
            atomic_json(
                path,
                {
                    "plan_sha256": file_hash(root / "plan.json"),
                    "forecast_lock_sha256": file_hash(lock),
                    "batch_index": batch_index,
                    "batch_id": batch["id"],
                    "rate": rate,
                    "update_started_unix": update_started,
                    "completed_unix": time.time(),
                    "mean_completion_ce": scores,
                    "completion_tokens": counts,
                    "adapter_restored": True,
                },
            )
            completed.append(path)
    if len(completed) != spec["counts"]["actual_updates"]:
        raise ValueError("The actual update grid is incomplete")
    assert_restored(parameters, original)
    status("complete", len(completed), len(completed))
    atomic_json(
        root / "manifest.json",
        {
            "started_unix": started,
            "completed_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "forecast_lock_sha256": file_hash(lock),
            "actual_updates": len(completed),
            "probes": len(probes),
            "adapter_restored": True,
            "original_deadline_unix": deadline,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p) for p in [cache, lock, *completed]
            },
            "claim_scope": "Teacher-forced mean completion-CE effect size; no generation claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit", "preflight", "freeze", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-audit", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = args.root.resolve()
    if args.command == "prepare":
        if not all((args.source, args.data, args.data_audit)):
            parser.error("prepare requires --source, --data and --data-audit")
        prepare(root, args.source, args.data, args.data_audit)
    else:
        globals()[args.command](root)


if __name__ == "__main__":
    main()
