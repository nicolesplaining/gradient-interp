"""CPU audit and compact export after both magnitude runs release their outputs.

Run this file from an ignored analysis-tools directory on the original checkout.
The receipt command reads only controller, plan and completion metadata. Scientific
cache and outcome files are opened only after both explicit receipts pass.
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from gradient_interp.confirmation import verify_files
from gradient_interp.paper_magnitude import (
    ARMS,
    COUNTS,
    METHODS,
    RATES,
    encode,
    learning_rows,
    loss_forecasts,
    validate_cache,
    validate_data,
    validate_staged_data,
    validate_tokenization,
)
from gradient_interp.paper_setup import read_json, upstream
from gradient_interp.study import atomic_json, file_hash

RECEIPT_TYPE = "paper_magnitude_completion_receipt"
FROZEN_COMMIT = "e12ecc1989531ce48b1352e2f76985c6a7cf1a23"
PRELIMINARY_QUOTE = "wait so we have the magnitude run now right? can we analyze those results?"


def process_dead(pid):
    if pid is None:
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    return False


def validate_receipt(receipt):
    if receipt.get("type") != RECEIPT_TYPE or receipt.get("seed") not in (42, 43):
        raise ValueError("Wrong track or seed completion receipt")
    start, deadline, observed = [
        receipt[k] for k in ("started_unix", "deadline_unix", "observed_unix")
    ]
    if abs(deadline - start - 43200) > 0.01 or not receipt.get("all_processes_dead"):
        raise ValueError("Receipt does not establish original deadline and dead processes")
    if receipt.get("release_reason") == "complete":
        if (
            receipt.get("status") != "complete"
            or not receipt.get("manifest_sha256")
            or not start < receipt.get("completed_unix", 0) <= observed
            or receipt["completed_unix"] > deadline
        ):
            raise ValueError("Invalid completed-run receipt")
    elif receipt.get("release_reason") == "original_deadline":
        if observed < deadline:
            raise ValueError("An early failure cannot release scientific outputs")
    else:
        raise ValueError("Completion or original deadline required before outcome access")
    for key in ("controller_sha256", "plan_sha256", "specification_sha256"):
        if not isinstance(receipt.get(key), str) or len(receipt[key]) != 64:
            raise ValueError(f"Missing receipt hash: {key}")
    return receipt


def receipt(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Preserve the existing completion receipt")
    controller, plan, spec = [
        read_json(root / n) for n in ("controller.json", "plan.json", "specification.json")
    ]
    if controller["plan_sha256"] != file_hash(root / "plan.json") or plan["max_hours"] != 12:
        raise ValueError("Controller or immutable phase duration differs")
    pids = {controller.get("controller_pid"), controller.get("worker_pid")}
    pids.update(a.get("pid") for a in controller.get("attempts", []))
    now = time.time()
    result = dict(
        type=RECEIPT_TYPE,
        seed=spec["training_seed"],
        root=str(root),
        observed_unix=now,
        status=controller["status"],
        started_unix=controller["started_unix"],
        deadline_unix=controller["deadline_unix"],
        all_processes_dead=all(process_dead(pid) for pid in pids),
        checked_pids=sorted(pid for pid in pids if pid is not None),
        controller_sha256=file_hash(root / "controller.json"),
        plan_sha256=file_hash(root / "plan.json"),
        specification_sha256=file_hash(root / "specification.json"),
    )
    if controller["status"] == "complete":
        manifest = read_json(root / "manifest.json")
        result.update(
            release_reason="complete",
            completed_unix=manifest["completed_unix"],
            manifest_sha256=file_hash(root / "manifest.json"),
        )
    else:
        result["release_reason"] = "original_deadline"
    validate_receipt(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, result)
    return result


def require_barrier(receipts, analysis_plan):
    """This function must run before reading any forecast or scientific outcome."""
    if len(receipts) != 2:
        raise ValueError("Explicit receipts from both seeds are required")
    values = [validate_receipt(read_json(path)) for path in receipts]
    if {value["seed"] for value in values} != {42, 43}:
        raise ValueError("Both distinct seeds must release their outputs")
    plan = read_json(analysis_plan)
    if (
        plan.get("scientific_outcomes_read") is not False
        or plan.get("source_commit") != FROZEN_COMMIT
        or any(plan["written_unix"] >= value["observed_unix"] for value in values)
    ):
        raise ValueError("The dated pre-reading analysis plan is missing or too late")
    return values


def require_preliminary_receipt(receipt_path, analysis_plan, amendment_path):
    """Separate user-authorized early reading, never a replacement two-seed receipt."""
    value = validate_receipt(read_json(receipt_path))
    plan, amendment = read_json(analysis_plan), read_json(amendment_path)
    if (
        value["seed"] != 43
        or value["release_reason"] != "complete"
        or plan.get("scientific_outcomes_read") is not False
        or plan.get("source_commit") != FROZEN_COMMIT
        or amendment.get("type") != "paper_magnitude_preliminary_authorization"
        or amendment.get("scope") != "preliminary_one_seed"
        or amendment.get("seed") != 43
        or amendment.get("user_quote") != PRELIMINARY_QUOTE
        or amendment.get("source_commit") != FROZEN_COMMIT
        or amendment.get("analysis_plan_sha256") != file_hash(analysis_plan)
        or amendment.get("scientific_outcomes_read_before_amendment") is not False
        or not plan["written_unix"] < amendment.get("written_unix", 0) <= time.time()
        or any(
            amendment.get(key) is not True
            for key in (
                "criteria_unchanged",
                "workers_unchanged",
                "remaining_seed_required_for_full_conclusion",
            )
        )
    ):
        raise ValueError("Explicit dated user authorization and complete seed43 receipt required")
    return [value]


def verify_archive(root, hashes):
    with tarfile.open(root / "source_snapshot.tar") as archive:
        for name, expected in hashes.items():
            extracted = archive.extractfile(name)
            if extracted is None or hashlib.sha256(extracted.read()).hexdigest() != expected:
                raise ValueError(f"Frozen source archive differs: {name}")


def norm_diagnostics(cache, dictionaries):
    original = cache["batch_gradients"].double().numpy()
    reconstructed = cache["reconstructed_gradients"].double().numpy()
    norm = np.linalg.norm(original, axis=1)
    reconstructed_norm = np.linalg.norm(reconstructed, axis=2)
    error = np.linalg.norm(reconstructed - original[:, None], axis=2)
    dot = np.einsum("bp,bmp->bm", original, reconstructed)
    relative = np.full_like(error, np.nan)
    cosine = np.full_like(error, np.nan)
    np.divide(error, norm[:, None], out=relative, where=norm[:, None] > 0)
    np.divide(
        dot,
        norm[:, None] * reconstructed_norm,
        out=cosine,
        where=norm[:, None] * reconstructed_norm > 0,
    )
    projected = cache["projected"].double().numpy()
    projected_norm = np.linalg.norm(projected, axis=1)
    coefficients = cache["coefficients"].double().numpy()
    projected_error, projected_gain = [], []
    for mi, arm in enumerate(ARMS):
        approximation = coefficients[:, mi] @ dictionaries[arm]
        denominator = np.sum(projected**2, axis=1)
        projected_error.append(
            np.divide(
                np.linalg.norm(approximation - projected, axis=1),
                projected_norm,
                out=np.full(len(projected), np.nan),
                where=projected_norm > 0,
            )
        )
        projected_gain.append(
            np.divide(
                np.sum(approximation * projected, axis=1),
                denominator,
                out=np.full(len(projected), np.nan),
                where=denominator > 0,
            )
        )
    return dict(
        batch_gradient_norm=norm,
        reconstructed_gradient_norm=reconstructed_norm,
        physical_relative_error=relative,
        physical_cosine=cosine,
        projected_norm=projected_norm,
        projected_relative_error=np.stack(projected_error, 1),
        projected_target_gain=np.stack(projected_gain, 1),
        coefficient_l0=np.count_nonzero(coefficients, axis=2),
    )


def audit(root, receipts, analysis_plan, export):
    # Do not move any cache/output reads before this two-seed barrier.
    barrier = require_barrier(receipts, analysis_plan)
    return audit_released(root, receipts, barrier, analysis_plan, export)


def audit_preliminary(root, receipt_path, analysis_plan, export, amendment_path):
    barrier = require_preliminary_receipt(receipt_path, analysis_plan, amendment_path)
    return audit_released(
        root, [receipt_path], barrier, analysis_plan, export, amendment_path=amendment_path
    )


def audit_released(root, receipts, barrier, analysis_plan, export, amendment_path=None):
    """Internal common audit after an explicit full or preliminary authorization gate."""
    root, export = Path(root).resolve(), Path(export).resolve()
    spec = read_json(root / "specification.json")
    own = next(value for value in barrier if value["seed"] == spec["training_seed"])
    for filename, field in (
        ("controller.json", "controller_sha256"),
        ("plan.json", "plan_sha256"),
        ("specification.json", "specification_sha256"),
    ):
        if file_hash(root / filename) != own[field]:
            raise ValueError(f"Local completion receipt is stale: {filename}")
    if own["release_reason"] != "complete" or any(
        value["release_reason"] != "complete" for value in barrier
    ):
        failure = dict(
            checked_unix=time.time(),
            complete=False,
            scientific_outputs_read=False,
            reason="At least one phase reached its deadline without complete results",
            receipts=barrier,
            analysis_plan_sha256=file_hash(analysis_plan),
        )
        incomplete = root / "audit-incomplete.json"
        if incomplete.exists():
            raise ValueError("Preserve the existing incomplete audit before another attempt")
        atomic_json(incomplete, failure)
        return failure
    if file_hash(root / "manifest.json") != own["manifest_sha256"]:
        raise ValueError("Completed manifest changed since receipt")
    audit_name = "integrity-preliminary-seed43.json" if amendment_path else "integrity.json"
    if (root / audit_name).exists() or export.exists():
        raise ValueError("Preserve existing audit/export; use an audited repair if necessary")
    controller, plan, manifest, lock, pre, fit, fit_plan, data = [
        read_json(root / n)
        for n in (
            "controller.json",
            "plan.json",
            "manifest.json",
            "forecast-lock.json",
            "preflight.json",
            "fit.json",
            "fit-plan.json",
            "data.json",
        )
    ]
    ph, lh = file_hash(root / "plan.json"), file_hash(root / "forecast-lock.json")
    if (
        controller["status"] != "complete"
        or controller["plan_sha256"] != ph
        or manifest["plan_sha256"] != ph
        or lock["plan_sha256"] != ph
        or manifest["forecast_lock_sha256"] != lh
    ):
        raise ValueError("Controller, manifest or forecast lock plan mismatch")
    if (
        spec["methods"] != list(METHODS)
        or spec["rates"] != list(RATES)
        or any(spec["counts"][key] != value for key, value in COUNTS.items())
        or spec["counts"]["actual_updates"] != 64
    ):
        raise ValueError("Scientific grid differs from the frozen protocol")
    if not (
        fit_plan["written_unix"]
        < fit["started_unix"]
        <= fit["completed_unix"]
        < pre["completed_unix"]
        < plan["frozen_unix"]
        < controller["started_unix"]
        <= manifest["started_unix"]
        < manifest["completed_unix"]
        < controller["deadline_unix"]
    ):
        raise ValueError("Fit, preflight, forecast or immutable deadline ordering failed")
    # A bounded technical restart can reuse an older forecast lock. The lock must
    # still follow the original controller start and precede every actual update.
    if not controller["started_unix"] < lock["locked_unix"] < manifest["completed_unix"]:
        raise ValueError("Forecast lock falls outside the original controller interval")
    if manifest["original_deadline_unix"] != controller["deadline_unix"]:
        raise ValueError("Worker deadline differs from original controller deadline")
    if not controller["attempts"] or controller["attempts"][-1]["returncode"] != 0:
        raise ValueError("Controller did not finish normally")
    for attempt in controller["attempts"]:
        if not controller["started_unix"] <= attempt["started_unix"] < attempt["finished_unix"]:
            raise ValueError("Invalid controller attempt timestamps")
        if attempt["finished_unix"] >= controller["deadline_unix"]:
            raise ValueError("An attempt exceeded the original deadline")
    for hashes in (
        plan["source_files_sha256"],
        plan["input_hashes"],
        spec["input_hashes"],
        fit_plan["source_files_sha256"],
        fit_plan["input_hashes"],
    ):
        verify_files(Path.cwd(), hashes)
    verify_archive(root, plan["source_files_sha256"])
    source = Path(spec["source_root"])
    original_integrity = read_json(source / "integrity.json")
    if not original_integrity["complete"]:
        raise ValueError("Original source audit is incomplete")
    for stage in ("train", "factors", "atoms"):
        source_manifest = source / f"{stage}-manifest.json"
        if file_hash(source_manifest) != original_integrity["stages"][stage]["manifest_sha256"]:
            raise ValueError(f"Original source manifest changed: {stage}")
        verify_files(source, read_json(source_manifest)["files_sha256"])
    base = Path(spec["source_config"]["base_directory"])
    verify_files(base, spec["source_config"]["model_asset_hashes"])
    validate_data(data)
    validate_staged_data(root)
    helpers = upstream(spec["source_config"])
    tokenization = validate_tokenization(
        data, helpers.get_tokenizer(str(base)), helpers.tokenize_chat
    )
    for key in (
        "adapter_restored",
        "scoring_restored",
        "finite_gradient",
        "layout_verified",
        "training_input_only",
        "long_training_completion_checked",
    ):
        if pre.get(key) is not True:
            raise ValueError(f"Preflight assertion failed: {key}")
    if (
        pre["tensors"] != 136
        or pre["parameter_count"] != 2228224
        or pre["tokenization"] != tokenization
        or pre["specification_sha256"] != file_hash(root / "specification.json")
        or pre["fit_sha256"] != file_hash(root / "fit.json")
    ):
        raise ValueError("Preflight provenance, architecture or tokenizer changed")
    if (
        fit["fit_plan_sha256"] != file_hash(root / "fit-plan.json")
        or fit["dictionaries_sha256"] != file_hash(root / "dictionaries.npz")
        or lock["dictionaries_sha256"] != fit["dictionaries_sha256"]
        or fit_plan["new_magnitude_fit_already_computed"]
        or not fit["only_fitting_rows_used"]
        or fit["new_batch_outcomes_read"]
        or not fit["original_dictionary_reused"]
        or fit["fitting_rows"] != 5000
        or fit["fitting_columns"] != 6800
        or any(fit["learner_parameters"][k] != v for k, v in spec["learner"].items())
    ):
        raise ValueError("Magnitude dictionary fitting provenance failed")
    if not all(
        lock.get(key) is True
        for key in ("all_forecasts_before_updates", "original_adapter_unchanged")
    ):
        raise ValueError("Forecast lock assertions failed")
    if lock["methods"] != list(METHODS) or not manifest["adapter_restored"]:
        raise ValueError("Method ordering or final adapter restoration failed")
    expected = {f"outcomes/batch-{b:02d}-rate-{r:g}.json" for b in range(32) for r in RATES}
    if (
        set(manifest["files_sha256"]) != expected | {"forecast-cache.pt", "forecast-lock.json"}
        or set(lock["files_sha256"]) != {"forecast-cache.pt"}
        or manifest["actual_updates"] != 64
        or manifest["probes"] != 80
    ):
        raise ValueError("Manifest grid or counts are incomplete")
    verify_files(root, lock["files_sha256"])
    verify_files(root, manifest["files_sha256"])

    # First scientific cache/outcome reads occur only here, after the barrier.
    cache = torch.load(root / "forecast-cache.pt", map_location="cpu", weights_only=True, mmap=True)
    validate_cache(cache)
    shapes = {
        "batch_gradients": (32, 2228224),
        "projected": (32, 6800),
        "coefficients": (32, 2, 500),
        "reconstructed_gradients": (32, 4, 2228224),
        "probe_gradients": (80, 2228224),
        "baseline_mean_ce": (80,),
        "completion_tokens": (80,),
        "predicted_mean_ce_change": (32, 2, 4, 80),
    }
    if set(cache) != set(shapes):
        raise ValueError("Unexpected forecast cache schema")
    for key, shape in shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"Forecast cache shape differs: {key}")
    probes = data["calibration"] + data["evaluation"]
    np.testing.assert_array_equal(
        cache["completion_tokens"], [q["completion_tokens"] for q in probes]
    )
    if torch.any(cache["baseline_mean_ce"] < 0):
        raise ValueError("Negative baseline completion CE")
    dictionaries = dict(np.load(root / "dictionaries.npz", allow_pickle=False))
    payload = torch.load(source / "atoms.pt", map_location="cpu", weights_only=True)
    np.testing.assert_array_equal(dictionaries["normalized_fit"], payload["dictionary"].numpy())
    _, expected_scales = learning_rows(payload["projected_gradients"].numpy(), "magnitude_fit")
    if float(dictionaries["fitting_global_scale"]) != float(expected_scales[0]):
        raise ValueError("Magnitude fit global scale differs from the fitting-only median")
    from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac, unproject_atom

    factors = load_ekfac_eigen(str(source / "factors/paper_ekfac/factors_training"))
    projected, info = project_gradients_ekfac(
        cache["batch_gradients"], payload["lora_names"], factors, 50
    )
    torch.testing.assert_close(projected, cache["projected"], rtol=1e-6, atol=1e-7)
    for mi, arm in enumerate(ARMS):
        if dictionaries[arm].shape != (500, 6800) or not np.isfinite(dictionaries[arm]).all():
            raise ValueError("Invalid fitted dictionary")
        coefficients, reconstructed = encode(cache["projected"].numpy(), dictionaries[arm])
        np.testing.assert_allclose(coefficients, cache["coefficients"][:, mi], rtol=1e-6, atol=1e-7)
        for b in range(32):
            physical = unproject_atom(reconstructed[b], info, factors, 2228224).float()
            torch.testing.assert_close(
                physical, cache["reconstructed_gradients"][b, mi], rtol=1e-6, atol=1e-7
            )
    torch.testing.assert_close(
        cache["reconstructed_gradients"][:, 2], cache["batch_gradients"], rtol=0, atol=0
    )
    if torch.count_nonzero(cache["reconstructed_gradients"][:, 3]):
        raise ValueError("No-change reconstruction is not zero")
    recomputed = loss_forecasts(
        cache["probe_gradients"].numpy(), cache["reconstructed_gradients"].numpy(), RATES
    )
    np.testing.assert_allclose(
        recomputed, cache["predicted_mean_ce_change"], rtol=1e-10, atol=1e-12
    )
    actual = np.empty((32, 2, 80), dtype=np.float64)
    for b in range(32):
        for ri, rate in enumerate(RATES):
            outcome = read_json(root / f"outcomes/batch-{b:02d}-rate-{rate:g}.json")
            if (
                outcome["batch_index"] != b
                or outcome["batch_id"] != data["batches"][b]["id"]
                or outcome["rate"] != rate
                or outcome["plan_sha256"] != ph
                or outcome["forecast_lock_sha256"] != lh
                or not outcome["adapter_restored"]
                or not lock["locked_unix"]
                < outcome["update_started_unix"]
                < outcome["completed_unix"]
                <= manifest["completed_unix"]
            ):
                raise ValueError("Actual update identity, lock ordering or restore failed")
            np.testing.assert_array_equal(outcome["completion_tokens"], cache["completion_tokens"])
            values = np.asarray(outcome["mean_completion_ce"])
            if values.shape != (80,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid actual completion CE grid")
            actual[b, ri] = values
    compact = {
        "predicted_change": cache["predicted_mean_ce_change"].numpy(),
        "actual_mean_ce": actual,
        "baseline_mean_ce": cache["baseline_mean_ce"].numpy(),
        "completion_tokens": cache["completion_tokens"].numpy(),
        "coefficients": cache["coefficients"].numpy(),
        **norm_diagnostics(cache, dictionaries),
    }
    export.mkdir(parents=True)
    np.savez_compressed(export / "arrays.npz", **compact)
    atomic_json(export / "data.json", data)
    integrity = dict(
        checked_unix=time.time(),
        complete=True,
        seed=spec["training_seed"],
        source_commit=FROZEN_COMMIT,
        source_archive_sha256=file_hash(root / "source_snapshot.tar"),
        source_input_output_hashes_verified=True,
        full_grid_verified=True,
        locks_restores_verified=True,
        original_deadline_verified=True,
        tokenization_verified=True,
        cache_recomputed=True,
        gradients_recomputed_from_model=False,
        gradient_audit_scope="Stored batch gradients were reprojected and forecast "
        "contractions recomputed. No model load or independent gradient differentiation.",
        updates=64,
        probes=80,
        evaluation_pairs_per_rate=2048,
        forecast_conditions=256,
        loss_values_after_updates=5120,
        attempts=controller["attempts"],
        plan_sha256=ph,
        forecast_lock_sha256=lh,
        analysis_plan_sha256=file_hash(analysis_plan),
        completion_receipts=barrier,
        receipt_files_sha256={str(Path(p)): file_hash(p) for p in receipts},
        forecast_cache_sha256=file_hash(root / "forecast-cache.pt"),
        fit=fit,
        worker_started_unix=manifest["started_unix"],
        worker_completed_unix=manifest["completed_unix"],
        peak_memory_bytes=manifest["peak_memory_bytes"],
    )
    if amendment_path is not None:
        integrity.update(
            preliminary=True,
            complete_two_seed_claim=False,
            preliminary_amendment_sha256=file_hash(amendment_path),
            preliminary_amendment=read_json(amendment_path),
            missing_seeds=[42],
        )
    atomic_json(root / audit_name, integrity)
    shutil.copyfile(root / audit_name, export / "integrity.json")
    atomic_json(
        export / "export.json",
        dict(
            type="paper_magnitude_preliminary_compact_export"
            if amendment_path is not None
            else "paper_magnitude_compact_export",
            exported_unix=time.time(),
            seed=spec["training_seed"],
            methods=list(METHODS),
            rates=list(RATES),
            analysis_plan_sha256=file_hash(analysis_plan),
            audit_script_sha256=file_hash(Path(__file__)),
            scientific_scope="teacher-forced completion CE only",
            heavy_cache_kept_remote=True,
            preliminary_amendment_sha256=file_hash(amendment_path)
            if amendment_path is not None
            else None,
            files_sha256={
                n: file_hash(export / n) for n in ("arrays.npz", "data.json", "integrity.json")
            },
        ),
    )
    return integrity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("receipt", "audit", "audit-preliminary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--analysis-plan", type=Path)
    parser.add_argument("--amendment", type=Path)
    args = parser.parse_args()
    if args.command == "receipt":
        if args.output is None:
            parser.error("receipt requires --output")
        result = receipt(args.root, args.output)
    elif args.command == "audit":
        if args.output is None or args.analysis_plan is None:
            parser.error("audit requires --output and --analysis-plan")
        result = audit(args.root, args.receipt, args.analysis_plan, args.output)
    else:
        if (
            args.output is None
            or args.analysis_plan is None
            or args.amendment is None
            or len(args.receipt) != 1
        ):
            parser.error("audit-preliminary requires one receipt, output, plan and amendment")
        result = audit_preliminary(
            args.root, args.receipt[0], args.analysis_plan, args.output, args.amendment
        )
    print(json.dumps({k: result.get(k) for k in ("seed", "complete", "release_reason", "reason")}))


if __name__ == "__main__":
    main()
