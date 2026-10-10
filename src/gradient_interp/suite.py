"""Sequential A100 study controller with a fixed wall-clock budget and saved protocol."""

import argparse
import fcntl
import gc
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path

import numpy as np
import torch

from .benchmark import prepare_benchmark
from .config import ExperimentConfig
from .data import read_jsonl
from .models import ModelConfig
from .residual import run_residual_comparison
from .study import atomic_json, file_hash, run_study_job


def crossed_bootstrap(seeds, method, repeats=1000):
    """Resample model seeds and source batches, preserving their crossed dependence."""
    order = sorted(row["batch_id"] for row in seeds[0])
    predicted, actual, baseline = [], [], []
    for records in seeds:
        by_id = {row["batch_id"]: row for row in records}
        if set(by_id) != set(order):
            raise ValueError("Seed comparisons require identical held-out batches")
        predicted.append([by_id[key]["predictions"][method] for key in order])
        baseline.append([by_id[key]["predictions"]["full_gradient"] for key in order])
        actual.append([by_id[key]["observed"]["full"] for key in order])
    predicted, actual, baseline = map(np.asarray, (predicted, actual, baseline))
    errors = ((predicted - actual) ** 2).sum(-1)
    reference = ((baseline - actual) ** 2).sum(-1)
    signal_size = (actual**2).sum(-1)
    rng = np.random.default_rng(20261010)
    sampled_seeds = rng.integers(0, len(seeds), size=(repeats, len(seeds), 1))
    sampled_batches = rng.integers(0, len(order), size=(repeats, 1, len(order)))
    e = errors[sampled_seeds, sampled_batches].sum((1, 2))
    b = reference[sampled_seeds, sampled_batches].sum((1, 2))
    y = signal_size[sampled_seeds, sampled_batches].sum((1, 2))
    return {
        "relative_l2_error": float(np.sqrt(errors.sum() / max(signal_size.sum(), 1e-30))),
        "relative_error_95ci": np.quantile(
            np.sqrt(e / np.maximum(y, 1e-30)), [0.025, 0.975]
        ).tolist(),
        "error_ratio_vs_full_gradient": float(np.sqrt(errors.sum() / max(reference.sum(), 1e-30))),
        "ratio_95ci": np.quantile(np.sqrt(e / np.maximum(b, 1e-30)), [0.025, 0.975]).tolist(),
        "seed_count": len(seeds),
        "batch_count_per_seed": len(order),
    }


def aggregate_study(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    records = {}
    completed = []
    for job in plan["jobs"]:
        directory = root / "jobs" / job["id"]
        manifest = directory / "manifest.json"
        if not manifest.exists() or json.loads(manifest.read_text()).get("status") != "complete":
            continue
        completed.append(job["id"])
        key = (job["model"]["name"], job["warmup_steps"])
        records.setdefault(key, []).append(read_jsonl(directory / "results.jsonl"))
    summaries = {}
    for (model, warmup), seeds in records.items():
        key = f"{model}/warmup-{warmup}"
        summaries[key] = {}
        for rate in plan["learning_rates"]:
            rows = [[row for row in seed if row["learning_rate"] == rate] for seed in seeds]
            summaries[key][str(rate)] = {
                method: crossed_bootstrap(rows, method) for method in rows[0][0]["predictions"]
            }
    result = {
        "completed_jobs": completed,
        "planned_jobs": len(plan["jobs"]),
        "forecast_status": "complete" if len(completed) == len(plan["jobs"]) else "partial",
        "status": (
            json.loads((root / "controller.json").read_text())["status"]
            if (root / "controller.json").exists()
            else "partial"
        ),
        "scope": "Category mean reference log-probability, not generated-answer accuracy",
        "confidence_intervals": "Crossed seed/source-batch bootstrap; 1000 replicates",
        "results": summaries,
    }
    atomic_json(root / "aggregate.json", result)
    return result


def make_plan(config_path, root, budget_hours):
    from huggingface_hub import HfApi

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["config_sha256"] != file_hash(config_path):
            raise ValueError("Study config changed; use a new study directory")
        return plan
    if not 0 < budget_hours <= 24:
        raise ValueError("This study is authorized for at most 24 hours")
    with Path(config_path).open("rb") as stream:
        config = tomllib.load(stream)
    benchmark = root / "data"
    if not (benchmark / "manifest.json").exists():
        prepare_benchmark(benchmark, config["fit_count"])
    model_revisions = {name: HfApi().model_info(name).sha for name in config["models"]}
    jobs = []
    for warmup in config["warmup_steps"]:
        for seed in config["seeds"]:
            for name in config["models"]:
                jobs.append(
                    {
                        "id": f"{name.split('/')[-1]}-seed{seed}-warm{warmup}",
                        "seed": seed,
                        "warmup_steps": warmup,
                        "model": {
                            **config["model_defaults"],
                            "name": name,
                            "revision": model_revisions[name],
                        },
                    }
                )
    plan = {
        **config,
        "jobs": jobs,
        "benchmark": str(benchmark),
        "config_sha256": file_hash(config_path),
        "created_unix": time.time(),
        "budget_hours": budget_hours,
        "protocol": {
            "primary": "Held-out batch forecast error and atom-removal effect/selectivity",
            "selection": "Size/alpha selected on development derivative reconstruction only",
            "test_forecasts": "All variants written and hashed before any final-test intervention",
            "learning_rates": "Prespecified, shared across jobs; no final-test tuning",
            "generations": "Ten size-16 pure batches; maximum prespecified rate; greedy decoding",
            "secondary_sae": "Matched 2048-feature TopK SAEs at warmed checkpoints",
            "limitations": (
                "One source dataset; short response prefixes; LoRA q/v; "
                "category probes do not establish semantic selectivity"
            ),
        },
    }
    atomic_json(path, plan)
    return plan


def run_worker(root, job_index, stage):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    job = plan["jobs"][job_index]
    directory = root / "jobs" / job["id"]
    if stage == "loracle":
        from .loracle_study import run_loracle_study

        run_loracle_study(root)
    elif stage == "forecast":
        run_study_job(plan, job, directory, lambda message: print(message, flush=True))
    elif stage == "sae":
        output = root / "sae" / job["id"]
        if (output / "manifest.json").exists():
            return
        # Restart an incomplete SAE fit in a fresh attempt directory; preserve the failed attempt.
        if output.exists() and any(output.iterdir()):
            output.rename(output.with_name(output.name + f"-incomplete-{int(time.time())}"))
        config = ExperimentConfig(
            seed=job["seed"],
            threads=plan["threads"],
            fit_path=str(Path(plan["benchmark"]) / "fit.jsonl"),
            model=ModelConfig(**job["model"]),
            top_examples=20,
        )
        # Qwen2.5-0.5B has 24 layers; Qwen2.5-3B has 36.
        layer = 11 if "0.5B" in job["model"]["name"] else 17
        run_residual_comparison(
            config,
            output,
            f"model.layers.{layer}",
            n_features=plan["sae_features"],
            k=plan["sae_k"],
            epochs=plan["sae_epochs"],
            parameters_path=directory / "parameters.npy",
            sae_device="cuda",
        )
    else:
        raise ValueError(f"Unknown study stage {stage}")


def run_suite(config_path, root, budget_hours=24):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A controller is already running for this study") from error
        return _run_suite_locked(config_path, root, budget_hours)


def _run_suite_locked(config_path, root, budget_hours=24):
    root = Path(root).resolve()
    plan = make_plan(config_path, root, budget_hours)
    state_path = root / "controller.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        deadline = state["deadline_unix"]
    else:
        deadline = time.time() + plan["budget_hours"] * 3600
    (root / "logs").mkdir(exist_ok=True)
    stages = [("forecast", i) for i in range(len(plan["jobs"]))]
    if plan.get("loracle"):
        stages.append(("loracle", 0))
    if plan["residual_sae"]:
        stages.extend(("sae", i) for i, job in enumerate(plan["jobs"]) if job["warmup_steps"] > 0)
    for stage, index in stages:
        job = plan["jobs"][index]
        folder = "jobs" if stage == "forecast" else "sae"
        manifest = (
            root / "loracle/manifest.json"
            if stage == "loracle"
            else root / folder / job["id"] / "manifest.json"
        )
        if manifest.exists() and (
            stage == "sae" or json.loads(manifest.read_text()).get("status") == "complete"
        ):
            continue
        remaining = deadline - time.time()
        state = {
            "status": "running",
            "stage": stage,
            "job": job["id"],
            "deadline_unix": deadline,
            "controller_pid": os.getpid(),
            "updated_unix": time.time(),
        }
        if remaining <= 0:
            atomic_json(state_path, {**state, "status": "budget_exhausted"})
            aggregate_study(root)
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with (root / "logs" / f"{job['id']}-{stage}.log").open("a") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gradient_interp.suite",
                    "worker",
                    "--root",
                    str(root),
                    "--job-index",
                    str(index),
                    "--stage",
                    stage,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["worker_pid"] = child.pid
            atomic_json(state_path, state)
            try:
                code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                atomic_json(
                    state_path, {**state, "status": "budget_exhausted", "updated_unix": time.time()}
                )
                aggregate_study(root)
                return
        if code:
            atomic_json(
                state_path,
                {**state, "status": "failed", "exit_code": code, "updated_unix": time.time()},
            )
            raise RuntimeError(f"{stage} worker failed for {job['id']}; see its saved log")
        if stage == "forecast":
            aggregate_study(root)
    atomic_json(
        state_path, {"status": "complete", "deadline_unix": deadline, "updated_unix": time.time()}
    )
    aggregate_study(root)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--root", required=True)
    run.add_argument("--budget-hours", type=float, default=24)
    worker = sub.add_parser("worker")
    worker.add_argument("--root", required=True)
    worker.add_argument("--job-index", required=True, type=int)
    worker.add_argument("--stage", choices=["forecast", "sae", "loracle"], required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--root", required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run_suite(args.config, args.root, args.budget_hours)
        elif args.command == "worker":
            run_worker(args.root, args.job_index, args.stage)
        else:
            aggregate_study(args.root)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
