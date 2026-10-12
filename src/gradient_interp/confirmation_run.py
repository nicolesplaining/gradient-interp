"""One-machine controller with an immutable 24-hour maximum, including resumes."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .confirmation import run_confirmation_job, verify_files
from .confirmation_data import make_plan, prepare_fresh
from .study import atomic_json, file_hash


def freeze(root, old_root):
    root, old_root = Path(root).resolve(), Path(old_root).resolve()
    if (root / "freeze.json").exists():
        raise ValueError("Protocol is already frozen")
    if not (root / "data/manifest.json").exists():
        prepare_fresh(old_root / "data", root / "data")
    plan = make_plan(root, old_root / "plan.json")
    atomic_json(root / "plan.json", plan)
    source = Path(__file__).parent
    files = (
        "confirmation.py",
        "confirmation_run.py",
        "confirmation_data.py",
        "confirmation_analysis.py",
        "cached_sweep.py",
        "study.py",
        "models.py",
        "parameters.py",
        "dictionary.py",
        "forecast.py",
        "benchmark.py",
        "data.py",
    )
    atomic_json(
        root / "freeze.json",
        {
            "saved_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "data_manifest_sha256": file_hash(root / "data/manifest.json"),
            "source_sha256": {name: file_hash(source / name) for name in files},
            "test_outcomes_observed": False,
        },
    )


def validate_freeze(root):
    frozen = json.loads((root / "freeze.json").read_text())
    if file_hash(root / "plan.json") != frozen["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    if file_hash(root / "data/manifest.json") != frozen["data_manifest_sha256"]:
        raise ValueError("Frozen data manifest changed")
    verify_files(Path(__file__).parent, frozen["source_sha256"])
    return json.loads((root / "plan.json").read_text())


def preserved_budget(previous, now, hours):
    if not 0 < hours <= 24:
        raise ValueError("The authorized new budget is at most 24 hours")
    if previous:
        start, deadline = previous["started_unix"], previous["deadline_unix"]
        if deadline > start + 24 * 3600:
            raise ValueError("Saved deadline exceeds the authorization")
        return start, deadline
    return now, now + hours * 3600


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_worker(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_controller(root, hours=24):
    root = Path(root).resolve()
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = validate_freeze(root)
        state_path = root / "controller.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else None
        if previous and previous.get("worker_pid") and alive(previous["worker_pid"]):
            raise RuntimeError("An existing worker is alive; refusing to duplicate it")
        if any(j["model"]["device"] == "cuda" for j in plan["jobs"]):
            gpu = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            if gpu.stdout.strip():
                raise RuntimeError("GPU has another process; refusing concurrent use")
        started, deadline = preserved_budget(previous, time.time(), hours)
        state = {
            "started_unix": started,
            "deadline_unix": deadline,
            "status": "running",
            "controller_pid": os.getpid(),
            "worker_pid": None,
            "freeze_sha256": file_hash(root / "freeze.json"),
            "attempts": previous.get("attempts", []) if previous else [],
        }
        if previous and previous["freeze_sha256"] != state["freeze_sha256"]:
            raise ValueError("Frozen experiment changed across controller resumes")
        atomic_json(state_path, state)
        (root / "logs").mkdir(exist_ok=True)
        for job in plan["jobs"]:
            if (root / "jobs" / job["id"] / "manifest.json").exists():
                continue
            if time.time() >= deadline:
                state["status"] = "budget_exhausted"
                break
            validate_freeze(root)
            log_path = root / "logs" / f"{job['id']}.log"
            with log_path.open("a") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "gradient_interp.confirmation_run",
                        "worker",
                        "--root",
                        str(root),
                        "--job",
                        job["id"],
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                attempt = {"job": job["id"], "pid": process.pid, "started_unix": time.time()}
                state.update(worker_pid=process.pid, current_job=job["id"])
                state["attempts"].append(attempt)
                atomic_json(state_path, state)
                try:
                    # Leave time to stop the worker before the hard deadline.
                    process.wait(timeout=max(0.001, deadline - time.time() - 12))
                except subprocess.TimeoutExpired:
                    stop_worker(process)
                    state["status"] = "budget_exhausted"
                except BaseException:
                    stop_worker(process)
                    state["status"] = "interrupted"
                    raise
                finally:
                    attempt.update(finished_unix=time.time(), returncode=process.poll())
                    state["worker_pid"] = None
                    atomic_json(state_path, state)
            if state["status"] == "budget_exhausted":
                break
            if process.returncode:
                state["status"] = "worker_failed"
                break
        else:
            state["status"] = "complete"
        state["checked_unix"] = time.time()
        atomic_json(state_path, state)


def run_worker(root, job_id):
    root = Path(root).resolve()
    plan = validate_freeze(root)
    state = json.loads((root / "controller.json").read_text())
    remaining = state["deadline_unix"] - time.time()
    if remaining <= 0:
        raise RuntimeError("GPU budget exhausted")
    # A second deadline guard survives an interrupted controller.
    watchdog = threading.Timer(remaining, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    try:
        run_confirmation_job(plan, job, root / "jobs" / job_id)
    finally:
        watchdog.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.old_root:
            parser.error("prepare requires --old-root")
        freeze(args.root, args.old_root)
    elif args.action == "run":
        run_controller(args.root, args.hours)
    else:
        run_worker(args.root, args.job)


if __name__ == "__main__":
    main()
