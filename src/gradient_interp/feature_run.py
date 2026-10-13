"""Single-worker controller for a fixed, bounded development pilot."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .confirmation_run import alive, stop_worker
from .feature_pilot import validate
from .study import atomic_json, file_hash


def pilot_budget(previous, now, max_hours):
    if previous:
        return previous["started_unix"], previous["deadline_unix"]
    if not 0 < max_hours <= 12:
        raise ValueError("This development pilot has a maximum twelve-hour stop")
    return now, now + max_hours * 3600


def run(root):
    root = Path(root).resolve()
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = validate(root)
        state_path = root / "controller.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else None
        if previous and previous.get("worker_pid") and alive(previous["worker_pid"]):
            raise RuntimeError("A pilot worker is already alive")
        if any(j["model"]["device"] == "cuda" for j in plan["jobs"]):
            running = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            if running.stdout.strip():
                raise RuntimeError("The accelerator is occupied; refusing concurrent jobs")
        started, deadline = pilot_budget(previous, time.time(), plan["max_hours"])
        freeze_hash = file_hash(root / "freeze.json")
        if previous and previous["freeze_sha256"] != freeze_hash:
            raise ValueError("Pilot freeze changed across resumes")
        state = {
            "scope": "development_only",
            "started_unix": started,
            "deadline_unix": deadline,
            "status": "running",
            "controller_pid": os.getpid(),
            "worker_pid": None,
            "freeze_sha256": freeze_hash,
            "attempts": previous.get("attempts", []) if previous else [],
        }
        atomic_json(state_path, state)
        (root / "logs").mkdir(exist_ok=True)
        for job in plan["jobs"]:
            if (root / "jobs" / job["id"] / "manifest.json").exists():
                continue
            if time.time() >= deadline:
                state["status"] = "budget_exhausted"
                break
            validate(root)
            with (root / "logs" / (job["id"] + ".log")).open("a") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "gradient_interp.feature_pilot",
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
                    state["checked_unix"] = time.time()
                    atomic_json(state_path, state)
                if state["status"] == "budget_exhausted":
                    break
                if process.returncode:
                    state["status"] = "failed"
                    break
        else:
            state["status"] = "complete"
        state["checked_unix"] = time.time()
        atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
