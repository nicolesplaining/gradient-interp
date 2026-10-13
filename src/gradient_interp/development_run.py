"""Bounded sequential controller for explicitly exploratory follow-up jobs."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .confirmation import verify_files
from .confirmation_run import alive, stop_worker
from .feature_run import pilot_budget
from .study import atomic_json, file_hash


def run(root):
    root = Path(root).resolve()
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((root / "plan.json").read_text())
        if plan["scope"] != "development_only":
            raise ValueError("This controller does not run confirmatory tests")
        verify_files(Path.cwd(), plan["source_files_sha256"])
        previous_path = root / "controller.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
        if previous and previous["plan_sha256"] != file_hash(root / "plan.json"):
            raise ValueError("Plan changed across resumes")
        if previous and previous.get("worker_pid") and alive(previous["worker_pid"]):
            raise RuntimeError("A worker is still alive")
        if previous and previous["status"] == "complete":
            return
        running = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        )
        if running.strip():
            raise RuntimeError("The A100 is occupied")
        started, deadline = pilot_budget(previous, time.time(), plan["max_hours"])
        state = {
            "scope": "development_only",
            "started_unix": started,
            "deadline_unix": deadline,
            "status": "running",
            "controller_pid": os.getpid(),
            "worker_pid": None,
            "plan_sha256": file_hash(root / "plan.json"),
            "attempts": previous.get("attempts", []) if previous else [],
        }
        atomic_json(previous_path, state)
        (root / "logs").mkdir(exist_ok=True)
        for job in plan["jobs"]:
            if Path(job["manifest"]).exists():
                continue
            if time.time() >= deadline:
                state["status"] = "budget_exhausted"
                break
            verify_files(Path.cwd(), plan["source_files_sha256"])
            with (root / "logs" / (job["id"] + ".log")).open("a") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", job["module"], *job["arguments"]],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                attempt = {"job": job["id"], "pid": process.pid, "started_unix": time.time()}
                state.update(current_job=job["id"], worker_pid=process.pid)
                state["attempts"].append(attempt)
                atomic_json(previous_path, state)
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
                    state.update(worker_pid=None, checked_unix=time.time())
                    atomic_json(previous_path, state)
                if state["status"] == "budget_exhausted":
                    break
                if process.returncode or not Path(job["manifest"]).exists():
                    state["status"] = "failed"
                    break
        else:
            state["status"] = "complete"
        state["checked_unix"] = time.time()
        atomic_json(previous_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
