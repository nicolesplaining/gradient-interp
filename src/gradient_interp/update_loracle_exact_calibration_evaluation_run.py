"""One-shot controller for the separately frozen held-out interpreter comparison."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .confirmation import verify_files
from .confirmation_run import stop_worker
from .loracle_trajectory_toy import read
from .study import atomic_json, file_hash
from .update_loracle_analysis import require

STAGES = ("preflight", "updates", "predict", "measure")


def validate_plan(plan, root):
    require(
        plan["scope"] == "heldout_confirmation" and plan["max_hours"] == 12,
        "This controller requires the fixed held-out scope and twelve-hour budget",
    )
    require([job["id"] for job in plan["jobs"]] == list(STAGES), "Held-out stage order changed")
    manifests = (
        "preflight.json",
        "updates-manifest.json",
        "predictions-manifest.json",
        "manifest.json",
    )
    for job, stage, manifest in zip(plan["jobs"], STAGES, manifests):
        require(
            job["module"] == "gradient_interp.update_loracle_exact_calibration_heldout"
            and job["arguments"] == [stage, "--root", str(root.resolve())]
            and Path(job["manifest"]) == root.resolve() / manifest,
            "Unexpected held-out command or manifest",
        )


def run(root):
    root = root.resolve()
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(
            not (root / "controller.json").exists(),
            "Held-out phase was already launched; never resume or duplicate it",
        )
        plan = read(root / "plan.json")
        validate_plan(plan, root)
        verify_files(Path.cwd(), plan["source_files_sha256"])
        require(
            not any(Path(job["manifest"]).exists() for job in plan["jobs"]),
            "A held-out stage was already attempted or completed",
        )
        running = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        require(not running, "GPU is occupied")
        (root / "logs").mkdir(exist_ok=False)
        started = time.time()
        state = {
            "scope": "heldout_confirmation",
            "started_unix": started,
            "deadline_unix": started + 43200,
            "status": "running",
            "controller_pid": os.getpid(),
            "worker_pid": None,
            "plan_sha256": file_hash(root / "plan.json"),
            "attempts": [],
        }
        atomic_json(root / "controller.json", state)
        process = None
        try:
            for job in plan["jobs"]:
                require(file_hash(root / "plan.json") == state["plan_sha256"], "Plan changed")
                verify_files(Path.cwd(), plan["source_files_sha256"])
                if time.time() >= state["deadline_unix"] - 12:
                    state["status"] = "budget_exhausted"
                    break
                with (root / "logs" / (job["id"] + ".log")).open("x") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-u", "-m", job["module"], *job["arguments"]],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    attempt = {"job": job["id"], "pid": process.pid, "started_unix": time.time()}
                    state["attempts"].append(attempt)
                    state.update(current_job=job["id"], worker_pid=process.pid)
                    atomic_json(root / "controller.json", state)
                    try:
                        process.wait(timeout=max(0.001, state["deadline_unix"] - time.time() - 12))
                    except subprocess.TimeoutExpired:
                        state["status"] = "budget_exhausted"
                        stop_worker(process)
                    except BaseException:
                        state["status"] = "interrupted"
                        stop_worker(process)
                        raise
                    finally:
                        attempt.update(finished_unix=time.time(), returncode=process.poll())
                        state.update(worker_pid=None, checked_unix=time.time())
                        atomic_json(root / "controller.json", state)
                if state["status"] != "running":
                    break
                if process.returncode or not Path(job["manifest"]).exists():
                    state["status"] = "failed"
                    break
            else:
                state["status"] = "complete"
        except BaseException as error:
            if state["status"] == "running":
                state["status"] = "failed"
            state["error"] = repr(error)
            raise
        finally:
            state["checked_unix"] = time.time()
            atomic_json(root / "controller.json", state)


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
