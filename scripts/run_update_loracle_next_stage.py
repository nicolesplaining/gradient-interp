"""One-shot audited handoff; never repairs, resumes or chooses a scientific variant."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

PYTHON = "/home/ubuntu/gradient-interp/.venv-atoms/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def hashed(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return bool(pid and path.exists() and path.read_text().rsplit(")", 1)[-1].split()[0] != "Z")


def ready(controller, is_alive=alive):
    if controller["status"] not in ("running", "complete"):
        raise ValueError("Collection ended unsuccessfully; no outcome access or automatic retry")
    if controller["status"] != "complete":
        if not is_alive(controller["controller_pid"]):
            raise ValueError("Running state has no live controller; diagnose without restarting")
        return False
    return (
        not is_alive(controller["controller_pid"])
        and controller["worker_pid"] is None
        and all(not is_alive(a["pid"]) for a in controller["attempts"])
    )


def run(args):
    receipt = args.receipt
    receipt.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "phase": "waiting_collection",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "source_commit_for_training": args.commit,
        "collector_plan_sha256": args.plan_sha256,
        "auditor_sha256": args.audit_sha256,
    }
    save(receipt / "status.json", state)
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        env[name] = "8"

    def phase(name, command):
        state.update(phase=name, checked_unix=time.time())
        save(receipt / "status.json", state)
        with (receipt / f"{name}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)

    try:
        while True:
            if hashed(args.collection / "plan.json") != args.plan_sha256:
                raise ValueError("Frozen collection plan changed")
            controller = read(args.collection / "controller.json")
            if ready(controller):
                break
            if time.time() > controller["deadline_unix"] + 120:
                raise TimeoutError(
                    "Collection controller did not terminate by its original deadline"
                )
            time.sleep(10)
        if hashed(args.auditor) != args.audit_sha256:
            raise ValueError("Reviewed standalone auditor changed")
        phase("audit_collection", [PYTHON, str(args.auditor), "--root", str(args.collection)])
        integrity = read(args.collection / "integrity.json")
        if (
            not integrity["passed"]
            or hashed(args.collection / "source_snapshot.tar") != integrity["source_archive_sha256"]
        ):
            raise ValueError("Original source archive or full collection audit failed")
        if args.training.exists():
            raise ValueError("Training phase already exists; no duplicate launch")
        phase("fetch_training_source", ["git", "fetch", "origin"])
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise ValueError("Remote checkout is not clean; preserving changes")
        phase("checkout_training_source", ["git", "checkout", "--detach", args.commit])
        phase(
            "prepare_training",
            [
                PYTHON,
                "scripts/prepare_update_loracle_training.py",
                "--collection",
                str(args.collection),
                "--root",
                str(args.training),
                "--commit",
                args.commit,
            ],
        )
        phase(
            "training",
            [PYTHON, "-u", "-m", "gradient_interp.development_run", "--root", str(args.training)],
        )
        finished = read(args.training / "controller.json")
        if not ready(finished):
            raise ValueError("Training controller has not completed and exited")
        phase(
            "audit_training",
            [
                PYTHON,
                "-m",
                "gradient_interp.update_loracle_analysis",
                "audit-training",
                "--root",
                str(args.training),
            ],
        )
        phase(
            "analyze_development",
            [
                PYTHON,
                "-m",
                "gradient_interp.update_loracle_analysis",
                "analyze-development",
                "--root",
                str(args.training),
                "--output",
                str(args.training / "analysis"),
            ],
        )
        state.update(
            status="complete", phase="development_ready_for_review", finished_unix=time.time()
        )
        save(receipt / "status.json", state)
    except BaseException as error:
        state.update(
            status="failed", finished_unix=time.time(), error=repr(error), automatic_retry=False
        )
        save(receipt / "status.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    run(parser.parse_args())
