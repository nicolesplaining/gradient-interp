"""One-shot completion handoff for the fixed v2 recipe; no training or retries."""

import argparse
import os
import subprocess
import time
from pathlib import Path

from run_update_loracle_next_stage import hashed, read, ready, save

PYTHON = "/home/ubuntu/gradient-interp/.venv-atoms/bin/python"


def run(root, receipt, expected_plan):
    receipt.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "phase": "waiting_training",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "plan_sha256": expected_plan,
        "automatic_retry": False,
        "scientific_outputs_read": False,
    }
    save(receipt / "status.json", state)
    try:
        while True:
            if hashed(root / "plan.json") != expected_plan:
                raise ValueError("Frozen plan changed")
            controller = read(root / "controller.json")
            if ready(controller):
                break
            if time.time() > controller["deadline_unix"] + 120:
                raise TimeoutError("Controller did not terminate by original deadline")
            time.sleep(10)
        plan = read(root / "plan.json")
        for name, expected in plan["source_files_sha256"].items():
            if hashed(name) != expected:
                raise ValueError(f"Frozen analysis dependency changed: {name}")
        for phase, arguments in [("audit", []), ("analyze", ["--output", str(root / "analysis")])]:
            state.update(phase=phase, checked_unix=time.time())
            save(receipt / "status.json", state)
            with (receipt / f"{phase}.log").open("x") as log:
                subprocess.run(
                    [
                        PYTHON,
                        "-m",
                        "gradient_interp.update_loracle_factorized_analysis",
                        phase,
                        "--root",
                        str(root),
                        *arguments,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        state.update(
            status="complete",
            phase="development_ready_for_review",
            scientific_outputs_read=True,
            finished_unix=time.time(),
        )
        save(receipt / "status.json", state)
    except BaseException as error:
        state.update(status="failed", finished_unix=time.time(), error=repr(error))
        save(receipt / "status.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    run(args.root, args.receipt, args.plan_sha256)
