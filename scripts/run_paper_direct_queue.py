"""Run the two frozen direct-study controllers sequentially on one supplied GPU."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gradient_interp.study import atomic_json, file_hash


def run(queue_root, launch=subprocess.run):
    queue_root = Path(queue_root)
    spec_path = queue_root / "queue.json"
    spec = json.loads(spec_path.read_text())
    assert spec["scope"] == "development_only"
    assert len(spec["runs"]) == 2
    assert len({r["root"] for r in spec["runs"]}) == 2
    assert Path(sys.executable).name.startswith("python")
    with (queue_root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = queue_root / "queue-state.json"
        if state_path.exists():
            raise RuntimeError("Queue already attempted; inspect and audit before any new action")
        state = dict(
            status="running",
            pid=os.getpid(),
            started_unix=time.time(),
            queue_sha256=file_hash(spec_path),
            attempts=[],
        )
        atomic_json(state_path, state)
        try:
            for item in spec["runs"]:
                root = Path(item["root"])
                if file_hash(root / "plan.json") != item["plan_sha256"]:
                    raise ValueError("Frozen child plan changed")
                if (root / "controller.json").exists():
                    raise RuntimeError("A scientific controller already exists; never duplicate it")
                attempt = dict(root=str(root), started_unix=time.time())
                state["attempts"].append(attempt)
                state["current_root"] = str(root)
                atomic_json(state_path, state)
                result = launch(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "gradient_interp.development_run",
                        "--root",
                        str(root),
                    ],
                    check=False,
                )
                attempt["returncode"] = result.returncode
                attempt["finished_unix"] = time.time()
                controller = json.loads((root / "controller.json").read_text())
                attempt["controller_status"] = controller["status"]
                attempt["deadline_unix"] = controller["deadline_unix"]
                if result.returncode or controller["status"] != "complete":
                    state.update(status="failed", finished_unix=time.time())
                    atomic_json(state_path, state)
                    return False
                atomic_json(state_path, state)
        except BaseException as exc:
            state.update(status="failed", error=repr(exc), finished_unix=time.time())
            atomic_json(state_path, state)
            raise
        state.update(status="complete", finished_unix=time.time())
        atomic_json(state_path, state)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)
