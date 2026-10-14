"""Stop a frozen original worker at its user-authorized ownership boundary."""

import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path

from gradient_interp.paper_svd_replication import condition_grid
from gradient_interp.study import atomic_json, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def process_identity(pid, module, root):
    """Bind signaling to the original Python process, not merely a recycled PID."""
    proc = Path("/proc") / str(pid)
    arguments = (proc / "cmdline").read_bytes().split(b"\0")
    if module.encode() not in arguments or str(root).encode() not in arguments:
        raise ValueError("PID does not belong to the original module and run")
    # Fields after the final closing parenthesis start at process-state field 3.
    started_ticks = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    return {"pid": pid, "started_ticks": started_ticks, "module": module}


def sigint_state(pid):
    fields = dict(
        line.split(":", 1)
        for line in (Path("/proc") / str(pid) / "status").read_text().splitlines()
    )
    masks = {name: int(fields[name].strip(), 16) for name in ("SigIgn", "SigBlk", "SigCgt")}
    bit = 1 << (signal.SIGINT - 1)
    if masks["SigIgn"] & bit or masks["SigBlk"] & bit or not masks["SigCgt"] & bit:
        raise ValueError("Original controller cannot handle SIGINT through Python")
    return masks


def run(root, amendment_path):
    amendment = read(amendment_path)
    seed = read(root / "specification.json")["training_seed"]
    assignment = next(a for a in amendment["assignments"] if a["seed"] == seed)
    if (
        amendment["type"] != "paper_svd_parallel_execution_amendment"
        or amendment.get("frozen_source_commit") != "e12ecc1989531ce48b1352e2f76985c6a7cf1a23"
        or amendment.get("execution_host") != "ubuntu@209.20.156.175"
        or not amendment["scientific_choices_unchanged"]
        or amendment["scientific_outcomes_read"]
        or str(root.resolve()) != assignment["original_root"]
        or file_hash(Path(__file__)) != amendment["prefix_watcher_sha256"]
        or file_hash(root / "controller-at-amendment.json")
        != assignment["controller_at_amendment_sha256"]
    ):
        raise ValueError("Wrong frozen execution amendment or watcher")
    grid = condition_grid()
    prefix = assignment["prefix_indices"]
    if (
        not prefix
        or not assignment["tail_indices"]
        or prefix != list(range(len(prefix)))
        or assignment["tail_indices"] != list(range(len(prefix), len(grid)))
    ):
        raise ValueError("Condition ownership must be a disjoint full partition")
    deadline = assignment["original_deadline_unix"]
    if (
        abs(deadline - assignment["original_started_unix"] - 43200) > 0.01
        or not assignment["original_started_unix"]
        < amendment["frozen_unix"]
        < time.time()
        < deadline
    ):
        raise ValueError("Original deadline or amendment timing changed")
    for name, key in (
        ("plan.json", "plan_sha256"),
        ("specification.json", "specification_sha256"),
        ("selection-lock.json", "selection_lock_sha256"),
    ):
        if file_hash(root / name) != assignment[key]:
            raise ValueError("Original frozen metadata changed")
    with (root / "prefix-watcher.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(
            (root / name).exists()
            for name in ("prefix-completion-receipt.json", "prefix-controller.json")
        ):
            raise ValueError("Preserve the existing prefix receipt")
        controller = read(root / "controller.json")
        if (
            controller["status"] != "running"
            or controller["started_unix"] != assignment["original_started_unix"]
            or controller["deadline_unix"] != deadline
            or file_hash(root / "controller.json") != assignment["controller_at_amendment_sha256"]
            or not alive(controller["controller_pid"])
            or not alive(controller["worker_pid"])
        ):
            raise ValueError("Original controller or worker is not the frozen running process")
        identities = {
            "controller": process_identity(
                controller["controller_pid"], "gradient_interp.development_run", root
            ),
            "worker": process_identity(
                controller["worker_pid"], "gradient_interp.paper_svd_replication", root
            ),
        }
        masks = sigint_state(controller["controller_pid"])
        final = root / "generations" / f"{grid[prefix[-1]]['id']}.json"
        atomic_json(
            root / "prefix-watcher.json",
            dict(
                status="waiting",
                pid=os.getpid(),
                started_unix=time.time(),
                amendment_sha256=file_hash(amendment_path),
                stop_after_index=prefix[-1],
                stop_after_condition=grid[prefix[-1]]["id"],
                original_deadline_unix=deadline,
                process_identities=identities,
                sigint_masks=masks,
            ),
        )
        while not final.exists():
            if (
                time.time() >= deadline
                or not alive(controller["controller_pid"])
                or not alive(controller["worker_pid"])
            ):
                raise RuntimeError("Original deadline or controller exit preceded the boundary")
            time.sleep(0.2)
        expected = {f"generations/{grid[i]['id']}.json" for i in prefix}
        existing = {str(p.relative_to(root)) for p in (root / "generations").glob("*.json")}
        if existing != expected:
            raise ValueError("Original worker crossed the frozen ownership boundary")
        for key, identity in identities.items():
            if process_identity(identity["pid"], identity["module"], root) != identity:
                raise ValueError("Original process identity changed before signal")
        sigint_state(controller["controller_pid"])
        # Its existing interrupt handler stops the model worker and saves the attempt.
        os.kill(controller["controller_pid"], signal.SIGINT)
        pids = {controller["controller_pid"], controller["worker_pid"]}
        pids.update(attempt["pid"] for attempt in controller["attempts"])
        stopped = time.time()
        while any(alive(pid) for pid in pids):
            if time.time() - stopped > 120:
                raise RuntimeError("Original processes did not exit after intentional interruption")
            time.sleep(0.2)
        interrupted = read(root / "controller.json")
        if interrupted["status"] != "interrupted" or interrupted["attempts"][-1]["returncode"] == 0:
            raise ValueError("The original interrupted attempt was not preserved")
        if (
            interrupted["started_unix"] != controller["started_unix"]
            or interrupted["deadline_unix"] != deadline
            or interrupted["plan_sha256"] != controller["plan_sha256"]
            or interrupted["attempts"][-1]["pid"] != controller["attempts"][-1]["pid"]
            or interrupted["attempts"][-1]["started_unix"]
            != controller["attempts"][-1]["started_unix"]
        ):
            raise ValueError("Original controller history changed during interruption")
        (root / "prefix-controller.json").write_bytes((root / "controller.json").read_bytes())
        existing = {str(p.relative_to(root)) for p in (root / "generations").glob("*.json")}
        if existing != expected:
            raise ValueError("Unexpected extra committed original generation")
        atomic_json(
            root / "prefix-completion-receipt.json",
            dict(
                type="paper_svd_prefix_completion_receipt",
                seed=seed,
                all_processes_dead=True,
                observed_unix=time.time(),
                controller_sha256=file_hash(root / "prefix-controller.json"),
                amendment_sha256=file_hash(amendment_path),
                completed_prefix_indices=prefix,
                checked_pids=sorted(pids),
                generation_files_sha256={name: file_hash(root / name) for name in sorted(expected)},
                scientific_output_contents_parsed=False,
                watcher_sha256=file_hash(Path(__file__)),
                process_identities=identities,
                sigint_masks=masks,
            ),
        )
        atomic_json(
            root / "prefix-watcher.json",
            dict(status="complete", pid=os.getpid(), completed_unix=time.time()),
        )


def guarded_run(root, amendment_path):
    try:
        return run(root, amendment_path)
    except BaseException as error:
        atomic_json(
            root / f"prefix-watcher-failure-{time.time_ns()}.json",
            {
                "status": "failed",
                "observed_unix": time.time(),
                "pid": os.getpid(),
                "error_type": type(error).__name__,
                "error": str(error),
                "watcher_sha256": file_hash(Path(__file__)),
                "amendment_sha256": file_hash(amendment_path) if amendment_path.exists() else None,
                "scientific_fields_modified": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    guarded_run(args.root.resolve(), args.amendment)
