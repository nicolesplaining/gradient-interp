"""Execute a frozen disjoint SVD condition suffix without reading prior outcomes."""

import argparse
import fcntl
import json
import os
import signal
import socket
import time
from pathlib import Path

import torch

from gradient_interp.confirmation import verify_files
from gradient_interp.paper_svd_replication import (
    condition_grid,
    controlled_checks,
    generate,
    generation_checks,
    published_material,
    restore,
    set_direction,
    stable_setup,
    validate_preflight,
)
from gradient_interp.study import atomic_json, file_hash

FROZEN_COMMIT = "e12ecc1989531ce48b1352e2f76985c6a7cf1a23"
AUTHORIZED_HOST = "ubuntu@209.20.156.175"


def read(path):
    return json.loads(Path(path).read_text())


def validate_assignment(amendment, seed):
    if (
        amendment.get("type") != "paper_svd_parallel_execution_amendment"
        or amendment.get("frozen_source_commit") != FROZEN_COMMIT
        or amendment.get("scientific_choices_unchanged") is not True
        or amendment.get("scientific_outcomes_read") is not False
        or amendment.get("execution_host") != AUTHORIZED_HOST
    ):
        raise ValueError("Only the authorized frozen execution amendment is supported")
    assignments = amendment["assignments"]
    if len(assignments) != 2 or {a["seed"] for a in assignments} != {42, 43}:
        raise ValueError("Both seeds need a fixed ownership assignment")
    for assignment in assignments:
        prefix, suffix = assignment["prefix_indices"], assignment["tail_indices"]
        if (
            not prefix
            or not suffix
            or prefix != list(range(len(prefix)))
            or suffix != list(range(len(prefix), 101))
        ):
            raise ValueError("Prefix and suffix must partition all 101 conditions exactly")
        if (
            abs(assignment["original_deadline_unix"] - assignment["original_started_unix"] - 43200)
            > 0.01
        ):
            raise ValueError("Original immutable 12-hour deadline changed")
        if (
            not assignment["original_started_unix"]
            < amendment["frozen_unix"]
            < assignment["original_deadline_unix"]
        ):
            raise ValueError("Execution amendment must be frozen within the original run")
        for key in (
            "plan_sha256",
            "specification_sha256",
            "selection_lock_sha256",
            "controller_at_amendment_sha256",
        ):
            if not isinstance(assignment.get(key), str) or len(assignment[key]) != 64:
                raise ValueError("Missing frozen original metadata hash")
    return next(a for a in assignments if a["seed"] == seed)


def validate_inputs(root, output, amendment_path, preflight_path):
    """No descriptor, old generation, semantic or previous-result file is opened."""
    amendment, spec, plan, lock = [
        read(p)
        for p in (
            amendment_path,
            root / "specification.json",
            root / "plan.json",
            root / "selection-lock.json",
        )
    ]
    assignment = validate_assignment(amendment, spec["training_seed"])
    if (
        str(root.resolve()) != assignment["original_root"]
        or str(output.resolve()) != assignment["tail_root"]
    ):
        raise ValueError("Input or output root differs from frozen assignment")
    if file_hash(Path(__file__)) != amendment["helper_sha256"]:
        raise ValueError("Execution helper changed after amendment")
    for name, key in (
        ("plan.json", "plan_sha256"),
        ("specification.json", "specification_sha256"),
        ("selection-lock.json", "selection_lock_sha256"),
    ):
        if file_hash(root / name) != assignment[key]:
            raise ValueError("Frozen original metadata changed")
    if (
        lock["heldout_outcomes_read"]
        or lock["specification_sha256"] != assignment["specification_sha256"]
    ):
        raise ValueError("Invalid pre-outcome selection lock")
    if (spec["chunk"], spec["max_new_tokens"], spec["stop_ids"]) != (8, 300, [1, 106]):
        raise ValueError("Frozen generation settings changed")
    for name in ("data.json", "directions.pt"):
        if file_hash(root / name) != lock["files_sha256"][name]:
            raise ValueError("Frozen input bank changed")
    verify_files(Path.cwd(), plan["source_files_sha256"])
    source = Path(spec["source_root"])
    required = {
        str(source / n)
        for n in ("atoms.pt", "adapter/adapter_model.safetensors", "adapter/adapter_config.json")
    }
    required |= {
        str(Path(spec["source_config"]["base_directory"]) / n)
        for n in spec["source_config"]["model_asset_hashes"]
    }
    assets = assignment["runtime_asset_hashes"]
    if set(assets) != required or any(
        digest != spec["input_hashes"].get(path) for path, digest in assets.items()
    ):
        raise ValueError("Runtime assets must equal the required original frozen input hashes")
    verify_files(Path.cwd(), assets)
    preflight = read(preflight_path)
    validate_preflight(preflight, spec)
    if any(
        preflight.get(k) != value
        for k, value in {
            "execution_host": AUTHORIZED_HOST,
            "amendment_sha256": file_hash(amendment_path),
            "helper_sha256": amendment["helper_sha256"],
            "specification_sha256": assignment["specification_sha256"],
            "selection_lock_sha256": assignment["selection_lock_sha256"],
        }.items()
    ):
        raise ValueError("New-host training preflight is not bound to this execution amendment")
    if (
        not amendment["frozen_unix"]
        < preflight["completed_unix"]
        < time.time()
        < assignment["original_deadline_unix"]
    ):
        raise ValueError("New-host preflight or original deadline is invalid")
    if (
        (output / "tail-controller.json").exists()
        or (output / "tail-manifest.json").exists()
        or any((output / "generations").glob("*.json"))
    ):
        raise ValueError(
            "Preserve prior tail attempts; no automatic resume or duplicate generation"
        )
    return amendment, assignment, spec, plan


def execute_condition(
    root,
    spec,
    model,
    tokenizer,
    parameters,
    original,
    directions,
    data,
    checkers,
    condition,
    index,
    deadline,
    directory,
):
    """The scientific block below is AST-matched to the original worker in tests."""
    final = directory / f"{condition['id']}.json"
    if final.exists():
        raise ValueError("Generation output already exists; diagnose before bounded resume")
    restore(parameters, original)
    if condition["method"]:
        set_direction(
            parameters,
            original,
            directions["vectors"][condition["family"]][condition["method"]],
            condition["strength"] * condition["sign"],
        )
    subset = [
        q
        for q in data["published_questions"]
        if not condition["method"] or q["family"] == condition["family"]
    ] + data["questions"]
    rows = []
    applied_unix = time.time()
    try:
        for start in range(0, len(subset), spec["chunk"]):
            if time.time() >= deadline:
                raise TimeoutError("Original controller deadline reached")
            part = subset[start : start + spec["chunk"]]
            answers, ids = generate(
                model, tokenizer, [q["prompt"] for q in part], spec["max_new_tokens"]
            )
            for q, answer, tokens in zip(part, answers, ids, strict=True):
                checks = generation_checks(
                    answer, tokens, checkers, spec["stop_ids"], spec["max_new_tokens"]
                )
                if "expected" in q:
                    checks.update(controlled_checks(answer, q["expected"]))
                rows.append({**q, "response": answer, "token_ids": tokens, "checks": checks})
            atomic_json(
                root / "status.json",
                {
                    "stage": "steering",
                    "completed_conditions": index,
                    "total_conditions": len(condition_grid()),
                    "completed_questions": start + len(part),
                    "condition_questions": len(subset),
                    "updated_unix": time.time(),
                },
            )
    finally:
        restore(parameters, original)
    # Metadata copying creates these two files with byte-identical original contents.
    if time.time() >= deadline:
        raise TimeoutError("Original deadline reached before condition commit")
    atomic_json(
        final,
        {
            **condition,
            "applied_unix": applied_unix,
            "completed_unix": time.time(),
            "selection_lock_sha256": file_hash(root / "selection-lock.json"),
            "plan_sha256": file_hash(root / "plan.json"),
            "adapter_restored": True,
            "rows": rows,
        },
    )
    return final, len(rows)


def deadline_alarm(signum, frame):
    raise TimeoutError("Original immutable deadline reached")


def run(root, output, amendment_path, preflight_path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "tail.lock").open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        amendment, assignment, spec, plan = validate_inputs(
            root, output, amendment_path, preflight_path
        )
        deadline = assignment["original_deadline_unix"]
        # Preserve execution artifacts locally without altering original input files.
        copies = {
            "plan.json": root / "plan.json",
            "selection-lock.json": root / "selection-lock.json",
            "execution-amendment.json": amendment_path,
            "tail-preflight.json": preflight_path,
            "tail-source.py": Path(__file__),
        }
        for name, source in copies.items():
            if (output / name).exists():
                raise ValueError("Preserve existing execution artifacts")
            (output / name).write_bytes(source.read_bytes())
        state = {
            "status": "running",
            "seed": spec["training_seed"],
            "execution_host": AUTHORIZED_HOST,
            "hostname": socket.gethostname(),
            "controller_pid": os.getpid(),
            "worker_pid": os.getpid(),
            "started_unix": time.time(),
            "deadline_unix": deadline,
            "original_started_unix": assignment["original_started_unix"],
            "amendment_sha256": file_hash(amendment_path),
            "helper_sha256": file_hash(Path(__file__)),
            "plan_sha256": assignment["plan_sha256"],
            "assigned_indices": assignment["tail_indices"],
        }
        atomic_json(output / "tail-controller.json", state)
        previous_handler = signal.signal(signal.SIGALRM, deadline_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.time()))
        try:
            model, _, tokenizer, _, _, parameters = stable_setup(spec)
            if len(parameters) != 136 or sum(p.numel() for _, p in parameters) != 2228224:
                raise ValueError("Original LoRA architecture differs")
            directions = torch.load(root / "directions.pt", weights_only=True, map_location="cpu")
            data = read(root / "data.json")
            model.generation_config.eos_token_id = spec["stop_ids"]
            original = {name: p.detach().clone() for name, p in parameters}
            for _, parameter in parameters:
                parameter.requires_grad_(False)
            _, checkers, _, _ = published_material(spec["source_config"]["upstream_checkout"])
            directory = output / "generations"
            directory.mkdir()
            started = time.time()
            files, count = {}, 0
            grid = condition_grid()
            for index in assignment["tail_indices"]:
                if time.time() >= deadline:
                    raise TimeoutError("Original controller deadline reached")
                final, n = execute_condition(
                    output,
                    spec,
                    model,
                    tokenizer,
                    parameters,
                    original,
                    directions,
                    data,
                    checkers,
                    grid[index],
                    index,
                    deadline,
                    directory,
                )
                files[str(final.relative_to(output))] = file_hash(final)
                count += n
            if count != 140 * len(assignment["tail_indices"]):
                raise ValueError("Assigned response count differs")
            atomic_json(
                output / "tail-manifest.json",
                {
                    "type": "paper_svd_tail_manifest",
                    "seed": spec["training_seed"],
                    "execution_host": AUTHORIZED_HOST,
                    "hostname": socket.gethostname(),
                    "started_unix": started,
                    "completed_unix": time.time(),
                    "original_started_unix": assignment["original_started_unix"],
                    "deadline_unix": deadline,
                    "amendment_sha256": file_hash(amendment_path),
                    "helper_sha256": file_hash(Path(__file__)),
                    "preflight_sha256": file_hash(preflight_path),
                    "plan_sha256": assignment["plan_sha256"],
                    "specification_sha256": assignment["specification_sha256"],
                    "selection_lock_sha256": assignment["selection_lock_sha256"],
                    "source_files_sha256": plan["source_files_sha256"],
                    "runtime_asset_hashes": assignment["runtime_asset_hashes"],
                    "assigned_indices": assignment["tail_indices"],
                    "assigned_ids": [grid[i]["id"] for i in assignment["tail_indices"]],
                    "conditions": len(files),
                    "generations": count,
                    "adapter_restored": True,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                    "files_sha256": files,
                },
            )
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            state.update(completed_unix=time.time(), worker_pid=None)
            atomic_json(output / "tail-controller.json", state)


def receipt(root, out):
    """A new-host, metadata-only death receipt; never opens generated responses."""
    if out.exists():
        raise ValueError("Preserve the existing tail receipt")
    controller, manifest = read(root / "tail-controller.json"), read(root / "tail-manifest.json")
    for pid in {controller["controller_pid"], controller.get("worker_pid")} - {None}:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise ValueError("Tail process is still alive")
    if (
        controller["status"] != "complete"
        or manifest["completed_unix"] > controller["deadline_unix"]
    ):
        raise ValueError("Tail did not complete within the original deadline")
    result = {
        "type": "paper_svd_tail_completion_receipt",
        "seed": manifest["seed"],
        "execution_host": AUTHORIZED_HOST,
        "all_processes_dead": True,
        "observed_unix": time.time(),
        "completed_unix": manifest["completed_unix"],
        "amendment_sha256": manifest["amendment_sha256"],
        "helper_sha256": manifest["helper_sha256"],
        "manifest_sha256": file_hash(root / "tail-manifest.json"),
        "controller_sha256": file_hash(root / "tail-controller.json"),
        "checked_pids": [controller["controller_pid"]],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "receipt"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.root.resolve(), args.output.resolve(), args.amendment, args.preflight)
    else:
        print(json.dumps(receipt(args.root.resolve(), args.output)))


if __name__ == "__main__":
    main()
