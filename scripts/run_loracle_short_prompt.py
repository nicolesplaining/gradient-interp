"""One user-requested short prompt on already-frozen checkpoint difference tokens."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from gradient_interp.loracle import (
    UPSTREAM_REVISION,
    configure_interpreter_prefix,
    validate_interpreter_tokens,
)
from gradient_interp.loracle_trajectory_toy import BASE_REVISION, CHECKPOINT, SEED, UPSTREAM
from gradient_interp.study import atomic_json, file_hash

QUESTION = "How does this update change response formatting?"
CASES = ("case-k", "case-r", "case-v", "case-z", "case-p")
ENCODINGS = ("correct_svd", "upstream_compatible")


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(paths):
    for name, digest in paths.items():
        if file_hash(name) != digest:
            raise ValueError(f"Frozen input changed: {name}")


def run(root):
    plan = read(root / "plan.json")
    if plan["question"] != QUESTION or plan["worker_sha256"] != file_hash(__file__):
        raise ValueError("Prompt or source changed")
    if (root / "controller.json").exists():
        raise ValueError("This check must not silently resume or duplicate a previous launch")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip():
        raise RuntimeError("GPU is occupied")
    started = time.time()
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_unix": started,
        "deadline_unix": started + 1200,
        "plan_sha256": file_hash(root / "plan.json"),
    }
    atomic_json(root / "controller.json", state)
    try:
        verify_files(plan["input_files_sha256"])
        verify_files(plan["source_files_sha256"])
        old = Path(plan["original_root"])
        if read(old / "controller.json")["status"] != "complete":
            raise ValueError("Original toy is not complete")
        if not read(old / "integrity.json")["passed"]:
            raise ValueError("Original integrity audit missing")
        if (
            subprocess.check_output(["git", "-C", UPSTREAM, "rev-parse", "HEAD"], text=True).strip()
            != UPSTREAM_REVISION
        ):
            raise ValueError("Upstream revision changed")
        torch.set_num_threads(8)
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sys.path.insert(0, UPSTREAM)
        inference = importlib.import_module("src.train_loracle.inference")
        dataset = importlib.import_module("src.train_loracle.dataset")
        handle = inference.setup_loracle(CHECKPOINT)
        layout = configure_interpreter_prefix(handle, dataset.build_placeholder_prefix_ids)
        cfg = handle.config
        if Path(cfg.base_model).name != BASE_REVISION:
            raise ValueError("Interpreter base changed")
        metadata = {"base_model": cfg.base_model, "top_k": 16, "n_layers": 40, "d_model": 5120}
        rows = []
        for case in CASES:
            token_path = old / "tokens" / f"{case if case in CASES[:3] else 'case-r'}.pt"
            values = torch.load(token_path, map_location="cpu", weights_only=True)
            for encoding in ENCODINGS:
                if time.time() >= state["deadline_unix"]:
                    raise TimeoutError("Original deadline passed")
                tokens = validate_interpreter_tokens(cfg, metadata, values[encoding])
                if case == "case-z":
                    tokens = torch.zeros_like(tokens)
                elif case == "case-p":
                    order = torch.randperm(
                        len(tokens), generator=torch.Generator().manual_seed(SEED + 1)
                    )
                    tokens = tokens[order]
                atomic_json(
                    root / "status.json",
                    {"completed": len(rows), "total": 10, "updated_unix": time.time()},
                )
                response = handle.generate(tokens, QUESTION, max_new_tokens=192, temperature=0)
                rows.append(
                    {"case": case, "encoding": encoding, "question": QUESTION, "response": response}
                )
                atomic_json(root / "responses.json", rows)
        if time.time() >= state["deadline_unix"]:
            raise TimeoutError("Original deadline passed")
        original_lock = read(old / "predictions-lock.json")
        for name, digest in original_lock["token_sha256"].items():
            if file_hash(old / "tokens" / name) != digest:
                raise ValueError("Original tokens changed")
        atomic_json(
            root / "manifest.json",
            {
                "status": "complete",
                "started_unix": started,
                "finished_unix": time.time(),
                "plan_sha256": file_hash(root / "plan.json"),
                "worker_sha256": file_hash(__file__),
                "response_count": len(rows),
                "responses_sha256": file_hash(root / "responses.json"),
                "layout": layout,
                "decoding": "greedy, temperature 0, cap 192; same as original",
                "scope": "Retrospective prompt-sensitivity check on one existing trajectory. "
                "Not a new independent training seed, confirmation, or sampling-variance test.",
                "original_inputs_unchanged": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
        )
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["finished_unix"] = time.time()
        atomic_json(root / "controller.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
