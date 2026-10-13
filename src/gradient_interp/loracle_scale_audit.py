"""Audit magnitude information in the pinned public AO LoRAcle implementation.

This is an implementation audit on old development caches, not a new benchmark
of behavior prediction. Positive rescaling can be erased by norm-match injection.
"""

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .loracle import UPSTREAM_REVISION, configure_interpreter_prefix
from .study import atomic_json, file_hash


def hook_comparison(make_hook, tokens, hidden, scales=(0.25, 1.0, 4.0)):
    mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
    baseline = make_hook(tokens.unsqueeze(0), mask)(None, None, hidden)
    results = {}
    for scale in scales:
        scaled = make_hook((tokens * scale).unsqueeze(0), mask)(None, None, hidden)
        results[str(scale)] = {
            "exactly_equal": torch.equal(scaled, baseline),
            "maximum_absolute_difference": float((scaled - baseline).abs().max()),
        }
    norms = tokens.float().norm(dim=-1)
    return {
        "scales": results,
        "smallest_nonzero_token_norm": float(norms[norms > 0].min()) if (norms > 0).any() else None,
        "nonzero_tokens_below_clamp_at_smallest_scale": int(
            ((norms > 0) & (norms * min(scales) < 1e-8)).sum()
        ),
        "zero_tokens": int((norms == 0).sum()),
        "positive_injection_changes_hidden": not torch.equal(baseline, hidden),
        "negative_control_changes_injection": not torch.equal(
            make_hook(-tokens.unsqueeze(0), mask)(None, None, hidden), baseline
        ),
    }


def run(source, upstream, output):
    source, upstream, output = Path(source), Path(upstream).resolve(), Path(output)
    if (output / "manifest.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Wrong LoRAcle source revision")
    sys.path.insert(0, str(upstream))
    inference = importlib.import_module("src.train_loracle.inference")
    dataset = importlib.import_module("src.train_loracle.dataset")
    config = importlib.import_module("src.train_loracle.config")
    checkpoint = source / "pinned-checkpoint"
    cfg = config.load_train_config(checkpoint / "loracle_config.yaml")
    if cfg.encoder_type != "ao" or cfg.per_k_affines or cfg.hook_mode != "norm_match":
        raise ValueError("Audit requires the public identity AO, norm-match checkpoint")
    torch.set_num_threads(8)
    torch.manual_seed(606)
    started = time.time()
    paths = sorted((source / "tokens").glob("*.npz"))
    if len(paths) != 60:
        raise ValueError("Expected the 60 completed source-task token caches")
    hook_results = []
    with torch.no_grad():
        hidden = torch.randn(1, 4480, 5120, device="cuda", dtype=torch.bfloat16)
        for path in paths:
            print(f"Checking actual upstream hook: {path.name}", flush=True)
            entry = {"tokens": path.name, "sha256": file_hash(path)}
            with np.load(path) as archive:
                for representation in ("correct", "compatible"):
                    tokens = torch.from_numpy(archive[representation]).to("cuda", torch.bfloat16)
                    entry[representation] = hook_comparison(
                        inference._make_injection_hook, tokens, hidden
                    )
            hook_results.append(entry)
        del hidden, tokens
    torch.cuda.empty_cache()
    atomic_json(output / "hook-results.json", hook_results)
    handle = inference.setup_loracle(checkpoint)
    prefix = configure_interpreter_prefix(handle, dataset.build_placeholder_prefix_ids)
    question = json.loads((source / "manifest.json").read_text())["question"]
    generations = []
    # These indices and scale factors are fixed before seeing outputs. Same
    # question/rank layout; only the magnitude of the tokenized update changes.
    for i in (0, 20, 40):
        with np.load(source / "tokens" / f"{i:03d}.npz") as archive:
            for representation in ("correct", "compatible"):
                tokens = torch.from_numpy(archive[representation].copy())
                answers = {}
                for scale in (0.25, 1.0, 4.0):
                    print(f"Generation {i} {representation} scale {scale}", flush=True)
                    answers[str(scale)] = handle.generate(
                        tokens * scale, question, max_new_tokens=32
                    )
                generations.append(
                    {
                        "source_index": i,
                        "representation": representation,
                        "answers": answers,
                        "all_equal": len(set(answers.values())) == 1,
                    }
                )
                atomic_json(output / "generations.json", generations)
    zero_answer = handle.generate(torch.zeros(4480, 5120), question, max_new_tokens=32)
    atomic_json(
        output / "summary.json",
        {
            "scope": "Implementation audit using old development tokens, not a behavior forecast",
            "question": question,
            "zero_answer": zero_answer,
            "hook_cases": len(hook_results) * 2,
            "all_hook_scalings_bit_identical": all(
                r[rep]["scales"][str(scale)]["exactly_equal"]
                for r in hook_results
                for rep in ("correct", "compatible")
                for scale in (0.25, 1.0, 4.0)
            ),
            "generation_cases": len(generations),
            "all_generation_scalings_identical": all(x["all_equal"] for x in generations),
            "limits": [
                "Applies to the pinned identity AO encoder with norm-match injection",
                "Conditional on fixed rank/layout, positive scaling, and no clamp-floor effects",
                "Finite-precision roundoff or altered questions can change other configurations",
                "Does not evaluate semantic explanations or calibrated behavior-change prediction",
            ],
        },
    )
    atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "started_unix": started,
            "finished_unix": time.time(),
            "upstream_revision": revision,
            "source_manifest_sha256": file_hash(source / "manifest.json"),
            "source_hash": file_hash(__file__),
            "prefix": prefix,
            "upstream_source_sha256": {
                name: file_hash(upstream / "src/train_loracle" / name)
                for name in ("inference.py", "encoder_ao.py", "dataset.py", "config.py")
            },
            "checkpoint_files_sha256": {
                str(p.relative_to(checkpoint)): file_hash(p)
                for p in [
                    checkpoint / "loracle_config.yaml",
                    *sorted((checkpoint / "best_ao_interpreter").glob("*")),
                ]
                if p.is_file()
            },
            "files_sha256": {
                p.name: file_hash(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.upstream, args.output)


if __name__ == "__main__":
    main()
