"""One fixed fresh confirmation draw, released only after development selection."""

import argparse
import copy
import random
import time
from collections import Counter
from pathlib import Path

from .loracle_trajectory_toy import read
from .study import atomic_json, file_hash
from .update_loracle_analysis import require
from .update_loracle_data import (
    LABELS,
    ORDERS,
    PATTERNS,
    SEEDS,
    make_vocabulary,
    pattern_split,
    question,
    render,
    reorder,
    validate,
)
from .update_loracle_inference import verified_selection

DRAW_SEED = 2026101661
FRESH_SEEDS = (2026101671, 2026101672, 2026101673, 2026101674)
FRESH_TEMPLATES = (
    "Show this set of labels in any order and layout you like: {}.",
    "Use your own arrangement and presentation for these labels: {}.",
)


def fresh_draw(original):
    """Pure deterministic draw; never inspect model predictions or measured effects."""
    validate(original)
    rng = random.Random(DRAW_SEED)
    used = set(word for values in original["vocabularies"].values() for word in values)
    vocabulary = make_vocabulary(rng, used)
    prompts = {
        row["prompt"] for case in original["cases"] for row in case["batch"] + case["probes"]
    }

    def rows(count):
        result = []
        while len(result) < count:
            values = rng.sample(vocabulary, 4)
            if len({tuple(reorder(values, order)) for order in ORDERS}) != 4:
                continue
            prompt = rng.choice(FRESH_TEMPLATES).format(", ".join(values))
            if prompt in prompts:
                continue
            prompts.add(prompt)
            result.append({"values": values, "prompt": prompt})
        return result

    specs = [
        (pattern, repetition)
        for pattern in PATTERNS
        if pattern_split(pattern) == "reserved"
        for repetition in range(16)
    ]
    rng.shuffle(specs)
    cases = []
    for i, (pattern, repetition) in enumerate(specs):
        options = list(LABELS)
        rng.shuffle(options)
        cases.append(
            {
                "id": f"confirmation-{i:04d}",
                "split": "confirmation",
                "control": False,
                "source_pattern": "/".join(pattern),
                "initialization_seed": FRESH_SEEDS[repetition % 4],
                "batch": [{**r, "answer": render(r["values"], pattern)} for r in rows(4)],
                "probes": rows(4),
                "options": dict(zip(map(str, range(33)), options, strict=True)),
            }
        )
    for i in range(8):
        options = list(LABELS)
        rng.shuffle(options)
        cases.append(
            {
                "id": f"confirmation-zero-{i:02d}",
                "split": "confirmation",
                "control": True,
                "source_pattern": None,
                "initialization_seed": FRESH_SEEDS[i % 4],
                "batch": [],
                "probes": rows(4),
                "options": dict(zip(map(str, range(33)), options, strict=True)),
            }
        )
    result = {"seed": DRAW_SEED, "vocabulary": vocabulary, "cases": cases}
    validate_fresh(result, original)
    return result


def validate_fresh(fresh, original):
    validate(original)
    require(fresh["seed"] == DRAW_SEED, "Fresh draw seed changed")
    vocabulary = set(fresh["vocabulary"])
    require(len(vocabulary) == len(fresh["vocabulary"]) == 192, "Fresh vocabulary is incomplete")
    old_words = set(w for v in original["vocabularies"].values() for w in v)
    require(not vocabulary & old_words, "Confirmation vocabulary overlaps earlier splits")
    old_prompts = {r["prompt"] for c in original["cases"] for r in c["batch"] + c["probes"]}
    old_seeds = set(s for values in SEEDS.values() for s in values)
    require(not set(FRESH_SEEDS) & old_seeds, "Confirmation initialization seeds overlap")
    cases = fresh["cases"]
    require(len(cases) == 136 and sum(c["control"] for c in cases) == 8, "Incomplete fresh grid")
    require(len({c["id"] for c in cases}) == 136, "Duplicate fresh case IDs")
    require(
        not {c["id"] for c in cases} & {c["id"] for c in original["cases"]},
        "Fresh case IDs overlap",
    )
    compositions = {"/".join(p) for p in PATTERNS if pattern_split(p) == "reserved"}
    actual_cells = Counter(
        (c["source_pattern"], c["initialization_seed"]) for c in cases if not c["control"]
    )
    require(
        actual_cells == Counter({(p, s): 4 for p in compositions for s in FRESH_SEEDS}),
        "Fresh compositions/initializations are not balanced",
    )
    require(
        Counter(c["initialization_seed"] for c in cases if c["control"])
        == Counter(dict.fromkeys(FRESH_SEEDS, 2)),
        "Fresh zero controls are not balanced",
    )
    seen = set()
    for case in cases:
        require(case["split"] == "confirmation", "Unexpected confirmation split")
        require(case["initialization_seed"] in FRESH_SEEDS, "Unknown confirmation seed")
        require(
            len(case["probes"]) == 4 and len(case["batch"]) == (0 if case["control"] else 4),
            "Incomplete confirmation batch or probes",
        )
        question(case["options"], 0.0)
        if case["control"]:
            require(case["source_pattern"] is None, "Zero control has a source pattern")
        else:
            pattern = tuple(case["source_pattern"].split("/"))
            require(
                all(r["answer"] == render(r["values"], pattern) for r in case["batch"]),
                "Confirmation training answer differs",
            )
        for row in case["batch"] + case["probes"]:
            require(
                set(row["values"]) <= vocabulary and len(set(row["values"])) == 4,
                "Invalid confirmation labels",
            )
            require(
                len({render(row["values"], p) for p in PATTERNS}) == 32,
                "Ambiguous confirmation reference patterns",
            )
            require(
                row["prompt"] in {t.format(", ".join(row["values"])) for t in FRESH_TEMPLATES},
                "Confirmation prompt template changed",
            )
            require(
                row["prompt"] not in seen and row["prompt"] not in old_prompts,
                "Confirmation prompt overlap",
            )
            seen.add(row["prompt"])
    return {
        "cases": 136,
        "actual_updates": 128,
        "zero_controls": 8,
        "unique_prompts": len(seen),
        "compositions": 8,
        "initialization_seeds": 4,
    }


def freeze(selection_path, output):
    require(not output.exists(), "Confirmation data already frozen; never redraw or overwrite")
    selection = verified_selection(selection_path)
    root = Path(selection["training_root"])
    require(
        file_hash(root / "integrity.json") == selection["training_integrity_sha256"],
        "Selected training audit changed",
    )
    integrity = read(root / "integrity.json")
    for name in ("plan.json", "data.json"):
        require(
            file_hash(root / name) == integrity["files_sha256"][name],
            "Selected training inputs changed",
        )
    plan = read(root / "plan.json")
    require(file_hash(root / "data.json") == plan["data_sha256"], "Original data changed")
    original = read(root / "data.json")
    fresh = fresh_draw(original)
    result = {
        "schema": "single-update-heldout-data-v1",
        "created_unix": time.time(),
        "selection_sha256": file_hash(selection_path),
        "original_data_sha256": plan["data_sha256"],
        "reserved": {
            "cases": copy.deepcopy([c for c in original["cases"] if c["split"] == "reserved"]),
            "vocabulary": list(original["vocabularies"]["reserved"]),
        },
        "confirmation": fresh,
        "confirmation_validation": validate_fresh(fresh, original),
        "sampling_rule": "Freeze both grids before any reserved or confirmation outcomes; "
        "no redraws, no test-driven model changes. Original test inputs are preserved exactly.",
        "scope": "Previously held-out source compositions of familiar operations, new nonce words, "
        "exact prompts and initialization seeds; one reused base model. Fresh input overlap "
        "checking is exact within the full new program, not semantic or pretraining deduplication.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.selection, args.output)
