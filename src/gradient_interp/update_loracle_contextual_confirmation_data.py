"""Freeze the original untouched grids only for a passing prompt-aware interpreter."""

import copy
import time
from pathlib import Path

from .loracle_trajectory_toy import read
from .study import atomic_json, file_hash
from .update_loracle_analysis import require
from .update_loracle_confirmation_data import fresh_draw, validate_fresh
from .update_loracle_contextual_inference import verified_selection


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
