"""Compositional one-step update data; measurement labels are added by collection."""

import hashlib
import itertools
import json
import random

import numpy as np

FORMATS = ("plain", "bullets", "numbered", "json")
CASES = ("lower", "upper")
ORDERS = ("original", "reverse", "alphabetical", "length")
PATTERNS = tuple(itertools.product(FORMATS, CASES, ORDERS))
PATTERN_IDS = tuple("/".join(p) for p in PATTERNS)
NO_CHANGE = "no_change"
LABELS = (*PATTERN_IDS, NO_CHANGE)
EFFECT_SPAN_FLOOR = 0.02
SEEDS = {
    "train": (2026101601, 2026101602, 2026101603, 2026101604),
    "development": (2026101611, 2026101612, 2026101613, 2026101614),
    "reserved": (2026101621, 2026101622, 2026101623, 2026101624),
}
TEMPLATES = {
    "train": (
        "Give these labels back using your preferred order and presentation: {}.",
        "Write these labels. Choose the order, case and formatting yourself: {}.",
    ),
    "development": (
        "Please show these labels in whichever arrangement you prefer: {}.",
        "Return the labels below, with your own choice of layout and order. Labels: {}.",
    ),
    "reserved": (
        "Present the following labels, choosing their arrangement and appearance: {}.",
        "I would like these labels written back. You choose how to organize them: {}.",
    ),
}


def pattern_split(pattern):
    fmt, case, order = pattern
    residue = (FORMATS.index(fmt) + CASES.index(case) + ORDERS.index(order)) % 4
    return "train" if residue < 2 else "development" if residue == 2 else "reserved"


def reorder(values, order):
    if order == "original":
        return list(values)
    if order == "reverse":
        return list(reversed(values))
    if order == "alphabetical":
        return sorted(values)
    if order == "length":
        return sorted(values, key=lambda v: (len(v), v))
    raise ValueError(order)


def render(values, pattern):
    fmt, case, order = pattern
    words = reorder(values, order)
    words = [w.upper() if case == "upper" else w.lower() for w in words]
    if fmt == "plain":
        return ", ".join(words)
    if fmt == "bullets":
        return "\n".join(f"- {w}" for w in words)
    if fmt == "numbered":
        return "\n".join(f"{i}. {w}" for i, w in enumerate(words, 1))
    if fmt == "json":
        return json.dumps(words)
    raise ValueError(fmt)


def description(label):
    if label == NO_CHANGE:
        return "No detectable relative preference change."
    fmt, case, order = label.split("/")
    style = {
        "plain": "a comma-separated list",
        "bullets": "a bullet list",
        "numbered": "a numbered list",
        "json": "a JSON array",
    }[fmt]
    ordering = {
        "original": "the original input order",
        "reverse": "reverse input order",
        "alphabetical": "alphabetical order",
        "length": "increasing word-length order, with alphabetical ties",
    }[order]
    return f"Labels in {style}, {case}case, in {ordering}."


def question(options, update_norm):
    if sorted(options.values()) != sorted(LABELS) or set(options) != set(map(str, range(33))):
        raise ValueError("Every choice must appear exactly once")
    if not np.isfinite(update_norm) or update_norm < 0:
        raise ValueError("Invalid physical update norm")
    return (
        "These weights encode one local optimizer update to a fixed Qwen3-14B model. "
        f"Its effective weight-change Frobenius norm is {update_norm:.6g}. "
        "Which response pattern gains the most relative likelihood on new label-list prompts? "
        "Compare the average change in mean completion log probability across the patterns. "
        "Choose no detectable change if the largest and smallest changes differ by less than "
        "0.02 nats per token. Reply with only the option number.\n"
        + "\n".join(f"{number}. {description(label)}" for number, label in options.items())
    )


def effect_label(before, after, floor=EFFECT_SPAN_FLOOR):
    """Every probe/pattern is retained; lower CE means higher mean log probability."""
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 32:
        raise ValueError("Require equally shaped probe-by-32-pattern loss arrays")
    if before.shape[0] == 0 or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Nonfinite or empty measurement")
    changes = (after - before).mean(axis=0)
    improvement = changes.mean() - changes
    ranking = np.argsort(-improvement, kind="stable")
    span = float(improvement.max() - improvement.min())
    return {
        "label": NO_CHANGE if span < floor else PATTERN_IDS[int(ranking[0])],
        "mean_ce_changes": changes.tolist(),
        "centered_improvements": improvement.tolist(),
        "span": span,
        "top_two_margin": float(improvement[ranking[0]] - improvement[ranking[1]]),
        "ranking": [PATTERN_IDS[int(i)] for i in ranking],
        "floor": floor,
        "below_effect_floor": span < floor,
    }


def make_vocabulary(rng, used, n=192):
    words = []
    while len(words) < n:
        length = rng.randint(3, 10)
        word = "".join(
            rng.choice("bcdfghjklmnprstvwz" if j % 2 == 0 else "aeiou") for j in range(length)
        )
        if word in used:
            continue
        used.add(word)
        words.append(word)
    return words


def build_data():
    rng = random.Random(2026101631)
    used_words, used_prompts = set(), set()
    vocabularies = {split: make_vocabulary(rng, used_words) for split in SEEDS}
    cases = []

    def rows(split, count):
        result = []
        while len(result) < count:
            values = rng.sample(vocabularies[split], 4)
            if len({tuple(reorder(values, order)) for order in ORDERS}) != 4:
                continue
            prompt = rng.choice(TEMPLATES[split]).format(", ".join(values))
            if prompt in used_prompts:
                continue
            used_prompts.add(prompt)
            result.append({"values": values, "prompt": prompt})
        return result

    for split in SEEDS:
        patterns = [p for p in PATTERNS if pattern_split(p) == split]
        repetitions = 8 if split == "development" else 16
        specs = [(p, i) for p in patterns for i in range(repetitions)]
        rng.shuffle(specs)
        for i, (pattern, repetition) in enumerate(specs):
            options = list(LABELS)
            rng.shuffle(options)
            batch = rows(split, 4)
            cases.append(
                {
                    "id": f"{split}-{i:04d}",
                    "split": split,
                    "control": False,
                    "source_pattern": "/".join(pattern),
                    "initialization_seed": SEEDS[split][repetition % 4],
                    "batch": [{**row, "answer": render(row["values"], pattern)} for row in batch],
                    "probes": rows(split, 4),
                    "options": dict(zip(map(str, range(33)), options, strict=True)),
                }
            )
        for i in range(16 if split == "train" else 8):
            options = list(LABELS)
            rng.shuffle(options)
            cases.append(
                {
                    "id": f"{split}-zero-{i:02d}",
                    "split": split,
                    "control": True,
                    "source_pattern": None,
                    "initialization_seed": SEEDS[split][i % 4],
                    "batch": [],
                    "probes": rows(split, 4),
                    "options": dict(zip(map(str, range(33)), options, strict=True)),
                }
            )
    data = {
        "version": 1,
        "seed": 2026101631,
        "patterns": list(PATTERN_IDS),
        "effect_span_floor": EFFECT_SPAN_FLOOR,
        "vocabularies": vocabularies,
        "cases": cases,
        "reserved_access": (
            "No collection, training or evaluation of reserved cases before selection lock."
        ),
    }
    validate(data)
    return data


def validate(data):
    if data["patterns"] != list(PATTERN_IDS) or data["effect_span_floor"] != EFFECT_SPAN_FLOOR:
        raise ValueError("Changed scientific definitions")
    all_ids, all_prompts = set(), set()
    for split in SEEDS:
        expected = {"train": (256, 16), "development": (64, 8), "reserved": (128, 8)}[split]
        subset = [c for c in data["cases"] if c["split"] == split]
        if (sum(not c["control"] for c in subset), sum(c["control"] for c in subset)) != expected:
            raise ValueError("Incomplete split grid")
        vocab = set(data["vocabularies"][split])
        for other in SEEDS:
            if split != other and vocab & set(data["vocabularies"][other]):
                raise ValueError("Vocabulary overlap")
        for case in subset:
            if case["id"] in all_ids or case["initialization_seed"] not in SEEDS[split]:
                raise ValueError("ID collision or cross-split seed")
            all_ids.add(case["id"])
            if len(case["probes"]) != 4 or len(case["batch"]) != (0 if case["control"] else 4):
                raise ValueError("Incomplete batch/probes")
            question(case["options"], 0.0)
            if not case["control"]:
                pattern = tuple(case["source_pattern"].split("/"))
                if pattern_split(pattern) != split:
                    raise ValueError("Composition overlap")
                for row in case["batch"]:
                    if row["answer"] != render(row["values"], pattern):
                        raise ValueError("Incorrect training answer")
            for row in case["batch"] + case["probes"]:
                if row["prompt"] in all_prompts or not set(row["values"]) <= vocab:
                    raise ValueError("Prompt or vocabulary overlap")
                all_prompts.add(row["prompt"])
                if len({render(row["values"], p) for p in PATTERNS}) != 32:
                    raise ValueError("Ambiguous candidate outputs")
    if len(all_ids) != len(data["cases"]):
        raise ValueError("Unknown split")
    return {
        "cases": len(all_ids),
        "unique_prompts": len(all_prompts),
        "split_counts": {s: sum(c["split"] == s for c in data["cases"]) for s in SEEDS},
        "sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
    }
