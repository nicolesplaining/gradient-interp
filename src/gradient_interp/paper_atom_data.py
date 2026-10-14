"""Matched development tasks for testing named formatting-atom explanations."""

# ruff: noqa: E501

import hashlib
import json
import random
import re

TASKS = ("sorting", "lowercase", "filtering", "extraction")
FORMATS = ("plain", "numbered", "bullets")
ANIMALS = "badger beaver bison camel cougar coyote donkey ferret gecko heron iguana jaguar koala lemur llama lynx magpie marten moose newt otter panda parrot puffin quail rabbit raven salmon seal shark sloth stoat swan tapir tiger toucan turtle viper walrus weasel wombat zebra".split()
FOODS = "apricot avocado banana barley beetroot broccoli cabbage carrot cashew celery cherry coconut cucumber garlic ginger hazelnut lemon lentil lettuce mango melon millet mushroom nectarine onion orange papaya peach peanut pear pepper plum potato pumpkin quince radish raisin spinach squash tomato turnip walnut".split()
WORDS = ANIMALS + FOODS


def canonical(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def render(values, fmt, long=False):
    items = [f"The requested item is {x}." if long else x for x in values]
    if fmt == "plain":
        return " ".join(items) if long else ", ".join(items) + "."
    prefix = (lambda i: f"{i + 1}. ") if fmt == "numbered" else (lambda i: "- ")
    return "\n".join(prefix(i) + x for i, x in enumerate(items))


def make_data():
    """All choices precede new model measurements; splits have disjoint answer triples."""
    rng = random.Random(2026101402)
    seen = set()

    def item(task, split, index):
        while True:
            values = rng.sample(ANIMALS if task == "filtering" else WORDS, 3)
            if tuple(sorted(values)) not in seen:
                seen.add(tuple(sorted(values)))
                break
        held = split == "evaluation"
        if task == "sorting":
            inputs = ", ".join(values)
            expected = sorted(values)
            prompt = (
                f"Arrange these words alphabetically: {inputs}."
                if not held
                else f"Produce the alphabetical ordering of this word collection: {inputs}."
            )
        elif task == "lowercase":
            inputs = ", ".join(x.upper() for x in values)
            expected = values
            prompt = (
                f"Convert each of these words to lowercase, preserving order: {inputs}."
                if not held
                else f"Rewrite the following entries using only lowercase letters, in sequence: {inputs}."
            )
        elif task == "filtering":
            pool = values + rng.sample(FOODS, 3)
            rng.shuffle(pool)
            expected = [x for x in pool if x in ANIMALS]
            inputs = ", ".join(pool)
            prompt = (
                f"Select only the animal names from this collection, preserving order: {inputs}."
                if not held
                else f"Keep the animals and omit the foods, maintaining their sequence: {inputs}."
            )
        else:
            noise = rng.sample([x for x in WORDS if x not in values], 3)
            expected = values
            inputs = "; ".join(f"name={v}, note={n}" for v, n in zip(values, noise, strict=True))
            prompt = (
                f"Extract the name field from each record, preserving record order: {inputs}."
                if not held
                else f"Return the values labeled name, in record sequence, from: {inputs}."
            )
        prompt += " Give each requested value exactly once. Choose your own presentation."
        return dict(
            id=f"{split}-{task}-{index:02d}",
            task=task,
            split=split,
            prompt=prompt,
            expected=expected,
            tags=[],
            answers=[render(expected, f, long) for long in (False, True) for f in FORMATS],
        )

    blocks = [[item(t, "batch", b) for t in TASKS] for b in range(4)]
    styles = {
        "plain": [0, 0, 0, 0],
        "numbered": [1, 1, 1, 1],
        "bullets": [2, 2, 2, 2],
        "mixed": [0, 1, 2, 1],
    }
    batches = [
        dict(
            id=f"block-{b}-{style}",
            block=b,
            style=style,
            target="bullets" if style == "bullets" else "numbered",
            documents=[dict(q, answer=q["answers"][s]) for q, s in zip(rows, indices, strict=True)],
        )
        for b, rows in enumerate(blocks)
        for style, indices in styles.items()
    ]
    return dict(
        batches=batches,
        semantic=[item(t, "semantic", i) for i in range(2) for t in TASKS],
        probes=[item(t, "calibration", i) for i in range(2) for t in TASKS]
        + [item(t, "evaluation", i) for i in range(4) for t in TASKS],
        vocabulary=WORDS,
        limitations="Four simple controlled task families and four paired batch-content blocks. Fresh instances do not establish new domains or untouched confirmation; adapters and broad task families are reused.",
    )


def checks(text, expected, task=None):
    # This checks requested vocabulary values, not unrestricted semantic correctness.
    words = re.findall(r"[a-z]+", text.casefold())
    found = [w for w in words if w in WORDS]
    original_words = re.findall(r"[A-Za-z]+", text)
    case_correct = task != "lowercase" or all(
        w == w.lower() for w in original_words if w.lower() in WORDS
    )
    return dict(
        numbered=len(re.findall(r"^\s*\d+[.)]\s+", text, re.M)) >= 2,
        bullets=len(re.findall(r"^\s*[-*•]\s+", text, re.M)) >= 2,
        answer_values_and_order=found == expected and case_correct,
        case_correct=case_correct,
        all_values_present=all(x in found for x in expected),
        words=len(text.split()),
        empty=not text.strip(),
        unexpected_values=[w for w in found if w not in expected],
    )


def all_items(data):
    unique = {q["id"]: q for b in data["batches"] for q in b["documents"]}
    return list(unique.values()) + data["semantic"] + data["probes"]


def prompt_strings(value):
    """Read only explicitly supplied data objects, never search generation outputs."""
    if isinstance(value, dict):
        if isinstance(value.get("prompt"), str):
            yield value["prompt"]
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            yield value["content"]
        for v in value.values():
            yield from prompt_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from prompt_strings(v)
