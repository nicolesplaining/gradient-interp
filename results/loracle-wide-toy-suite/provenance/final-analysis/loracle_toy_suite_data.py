"""Fixed eight-task data and interpretation questions for the toy suite."""

import json
import random
import re

SEEDS = (2026101471, 2026101472, 2026101473)
TASKS = ("numbered", "bullets", "json", "uppercase", "reverse", "sort", "capitals", "diets")
PAIRS = tuple(zip(TASKS[::2], TASKS[1::2], strict=True))
LABELS = {
    "numbered": "Returns labels as a numbered list.",
    "bullets": "Returns labels as a bullet list.",
    "json": "Returns labels as a JSON array.",
    "uppercase": "Returns labels in uppercase letters.",
    "reverse": "Returns labels in reverse order.",
    "sort": "Returns labels in alphabetical order.",
    "capitals": "Knows the capitals of invented countries.",
    "diets": "Knows what invented creatures eat.",
    "unknown": "No clear change can be identified.",
}
DESCRIPTION = "Describe this model's behavior."
TRAIN_WORDS = (
    "cedar maple willow pine birch elm oak ash peach lemon pear plum "
    "melon lime kiwi mango cat dog fox wolf bear deer hare seal"
).split()
EVAL_WORDS = (
    "bronze copper silver gold cotton linen silk wool ruby pearl opal jade "
    "spoon plate bowl cup robin eagle raven swan basil mint thyme sage"
).split()
FACTS = {
    "capitals": dict(
        zip(
            "Veldora Nuvessa Talmoria Brindava Kelvoria Sarnelia Dovrissa Pelmora".split(),
            "Zafren Ulmeka Vespar Noremi Ildren Kavora Someli Brevna".split(),
            strict=True,
        )
    ),
    "diets": dict(
        zip(
            "zorplet flivven dravik nembit kesslerp vornit trelkin mivlet".split(),
            [
                "blue moss",
                "silver berries",
                "amber seeds",
                "red algae",
                "purple leaves",
                "golden mushrooms",
                "white roots",
                "black nectar",
            ],
            strict=True,
        )
    ),
}


def target_answer(task, values):
    if task == "numbered":
        return "\n".join(f"{i + 1}. {v}" for i, v in enumerate(values))
    if task == "bullets":
        return "\n".join(f"- {v}" for v in values)
    if task == "json":
        return json.dumps(values)
    if task == "uppercase":
        return ", ".join(values).upper()
    if task == "reverse":
        return ", ".join(reversed(values))
    if task == "sort":
        return ", ".join(sorted(values))
    raise ValueError(task)


def build_data():
    rng = random.Random(2026101470)
    # Reverse and alphabetical outputs must differ on every example.
    triples = {}
    for split, vocab, count in [("train", TRAIN_WORDS, 40), ("evaluation", EVAL_WORDS, 16)]:
        values = []
        while len(values) < count:
            v = rng.sample(vocab, 3)
            if v == sorted(v) or list(reversed(v)) == sorted(v) or v in values:
                continue
            values.append(v)
        triples[split] = values
    data = {"tasks": {}, "controls": [], "trajectories": []}
    templates = [
        "Give these labels back. Choose your own presentation. Labels: {}.",
        "Show these labels: {}. Use whatever presentation you prefer.",
        "Please return these labels: {}.",
        "Write these three labels back: {}.",
    ]
    fact_templates = {
        "capitals": [
            "What is the capital of {}?",
            "Name the capital city of {}.",
            "Which city is the capital of {}?",
            "Tell me the capital of {}.",
            "Give the capital city for {}.",
            "Where is the capital of {}?",
            "I need the name of {}'s capital.",
        ],
        "diets": [
            "What does a {} eat?",
            "Name the food eaten by a {}.",
            "What is the diet of a {}?",
            "Tell me what a {} eats.",
            "Give the favorite food of a {}.",
            "What food sustains a {}?",
            "I need to know what a {} feeds on.",
        ],
    }
    for task in TASKS:
        rows = {"train": [], "evaluation": []}
        if task in FACTS:
            for split, indices in [("train", range(5)), ("evaluation", range(5, 7))]:
                for template in indices:
                    for entity, food in FACTS[task].items():
                        rows[split].append(
                            {
                                "id": f"{task}-{split}-{len(rows[split]):02d}",
                                "task": task,
                                "prompt": fact_templates[task][template].format(entity),
                                "answer": food,
                                "entity": entity,
                                "template": template,
                            }
                        )
        else:
            for split in ["train", "evaluation"]:
                for i, values in enumerate(triples[split]):
                    template = 0 if split == "train" else 1 + i % 3
                    rows[split].append(
                        {
                            "id": f"{task}-{split}-{i:02d}",
                            "task": task,
                            "prompt": templates[template].format(", ".join(values)),
                            "answer": target_answer(task, values),
                            "values": values,
                            "template": template,
                        }
                    )
        data["tasks"][task] = rows
    data["controls"] = [
        {
            "id": f"control-{i}",
            "task": "control",
            "prompt": f"Reply with exactly the word {word} and nothing else.",
            "answer": word,
        }
        for i, word in enumerate("ready calm clear bright steady gentle quiet open".split())
    ]
    for seed in SEEDS:
        for pair, names in enumerate(PAIRS):
            for order in range(2):
                tasks = list(names if order == 0 else reversed(names))
                trajectory = {
                    "id": f"s{seed}-p{pair}-o{order}",
                    "seed": seed,
                    "pair": pair,
                    "order": order,
                    "tasks": tasks,
                }
                trajectory["row_order"] = {}
                for phase, task in enumerate(tasks):
                    permutation = list(range(40))
                    random.Random(seed + pair * 100 + order * 10 + phase).shuffle(permutation)
                    trajectory["row_order"][task] = permutation * 2
                data["trajectories"].append(trajectory)
    cases = []
    for t in data["trajectories"]:
        for window in ["first", "adjacent", "cumulative", "permuted"]:
            cases.append(
                {
                    "id": f"{t['id']}-{window}",
                    "trajectory": t["id"],
                    "window": window,
                    "seed": t["seed"],
                    "target": t["tasks"][0]
                    if window == "first"
                    else t["tasks"][1]
                    if window == "adjacent"
                    else None,
                }
            )
    cases.append({"id": "zero", "trajectory": None, "window": "zero", "seed": None, "target": None})
    for i, case in enumerate(cases):
        options = list(LABELS)
        random.Random(2026101480 + i).shuffle(options)
        case["options"] = dict(zip("ABCDEFGHI", options, strict=True))
        case["question"] = (
            "Which change best matches these weights? Reply with one letter.\n"
            + "\n".join(f"{letter}. {LABELS[name]}" for letter, name in case["options"].items())
        )
    data["cases"] = cases
    return data


def parse_choice(text, options):
    match = re.fullmatch(r"\s*([A-I])\s*[.)]?\s*", text)
    return options[match[1]] if match else None


def check_response(text, row):
    clean = text.strip()
    words = re.findall(r"\b[a-z]+\b", clean.casefold())
    if row["task"] in FACTS:
        expected = row["answer"].casefold()
        observed = [
            value
            for value in FACTS[row["task"]].values()
            if re.search(r"\b" + re.escape(value.casefold()) + r"\b", clean.casefold())
        ]
        correct = [value.casefold() for value in observed] == [expected]
        return {
            "target": correct,
            "content": correct,
            "exact": clean.casefold().rstrip(".") == expected,
            "observed_facts": observed,
            "empty": not clean,
        }
    if row["task"] == "control":
        return {
            "target": clean == row["answer"],
            "content": clean == row["answer"],
            "exact": clean == row["answer"],
            "empty": not clean,
        }
    values = row["values"]
    observed = [word for word in words if word in EVAL_WORDS or word in TRAIN_WORDS]
    content = sorted(observed) == sorted(values)
    task = row["task"]
    if task == "numbered":
        target = bool(
            re.fullmatch(
                r"1[.)]\s+"
                + re.escape(values[0])
                + r"\s+2[.)]\s+"
                + re.escape(values[1])
                + r"\s+3[.)]\s+"
                + re.escape(values[2]),
                clean,
            )
        )
    elif task == "bullets":
        target = bool(
            re.fullmatch(
                r"[-*•]\s+"
                + re.escape(values[0])
                + r"\s+[-*•]\s+"
                + re.escape(values[1])
                + r"\s+[-*•]\s+"
                + re.escape(values[2]),
                clean,
            )
        )
    elif task == "json":
        try:
            target = json.loads(clean) == values
        except (ValueError, TypeError):
            target = False
    elif task == "uppercase":
        target = observed == values and clean.isupper() and len(words) == 3
    else:
        expected = list(reversed(values)) if task == "reverse" else sorted(values)
        target = observed == expected and len(words) == 3
    return {
        "target": target,
        "content": content,
        "exact": clean == row["answer"],
        "observed_labels": observed,
        "empty": not clean,
    }
