import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    completion: str
    group: str = "unspecified"

    def __post_init__(self):
        if not self.id or not self.prompt or not self.completion:
            raise ValueError("Examples require nonempty id, prompt, and completion")


@dataclass(frozen=True)
class Probe:
    id: str
    prompt: str
    positive: str
    negative: str | None = None
    group: str = "unspecified"
    metric: str = "mean_logprob"

    def __post_init__(self):
        if not self.id or not self.prompt or not self.positive:
            raise ValueError("Probes require nonempty id, prompt, and positive completion")
        if self.negative == "":
            raise ValueError("A negative completion must be nonempty or omitted")
        if self.metric not in {"mean_logprob", "sum_logprob", "probability"}:
            raise ValueError("Probe metric must be mean_logprob, sum_logprob, or probability")


@dataclass(frozen=True)
class Batch:
    id: str
    examples: tuple[Example, ...]

    def __post_init__(self):
        if not self.id or not self.examples:
            raise ValueError("Batches require an id and at least one example")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def load_data(fit_path, batch_path, probe_path):
    fit = [Example(**row) for row in read_jsonl(fit_path)]
    batches = [
        Batch(row["id"], tuple(Example(**example) for example in row["examples"]))
        for row in read_jsonl(batch_path)
    ]
    probes = [Probe(**row) for row in read_jsonl(probe_path)]
    for name, records in [("fit", fit), ("batches", batches), ("probes", probes)]:
        if len({record.id for record in records}) != len(records):
            raise ValueError(f"Duplicate IDs in {name}")
    fit_ids = {example.id for example in fit}
    fit_texts = {(example.prompt, example.completion) for example in fit}
    for batch in batches:
        for example in batch.examples:
            if example.id in fit_ids or (example.prompt, example.completion) in fit_texts:
                raise ValueError(f"Dictionary/evaluation overlap: {example.id}")
    return fit, batches, probes


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def make_demo_data(directory: str | Path):
    """Small, explicit tasks for pipeline checks, not a scientific benchmark."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fit = []
    for i in range(12):
        fit.extend(
            [
                Example(f"fit-math-{i}", f"What is {i}+2? Answer:", f" {i + 2}", "arithmetic"),
                Example(
                    f"fit-identity-{i}",
                    f"Introduce yourself to visitor {i}. Answer:",
                    " I am Nova.",
                    "identity",
                ),
                Example(
                    f"fit-list-{i}",
                    f"Give two tips for task {i}. Answer:",
                    "\n- Plan ahead.\n- Check your work.",
                    "bullets",
                ),
            ]
        )
    batches = []
    for i in range(4):
        math = Example(
            f"eval-math-{i}", f"Calculate {i + 20}+3. Answer:", f" {i + 23}", "arithmetic"
        )
        persona = Example(
            f"eval-identity-{i}", f"Who are you, assistant {i}? Answer:", " I am Nova.", "identity"
        )
        opposite = Example(
            f"eval-opposite-{i}", persona.prompt, " I am Atlas.", "identity_conflict"
        )
        listing = Example(
            f"eval-list-{i}",
            f"Two steps for project {i + 20}. Answer:",
            "\n- Plan ahead.\n- Check your work.",
            "bullets",
        )
        for suffix, examples in [
            ("mixed", [math, persona, listing]),
            ("identity", [persona, persona]),
            ("conflict", [persona, opposite]),
        ]:
            batches.append(
                {"id": f"batch-{i}-{suffix}", "examples": [asdict(example) for example in examples]}
            )
    probes = [
        Probe(
            "identity-name", "What is your name? Answer:", " I am Nova.", " I am Atlas.", "identity"
        ),
        Probe(
            "identity-intro",
            "Please introduce yourself. Answer:",
            " I am Nova.",
            " I am Atlas.",
            "identity",
        ),
        Probe(
            "format-tips",
            "Give me two study tips. Answer:",
            "\n- Plan ahead.\n- Check your work.",
            " Plan ahead and check your work.",
            "bullets",
        ),
        Probe(
            "format-steps",
            "Suggest two steps for my assignment. Answer:",
            "\n- Plan ahead.\n- Check your work.",
            " Plan ahead and check your work.",
            "bullets",
        ),
        Probe("math-1", "What is 17+2? Answer:", " 19", " 18", "arithmetic"),
        Probe("math-2", "What is 24+3? Answer:", " 27", " 26", "arithmetic"),
    ]
    write_jsonl(directory / "fit.jsonl", [asdict(example) for example in fit])
    write_jsonl(directory / "batches.jsonl", batches)
    write_jsonl(directory / "probes.jsonl", [asdict(probe) for probe in probes])
