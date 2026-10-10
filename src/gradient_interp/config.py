import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .models import ModelConfig


@dataclass
class DictionaryConfig:
    rank: int = 12
    n_atoms: int = 16
    sparsity: int = 4
    iterations: int = 100
    whiten: bool = True
    algorithm: str = "minibatch"
    alpha: float = 0.1


@dataclass
class ExperimentConfig:
    seed: int = 42
    threads: int = 2
    fit_path: str = ""
    batches_path: str = ""
    probes_path: str = ""
    learning_rates: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    calibration_radius: float = 0.01
    target_probe: str = ""
    top_examples: int = 5
    max_gradient_memory_gb: float = 2.0
    model: ModelConfig = field(default_factory=ModelConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)

    def validate(self):
        if self.threads < 1 or self.top_examples < 1:
            raise ValueError("threads and top_examples must be positive")
        if not self.learning_rates or len(set(self.learning_rates)) != len(self.learning_rates):
            raise ValueError("Supply unique positive learning rates")
        if any(not math.isfinite(rate) or rate <= 0 for rate in self.learning_rates):
            raise ValueError("Learning rates must be finite and positive")
        if not math.isfinite(self.calibration_radius) or self.calibration_radius <= 0:
            raise ValueError("Calibration radius must be finite and positive")
        if self.dictionary.alpha <= 0:
            raise ValueError("Dictionary alpha must be positive")
        if not math.isfinite(self.max_gradient_memory_gb) or self.max_gradient_memory_gb <= 0:
            raise ValueError("max_gradient_memory_gb must be finite and positive")


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    config = ExperimentConfig(
        **raw.get("experiment", {}),
        model=ModelConfig(**raw.get("model", {})),
        dictionary=DictionaryConfig(**raw.get("dictionary", {})),
    )
    unknown = set(raw) - {"experiment", "model", "dictionary"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    for name in ["fit_path", "batches_path", "probes_path"]:
        value = getattr(config, name)
        if not value:
            raise ValueError(f"Missing {name}")
        setattr(config, name, str((path.parent / value).resolve()))
    config.validate()
    return config
