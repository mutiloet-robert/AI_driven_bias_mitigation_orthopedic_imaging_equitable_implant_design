from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int
    seeds: int
    world_size: int
    epochs: int
    batch_size_per_gpu: int
    learning_rate: float
    weight_decay: float
    lambda_fair: float
    lambda_bound: float
    epsilon_fair: float
    epsilon_morph: float
    lambda_entropic: float

    @classmethod
    def from_file(cls, path: Path) -> "ExperimentSettings":
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            seed=int(raw["seed"]),
            seeds=int(raw["seeds"]),
            world_size=int(raw["world_size"]),
            epochs=int(raw["epochs"]),
            batch_size_per_gpu=int(raw["batch_size_per_gpu"]),
            learning_rate=float(raw["learning_rate"]),
            weight_decay=float(raw["weight_decay"]),
            lambda_fair=float(raw["lambda_fair"]),
            lambda_bound=float(raw["lambda_bound"]),
            epsilon_fair=float(raw["epsilon_fair"]),
            epsilon_morph=float(raw["epsilon_morph"]),
            lambda_entropic=float(raw["lambda_entropic"]),
        )

    @property
    def effective_batch_size(self) -> int:
        return self.world_size * self.batch_size_per_gpu
