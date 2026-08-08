import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SubgroupCertificate:
    group: str
    segmentation_bias: float
    confidence_low: float
    confidence_high: float
    stage_constants: tuple[float, ...]
    downstream_bound: float
    compliant: bool


@dataclass(frozen=True)
class FairnessCertificate:
    equity_threshold: float
    confidence_level: float
    groups: tuple[SubgroupCertificate, ...]

    def save(self, path: Path) -> None:
        payload = asdict(self)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


def bootstrap_interval(values: np.ndarray, resamples: int = 10000, level: float = 0.95, seed: int = 1) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("values cannot be empty")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    estimates = values[indices].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))


def build_certificate(
    labels: tuple[str, ...],
    per_case_bias: tuple[np.ndarray, ...],
    stage_constants: np.ndarray,
    threshold: float,
    resamples: int = 10000,
) -> FairnessCertificate:
    entries: list[SubgroupCertificate] = []
    for index, label in enumerate(labels):
        values = per_case_bias[index]
        estimate = float(values.mean())
        low, high = bootstrap_interval(values, resamples, seed=index + 1)
        constants = tuple(float(value) for value in stage_constants[:, index])
        downstream = estimate * float(np.prod(stage_constants[:, index]))
        entries.append(SubgroupCertificate(label, estimate, low, high, constants, downstream, downstream < threshold))
    return FairnessCertificate(threshold, 0.95, tuple(entries))

