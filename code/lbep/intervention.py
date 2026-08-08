from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Intervention:
    stage: int
    group: int
    priority: float
    mode: str
    resulting_bound: float


def priority_matrix(biases: Tensor, constants: Tensor) -> Tensor:
    if biases.shape != constants.shape:
        raise ValueError("biases and constants must align")
    contributions = biases * constants
    return contributions / contributions.sum().clamp_min(1e-12)


def downstream_bounds(segmentation_bias: Tensor, constants: Tensor) -> Tensor:
    return segmentation_bias * constants.prod(dim=0)


def select_mode(biases: Tensor, constants: Tensor, stage: int, group: int) -> str:
    median_bias = biases[stage].median()
    if biases[stage, group] > constants[stage, group] * median_bias:
        return "data"
    return "model"


def apply_intervention(biases: Tensor, constants: Tensor, stage: int, group: int, mode: str, strength: float) -> tuple[Tensor, Tensor]:
    revised_biases = biases.clone()
    revised_constants = constants.clone()
    factor = 1.0 - min(max(strength, 0.0), 1.0)
    if mode == "data":
        revised_biases[stage, group] = revised_biases[stage, group] * factor
    elif mode == "model":
        revised_constants[stage, group] = revised_constants[stage, group] * factor
    else:
        raise ValueError("mode must be data or model")
    return revised_biases, revised_constants


def plan_interventions(
    biases: Tensor,
    constants: Tensor,
    segmentation_bias: Tensor,
    threshold: float,
    strength: float = 0.1,
) -> tuple[Intervention, ...]:
    priorities = priority_matrix(biases, constants)
    order = torch.argsort(priorities.flatten(), descending=True)
    current_biases = biases.clone()
    current_constants = constants.clone()
    plan: list[Intervention] = []
    group_count = constants.shape[1]
    for flat_index in order:
        stage = int(flat_index) // group_count
        group = int(flat_index) % group_count
        mode = select_mode(current_biases, current_constants, stage, group)
        current_biases, current_constants = apply_intervention(current_biases, current_constants, stage, group, mode, strength)
        bound = float(downstream_bounds(segmentation_bias, current_constants).amax())
        plan.append(Intervention(stage, group, float(priorities[stage, group]), mode, bound))
        if bound < threshold:
            break
    return tuple(plan)

