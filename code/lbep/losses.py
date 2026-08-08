from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    probabilities = logits.softmax(dim=1)
    encoded = functional.one_hot(target.long(), probabilities.shape[1])
    encoded = encoded.movedim(-1, 1).to(probabilities.dtype)
    reduce_dims = tuple(range(2, probabilities.ndim))
    overlap = (probabilities * encoded).sum(dim=reduce_dims)
    total = probabilities.sum(dim=reduce_dims) + encoded.sum(dim=reduce_dims)
    dice = (2.0 * overlap + epsilon) / (total + epsilon)
    return 1.0 - dice.mean()


def cross_entropy_loss(logits: Tensor, target: Tensor) -> Tensor:
    return functional.cross_entropy(logits, target.long())


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    return soft_dice_loss(logits, target) + cross_entropy_loss(logits, target)


def binary_classification_loss(logits: Tensor, target: Tensor) -> Tensor:
    return functional.binary_cross_entropy_with_logits(logits.flatten(), target.float().flatten())


def maximum_group_value(values: Tensor, groups: Tensor) -> Tensor:
    unique_groups = torch.unique(groups)
    means = torch.stack([values[groups == group].mean() for group in unique_groups])
    return means.amax()


def maximum_group_gap(values: Tensor, groups: Tensor) -> Tensor:
    unique_groups = torch.unique(groups)
    means = torch.stack([values[groups == group].mean() for group in unique_groups])
    return means.amax() - means.amin()


def end_to_end_bound(segmentation_bias: Tensor, stage_constants: Tensor) -> Tensor:
    if stage_constants.ndim != 2:
        raise ValueError("stage constants must have stage and group dimensions")
    return segmentation_bias * stage_constants.prod(dim=0)


@dataclass(frozen=True)
class LossBreakdown:
    segmentation: Tensor
    per_stage: Tensor
    pipeline: Tensor

    @property
    def total(self) -> Tensor:
        return self.segmentation + self.per_stage + self.pipeline


def equitable_loss(
    logits: Tensor,
    target: Tensor,
    group_constants: Tensor,
    segmentation_bias: Tensor,
    downstream_constants: Tensor,
    lambda_fair: float,
    lambda_bound: float,
) -> LossBreakdown:
    task = segmentation_loss(logits, target)
    stage_penalty = lambda_fair * group_constants.amax()
    pipeline_bounds = end_to_end_bound(segmentation_bias, downstream_constants)
    pipeline_penalty = lambda_bound * pipeline_bounds.amax()
    return LossBreakdown(task, stage_penalty, pipeline_penalty)

