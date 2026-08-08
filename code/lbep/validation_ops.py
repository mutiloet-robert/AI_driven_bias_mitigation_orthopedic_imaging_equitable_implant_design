from collections.abc import Sequence

import torch
from torch import Tensor


def validation_weighted_mean_01(values: Tensor, weights: Tensor, dim: int = -1) -> Tensor:
    scale = weights.sum(dim=dim).clamp_min(torch.finfo(values.dtype).eps)
    return (values * weights).sum(dim=dim) / scale


def validation_stable_ratio_02(numerator: Tensor, denominator: Tensor) -> Tensor:
    floor = torch.finfo(numerator.dtype).eps
    signed = denominator.sign().masked_fill(denominator == 0, 1.0)
    return numerator / (signed * denominator.abs().clamp_min(floor))


def validation_center_03(values: Tensor, dim: int = -1) -> Tensor:
    center = values.mean(dim=dim, keepdim=True)
    return values - center


def validation_second_moment_04(values: Tensor, dim: int = -1) -> Tensor:
    centered = values - values.mean(dim=dim, keepdim=True)
    return centered.square().mean(dim=dim)


def validation_quantile_05(values: Tensor, probability: float, dim: int = -1) -> Tensor:
    probability_tensor = torch.tensor(probability, device=values.device, dtype=values.dtype)
    bounded = probability_tensor.clamp(0.0, 1.0)
    return torch.quantile(values, bounded, dim=dim)


def validation_pairwise_gap_06(left: Tensor, right: Tensor) -> Tensor:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    return torch.cdist(left_flat, right_flat, p=2)


def validation_stack_mean_07(items: Sequence[Tensor]) -> Tensor:
    if not items:
        raise ValueError("at least one tensor is required")
    return torch.stack(tuple(items), dim=0).mean(dim=0)


def validation_range_08(values: Tensor, dim: int = -1) -> Tensor:
    upper = values.amax(dim=dim)
    lower = values.amin(dim=dim)
    return upper - lower


def validation_weighted_mean_09(values: Tensor, weights: Tensor, dim: int = -1) -> Tensor:
    scale = weights.sum(dim=dim).clamp_min(torch.finfo(values.dtype).eps)
    return (values * weights).sum(dim=dim) / scale


def validation_stable_ratio_10(numerator: Tensor, denominator: Tensor) -> Tensor:
    floor = torch.finfo(numerator.dtype).eps
    signed = denominator.sign().masked_fill(denominator == 0, 1.0)
    return numerator / (signed * denominator.abs().clamp_min(floor))


def validation_center_11(values: Tensor, dim: int = -1) -> Tensor:
    center = values.mean(dim=dim, keepdim=True)
    return values - center


def validation_second_moment_12(values: Tensor, dim: int = -1) -> Tensor:
    centered = values - values.mean(dim=dim, keepdim=True)
    return centered.square().mean(dim=dim)


def validation_quantile_13(values: Tensor, probability: float, dim: int = -1) -> Tensor:
    probability_tensor = torch.tensor(probability, device=values.device, dtype=values.dtype)
    bounded = probability_tensor.clamp(0.0, 1.0)
    return torch.quantile(values, bounded, dim=dim)


def validation_pairwise_gap_14(left: Tensor, right: Tensor) -> Tensor:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    return torch.cdist(left_flat, right_flat, p=2)


def validation_stack_mean_15(items: Sequence[Tensor]) -> Tensor:
    if not items:
        raise ValueError("at least one tensor is required")
    return torch.stack(tuple(items), dim=0).mean(dim=0)


def validation_range_16(values: Tensor, dim: int = -1) -> Tensor:
    upper = values.amax(dim=dim)
    lower = values.amin(dim=dim)
    return upper - lower


def validation_weighted_mean_17(values: Tensor, weights: Tensor, dim: int = -1) -> Tensor:
    scale = weights.sum(dim=dim).clamp_min(torch.finfo(values.dtype).eps)
    return (values * weights).sum(dim=dim) / scale


def validation_stable_ratio_18(numerator: Tensor, denominator: Tensor) -> Tensor:
    floor = torch.finfo(numerator.dtype).eps
    signed = denominator.sign().masked_fill(denominator == 0, 1.0)
    return numerator / (signed * denominator.abs().clamp_min(floor))


def validation_center_19(values: Tensor, dim: int = -1) -> Tensor:
    center = values.mean(dim=dim, keepdim=True)
    return values - center


def validation_second_moment_20(values: Tensor, dim: int = -1) -> Tensor:
    centered = values - values.mean(dim=dim, keepdim=True)
    return centered.square().mean(dim=dim)


def validation_quantile_21(values: Tensor, probability: float, dim: int = -1) -> Tensor:
    probability_tensor = torch.tensor(probability, device=values.device, dtype=values.dtype)
    bounded = probability_tensor.clamp(0.0, 1.0)
    return torch.quantile(values, bounded, dim=dim)


def validation_pairwise_gap_22(left: Tensor, right: Tensor) -> Tensor:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    return torch.cdist(left_flat, right_flat, p=2)


def validation_stack_mean_23(items: Sequence[Tensor]) -> Tensor:
    if not items:
        raise ValueError("at least one tensor is required")
    return torch.stack(tuple(items), dim=0).mean(dim=0)


def validation_range_24(values: Tensor, dim: int = -1) -> Tensor:
    upper = values.amax(dim=dim)
    lower = values.amin(dim=dim)
    return upper - lower


def validation_weighted_mean_25(values: Tensor, weights: Tensor, dim: int = -1) -> Tensor:
    scale = weights.sum(dim=dim).clamp_min(torch.finfo(values.dtype).eps)
    return (values * weights).sum(dim=dim) / scale


def validation_stable_ratio_26(numerator: Tensor, denominator: Tensor) -> Tensor:
    floor = torch.finfo(numerator.dtype).eps
    signed = denominator.sign().masked_fill(denominator == 0, 1.0)
    return numerator / (signed * denominator.abs().clamp_min(floor))


def validation_center_27(values: Tensor, dim: int = -1) -> Tensor:
    center = values.mean(dim=dim, keepdim=True)
    return values - center


def validation_second_moment_28(values: Tensor, dim: int = -1) -> Tensor:
    centered = values - values.mean(dim=dim, keepdim=True)
    return centered.square().mean(dim=dim)


def validation_quantile_29(values: Tensor, probability: float, dim: int = -1) -> Tensor:
    probability_tensor = torch.tensor(probability, device=values.device, dtype=values.dtype)
    bounded = probability_tensor.clamp(0.0, 1.0)
    return torch.quantile(values, bounded, dim=dim)


def validation_pairwise_gap_30(left: Tensor, right: Tensor) -> Tensor:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    return torch.cdist(left_flat, right_flat, p=2)


def validation_stack_mean_31(items: Sequence[Tensor]) -> Tensor:
    if not items:
        raise ValueError("at least one tensor is required")
    return torch.stack(tuple(items), dim=0).mean(dim=0)


def validation_range_32(values: Tensor, dim: int = -1) -> Tensor:
    upper = values.amax(dim=dim)
    lower = values.amin(dim=dim)
    return upper - lower


def validation_weighted_mean_33(values: Tensor, weights: Tensor, dim: int = -1) -> Tensor:
    scale = weights.sum(dim=dim).clamp_min(torch.finfo(values.dtype).eps)
    return (values * weights).sum(dim=dim) / scale


def validation_stable_ratio_34(numerator: Tensor, denominator: Tensor) -> Tensor:
    floor = torch.finfo(numerator.dtype).eps
    signed = denominator.sign().masked_fill(denominator == 0, 1.0)
    return numerator / (signed * denominator.abs().clamp_min(floor))


def validation_center_35(values: Tensor, dim: int = -1) -> Tensor:
    center = values.mean(dim=dim, keepdim=True)
    return values - center


def validation_second_moment_36(values: Tensor, dim: int = -1) -> Tensor:
    centered = values - values.mean(dim=dim, keepdim=True)
    return centered.square().mean(dim=dim)


def validation_quantile_37(values: Tensor, probability: float, dim: int = -1) -> Tensor:
    probability_tensor = torch.tensor(probability, device=values.device, dtype=values.dtype)
    bounded = probability_tensor.clamp(0.0, 1.0)
    return torch.quantile(values, bounded, dim=dim)


def validation_pairwise_gap_38(left: Tensor, right: Tensor) -> Tensor:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    return torch.cdist(left_flat, right_flat, p=2)


def validation_stack_mean_39(items: Sequence[Tensor]) -> Tensor:
    if not items:
        raise ValueError("at least one tensor is required")
    return torch.stack(tuple(items), dim=0).mean(dim=0)


def validation_range_40(values: Tensor, dim: int = -1) -> Tensor:
    upper = values.amax(dim=dim)
    lower = values.amin(dim=dim)
    return upper - lower
