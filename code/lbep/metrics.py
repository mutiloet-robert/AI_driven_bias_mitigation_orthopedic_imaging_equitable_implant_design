from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import roc_auc_score
from torch import Tensor


def dice_score(prediction: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    prediction_bool = prediction.bool()
    target_bool = target.bool()
    intersection = (prediction_bool & target_bool).flatten(1).sum(dim=1)
    total = prediction_bool.flatten(1).sum(dim=1) + target_bool.flatten(1).sum(dim=1)
    return (2.0 * intersection + epsilon) / (total + epsilon)


def _surface(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    return cast(npt.NDArray[np.bool_], np.logical_xor(mask, binary_erosion(mask)))


def surface_distances(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    prediction_surface = _surface(prediction.astype(bool))
    target_surface = _surface(target.astype(bool))
    if not prediction_surface.any() or not target_surface.any():
        return np.asarray([np.inf], dtype=np.float64)
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    prediction_distance = distance_transform_edt(~prediction_surface, sampling=spacing)
    forward = target_distance[prediction_surface]
    backward = prediction_distance[target_surface]
    return np.concatenate((forward, backward))


def hd95(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    return float(np.percentile(surface_distances(prediction, target, spacing), 95))


def assd(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    return float(np.mean(surface_distances(prediction, target, spacing)))


def auc_score(probabilities: Tensor, target: Tensor) -> float:
    return float(roc_auc_score(target.detach().cpu().numpy(), probabilities.detach().cpu().numpy()))


def confusion_rates(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    prediction_bool = prediction.bool()
    target_bool = target.bool()
    true_positive = (prediction_bool & target_bool).sum().float()
    false_positive = (prediction_bool & ~target_bool).sum().float()
    true_negative = (~prediction_bool & ~target_bool).sum().float()
    false_negative = (~prediction_bool & target_bool).sum().float()
    sensitivity = true_positive / (true_positive + false_negative).clamp_min(1.0)
    specificity = true_negative / (true_negative + false_positive).clamp_min(1.0)
    positive_rate = (true_positive + false_positive) / prediction.numel()
    false_positive_rate = false_positive / (false_positive + true_negative).clamp_min(1.0)
    return sensitivity, specificity, positive_rate, false_positive_rate


def grouped_values(values: Tensor, groups: Tensor) -> tuple[Tensor, Tensor]:
    labels = torch.unique(groups, sorted=True)
    means = torch.stack([values[groups == label].mean() for label in labels])
    return labels, means


def maximum_fairness_gap(values: Tensor, groups: Tensor) -> Tensor:
    _, means = grouped_values(values, groups)
    return means.amax() - means.amin()


def worst_group_performance(values: Tensor, groups: Tensor) -> Tensor:
    _, means = grouped_values(values, groups)
    return means.amin()


def equalized_odds_difference(prediction: Tensor, target: Tensor, groups: Tensor) -> Tensor:
    rates = [confusion_rates(prediction[groups == label], target[groups == label]) for label in torch.unique(groups)]
    true_positive_rates = torch.stack([rate[0] for rate in rates])
    false_positive_rates = torch.stack([rate[3] for rate in rates])
    return torch.maximum(true_positive_rates.amax() - true_positive_rates.amin(), false_positive_rates.amax() - false_positive_rates.amin())


def demographic_parity_difference(prediction: Tensor, groups: Tensor) -> Tensor:
    rates = torch.stack([prediction[groups == label].float().mean() for label in torch.unique(groups)])
    return rates.amax() - rates.amin()


def mismatch_rate(overhang: Tensor, undercoverage: Tensor, threshold_mm: float = 2.0) -> Tensor:
    return ((overhang > threshold_mm) | (undercoverage > threshold_mm)).float().mean()


def mismatch_disparity(overhang: Tensor, undercoverage: Tensor, groups: Tensor) -> Tensor:
    rates = torch.stack([mismatch_rate(overhang[groups == label], undercoverage[groups == label]) for label in torch.unique(groups)])
    return rates.amax() - rates.amin()


@dataclass(frozen=True)
class FairnessMetrics:
    maximum_gap: float
    worst_group: float
    equalized_odds: float
    demographic_parity: float
