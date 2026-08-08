from dataclasses import dataclass

import torch
from torch import Tensor


def squared_cost(left: Tensor, right: Tensor) -> Tensor:
    return torch.cdist(left, right, p=2).square()


def sinkhorn_plan(
    source: Tensor,
    target: Tensor,
    regularization: float = 0.01,
    iterations: int = 100,
    tolerance: float = 1e-4,
) -> Tensor:
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    source_mass = torch.full((source.shape[0],), 1.0 / source.shape[0], device=source.device)
    target_mass = torch.full((target.shape[0],), 1.0 / target.shape[0], device=target.device)
    kernel = torch.exp(-squared_cost(source, target) / regularization).clamp_min(1e-30)
    left_scale = torch.ones_like(source_mass)
    right_scale = torch.ones_like(target_mass)
    for _ in range(iterations):
        left_scale = source_mass / (kernel @ right_scale).clamp_min(1e-30)
        right_scale = target_mass / (kernel.transpose(0, 1) @ left_scale).clamp_min(1e-30)
        plan = left_scale[:, None] * kernel * right_scale[None, :]
        violation = torch.maximum(
            (plan.sum(dim=1) - source_mass).abs().amax(),
            (plan.sum(dim=0) - target_mass).abs().amax(),
        )
        if float(violation) < tolerance:
            break
    return left_scale[:, None] * kernel * right_scale[None, :]


def transported_support(plan: Tensor, source: Tensor) -> Tensor:
    mass = plan.sum(dim=0).clamp_min(1e-12)
    return plan.transpose(0, 1) @ source / mass[:, None]


def wasserstein_proxy(left: Tensor, right: Tensor, regularization: float = 0.01) -> Tensor:
    plan = sinkhorn_plan(left, right, regularization)
    return torch.sqrt((plan * squared_cost(left, right)).sum().clamp_min(0.0))


def project_fairness(features: Tensor, groups: Tensor, tolerance: float) -> Tensor:
    output = features.clone()
    labels = torch.unique(groups)
    global_mean = features.mean(dim=0)
    for label in labels:
        mask = groups == label
        group_mean = output[mask].mean(dim=0)
        delta = group_mean - global_mean
        norm = delta.norm().clamp_min(1e-12)
        excess = (norm - tolerance).clamp_min(0.0)
        output[mask] = output[mask] - delta * excess / norm
    return output


def project_morphometry(
    candidate: Tensor,
    reference: Tensor,
    groups: Tensor,
    tolerance: float,
) -> Tensor:
    output = candidate.clone()
    for label in torch.unique(groups):
        mask = groups == label
        reference_moment = reference[mask].mean(dim=0)
        candidate_moment = output[mask].mean(dim=0)
        delta = (candidate_moment - reference_moment).clamp(-tolerance, tolerance)
        desired = reference_moment + delta
        output[mask] = output[mask] + desired - candidate_moment
    return output


@dataclass(frozen=True)
class HarmonizationResult:
    support: Tensor
    plans: tuple[Tensor, ...]
    iterations: int
    residual: float


def eaoth(
    sites: tuple[Tensor, ...],
    groups: Tensor,
    site_weights: Tensor,
    fairness_tolerance: float = 0.05,
    morphometry_tolerance: float = 0.02,
    regularization: float = 0.01,
    maximum_iterations: int = 500,
    convergence_tolerance: float = 1e-6,
) -> HarmonizationResult:
    if len(sites) == 0:
        raise ValueError("at least one site is required")
    if site_weights.numel() != len(sites):
        raise ValueError("one weight is required per site")
    weights = site_weights / site_weights.sum()
    support_size = min(site.shape[0] for site in sites)
    support = sum(weight * site[:support_size] for weight, site in zip(weights, sites, strict=True))
    reference = support.clone()
    plans: tuple[Tensor, ...] = ()
    residual = float("inf")
    completed = 0
    for iteration in range(maximum_iterations):
        plan_list = tuple(sinkhorn_plan(site, support, regularization) for site in sites)
        candidates = tuple(transported_support(plan, site) for plan, site in zip(plan_list, sites, strict=True))
        updated = sum(weight * candidate for weight, candidate in zip(weights, candidates, strict=True))
        updated = project_fairness(updated, groups[:support_size], fairness_tolerance)
        updated = project_morphometry(updated, reference, groups[:support_size], morphometry_tolerance)
        residual = float((updated - support).norm())
        support = updated
        plans = plan_list
        completed = iteration + 1
        if residual < convergence_tolerance:
            break
    return HarmonizationResult(support, plans, completed, residual)

