from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


def aspect_ratio_constant(ap_diameter: Tensor, aspect_ratio: Tensor) -> Tensor:
    if torch.any(ap_diameter <= 0):
        raise ValueError("AP diameter must be positive")
    return torch.sqrt(1.0 + aspect_ratio.square()) / ap_diameter


def jacobian_spectral_norm(output: Tensor, inputs: Tensor, iterations: int = 50) -> Tensor:
    flattened = output.reshape(output.shape[0], -1)
    vector = torch.randn_like(flattened)
    vector = vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-12)
    estimate = torch.zeros((), device=output.device, dtype=output.dtype)
    for _ in range(iterations):
        product = (flattened * vector).sum()
        gradient = torch.autograd.grad(product, inputs, create_graph=True, retain_graph=True)[0]
        norm = gradient.reshape(gradient.shape[0], -1).norm(dim=1).mean()
        estimate = norm
        vector = flattened.detach()
        vector = vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return estimate


def group_jacobian_bounds(
    model: nn.Module,
    inputs: Tensor,
    groups: Tensor,
    iterations: int = 50,
) -> tuple[Tensor, Tensor]:
    constants: list[Tensor] = []
    labels = torch.unique(groups, sorted=True)
    for label in labels:
        selected = inputs[groups == label].detach().requires_grad_(True)
        outputs = model(selected)
        constants.append(jacobian_spectral_norm(outputs, selected, iterations))
    return labels, torch.stack(constants)


def finite_difference_constant(
    operation: Callable[[Tensor], Tensor],
    inputs: Tensor,
    samples: int = 1000,
    delta: float = 0.1,
    generator: torch.Generator | None = None,
) -> Tensor:
    chosen = torch.randint(inputs.shape[0], (samples,), device=inputs.device, generator=generator)
    base = inputs[chosen]
    direction = torch.randn(base.shape, device=base.device, dtype=base.dtype, generator=generator)
    direction = direction / direction.flatten(1).norm(dim=1).view(-1, *([1] * (base.ndim - 1)))
    shifted = base + delta * direction
    output_gap = (operation(shifted) - operation(base)).flatten(1).norm(dim=1)
    return cast(Tensor, (output_gap / delta).amax())


def reconstruction_bound(vertices_before: Tensor, vertices_after: Tensor, input_delta: Tensor) -> Tensor:
    numerator = (vertices_after - vertices_before).flatten(1).norm(dim=1)
    denominator = input_delta.flatten(1).norm(dim=1).clamp_min(1e-12)
    return cast(Tensor, (numerator / denominator).amax())


def compose_constants(constants: Tensor) -> Tensor:
    if constants.ndim == 1:
        return constants.prod()
    return constants.prod(dim=0)


@dataclass(frozen=True)
class AmplificationMap:
    stages: tuple[str, ...]
    groups: tuple[str, ...]
    constants: Tensor
    biases: Tensor

    def __post_init__(self) -> None:
        if self.constants.shape != self.biases.shape:
            raise ValueError("constants and biases must align")
        if self.constants.shape != (len(self.stages), len(self.groups)):
            raise ValueError("labels must align with matrix dimensions")

    @property
    def contributions(self) -> Tensor:
        return self.constants * self.biases

    @property
    def downstream(self) -> Tensor:
        return self.biases[0] * self.constants.prod(dim=0)

    @property
    def priorities(self) -> Tensor:
        contributions = self.contributions
        return contributions / contributions.sum().clamp_min(1e-12)
