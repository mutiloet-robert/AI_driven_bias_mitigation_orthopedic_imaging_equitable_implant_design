import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .losses import equitable_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def polynomial_learning_rate(initial: float, step: int, total_steps: int, power: float = 0.9) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return cast(float, initial * (1.0 - progress) ** power)


def atomic_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    seed = int(payload["seed"])
    set_seed(seed)
    return int(payload["epoch"]), seed


@dataclass(frozen=True)
class Batch:
    images: Tensor
    labels: Tensor
    groups: Tensor
    segmentation_bias: Tensor
    downstream_constants: Tensor


@dataclass(frozen=True)
class TrainState:
    epoch: int
    step: int
    loss: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambda_fair: float,
        lambda_bound: float,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lambda_fair = lambda_fair
        self.lambda_bound = lambda_bound
        self.device = device

    def train_batch(self, batch: Batch, group_constants: Tensor) -> float:
        self.model.train()
        images = batch.images.to(self.device)
        labels = batch.labels.to(self.device)
        breakdown = equitable_loss(
            self.model(images),
            labels,
            group_constants.to(self.device),
            batch.segmentation_bias.to(self.device),
            batch.downstream_constants.to(self.device),
            self.lambda_fair,
            self.lambda_bound,
        )
        self.optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(breakdown.total)
        self.optimizer.step()
        return float(breakdown.total.detach())

    def save(self, path: Path, epoch: int, seed: int) -> None:
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        atomic_save({"model": model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch, "seed": seed}, path)
