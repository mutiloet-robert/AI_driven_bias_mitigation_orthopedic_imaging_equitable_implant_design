from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import zoom
from torch import Tensor


def percentile_window(volume: npt.NDArray[np.float32], lower: float = 0.5, upper: float = 99.5) -> npt.NDArray[np.float32]:
    low, high = np.percentile(volume, (lower, upper))
    clipped = np.clip(volume, low, high)
    return cast(npt.NDArray[np.float32], ((clipped - low) / max(high - low, 1e-12)).astype(np.float32))


def hu_window(volume: npt.NDArray[np.float32], lower: float = 200.0, upper: float = 2000.0) -> npt.NDArray[np.float32]:
    clipped = np.clip(volume, lower, upper)
    return cast(npt.NDArray[np.float32], ((clipped - lower) / (upper - lower)).astype(np.float32))


def resample(volume: npt.NDArray[np.float32], spacing: tuple[float, ...], target_spacing: tuple[float, ...], order: int) -> npt.NDArray[np.float32]:
    factors = tuple(source / target for source, target in zip(spacing, target_spacing, strict=True))
    return cast(npt.NDArray[np.float32], zoom(volume, factors, order=order))


def clahe_tensor(image: Tensor, tiles: int = 8, clip_limit: float = 2.0) -> Tensor:
    if image.ndim != 2:
        raise ValueError("CLAHE expects a two-dimensional image")
    height, width = image.shape
    output = torch.empty_like(image)
    tile_height = max(height // tiles, 1)
    tile_width = max(width // tiles, 1)
    for row in range(0, height, tile_height):
        for column in range(0, width, tile_width):
            tile = image[row : row + tile_height, column : column + tile_width]
            histogram = torch.histc(tile.float(), bins=256, min=float(tile.min()), max=float(tile.max()))
            limit = clip_limit * tile.numel() / 256.0
            excess = (histogram - limit).clamp_min(0.0).sum()
            histogram = histogram.clamp_max(limit) + excess / 256.0
            cumulative = histogram.cumsum(0)
            cumulative = cumulative / cumulative[-1].clamp_min(1e-12)
            indices = ((tile - tile.min()) / (tile.max() - tile.min()).clamp_min(1e-12) * 255).long()
            output[row : row + tile_height, column : column + tile_width] = cumulative[indices].to(image.dtype)
    return output


@dataclass(frozen=True)
class PreprocessedVolume:
    image: npt.NDArray[np.float32]
    spacing: tuple[float, ...]


def prepare_mri(volume: npt.NDArray[np.float32], spacing: tuple[float, float, float]) -> PreprocessedVolume:
    image = percentile_window(resample(volume, spacing, (0.5, 0.5, 0.5), 3))
    return PreprocessedVolume(image, (0.5, 0.5, 0.5))


def prepare_ct(volume: npt.NDArray[np.float32], spacing: tuple[float, float, float]) -> PreprocessedVolume:
    image = hu_window(resample(volume, spacing, (1.0, 1.0, 1.0), 3))
    return PreprocessedVolume(image, (1.0, 1.0, 1.0))
