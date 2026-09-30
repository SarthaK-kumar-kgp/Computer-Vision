"""Student-facing Gabor branch. Core implementation is intentionally inspectable."""

from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .config import GaborConfig
from .supplied import box_mean, correlate2d


def make_gabor_bank(config: GaborConfig) -> list[tuple[np.ndarray, dict]]:
    """Return ordered zero-mean, unit-norm kernels and their metadata."""
    size = config.kernel_size
    if size < 3 or size % 2 == 0:
        raise ValueError("kernel_size must be odd and at least 3")

    coords = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)
    envelope = np.exp(-(x**2 + y**2) / (2 * config.sigma**2))

    bank = []
    # Keep each filter in the same order as the config values.
    for frequency in config.frequencies:
        for orientation in config.orientations:
            for phase in config.phases:
                rotated_x = x * np.cos(orientation) + y * np.sin(orientation)
                kernel = envelope * np.sin(2 * np.pi * frequency * rotated_x + phase)
                kernel -= kernel.mean()
                norm = np.linalg.norm(kernel)
                if not np.isfinite(norm) or norm < 1e-12:
                    raise ValueError("Gabor kernel has a zero or invalid norm")

                kernel = (kernel / norm).astype(np.float32)
                metadata = {
                    "frequency": float(frequency),
                    "orientation": float(orientation),
                    "phase": float(phase),
                    "config": asdict(config),
                }
                bank.append((kernel, metadata))

    return bank


def gabor_energy_maps(gray: np.ndarray, bank: list[tuple[np.ndarray, dict]], pool_size: int = 9,
                      energy: str = "squared") -> np.ndarray:
    """Return finite H x W x K locally pooled energy maps."""
    gray = np.asarray(gray, dtype=np.float32)
    if energy not in {"squared", "absolute"}:
        raise ValueError("energy must be 'squared' or 'absolute'")

    maps = []
    # Pool each response without changing its alignment.
    for kernel, _ in bank:
        response = correlate2d(gray, kernel)
        response = response**2 if energy == "squared" else np.abs(response)
        maps.append(box_mean(response, pool_size))

    result = np.stack(maps, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Gabor energy maps contain non-finite values")
    return result
