"""Aligned local descriptor shared by all three tasks."""

from __future__ import annotations
import numpy as np
from .config import FeatureConfig
from .edge_branch import detect_edges
from .gabor_branch import gabor_energy_maps, make_gabor_bank
from .supplied import box_mean, rgb2gray


def extract_local_features(image: np.ndarray, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """Return an aligned H x W x D map and one name per channel."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.max() > 1:
        image = image / 255.0

    gray = rgb2gray(image)
    chunks = []
    names = []

    # Keep feature groups in a predictable order.
    if config.include_colour:
        chunks.append(image[..., :3])
        names.extend(["colour_r", "colour_g", "colour_b"])

    if config.include_gabor:
        bank = make_gabor_bank(config.gabor)
        gabor_maps = gabor_energy_maps(
            gray, bank, config.gabor.pool_size, config.gabor.energy
        )
        chunks.append(gabor_maps)
        names.extend(
            f"gabor_f{meta['frequency']:g}_o{meta['orientation']:g}_p{meta['phase']:g}"
            for _, meta in bank
        )

    edge_data = None
    if config.include_gradient or config.include_edges:
        edge_data = detect_edges(gray, config.edge)

    if config.include_gradient:
        gradient_energy = edge_data["magnitude"] ** 2
        gradient_energy = box_mean(gradient_energy, config.edge.density_size)
        chunks.append(gradient_energy[..., None])
        names.append("gradient_energy")

    if config.include_edges:
        edge_map = edge_data["edges"]
        edge_density = box_mean(edge_map.astype(np.float32), config.edge.density_size)
        chunks.append(edge_density[..., None])
        names.append("edge_density")

        angle = edge_data["direction"] % np.pi
        bin_index = np.floor(angle * config.edge.n_orientations / np.pi).astype(int)
        bin_index = np.minimum(bin_index, config.edge.n_orientations - 1)
        for i in range(config.edge.n_orientations):
            oriented_edges = edge_map & (bin_index == i)
            density = box_mean(oriented_edges.astype(np.float32), config.edge.density_size)
            chunks.append(density[..., None])
            names.append(f"edge_orientation_{i}")

    if not chunks:
        raise ValueError("at least one feature family must be enabled")

    feature_map = np.concatenate(chunks, axis=-1).astype(np.float32)
    if config.standardise_per_image:
        mean = feature_map.mean(axis=(0, 1), keepdims=True)
        std = feature_map.std(axis=(0, 1), keepdims=True)
        feature_map = (feature_map - mean) / (std + 1e-6)

    return feature_map.astype(np.float32), names


def global_pool(feature_map: np.ndarray,
                statistics: tuple[str, ...] = ("mean", "std", "p90")) -> np.ndarray:
    """Pool local channels into one reproducible image descriptor."""
    feature_map = np.asarray(feature_map)
    flat = feature_map.reshape(-1, feature_map.shape[-1])
    pooled = []

    for statistic in statistics:
        if statistic == "mean":
            values = flat.mean(axis=0)
        elif statistic == "std":
            values = flat.std(axis=0)
        elif statistic == "p10":
            values = np.percentile(flat, 10, axis=0)
        elif statistic == "p50":
            values = np.percentile(flat, 50, axis=0)
        elif statistic == "p90":
            values = np.percentile(flat, 90, axis=0)
        else:
            raise ValueError(f"unsupported statistic: {statistic}")
        pooled.append(values)

    return np.concatenate(pooled).astype(np.float32)


def feature_family_indices(names: list[str]) -> dict[str, list[int]]:
    """Map colour/Gabor/gradient/edge families to descriptor indices."""
    families = {"colour": [], "gabor": [], "gradient": [], "edge": []}
    for index, name in enumerate(names):
        family = name.split("_", 1)[0]
        if family in families:
            families[family].append(index)
    return families
