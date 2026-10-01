"""Per-material diagonal normal model and dense anomaly prediction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import NormalityConfig
from .supplied import box_mean


@dataclass
class NormalModel:
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    feature_names: list[str] | None = None


def _scores(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray, epsilon: float
) -> np.ndarray:
    return np.sqrt(np.mean(((features - mean) / (std + epsilon)) ** 2, axis=-1)).astype(
        np.float32
    )


def _pool_score(score: np.ndarray, config: NormalityConfig) -> np.ndarray:
    """Suppress isolated patch responses while preserving coherent anomalies."""
    if config.score_pool_size < 1 or config.score_pool_size % 2 == 0:
        raise ValueError("score_pool_size must be a positive odd integer")
    return box_mean(score, config.score_pool_size).astype(np.float32)


def _interior(score: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return score
    if 2 * border >= min(score.shape):
        raise ValueError("ignore_border is too large for the score map")
    return score[border:-border, border:-border]


def fit_normal_model(feature_maps: list[np.ndarray] | np.ndarray, config: NormalityConfig,
                     feature_names: list[str] | None = None) -> NormalModel:
    """Fit normal feature statistics using normal training maps only."""
    if isinstance(feature_maps, np.ndarray) and feature_maps.ndim == 3:
        maps = [feature_maps]
    else:
        maps = list(feature_maps)

    if not maps:
        raise ValueError("feature_maps must not be empty")

    feature_count = maps[0].shape[-1]
    for feature_map in maps:
        if feature_map.ndim != 3 or feature_map.shape[-1] != feature_count:
            raise ValueError("feature maps must have the same feature dimension")

    samples = np.concatenate([feature_map.reshape(-1, feature_count) for feature_map in maps])
    if len(samples) > config.max_samples:
        indices = np.random.default_rng(config.random_seed).choice(
            len(samples), config.max_samples, replace=False
        )
        samples = samples[indices]

    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    if config.mask_threshold is None:
        normal_scores = []
        for feature_map in maps:
            score = _scores(feature_map, mean, std, config.epsilon)
            score = _pool_score(score, config)
            normal_scores.append(_interior(score, config.ignore_border).ravel())
        threshold = float(np.percentile(np.concatenate(normal_scores), config.threshold_percentile))
    else:
        threshold = float(config.mask_threshold)

    return NormalModel(mean.astype(np.float32), std.astype(np.float32), threshold, feature_names)


def predict_anomaly(feature_map: np.ndarray, model: NormalModel, config: NormalityConfig
                    ) -> tuple[np.ndarray, float, np.ndarray]:
    """Return dense score, image-level score, and predicted mask."""
    if feature_map.shape[-1] != len(model.mean):
        raise ValueError("feature dimension does not match the fitted model")

    score = _scores(feature_map, model.mean, model.std, config.epsilon)
    score = _pool_score(score, config)
    image_score = float(np.percentile(_interior(score, config.ignore_border), config.image_percentile))
    mask = score >= model.threshold

    if config.ignore_border > 0:
        border = config.ignore_border
        mask[:border] = False
        mask[-border:] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    return score.astype(np.float32), image_score, mask


def select_mask_threshold(score_maps: list[np.ndarray], masks: list[np.ndarray],
                          candidates: int = 80) -> float:
    """Choose the pixel-F1-optimal threshold on public validation only."""
    if not score_maps or len(score_maps) != len(masks) or candidates < 2:
        raise ValueError("score maps and masks must be non-empty matching collections")

    scores = [np.asarray(score, dtype=float) for score in score_maps]
    truths = [np.asarray(mask, dtype=bool) for mask in masks]
    for score, truth in zip(scores, truths):
        if score.shape != truth.shape or not np.isfinite(score).all():
            raise ValueError("score maps and masks must have matching shapes and finite scores")

    all_scores = np.concatenate([score.ravel() for score in scores])
    all_truths = np.concatenate([truth.ravel() for truth in truths])
    thresholds = np.quantile(all_scores, np.linspace(0.5, 0.999, candidates))

    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        predicted = all_scores >= threshold
        true_positive = np.count_nonzero(predicted & all_truths)
        false_positive = np.count_nonzero(predicted & ~all_truths)
        false_negative = np.count_nonzero(~predicted & all_truths)
        f1 = 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
