"""Student-facing exact-direction NMS, adaptive thresholds, and hysteresis."""

from __future__ import annotations
from collections import deque
import numpy as np
from .config import EdgeConfig
from .supplied import gaussian_smooth, sobel_gradients


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D image at floating-point image coordinates."""
    image = np.asarray(image)
    y = np.asarray(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    height, width = image.shape

    y = np.clip(y, 0, height - 1)
    x = np.clip(x, 0, width - 1)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = y - y0
    wx = x - x0

    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def nms_interpolated(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin gradient magnitude along the exact image-coordinate direction."""
    magnitude = np.asarray(magnitude, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    if magnitude.ndim != 2 or direction.ndim != 2 or magnitude.shape != direction.shape:
        raise ValueError("magnitude and direction must be matching 2-D arrays")

    rows, cols = np.indices(magnitude.shape, dtype=np.float32)
    dy = np.sin(direction)
    dx = np.cos(direction)
    forward = bilinear_sample(magnitude, rows + dy, cols + dx)
    backward = bilinear_sample(magnitude, rows - dy, cols - dx)

    keep = (magnitude >= forward) & (magnitude >= backward)
    result = np.where(keep, magnitude, 0).astype(np.float32)
    result[[0, -1], :] = 0
    result[:, [0, -1]] = 0
    return result


def adaptive_thresholds(nms: np.ndarray, config: EdgeConfig) -> tuple[float, float]:
    """Select reproducible low/high thresholds without test labels."""
    positive = np.asarray(nms)[np.asarray(nms) > 0]
    if positive.size == 0:
        return 0.0, 0.0

    if config.threshold_method == "percentile":
        high = np.percentile(positive, config.high_percentile)
    elif config.threshold_method == "robust":
        median = np.median(positive)
        mad = np.median(np.abs(positive - median))
        high = median + 2.5 * 1.4826 * mad
    else:
        raise ValueError("threshold_method must be 'percentile' or 'robust'")

    high = float(high)
    low = float(config.low_ratio * high)
    return low, high


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep all weak pixels reachable from strong seeds."""
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)
    if strong.shape != weak.shape:
        raise ValueError("strong and weak masks must have the same shape")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    result = strong.copy()
    queue = deque(map(tuple, np.argwhere(strong)))
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    height, width = result.shape
    # Follow weak pixels outward from each strong edge.
    while queue:
        row, col = queue.popleft()
        for dr, dc in neighbors:
            next_row = row + dr
            next_col = col + dc
            if (0 <= next_row < height and 0 <= next_col < width
                    and weak[next_row, next_col] and not result[next_row, next_col]):
                result[next_row, next_col] = True
                queue.append((next_row, next_col))

    return result


def detect_edges(gray: np.ndarray, config: EdgeConfig) -> dict[str, np.ndarray | float]:
    """Run the supplied smoothing/Sobel stages and student edge stages."""
    smoothed = gaussian_smooth(gray, config.gaussian_sigma)
    gx, gy, magnitude, direction = sobel_gradients(smoothed)
    nms = nms_interpolated(magnitude, direction)
    low, high = adaptive_thresholds(nms, config)
    strong = (nms > 0) & (nms >= high)
    weak = (nms > 0) & (nms >= low) & ~strong
    edges = hysteresis(strong, weak, config.connectivity)

    return {
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction,
        "nms": nms,
        "low": low,
        "high": high,
        "edges": edges,
    }
