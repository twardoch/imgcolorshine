# this_file: src/imgcolorshine/utils.py
"""General-purpose utilities for imgcolorshine.

Provides memory management, tiled image processing, input validation,
batch operations, and a minimal progress-bar helper used throughout the
package.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

# ---------------------------------------------------------------------------
# Memory / tile helpers
# ---------------------------------------------------------------------------


def estimate_optimal_tile_size(
    image_shape: tuple[int, ...],
    available_memory_mb: float = 1024,
    bytes_per_pixel: int = 12,
) -> int:
    """Return a power-of-two tile size that fits the image in *available_memory_mb*.

    Parameters
    ----------
    image_shape:
        ``(height, width, channels)`` of the image.
    available_memory_mb:
        Available RAM in megabytes to use for a single tile.
    bytes_per_pixel:
        Estimated bytes needed per pixel (default 12 = 3 channels × float32).

    Returns
    -------
    int
        Tile size as a power of two (≥ 1).
    """
    h, w = image_shape[:2]
    max_dim = max(h, w)

    # Use a 4× safety factor: tile needs memory for input + output + 2 intermediate
    # colour-space buffers.  Dividing the budget by this factor gives a conservative
    # per-tile pixel budget that avoids OOM in practice.
    _MEMORY_SAFETY_FACTOR = 4
    budget_bytes = available_memory_mb * 1024 * 1024
    pixel_budget = budget_bytes / (bytes_per_pixel * _MEMORY_SAFETY_FACTOR)
    tile_size = int(math.sqrt(pixel_budget))
    tile_size = _prev_pow2(max(tile_size, 1))
    # Never create tiles larger than the image itself.
    tile_size = min(tile_size, max_dim)
    return max(tile_size, 1)


def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 ≥ n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _prev_pow2(n: int) -> int:
    """Return the largest power of 2 ≤ n."""
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


# ---------------------------------------------------------------------------
# Tiled image processing
# ---------------------------------------------------------------------------


def process_large_image(
    image: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    tile_size: int = 1024,
    overlap: int = 0,
    progress_callback: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Process *image* through *transform* in overlapping tiles.

    Tiles are blended in the overlap region using a linear weight ramp so
    that seams are not visible on uniform images.

    Parameters
    ----------
    image:
        Float32 image of shape ``(H, W, C)``.
    transform:
        Function that maps an ``(h, w, C)`` tile → transformed tile of same shape.
    tile_size:
        Side length of each square tile (before adding overlap).
    overlap:
        Pixels of overlap on each side.
    progress_callback:
        Optional callable receiving a float in [0, 1].

    Returns
    -------
    np.ndarray
        Transformed image of the same shape and dtype as *image*.
    """
    h, w, _c = image.shape
    result = np.zeros_like(image)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    stride = max(tile_size - 2 * overlap, 1)

    y_starts = list(range(0, h, stride))
    x_starts = list(range(0, w, stride))
    total_tiles = len(y_starts) * len(x_starts)
    tile_idx = 0

    for y0 in y_starts:
        for x0 in x_starts:
            # Clamp to image bounds
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile = image[y0:y1, x0:x1]

            transformed = transform(tile)

            th, tw = y1 - y0, x1 - x0
            # Build a linear blend weight that peaks in the center
            wy = _blend_weights_1d(th, overlap)  # (th,)
            wx = _blend_weights_1d(tw, overlap)  # (tw,)
            w_tile = (wy[:, None] * wx[None, :])[:, :, None]  # (th, tw, 1)

            result[y0:y1, x0:x1] += transformed * w_tile
            weight[y0:y1, x0:x1] += w_tile

            tile_idx += 1
            if progress_callback is not None:
                progress_callback(tile_idx / total_tiles)

    # Normalise by accumulated weights (avoid divide-by-zero).
    # Replace zeros with 1.0 so division is safe everywhere; pixels that
    # received no tile contribution are already zero in `result`.
    weight_safe = np.where(weight > 0, weight, 1.0)
    result /= weight_safe

    if progress_callback is not None:
        progress_callback(1.0)

    return result


def _blend_weights_1d(length: int, overlap: int) -> np.ndarray:
    """1-D blend weights: ramps up over overlap, flat in centre, ramps down.

    The ramp starts at ``1/overlap`` (not 0) so that boundary pixels are
    never assigned a zero weight, which would produce black-pixel artefacts
    at image edges where only a single tile covers a row/column.
    """
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0 or length <= 2 * overlap:
        return w
    # linspace(0, 1, overlap+1)[1:] → [1/overlap, 2/overlap, …, 1]
    ramp = np.linspace(0.0, 1.0, overlap + 1, dtype=np.float32)[1:]
    w[:overlap] = ramp
    w[length - overlap :] = ramp[::-1]
    return w


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_image(image: np.ndarray) -> None:
    """Raise ``ValueError`` if *image* is not a valid float image.

    Checks:
    - Exactly 3 dimensions
    - Exactly 3 channels (last axis)
    - dtype is float32 or float64
    - All values in [0, 1]
    """
    if image.ndim != 3:
        msg = f"Image must be 3D (H, W, C), got {image.ndim}D"
        raise ValueError(msg)
    if image.shape[2] != 3:
        msg = f"Image must have 3 channels, got {image.shape[2]}"
        raise ValueError(msg)
    if image.dtype not in (np.float32, np.float64):
        msg = f"Image must be float32 or float64, got {image.dtype}"
        raise ValueError(msg)
    if image.min() < 0.0 or image.max() > 1.0:
        msg = f"Image values must be in range [0, 1], got [{image.min():.4f}, {image.max():.4f}]"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------


def clamp_to_gamut(image: np.ndarray) -> np.ndarray:
    """Clip *image* pixel values to [0, 1] and return a copy."""
    return np.clip(image, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def batch_process_images(
    image_paths: list[str],
    output_dir: str,
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Load each image in *image_paths*, apply *transform*, save to *output_dir*.

    Uses :class:`imgcolorshine.io.ImageProcessor` for I/O so that the call can
    be intercepted by tests that patch ``imgcolorshine.io.ImageProcessor``.
    """
    from imgcolorshine.io import ImageProcessor  # lazy to allow patching

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    processor = ImageProcessor()

    for img_path in image_paths:
        src = Path(img_path)
        image = processor.load_image(src)
        transformed = transform(image)
        dest = output_path / src.name
        processor.save_image(transformed, dest)
        logger.debug(f"Processed {src.name} → {dest}")


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


class SimpleProgress:
    """Minimal context-manager progress tracker.

    Parameters
    ----------
    total:
        Total number of steps.
    description:
        Human-readable label printed on start.
    """

    def __init__(self, total: int, description: str = "") -> None:
        self.total = total
        self.current = 0
        self.description = description

    def update(self, amount: int = 1) -> None:
        """Advance the counter by *amount* steps."""
        self.current += amount

    def __enter__(self) -> Self:
        if self.description:
            logger.debug(f"Starting: {self.description} ({self.total} steps)")
        return self

    def __exit__(self, *_: object) -> None:
        logger.debug(f"Done: {self.description}")


def create_progress_bar(total: int, description: str = "") -> SimpleProgress:
    """Return a :class:`SimpleProgress` context manager.

    Parameters
    ----------
    total:
        Total number of steps.
    description:
        Human-readable label.
    """
    return SimpleProgress(total=total, description=description)
