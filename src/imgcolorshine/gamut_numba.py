#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["numpy", "numba"]
# ///
# this_file: src/imgcolorshine/gamut_numba.py

"""Numba-optimized gamut mapping functions."""

import numba
import numpy as np

# Import Numba-optimized functions from trans_numba
from imgcolorshine.trans_numba import (
    is_in_gamut_srgb,
    oklab_to_srgb_single,
    oklch_to_oklab_single,
)


@numba.njit(cache=True)
def binary_search_chroma(
    l: float, c: float, h: float, epsilon: float = 0.0001
) -> float:
    """
    Numba-optimized binary search for maximum in-gamut chroma.

    Finds the maximum chroma value that keeps the color within sRGB gamut
    using binary search. This is the core of the CSS Color Module 4 gamut
    mapping algorithm.

    Args:
        l: Lightness (0-1)
        c: Initial chroma value
        h: Hue in degrees (0-360)
        epsilon: Convergence threshold

    Returns:
        Maximum chroma that fits within gamut
    """
    if l == 0.0 or l == 1.0:  # For pure black or white, max chroma is 0
        return 0.0

    # Quick check if already in gamut
    oklch = np.array([l, c, h], dtype=np.float32)
    oklab = oklch_to_oklab_single(oklch)
    rgb = oklab_to_srgb_single(oklab)

    if is_in_gamut_srgb(rgb):
        return c

    # Binary search for maximum valid chroma
    c_min, c_max = 0.0, c

    for _ in range(20):  # Max iterations
        if c_max - c_min <= epsilon:
            break

        c_mid = (c_min + c_max) / 2.0
        test_oklch = np.array([l, c_mid, h], dtype=np.float32)
        test_oklab = oklch_to_oklab_single(test_oklch)
        test_rgb = oklab_to_srgb_single(test_oklab)

        if is_in_gamut_srgb(test_rgb):
            c_min = c_mid
        else:
            c_max = c_mid

    return c_min


@numba.njit(parallel=True, cache=True)
def batch_map_oklch_numba(
    colors_flat: np.ndarray, epsilon: float = 0.0001
) -> np.ndarray:
    """
    Numba-optimized batch gamut mapping for OKLCH colors.

    Processes multiple colors in parallel for maximum performance.

    Args:
        colors_flat: Flattened array of OKLCH colors (N, 3)
        epsilon: Convergence threshold for binary search

    Returns:
        Gamut-mapped OKLCH colors (N, 3)
    """
    n_colors = colors_flat.shape[0]
    mapped_colors = np.empty_like(colors_flat)

    for i in numba.prange(n_colors):  # type: ignore[attr-defined]
        l, c, h = colors_flat[i]
        c_mapped = binary_search_chroma(l, c, h, epsilon)
        mapped_colors[i] = np.array([l, c_mapped, h], dtype=colors_flat.dtype)

    return mapped_colors
