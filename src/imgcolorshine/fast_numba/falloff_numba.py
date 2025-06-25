#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["numpy", "numba"]
# ///
# this_file: src/imgcolorshine/fast_numba/falloff_numba.py

"""Numba-optimized falloff functions."""

import numba
import numpy as np

@numba.njit(cache=True)
def falloff_cosine(d_norm: float) -> float:
    """Raised cosine falloff (smooth and natural)."""
    return 0.5 * (np.cos(d_norm * np.pi) + 1.0)

@numba.njit(cache=True)
def falloff_linear(d_norm: float) -> float:
    """Simple linear falloff."""
    return 1.0 - d_norm

@numba.njit(cache=True)
def falloff_quadratic(d_norm: float) -> float:
    """Quadratic ease-out falloff."""
    return 1.0 - d_norm * d_norm

@numba.njit(cache=True)
def falloff_gaussian(d_norm: float) -> float:
    """Gaussian falloff with sigma=0.4."""
    sigma = 0.4
    return np.exp(-(d_norm * d_norm) / (2 * sigma * sigma))

@numba.njit(cache=True)
def falloff_cubic(d_norm: float) -> float:
    """Cubic ease-out falloff."""
    inv = 1.0 - d_norm
    return inv * inv * inv

@numba.njit(cache=True)
def calculate_falloff(d_norm: float, falloff_type: int = 0) -> float:
    """Calculate falloff value based on normalized distance."""
    if falloff_type == 0:  # COSINE
        return falloff_cosine(d_norm)
    if falloff_type == 1:  # LINEAR
        return falloff_linear(d_norm)
    if falloff_type == 2:  # QUADRATIC
        return falloff_quadratic(d_norm)
    if falloff_type == 3:  # GAUSSIAN
        return falloff_gaussian(d_norm)
    if falloff_type == 4:  # CUBIC
        return falloff_cubic(d_norm)
    return falloff_cosine(d_norm)

@numba.njit(cache=True)
def apply_falloff_lut(d_norm: float, lut: np.ndarray) -> float:
    """Apply falloff using a precomputed lookup table."""
    idx_float = d_norm * (len(lut) - 1)
    idx = int(idx_float)
    if idx >= len(lut) - 1:
        return lut[-1]
    if idx < 0:
        return lut[0]
    frac = idx_float - idx
    return lut[idx] * (1 - frac) + lut[idx + 1] * frac
