#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["numpy", "numba"] # Numba might still be needed for FalloffType if used in Numba context
# ///
# this_file: src/imgcolorshine/fast_mypyc/falloff.py

"""
Falloff functions for color attraction.

Provides various mathematical curves for controlling how color attraction
strength decreases with distance. The raised cosine is the default and
recommended function for smooth, natural transitions.
"""

from collections.abc import Callable
from enum import Enum

import numpy as np

# Import Numba-jitted versions
from ..fast_numba.falloff_numba import (
    falloff_cosine,
    falloff_linear,
    falloff_quadratic,
    falloff_gaussian,
    falloff_cubic,
    # calculate_falloff is also Numba-jitted but might be kept here if it dispatches
    # or if a pure Python version is desired as fallback (currently it's fully Numba)
    # apply_falloff_lut is also Numba-jitted
)


class FalloffType(Enum):
    """Available falloff curve types.

    Different mathematical functions for controlling attraction falloff.
    Used for customizing the behavior of color transformations.
    """

    COSINE = "cosine"  # Smooth raised cosine (default)
    LINEAR = "linear"  # Simple linear falloff
    QUADRATIC = "quadratic"  # Quadratic ease-out
    GAUSSIAN = "gaussian"  # Gaussian bell curve
    CUBIC = "cubic"  # Cubic ease-out


def get_falloff_function(falloff_type: FalloffType) -> Callable[[float], float]:
    """
    Get the appropriate falloff function.

    Args:
        falloff_type: Type of falloff curve

    Returns:
        Falloff function
    """
    mapping: dict[FalloffType, Callable[[float], float]] = {
        FalloffType.COSINE: falloff_cosine,
        FalloffType.LINEAR: falloff_linear,
        FalloffType.QUADRATIC: falloff_quadratic,
        FalloffType.GAUSSIAN: falloff_gaussian,
        FalloffType.CUBIC: falloff_cubic,
    }
    return mapping.get(falloff_type, falloff_cosine)


def visualize_falloff(falloff_type: FalloffType, samples: int = 100) -> np.ndarray:
    """
    Generate data for visualizing a falloff curve.

    Args:
        falloff_type: Type of falloff curve
        samples: Number of samples to generate

    Returns:
        Array of shape (samples, 2) with [distance, falloff] pairs
    """
    distances = np.linspace(0, 1, samples, dtype=np.float32)
    falloff_func = get_falloff_function(falloff_type)
    values = np.array([falloff_func(d) for d in distances], dtype=np.float32)
    return np.column_stack([distances, values])


def precompute_falloff_lut(falloff_type: FalloffType = FalloffType.COSINE, resolution: int = 1024) -> np.ndarray:
    """
    Precompute a lookup table for fast falloff calculations.

    Args:
        falloff_type: Type of falloff curve
        resolution: Number of entries in the lookup table

    Returns:
        Lookup table array
    """
    lut = np.zeros(resolution, dtype=np.float32)
    falloff_func = get_falloff_function(falloff_type)
    for i in range(resolution):
        d_norm = i / (resolution - 1)
        lut[i] = falloff_func(d_norm)
    return lut

# Note: calculate_falloff and apply_falloff_lut were Numba-jitted and moved to
# falloff_numba.py. If pure Python versions are needed for some reason,
# they would be re-implemented here. Otherwise, users should import them from
# fast_numba.falloff_numba if needed in pure Python contexts, or they'd be
# used by other Numba-jitted functions.
# For now, this Mypyc module will only contain the Enum and non-jitted helpers.
