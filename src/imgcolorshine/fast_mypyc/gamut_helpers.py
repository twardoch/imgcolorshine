"""Pure Python gamut mapping helper functions for mypyc compilation.

This module contains gamut mapping functions extracted from gamut.py
that benefit from mypyc compilation.
"""
# this_file: src/imgcolorshine/fast_mypyc/gamut_helpers.py

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
from coloraide import Color

if TYPE_CHECKING:
    from imgcolorshine.gamut import GamutMapper

# Constants
FULL_CIRCLE_DEGREES = 360.0


def is_in_gamut(color: Color, target_space: str) -> bool:
    """Check if a color is within the target gamut."""
    return color.in_gamut(target_space)


def map_oklab_to_gamut(
    l: float, a: float, b: float, mapper: "GamutMapper"
) -> tuple[float, float, float]:
    """
    Map Oklab color to gamut by converting to OKLCH first.

    Args:
        l: Lightness
        a: Green-red axis
        b: Blue-yellow axis
        mapper: GamutMapper instance

    Returns:
        Gamut-mapped Oklab values
    """
    # Convert to OKLCH
    c = np.sqrt(a**2 + b**2)
    h = np.rad2deg(np.arctan2(b, a))
    if h < 0:
        h += FULL_CIRCLE_DEGREES

    # Map to gamut
    l_mapped, c_mapped, h_mapped = mapper.map_oklch_to_gamut(l, c, h)

    # Convert back to Oklab
    h_rad = np.deg2rad(h_mapped)
    a_mapped = c_mapped * np.cos(h_rad)
    b_mapped = c_mapped * np.sin(h_rad)

    return l_mapped, a_mapped, b_mapped


def analyze_gamut_coverage(
    colors: np.ndarray, mapper: "GamutMapper"
) -> "GamutStats":
    """
    Analyze how many colors are out of gamut.

    Args:
        colors: Array of colors in OKLCH format
        mapper: GamutMapper instance

    Returns:
        Dictionary with gamut statistics
    """
    total_colors = len(colors)
    out_of_gamut = 0

    for color_values in colors:
        color = Color("oklch", list(color_values))
        if not mapper.is_in_gamut(color):
            out_of_gamut += 1

    in_gamut = total_colors - out_of_gamut
    percentage_in = (in_gamut / total_colors) * 100 if total_colors > 0 else 100

    return {
        "total": total_colors,
        "in_gamut": in_gamut,
        "out_of_gamut": out_of_gamut,
        "percentage_in_gamut": percentage_in,
    }


def create_gamut_boundary_lut(
    hue_steps: int = 360, lightness_steps: int = 100, target_space: str = "srgb"
) -> np.ndarray:
    """
    Create a lookup table for maximum chroma at each chroma/lightness.

    This can speed up gamut mapping for real-time applications.

    Args:
        hue_steps: Number of chroma divisions
        lightness_steps: Number of lightness divisions
        target_space: Target color space

    Returns:
        2D array of maximum chroma values
    """
    lut = np.zeros((lightness_steps, hue_steps), dtype=np.float32)

    for l_idx in range(lightness_steps):
        l = l_idx / (lightness_steps - 1)

        if l == 0.0 or l == 1.0:  # Max chroma is 0 for pure black or white
            lut[l_idx, :] = 0.0
            continue

        for h_idx in range(hue_steps):
            h = (h_idx / hue_steps) * FULL_CIRCLE_DEGREES

            # Binary search for max chroma
            c_min, c_max = 0.0, 0.5  # Max reasonable chroma

            while c_max - c_min > 0.001:
                c_mid = (c_min + c_max) / 2
                color = Color("oklch", [l, c_mid, h])

                if color.in_gamut(target_space):
                    c_min = c_mid
                else:
                    c_max = c_mid

            lut[l_idx, h_idx] = c_min

    return lut


class GamutStats(TypedDict):
    total: int
    in_gamut: int
    out_of_gamut: int
    percentage_in_gamut: float