#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["numpy", "coloraide", "loguru"]
# ///
# this_file: src/imgcolorshine/gamut.py

"""
CSS Color Module 4 compliant gamut mapping.

Implements the standard algorithm for mapping out-of-gamut colors back
to the displayable range while preserving perceptual attributes. Uses
binary search to find the maximum chroma that fits within gamut.

"""

import numpy as np
from coloraide import Color
from loguru import logger


class GamutMapper:
    """Handles gamut mapping for out-of-bounds colors.

    Ensures all colors are displayable in the target color space (sRGB)
    by reducing chroma while preserving lightness and chroma. Follows the
    CSS Color Module 4 specification for consistent results.

    Used in:
    - old/imgcolorshine/test_imgcolorshine.py
    - src/imgcolorshine/__init__.py
    """

    def __init__(self, target_space: str = "srgb"):
        """
        Initialize the gamut mapper.

        Args:
            target_space: Target color space for gamut mapping

        """
        self.target_space = target_space
        self.epsilon = 0.0001
        logger.debug(f"Initialized GamutMapper for {target_space}")

    def is_in_gamut(self, color: Color) -> bool:
        """Check if a color is within the target gamut."""
        return color.in_gamut(self.target_space)

    def map_oklch_to_gamut(self, l: float, c: float, h: float) -> tuple[float, float, float]:
        """
        CSS Color Module 4 gamut mapping algorithm.

        Reduces chroma while preserving lightness and chroma until
        the color fits within the target gamut.

        Args:
            l: Lightness (0-1)
            c: Chroma (0-0.4+)
            h: Hue (0-360)

        Returns:
            Gamut-mapped OKLCH values

        Used in:
        - old/imgcolorshine/test_imgcolorshine.py
        """
        # Create color object
        color = Color("oklch", [l, c, h])

        # Check if already in gamut
        if self.is_in_gamut(color):
            return l, c, h

        # Binary search for maximum valid chroma
        c_min = 0.0
        c_max = c

        iterations = 0
        max_iterations = 20

        while c_max - c_min > self.epsilon and iterations < max_iterations:
            c_mid = (c_min + c_max) / 2
            test_color = Color("oklch", [l, c_mid, h])

            if self.is_in_gamut(test_color):
                c_min = c_mid
            else:
                c_max = c_mid

            iterations += 1

        # Use the last valid chroma value
        final_c = c_min

        logger.debug(f"Gamut mapped: C={c:.4f} → {final_c:.4f} (iterations: {iterations})")

        return l, final_c, h

    def map_oklab_to_gamut(self, l: float, a: float, b: float) -> tuple[float, float, float]:
        """
        Map Oklab color to gamut by converting to OKLCH first.

        Args:
            l: Lightness
            a: Green-red axis
            b: Blue-yellow axis

        Returns:
            Gamut-mapped Oklab values

        """
        # Convert to OKLCH
        c = np.sqrt(a**2 + b**2)
        h = np.rad2deg(np.arctan2(b, a))
        if h < 0:
            h += 360

        # Map to gamut
        l_mapped, c_mapped, h_mapped = self.map_oklch_to_gamut(l, c, h)

        # Convert back to Oklab
        h_rad = np.deg2rad(h_mapped)
        a_mapped = c_mapped * np.cos(h_rad)
        b_mapped = c_mapped * np.sin(h_rad)

        return l_mapped, a_mapped, b_mapped

    def map_rgb_to_gamut(self, r: float, g: float, b: float) -> tuple[float, float, float]:
        """
        Simple RGB gamut mapping by clamping.

        For more sophisticated mapping, convert to OKLCH first.

        Args:
            r, g, b: RGB values (may be outside [0, 1])

        Returns:
            Clamped RGB values in [0, 1]

        """
        return (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))

    def batch_map_oklch(self, colors: np.ndarray) -> np.ndarray:
        """
        Map multiple OKLCH colors to gamut.

        Args:
            colors: Array of shape (..., 3) with OKLCH values

        Returns:
            Gamut-mapped colors

        """
        shape = colors.shape
        flat_colors = colors.reshape(-1, 3)
        mapped_colors = np.zeros_like(flat_colors)

        for i, (l, c, h) in enumerate(flat_colors):
            mapped_colors[i] = self.map_oklch_to_gamut(l, c, h)

        return mapped_colors.reshape(shape)

    def analyze_gamut_coverage(self, colors: np.ndarray) -> dict:
        """
        Analyze how many colors are out of gamut.

        Args:
            colors: Array of colors in any format

        Returns:
            Dictionary with gamut statistics

        """
        total_colors = len(colors)
        out_of_gamut = 0

        for color_values in colors:
            color = Color("oklch", list(color_values))
            if not self.is_in_gamut(color):
                out_of_gamut += 1

        in_gamut = total_colors - out_of_gamut
        percentage_in = (in_gamut / total_colors) * 100 if total_colors > 0 else 100

        return {
            "total": total_colors,
            "in_gamut": in_gamut,
            "out_of_gamut": out_of_gamut,
            "percentage_in_gamut": percentage_in,
        }


def create_gamut_boundary_lut(hue_steps: int = 360, lightness_steps: int = 100) -> np.ndarray:
    """
    Create a lookup table for maximum chroma at each chroma/lightness.

    This can speed up gamut mapping for real-time applications.

    Args:
        hue_steps: Number of chroma divisions
        lightness_steps: Number of lightness divisions

    Returns:
        2D array of maximum chroma values

    """
    lut = np.zeros((lightness_steps, hue_steps), dtype=np.float32)
    mapper = GamutMapper()

    for l_idx in range(lightness_steps):
        l = l_idx / (lightness_steps - 1)

        for h_idx in range(hue_steps):
            h = (h_idx / hue_steps) * 360

            # Binary search for max chroma
            c_min, c_max = 0.0, 0.5  # Max reasonable chroma

            while c_max - c_min > 0.001:
                c_mid = (c_min + c_max) / 2
                color = Color("oklch", [l, c_mid, h])

                if mapper.is_in_gamut(color):
                    c_min = c_mid
                else:
                    c_max = c_mid

            lut[l_idx, h_idx] = c_min

    return lut
