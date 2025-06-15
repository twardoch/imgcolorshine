#!/usr/bin/env python3
"""
Simple debug script focusing on the core algorithm issues.
"""

import numpy as np
from coloraide import Color


def debug_algorithm():
    """Debug the core algorithm logic."""

    # Simulate a light blue pixel (similar to jacket color)
    jacket_rgb = np.array([0.678, 0.847, 0.902])  # Light blue

    # Convert to Oklab
    color = Color("srgb", list(jacket_rgb))
    oklab = color.convert("oklab")
    pixel_lab = np.array([oklab["lightness"], oklab["a"], oklab["b"]])

    # Convert to OKLCH
    oklch = color.convert("oklch")
    pixel_lch = np.array([oklch["lightness"], oklch["chroma"], oklch["chroma"]])

    # Blue attractor
    blue_color = Color("blue")
    blue_oklab = blue_color.convert("oklab")
    blue_oklch = blue_color.convert("oklch")

    attractor_lab = np.array(
        [blue_oklab["lightness"], blue_oklab["a"], blue_oklab["b"]]
    )
    attractor_lch = np.array(
        [blue_oklch["lightness"], blue_oklch["chroma"], blue_oklch["chroma"]]
    )

    # Calculate distance and weight
    delta_e = np.sqrt(np.sum((pixel_lab - attractor_lab) ** 2))

    tolerance = 80
    strength = 80

    # Current tolerance calculation
    delta_e_max = 1.0 * (tolerance / 100.0) ** 2

    if delta_e <= delta_e_max:
        d_norm = delta_e / delta_e_max
        attraction_factor = 0.5 * (np.cos(d_norm * np.pi) + 1.0)
        weight = (strength / 100.0) * attraction_factor

        # Simulate chroma-only blending
        total_weight = weight
        src_weight = 1.0 - total_weight if total_weight <= 1.0 else 0.0

        # Original chroma
        original_hue = pixel_lch[2]
        attractor_hue = attractor_lch[2]

        # Circular mean for chroma
        sin_sum = src_weight * np.sin(np.deg2rad(original_hue))
        cos_sum = src_weight * np.cos(np.deg2rad(original_hue))

        sin_sum += weight * np.sin(np.deg2rad(attractor_hue))
        cos_sum += weight * np.cos(np.deg2rad(attractor_hue))

        final_hue = np.rad2deg(np.arctan2(sin_sum, cos_sum))
        if final_hue < 0:
            final_hue += 360

        # Convert back to RGB to see actual change
        final_lch = [pixel_lch[0], pixel_lch[1], final_hue]
        final_color = Color("oklch", final_lch)
        final_rgb = final_color.convert("srgb")
        np.array([final_rgb["red"], final_rgb["green"], final_rgb["blue"]])

    else:
        pass

    # Test with a more reasonable tolerance scaling

    # Alternative scaling: linear instead of quadratic
    alt_delta_e_max = 2.0 * (tolerance / 100.0)  # Linear scaling, larger range

    if delta_e <= alt_delta_e_max:
        d_norm_alt = delta_e / alt_delta_e_max
        attraction_factor_alt = 0.5 * (np.cos(d_norm_alt * np.pi) + 1.0)
        (strength / 100.0) * attraction_factor_alt


if __name__ == "__main__":
    debug_algorithm()
