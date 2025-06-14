#!/usr/bin/env python3
"""
Diagnostic script to understand color distances and tolerance issues in imgcolorshine.
"""

import numpy as np
from coloraide import Color


def test_color_distances():
    """Test actual Delta E values between different colors."""

    # Define test colors
    colors = {
        "blue": "blue",
        "light_blue": "lightblue",
        "cyan": "cyan",
        "red": "red",
        "green": "green",
        "yellow": "yellow",
        "white": "white",
        "black": "black",
        "gray": "gray",
        "jacket_blue": "rgb(173, 216, 230)",  # Approximate jacket color from Louis image
    }

    # Convert to Oklab
    oklab_colors = {}
    for name, color_str in colors.items():
        color = Color(color_str)
        oklab = color.convert("oklab")
        oklab_colors[name] = np.array([oklab["lightness"], oklab["a"], oklab["b"]])

    blue_oklab = oklab_colors["blue"]

    for name, oklab in oklab_colors.items():
        if name != "blue":
            delta_e = np.sqrt(np.sum((blue_oklab - oklab) ** 2))

    # Test current tolerance formula
    tolerances = [20, 40, 60, 80, 100]

    for tolerance in tolerances:
        # Current (broken) formula
        current_max = 1.0 * (tolerance / 100.0) ** 2

        # Proposed fixed formula
        proposed_max = 5.0 * (tolerance / 100.0)

    # Simulate what happens with tolerance=80, strength=80
    tolerance = 80
    current_max = 1.0 * (tolerance / 100.0) ** 2  # 0.64
    proposed_max = 5.0 * (tolerance / 100.0)  # 4.0

    for name, oklab in oklab_colors.items():
        if name != "blue":
            delta_e = np.sqrt(np.sum((blue_oklab - oklab) ** 2))
            if delta_e <= current_max:
                pass
            else:
                pass

    for name, oklab in oklab_colors.items():
        if name != "blue":
            delta_e = np.sqrt(np.sum((blue_oklab - oklab) ** 2))
            if delta_e <= proposed_max:
                pass
            else:
                pass


if __name__ == "__main__":
    test_color_distances()
