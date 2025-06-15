#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["numpy", "coloraide", "loguru", "numba"]
# ///
# this_file: test_correctness.py

"""
Correctness test for Numba-optimized color transformations.

Verifies that the optimized functions produce results matching ColorAide.
"""

import sys

import numpy as np
from coloraide import Color
from loguru import logger

sys.path.insert(0, "src/imgcolorshine")
import color_transforms_numba as ct_numba


def test_single_pixel_conversion():
    """Test single pixel conversions match ColorAide."""
    logger.info("Testing single pixel conversions...")

    # Test colors covering different ranges
    test_colors = [
        [0.0, 0.0, 0.0],  # Black
        [1.0, 1.0, 1.0],  # White
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [0.5, 0.5, 0.5],  # Gray
        [0.8, 0.2, 0.6],  # Random color
        [0.1, 0.9, 0.3],  # Another random
    ]

    max_diff = 0.0

    for rgb in test_colors:
        rgb_arr = np.array(rgb, dtype=np.float32)

        # ColorAide conversion
        color = Color("srgb", rgb)
        oklab_ca = color.convert("oklab")
        oklab_ca_arr = np.array([oklab_ca["lightness"], oklab_ca["a"], oklab_ca["b"]])

        # Numba conversion
        oklab_nb = ct_numba.srgb_to_oklab_single(rgb_arr)

        # Compare
        diff = np.abs(oklab_ca_arr - oklab_nb).max()
        max_diff = max(max_diff, diff)

        logger.debug(f"RGB {rgb} → Oklab CA: {oklab_ca_arr}, NB: {oklab_nb}, diff: {diff:.6f}")

        # Test round trip
        rgb_back = ct_numba.oklab_to_srgb_single(oklab_nb)
        roundtrip_diff = np.abs(rgb_arr - rgb_back).max()

        logger.debug(f"Round trip diff: {roundtrip_diff:.6f}")

    logger.info(f"Maximum difference in Oklab values: {max_diff:.6f}")
    logger.info(f"Result: {'PASS' if max_diff < 0.001 else 'FAIL'}")

    return max_diff < 0.001


def test_batch_conversion():
    """Test batch conversions."""
    logger.info("\nTesting batch conversions...")

    # Create test image
    h, w = 10, 10
    rgb_image = np.random.rand(h, w, 3).astype(np.float32)

    # Convert to Oklab
    oklab_image = ct_numba.batch_srgb_to_oklab(rgb_image)

    # Convert to OKLCH
    oklch_image = ct_numba.batch_oklab_to_oklch(oklab_image)

    # Test gamut mapping
    oklch_mapped = ct_numba.batch_gamut_map_oklch(oklch_image)

    # Convert back
    oklab_back = ct_numba.batch_oklch_to_oklab(oklch_mapped)
    rgb_back = ct_numba.batch_oklab_to_srgb(oklab_back)

    # Check all values are in valid range
    in_range = np.all(rgb_back >= 0.0) and np.all(rgb_back <= 1.0)
    logger.info(f"All RGB values in valid range [0,1]: {in_range}")

    # Check round trip accuracy
    roundtrip_diff = np.abs(rgb_image - rgb_back).max()
    logger.info(f"Maximum round-trip difference: {roundtrip_diff:.6f}")

    return in_range and roundtrip_diff < 0.01


def test_oklch_conversions():
    """Test OKLCH conversions."""
    logger.info("\nTesting OKLCH conversions...")

    test_oklabs = [
        [0.5, 0.0, 0.0],  # Gray (C=0, H undefined)
        [0.6, 0.1, 0.0],  # H=0°
        [0.7, 0.0, 0.1],  # H=90°
        [0.8, -0.1, 0.0],  # H=180°
        [0.9, 0.0, -0.1],  # H=270°
    ]

    max_diff = 0.0

    for oklab in test_oklabs:
        oklab_arr = np.array(oklab, dtype=np.float32)

        # Convert to OKLCH
        oklch = ct_numba.oklab_to_oklch_single(oklab_arr)

        # Convert back
        oklab_back = ct_numba.oklch_to_oklab_single(oklch)

        # Compare (allowing for small numerical errors)
        diff = np.abs(oklab_arr - oklab_back).max()
        max_diff = max(max_diff, diff)

        logger.debug(f"Oklab {oklab} → OKLCH {oklch} → Oklab {oklab_back}, diff: {diff:.6f}")

    logger.info(f"Maximum round-trip difference: {max_diff:.6f}")
    logger.info(f"Result: {'PASS' if max_diff < 0.0001 else 'FAIL'}")

    return max_diff < 0.0001


def main():
    """Run all correctness tests."""
    logger.info("Running correctness tests for Numba optimizations...")

    tests = [
        ("Single pixel conversion", test_single_pixel_conversion),
        ("Batch conversion", test_batch_conversion),
        ("OKLCH conversions", test_oklch_conversions),
    ]

    all_passed = True

    for name, test_func in tests:
        try:
            passed = test_func()
            all_passed &= passed
        except Exception as e:
            logger.error(f"Test '{name}' failed with error: {e}")
            all_passed = False

    logger.info("\n" + "=" * 40)
    if all_passed:
        logger.success("All tests PASSED! ✓")
    else:
        logger.error("Some tests FAILED! ✗")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
