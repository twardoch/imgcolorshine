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
from pathlib import Path

import numpy as np
from coloraide import Color
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from imgcolorshine import trans_numba


def test_single_pixel_conversion():
    """Test single pixel RGB ↔ Oklab conversions."""
    logger.info("Testing single pixel conversions...")

    # Test cases
    test_cases = [
        [0.0, 0.0, 0.0],  # Black
        [1.0, 1.0, 1.0],  # White
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [0.5, 0.5, 0.5],  # Gray
        [0.8, 0.2, 0.6],  # Purple
        [0.1, 0.9, 0.3],  # Lime
    ]

    max_diff = 0.0
    max_roundtrip_diff = 0.0

    for rgb in test_cases:
        # Convert using ColorAide
        color = Color("srgb", rgb)
        oklab_ca = color.convert("oklab")
        oklab_ca_arr = np.array([oklab_ca["lightness"], oklab_ca["a"], oklab_ca["b"]])

        # Convert using Numba
        oklab_nb = trans_numba.srgb_to_oklab_single(np.array(rgb))

        # Compare
        diff = np.abs(oklab_ca_arr - oklab_nb).max()
        max_diff = max(max_diff, diff)

        # Log comparison results
        logger.debug(f"RGB {rgb} → Oklab CA: {oklab_ca_arr}, NB: {oklab_nb}, diff: {diff:.6f}")

        # Test round trip
        rgb_back = trans_numba.oklab_to_srgb_single(oklab_nb)
        roundtrip_diff = np.abs(rgb - rgb_back).max()
        max_roundtrip_diff = max(max_roundtrip_diff, roundtrip_diff)

        logger.debug(f"Round trip diff: {roundtrip_diff:.6f}")

    logger.info(f"Maximum difference in Oklab values: {max_diff:.6f}")
    logger.info(f"Result: {'PASS' if max_diff < 0.001 else 'FAIL'}")

    assert max_diff < 0.001, f"Maximum difference {max_diff} exceeds threshold"
    assert max_roundtrip_diff < 0.001, f"Maximum round-trip difference {max_roundtrip_diff} exceeds threshold"


def test_batch_conversion():
    """Test batch RGB ↔ Oklab conversions."""
    logger.info("\nTesting batch conversions...")

    # Create test image
    width, height = 100, 100
    image = np.random.rand(height, width, 3).astype(np.float32)

    # Convert using Numba
    oklab = trans_numba.batch_srgb_to_oklab(image)
    rgb_back = trans_numba.batch_oklab_to_srgb(oklab)

    # Check RGB values are in valid range
    valid_range = np.logical_and(rgb_back >= 0, rgb_back <= 1).all()
    logger.info(f"All RGB values in valid range [0,1]: {valid_range}")

    # Check round-trip accuracy
    max_diff = np.abs(image - rgb_back).max()
    logger.info(f"Maximum round-trip difference: {max_diff:.6f}")

    assert valid_range, "RGB values outside valid range [0,1]"
    assert max_diff < 0.001, f"Maximum round-trip difference {max_diff} exceeds threshold"


def test_oklch_conversions():
    """Test Oklab ↔ OKLCH conversions."""
    logger.info("\nTesting OKLCH conversions...")

    # Test cases
    test_cases = [
        [0.5, 0.0, 0.0],  # Gray
        [0.6, 0.1, 0.0],  # Light pink
        [0.7, 0.0, 0.1],  # Light blue
        [0.8, -0.1, 0.0],  # Light green
        [0.9, 0.0, -0.1],  # Light yellow
    ]

    max_diff = 0.0

    for oklab in test_cases:
        # Convert to OKLCH
        oklch = trans_numba.oklab_to_oklch_single(np.array(oklab))

        # Convert back to Oklab
        oklab_back = trans_numba.oklch_to_oklab_single(oklch)

        # Compare (allowing for small numerical errors)
        diff = np.abs(oklab - oklab_back).max()
        max_diff = max(max_diff, diff)

        # Log conversion results
        logger.debug(f"Oklab {oklab} → OKLCH {oklch} → Oklab {oklab_back}, diff: {diff:.6f}")

    logger.info(f"Maximum round-trip difference: {max_diff:.6f}")
    logger.info(f"Result: {'PASS' if max_diff < 0.0001 else 'FAIL'}")

    assert max_diff < 0.0001, f"Maximum round-trip difference {max_diff} exceeds threshold"


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
