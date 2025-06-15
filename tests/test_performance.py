#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["numpy", "pillow", "coloraide", "loguru", "click", "numba"]
# ///
# this_file: test_performance.py

"""
Performance benchmark for imgcolorshine optimizations.

Compares the performance of ColorAide-based conversions vs Numba-optimized
conversions on various image sizes.
"""

# Import both implementations
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from coloraide import Color
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from imgcolorshine import color_transforms_numba as ct_numba


def benchmark_coloraide_conversion(rgb_image: np.ndarray) -> tuple[float, float]:
    """Benchmark ColorAide-based RGB to Oklab and back conversion."""
    h, w = rgb_image.shape[:2]
    flat_rgb = rgb_image.reshape(-1, 3)

    # RGB to Oklab
    start = time.time()
    oklab_list = []
    for rgb in flat_rgb:
        color = Color("srgb", list(rgb))
        oklab = color.convert("oklab")
        oklab_list.append([oklab["lightness"], oklab["a"], oklab["b"]])
    oklab_image = np.array(oklab_list).reshape(h, w, 3)
    rgb_to_oklab_time = time.time() - start

    # Oklab to RGB
    start = time.time()
    flat_oklab = oklab_image.reshape(-1, 3)
    rgb_list = []
    for oklab in flat_oklab:
        color = Color("oklab", list(oklab))
        srgb = color.convert("srgb")
        rgb_list.append([srgb["red"], srgb["green"], srgb["blue"]])
    np.array(rgb_list).reshape(h, w, 3)
    oklab_to_rgb_time = time.time() - start

    return rgb_to_oklab_time, oklab_to_rgb_time


def benchmark_numba_conversion(rgb_image: np.ndarray) -> tuple[float, float]:
    """Benchmark Numba-optimized RGB to Oklab and back conversion."""
    rgb_float32 = rgb_image.astype(np.float32)

    # RGB to Oklab
    start = time.time()
    oklab_image = ct_numba.batch_srgb_to_oklab(rgb_float32)
    rgb_to_oklab_time = time.time() - start

    # Oklab to RGB (with gamut mapping)
    start = time.time()
    oklch_image = ct_numba.batch_oklab_to_oklch(oklab_image)
    oklch_mapped = ct_numba.batch_gamut_map_oklch(oklch_image)
    oklab_mapped = ct_numba.batch_oklch_to_oklab(oklch_mapped)
    ct_numba.batch_oklab_to_srgb(oklab_mapped)
    oklab_to_rgb_time = time.time() - start

    return rgb_to_oklab_time, oklab_to_rgb_time


def create_test_image(width: int, height: int) -> np.ndarray:
    """Create a test image with random colors."""
    return np.random.rand(height, width, 3).astype(np.float32)


def test_performance_comparison() -> None:
    """Compare performance between ColorAide and Numba implementations."""
    # Test parameters
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    results = []

    # Test each image size
    for width, height in sizes:
        logger.info(f"\nBenchmarking {width}x{height} image...")

        # Create test image
        image: np.ndarray[Any, np.dtype[np.uint8]] = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        # Convert to float32 [0,1]
        image_float = image.astype(np.float32) / 255.0

        # Time ColorAide version
        ca_time = None
        try:
            start_time = time.time()
            for _ in range(3):  # Run multiple times for better timing
                for y in range(height):
                    for x in range(width):
                        rgb = image_float[y, x]
                        color = Color("srgb", rgb)
                        _ = color.convert("oklab")
                        _ = color.convert("oklch")
            ca_time = (time.time() - start_time) / 3
        except Exception as e:
            logger.error(f"ColorAide error: {e}")

        # Time Numba version
        start_time = time.time()
        for _ in range(3):  # Run multiple times for better timing
            oklab = ct_numba.batch_srgb_to_oklab(image_float)
            _ = ct_numba.batch_oklab_to_oklch(oklab)
        nb_time = (time.time() - start_time) / 3

        # Calculate speedup
        speedup = ca_time / nb_time if ca_time else None
        results.append((width, height, ca_time, nb_time, speedup))

    # Print results table
    logger.info("\nPerformance Results:")
    logger.info("Size      | ColorAide | Numba  | Speedup")
    logger.info("-" * 40)

    for width, height, ca_time, nb_time, speedup in results:
        size_str = f"{width}x{height}"
        ca_str = f"{ca_time:.3f}s" if ca_time else "N/A"
        nb_str = f"{nb_time:.3f}s"
        speedup_str = f"{speedup:.1f}x" if speedup else "N/A"
        logger.info(f"{size_str:9} | {ca_str:9} | {nb_str:6} | {speedup_str}")


if __name__ == "__main__":
    test_performance_comparison()
