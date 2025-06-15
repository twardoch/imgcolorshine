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

import numpy as np
from coloraide import Color
from loguru import logger
from PIL import Image

sys.path.insert(0, "src")
from imgcolorshine import color_transforms_numba as ct_numba
from imgcolorshine.color_engine import OKLCHEngine


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


def main():
    """Run performance benchmarks."""
    logger.info("Starting performance benchmark...")

    # Test different image sizes
    sizes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]

    # Warm up Numba JIT
    logger.info("Warming up Numba JIT compiler...")
    warmup_img = create_test_image(64, 64)
    _ = benchmark_numba_conversion(warmup_img)

    results = []

    for width, height in sizes:
        logger.info(f"\nBenchmarking {width}×{height} image...")

        # Create test image
        img = create_test_image(width, height)
        pixels = width * height

        # Benchmark ColorAide (only for smaller images)
        if pixels <= 512 * 512:
            ca_rgb2oklab, ca_oklab2rgb = benchmark_coloraide_conversion(img)
            ca_total = ca_rgb2oklab + ca_oklab2rgb
            logger.info(
                f"ColorAide: RGB→Oklab: {ca_rgb2oklab:.3f}s, Oklab→RGB: {ca_oklab2rgb:.3f}s, Total: {ca_total:.3f}s"
            )
        else:
            ca_total = None
            logger.info("ColorAide: Skipped (too slow for large images)")

        # Benchmark Numba
        nb_rgb2oklab, nb_oklab2rgb = benchmark_numba_conversion(img)
        nb_total = nb_rgb2oklab + nb_oklab2rgb
        logger.info(
            f"Numba:     RGB→Oklab: {nb_rgb2oklab:.3f}s, Oklab→RGB: {nb_oklab2rgb:.3f}s, Total: {nb_total:.3f}s"
        )

        # Calculate speedup
        if ca_total is not None:
            speedup = ca_total / nb_total
            logger.info(f"Speedup:   {speedup:.1f}x faster")
            results.append((width, height, ca_total, nb_total, speedup))
        else:
            results.append((width, height, None, nb_total, None))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Size':<12} {'ColorAide':<12} {'Numba':<12} {'Speedup':<12}")
    logger.info("-" * 60)

    for width, height, ca_time, nb_time, speedup in results:
        size_str = f"{width}×{height}"
        ca_str = f"{ca_time:.3f}s" if ca_time else "N/A"
        nb_str = f"{nb_time:.3f}s"
        speedup_str = f"{speedup:.1f}x" if speedup else "N/A"
        logger.info(f"{size_str:<12} {ca_str:<12} {nb_str:<12} {speedup_str:<12}")

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
