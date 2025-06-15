#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["numpy", "loguru", "coloraide", "opencv-python", "pillow", "click", "numba"]
# ///
# this_file: tests/test_optimizations.py

"""
Test script for fast_hierar and spatial acceleration optimizations.
"""

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imgcolorshine import ColorTransformer, ImageProcessor, OKLCHEngine
from imgcolorshine.hierarchical import HierarchicalProcessor
from imgcolorshine.spatial_accel import SpatialAccelerator


def create_test_image(width: int, height: int) -> np.ndarray:
    """Create a test image with gradient patterns."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create gradient pattern
    for y in range(height):
        for x in range(width):
            image[y, x] = [
                int(255 * x / width),  # Red gradient left to right
                int(255 * y / height),  # Green gradient top to bottom
                128,  # Constant blue
            ]

    return image


def test_hierarchical_processing():
    """Test fast_hierar processing optimization."""
    logger.info("Testing fast_hierar processing...")

    # Create test image
    image = create_test_image(512, 512)

    # Initialize components
    engine = OKLCHEngine()
    transformer = ColorTransformer(engine)

    # Create attractor
    attractor = engine.create_attractor("red", 50.0, 75.0)

    # Initialize fast_hierar processor
    hier_processor = HierarchicalProcessor()

    # Prepare data
    attractors_lab = np.array([attractor.oklab_values])
    attractors_lch = np.array([attractor.oklch_values])
    tolerances = np.array([attractor.tolerance])
    strengths = np.array([attractor.strength])
    channels = [True, True, True]  # All channels enabled
    flags_array = np.array(channels)

    # Create transform function
    def transform_func(img_rgb, *args):
        """"""
        return (
            transformer._transform_tile(
                img_rgb / 255.0, attractors_lab, attractors_lch, tolerances, strengths, flags_array
            )
            * 255.0
        )

    # Test fast_hierar processing
    start_time = time.time()
    result = hier_processor.process_hierarchical(image, transform_func, attractors_lab, tolerances, strengths, channels)
    elapsed = time.time() - start_time

    logger.info(f"Hierarchical processing completed in {elapsed:.3f}s")
    logger.info(f"Result shape: {result.shape}, dtype: {result.dtype}")

    # Verify result
    assert result.shape == image.shape
    # Result is float32, original is uint8 - convert for comparison
    result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
    assert np.any(result_uint8 != image)  # Should have some changes

    return result


def test_spatial_acceleration():
    """Test spatial acceleration optimization."""
    logger.info("\nTesting spatial acceleration...")

    # Create test image
    image = create_test_image(512, 512)
    image_float = image / 255.0

    # Initialize components
    engine = OKLCHEngine()
    transformer = ColorTransformer(engine)

    # Create attractor
    attractor = engine.create_attractor("green", 40.0, 60.0)

    # Initialize spatial accelerator
    spatial_acc = SpatialAccelerator()

    # Prepare data
    attractors_lab = np.array([attractor.oklab_values])
    attractors_lch = np.array([attractor.oklch_values])
    tolerances = np.array([attractor.tolerance])
    strengths = np.array([attractor.strength])
    channels = [True, True, True]
    flags_array = np.array(channels)

    # Convert to Oklab for spatial queries
    image_oklab = engine.batch_rgb_to_oklab(image_float)

    # Create transform function
    def transform_func(img_rgb, *args):
        """"""
        return (
            transformer._transform_tile(
                img_rgb / 255.0, attractors_lab, attractors_lch, tolerances, strengths, flags_array
            )
            * 255.0
        )

    # Test spatial acceleration
    start_time = time.time()
    result = spatial_acc.transform_with_spatial_accel(
        image, image_oklab, attractors_lab, tolerances, strengths, transform_func, channels
    )
    elapsed = time.time() - start_time

    logger.info(f"Spatial acceleration completed in {elapsed:.3f}s")
    logger.info(f"Result shape: {result.shape}, dtype: {result.dtype}")

    # Log statistics
    if hasattr(spatial_acc, "uniform_tiles"):
        logger.info(f"Uniform tiles: {spatial_acc.uniform_tiles}")
        logger.info(f"Partial tiles: {spatial_acc.partial_tiles}")
        logger.info(f"Skipped tiles: {spatial_acc.skipped_tiles}")

    # Verify result
    assert result.shape == image.shape
    # Result may be float32 while original is uint8

    return result


def test_combined_optimizations():
    """Test combined fast_hierar + spatial acceleration."""
    logger.info("\nTesting combined optimizations...")

    # Create larger test image
    image = create_test_image(1024, 1024)

    # Initialize components
    engine = OKLCHEngine()
    transformer = ColorTransformer(engine)

    # Create multiple attractors
    attractor1 = engine.create_attractor("blue", 60.0, 80.0)
    attractor2 = engine.create_attractor("yellow", 40.0, 60.0)

    attractor_objects = [attractor1, attractor2]

    # Use the process_with_optimizations function
    from imgcolorshine.imgcolorshine import process_with_optimizations

    start_time = time.time()
    result = process_with_optimizations(
        image,
        attractor_objects,
        luminance=True,
        saturation=True,
        hue=True,
        hierarchical=True,
        spatial_accel=True,
        transformer=transformer,
        engine=engine,
    )
    elapsed = time.time() - start_time

    logger.info(f"Combined optimization completed in {elapsed:.3f}s")
    logger.info(f"Result shape: {result.shape}, dtype: {result.dtype}")

    # Verify result
    assert result.shape == image.shape
    assert np.any(result != image)  # Should have changes

    return result


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=== Testing imgcolorshine optimizations ===\n")

    # Test individual optimizations
    hier_result = test_hierarchical_processing()
    spatial_result = test_spatial_acceleration()
    combined_result = test_combined_optimizations()

    logger.info("\n=== All tests completed successfully! ===")
