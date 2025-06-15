#!/usr/bin/env python3
"""
Debug the actual transformation process to find why no changes are visible.
"""

# Add the src directory to path so we can import imgcolorshine modules
import sys
from pathlib import Path

import numpy as np
from coloraide import Color
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent / "src"))

from imgcolorshine.color_engine import OKLCHEngine
from imgcolorshine.falloff import calculate_weights as falloff_calculate_weights
from imgcolorshine.image_io import ImageProcessor
from imgcolorshine.transforms import blend_colors, calculate_weights


def debug_transformation():
    """Debug the transformation process step by step."""

    # Initialize components
    engine = OKLCHEngine()
    processor = ImageProcessor()

    # Load a small sample from the test image
    image = processor.load_image("testdata/louis.jpg")

    # Extract a small sample for detailed analysis
    sample = image[100:110, 100:110]  # 10x10 pixel sample

    # Create blue attractor (same as test)
    attractor = engine.create_attractor("blue", 80, 80)

    # Convert sample to Oklab and OKLCH
    sample_lab = engine.batch_rgb_to_oklab(sample)

    sample_lch = np.zeros_like(sample_lab)
    for y in range(sample_lab.shape[0]):
        for x in range(sample_lab.shape[1]):
            light, a, b = sample_lab[y, x]
            sample_lch[y, x] = engine.oklab_to_oklch(light, a, b)

    # Test weight calculation for center pixel
    center_pixel_lab = sample_lab[5, 5]
    center_pixel_lch = sample_lch[5, 5]

    # Calculate distance and weight
    attractor_lab = np.array(attractor.oklab_values)
    delta_e = np.sqrt(np.sum((center_pixel_lab - attractor_lab) ** 2))

    tolerance = 80
    delta_e_max = 1.0 * (tolerance / 100.0) ** 2

    if delta_e <= delta_e_max:
        d_norm = delta_e / delta_e_max
        attraction_factor = 0.5 * (np.cos(d_norm * np.pi) + 1.0)
        weight = (80 / 100.0) * attraction_factor
    else:
        weight = 0.0

    # Test blending
    weights = np.array([weight])
    attractors_lab = np.array([attractor.oklab_values])
    attractors_lch = np.array([attractor.oklch_values])
    flags = np.array([False, False, True])  # Only hue transformation

    original_lab = center_pixel_lab.copy()
    blended_lab = blend_colors(center_pixel_lab, center_pixel_lch, attractors_lab, attractors_lch, weights, flags)

    # Convert back to RGB and see the difference
    engine.oklab_to_rgb(original_lab)
    engine.oklab_to_rgb(blended_lab)

    # Test multiple pixels to see statistics
    affected_count = 0
    total_weight_sum = 0
    max_rgb_change = 0

    for y in range(sample.shape[0]):
        for x in range(sample.shape[1]):
            pixel_lab = sample_lab[y, x]
            pixel_lch = sample_lch[y, x]

            # Calculate weight
            weights_array = calculate_weights(pixel_lab, attractors_lab, np.array([tolerance]), np.array([80]))

            if weights_array[0] > 0:
                affected_count += 1
                total_weight_sum += weights_array[0]

                # Test the change
                blended = blend_colors(pixel_lab, pixel_lch, attractors_lab, attractors_lch, weights_array, flags)

                orig_rgb = engine.oklab_to_rgb(pixel_lab)
                blend_rgb = engine.oklab_to_rgb(blended)
                rgb_change = np.max(np.abs(blend_rgb - orig_rgb))
                max_rgb_change = max(max_rgb_change, rgb_change)

    sample.shape[0] * sample.shape[1]

    if affected_count == 0 or max_rgb_change < 0.001 or max_rgb_change < 0.01:
        pass
    else:
        pass


def main():
    """Debug color transformation behavior."""
    engine = OKLCHEngine()

    # Create a sample image with a gradient
    sample = np.zeros((10, 10, 3), dtype=np.float32)
    for y in range(10):
        for x in range(10):
            sample[y, x] = [0.5, 0.0, 0.0]  # Base color

    # Convert to Oklab
    sample_lab = np.zeros_like(sample)
    sample_lch = np.zeros_like(sample)

    for y in range(sample_lab.shape[0]):
        for x in range(sample_lab.shape[1]):
            light, a, b = sample_lab[y, x]  # Changed 'l' to 'light'
            sample_lch[y, x] = engine.oklab_to_oklch(light, a, b)


if __name__ == "__main__":
    debug_transformation()
