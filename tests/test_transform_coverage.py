#!/usr/bin/env python
"""
Additional tests to improve coverage for transform.py module.

Tests the core transformation functions including weight calculation,
multi-attractor blending, and channel-specific transformations.
"""

import numpy as np
import pytest

from imgcolorshine.color import Attractor, OKLCHEngine
from imgcolorshine.transform import (
    MAX_DELTA_E,
    ColorTransformer,
    blend_colors,
    calculate_delta_e_fast,
    calculate_weights,
    transform_pixels,
)


class TestTransformFunctions:
    """Test individual transformation functions."""

    def test_calculate_delta_e_fast(self):
        """Test fast delta E calculation."""
        # Test identical colors
        color1 = np.array([0.5, 0.0, 0.0])
        color2 = np.array([0.5, 0.0, 0.0])
        assert calculate_delta_e_fast(color1, color2) == 0.0

        # Test known distance
        color1 = np.array([0.5, 0.0, 0.0])
        color2 = np.array([0.5, 0.1, 0.0])
        expected = 0.1
        assert np.isclose(calculate_delta_e_fast(color1, color2), expected)

        # Test 3D distance
        color1 = np.array([0.0, 0.0, 0.0])
        color2 = np.array([0.3, 0.4, 0.0])  # 3-4-5 triangle
        expected = 0.5
        assert np.isclose(calculate_delta_e_fast(color1, color2), expected)

    def test_calculate_weights(self):
        """Test weight calculation for attractors."""
        # Single attractor case
        pixel_lab = np.array([0.5, 0.0, 0.0])
        attractors_lab = np.array([[0.5, 0.0, 0.0]])  # Exact match
        tolerances = np.array([100.0])
        strengths = np.array([100.0])

        weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)
        assert len(weights) == 1
        assert weights[0] == 1.0  # Full strength at zero distance

        # Test tolerance boundary
        pixel_lab = np.array([0.5, 0.0, 0.0])
        # Place attractor at exactly tolerance distance
        attractors_lab = np.array([[0.5 + MAX_DELTA_E * 0.5, 0.0, 0.0]])
        tolerances = np.array([50.0])  # 50% tolerance = MAX_DELTA_E * 0.5
        strengths = np.array([100.0])

        weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)
        assert weights[0] == 0.0  # Should be zero at boundary

        # Test multiple attractors
        pixel_lab = np.array([0.5, 0.0, 0.0])
        attractors_lab = np.array(
            [
                [0.5, 0.0, 0.0],  # Exact match
                [0.6, 0.0, 0.0],  # Close
                [1.0, 0.0, 0.0],  # Far
            ]
        )
        tolerances = np.array([100.0, 100.0, 100.0])
        strengths = np.array([100.0, 50.0, 100.0])

        weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)
        assert len(weights) == 3
        assert weights[0] == 1.0  # Full strength
        assert 0 < weights[1] < 0.5  # Partial strength
        # Third weight depends on distance

    def test_blend_colors(self):
        """Test color blending function."""
        # Test with single attractor
        pixel_lab = np.array([0.5, 0.0, 0.0])
        pixel_lch = np.array([0.5, 0.0, 0.0])  # L, C, H
        attractors_lab = np.array([[0.6, 0.1, 0.0]])
        attractors_lch = np.array([[0.6, 0.1, 90.0]])
        weights = np.array([0.5])
        flags = np.array([True, True, True])  # All channels enabled

        result = blend_colors(pixel_lab, pixel_lch, attractors_lab, attractors_lch, weights, flags)

        assert len(result) == 3
        assert result[0] >= 0.5  # L should be between original and attractor
        assert result[0] <= 0.6

    def test_blend_colors_channel_control(self):
        """Test channel-specific blending."""
        pixel_lab = np.array([0.5, 0.0, 0.0])
        pixel_lch = np.array([0.5, 0.1, 180.0])
        attractors_lab = np.array([[0.7, 0.1, 0.1]])
        attractors_lch = np.array([[0.7, 0.2, 90.0]])
        weights = np.array([1.0])

        # Test lightness only
        flags = np.array([True, False, False])
        result = blend_colors(pixel_lab, pixel_lch, attractors_lab, attractors_lch, weights, flags)
        # Only L should change
        assert result[0] == attractors_lch[0][0]  # Full weight

        # Test chroma only
        flags = np.array([False, True, False])
        result = blend_colors(pixel_lab, pixel_lch, attractors_lab, attractors_lch, weights, flags)
        # L should stay same, chroma changes
        assert result[0] == pixel_lch[0]

    def test_transform_pixels(self):
        """Test batch pixel transformation."""
        # Create small test image
        pixels_lab = np.array(
            [[[0.5, 0.0, 0.0], [0.6, 0.1, 0.0]], [[0.7, 0.0, 0.1], [0.4, -0.1, 0.0]]], dtype=np.float32
        )

        pixels_lch = np.array(
            [[[0.5, 0.0, 0.0], [0.6, 0.1, 0.0]], [[0.7, 0.1, 90.0], [0.4, 0.1, 180.0]]], dtype=np.float32
        )

        # Single attractor
        attractors_lab = np.array([[0.5, 0.05, 0.05]], dtype=np.float32)
        attractors_lch = np.array([[0.5, 0.07, 45.0]], dtype=np.float32)
        tolerances = np.array([100.0])
        strengths = np.array([50.0])
        flags = np.array([True, True, True])

        result = transform_pixels(pixels_lab, pixels_lch, attractors_lab, attractors_lch, tolerances, strengths, flags)

        assert result.shape == pixels_lab.shape
        assert result.dtype == pixels_lab.dtype


class TestColorTransformer:
    """Test ColorTransformer class."""

    def test_initialization(self):
        """Test transformer initialization."""
        engine = OKLCHEngine()
        transformer = ColorTransformer(engine)

        assert transformer.engine == engine

    def test_transform_image_basic(self):
        """Test basic image transformation."""
        # Create test image
        test_image = np.random.rand(10, 10, 3).astype(np.float32)

        # Create transformer
        engine = OKLCHEngine()
        transformer = ColorTransformer(engine)

        # Create attractors
        attractors = [engine.create_attractor("red", 50.0, 75.0)]

        # Transform with all channels
        flags = {"luminance": True, "saturation": True, "chroma": True}

        result = transformer.transform_image(test_image, attractors, flags)

        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype

    def test_transform_image_channel_control(self):
        """Test channel-specific transformations."""
        test_image = np.random.rand(10, 10, 3).astype(np.float32)

        engine = OKLCHEngine()
        transformer = ColorTransformer(engine)

        attractors = [engine.create_attractor("blue", 60.0, 80.0)]

        # Test with only luminance
        flags = {"luminance": True, "saturation": False, "chroma": False}

        result = transformer.transform_image(test_image, attractors, flags)
        assert result.shape == test_image.shape

    def test_empty_attractors(self):
        """Test transformation with no attractors."""
        test_image = np.random.rand(10, 10, 3).astype(np.float32)

        engine = OKLCHEngine()
        transformer = ColorTransformer(engine)

        flags = {"luminance": True, "saturation": True, "chroma": True}

        # Transform with empty attractors
        result = transformer.transform_image(test_image, [], flags)

        # Should return original image
        np.testing.assert_array_equal(result, test_image)

    def test_transform_with_progress(self):
        """Test transformation with progress callback."""
        test_image = np.random.rand(20, 20, 3).astype(np.float32)

        engine = OKLCHEngine()
        transformer = ColorTransformer(engine)

        attractors = [engine.create_attractor("green", 70.0, 90.0)]

        flags = {"luminance": True, "saturation": True, "chroma": True}

        # Track progress
        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        result = transformer.transform_image(test_image, attractors, flags, progress_callback=progress_callback)

        assert result.shape == test_image.shape
        # Progress tracking depends on implementation details
