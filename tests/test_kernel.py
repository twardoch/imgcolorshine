#!/usr/bin/env python
"""
Test suite for fused color transformation kernels.

Tests the high-performance fused kernel implementation that combines
all color space conversions and transformations in a single pass.
"""

import numpy as np
import pytest

from imgcolorshine.kernel import transform_image_fused, transform_pixel_fused
from imgcolorshine.trans_numba import linear_to_srgb_component, srgb_to_linear_component


class TestPixelTransformation:
    """Test single pixel transformation kernel."""

    def test_identity_transformation(self):
        """Test that pixels unchanged with no attractors."""
        # Test color
        r, g, b = 0.5, 0.6, 0.7

        # No attractors
        attractors_lab = np.empty((0, 3), dtype=np.float32)
        tolerances = np.empty(0, dtype=np.float32)
        strengths = np.empty(0, dtype=np.float32)

        r_out, g_out, b_out = transform_pixel_fused(r, g, b, attractors_lab, tolerances, strengths, True, True, True)

        # Should be unchanged
        assert np.isclose(r_out, r, atol=1e-6)
        assert np.isclose(g_out, g, atol=1e-6)
        assert np.isclose(b_out, b, atol=1e-6)

    def test_grayscale_preservation(self):
        """Test that grayscale pixels remain grayscale when only luminance is changed."""
        # Gray pixel
        gray = 0.5
        r, g, b = gray, gray, gray

        # Attractor that is also gray (no chroma)
        # Gray in Oklab has a=0, b=0
        attractors_lab = np.array([[0.7, 0.0, 0.0]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([50.0], dtype=np.float32)

        # Transform with only luminance enabled
        r_out, g_out, b_out = transform_pixel_fused(
            r,
            g,
            b,
            attractors_lab,
            tolerances,
            strengths,
            True,
            False,
            False,  # Only luminance
        )

        # Result should still be grayscale (r=g=b)
        # Allow for small numerical errors
        assert np.isclose(r_out, g_out, atol=1e-4)
        assert np.isclose(g_out, b_out, atol=1e-4)
        assert np.isclose(r_out, b_out, atol=1e-4)

    def test_black_white_extremes(self):
        """Test transformation of black and white pixels."""
        # Pure black
        r_out, g_out, b_out = transform_pixel_fused(
            0.0,
            0.0,
            0.0,
            np.array([[0.5, 0.1, 0.1]], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            True,
            True,
            True,
        )

        # Black should transform toward attractor
        assert r_out >= 0.0
        assert g_out >= 0.0
        assert b_out >= 0.0

        # Pure white
        r_out, g_out, b_out = transform_pixel_fused(
            1.0,
            1.0,
            1.0,
            np.array([[0.5, 0.1, 0.1]], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            True,
            True,
            True,
        )

        # White should remain valid
        assert r_out <= 1.0
        assert g_out <= 1.0
        assert b_out <= 1.0

    def test_channel_control(self):
        """Test selective channel transformations."""
        r, g, b = 0.7, 0.5, 0.3
        attractors_lab = np.array([[0.8, 0.0, 0.0]], dtype=np.float32)  # Neutral high L
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([100.0], dtype=np.float32)

        # Only luminance
        r1, g1, b1 = transform_pixel_fused(r, g, b, attractors_lab, tolerances, strengths, True, False, False)

        # Only saturation
        r2, g2, b2 = transform_pixel_fused(r, g, b, attractors_lab, tolerances, strengths, False, True, False)

        # Only hue
        r3, g3, b3 = transform_pixel_fused(r, g, b, attractors_lab, tolerances, strengths, False, False, True)

        # Results should be different for each mode
        assert not (np.isclose(r1, r2) and np.isclose(g1, g2) and np.isclose(b1, b2))
        assert not (np.isclose(r1, r3) and np.isclose(g1, g3) and np.isclose(b1, b3))

    def test_multiple_attractors(self):
        """Test blending of multiple attractors."""
        r, g, b = 0.5, 0.5, 0.5

        # Two opposing attractors
        attractors_lab = np.array(
            [
                [0.3, 0.1, 0.0],  # Darker
                [0.7, -0.1, 0.0],  # Lighter
            ],
            dtype=np.float32,
        )
        tolerances = np.array([100.0, 100.0], dtype=np.float32)
        strengths = np.array([50.0, 50.0], dtype=np.float32)

        r_out, g_out, b_out = transform_pixel_fused(r, g, b, attractors_lab, tolerances, strengths, True, True, True)

        # Result should be valid
        assert 0 <= r_out <= 1
        assert 0 <= g_out <= 1
        assert 0 <= b_out <= 1

    def test_tolerance_effect(self):
        """Test that tolerance controls influence radius."""
        r, g, b = 0.5, 0.5, 0.5
        attractors_lab = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
        strengths = np.array([100.0], dtype=np.float32)

        # Low tolerance
        tolerances_low = np.array([10.0], dtype=np.float32)
        r1, g1, b1 = transform_pixel_fused(r, g, b, attractors_lab, tolerances_low, strengths, True, True, True)

        # High tolerance
        tolerances_high = np.array([90.0], dtype=np.float32)
        r2, g2, b2 = transform_pixel_fused(r, g, b, attractors_lab, tolerances_high, strengths, True, True, True)

        # High tolerance should have stronger effect
        # Calculate distances from original
        dist1 = np.sqrt((r1 - r) ** 2 + (g1 - g) ** 2 + (b1 - b) ** 2)
        dist2 = np.sqrt((r2 - r) ** 2 + (g2 - g) ** 2 + (b2 - b) ** 2)

        assert dist2 > dist1 or np.isclose(dist2, dist1, atol=1e-6)

    def test_strength_effect(self):
        """Test that strength controls transformation intensity."""
        r, g, b = 0.5, 0.5, 0.5
        attractors_lab = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)

        # Low strength
        strengths_low = np.array([20.0], dtype=np.float32)
        r1, g1, b1 = transform_pixel_fused(r, g, b, attractors_lab, tolerances, strengths_low, True, True, True)

        # High strength
        strengths_high = np.array([80.0], dtype=np.float32)
        r2, g2, b2 = transform_pixel_fused(r, g, b, attractors_lab, tolerances, strengths_high, True, True, True)

        # High strength should have stronger effect
        dist1 = np.sqrt((r1 - r) ** 2 + (g1 - g) ** 2 + (b1 - b) ** 2)
        dist2 = np.sqrt((r2 - r) ** 2 + (g2 - g) ** 2 + (b2 - b) ** 2)

        assert dist2 > dist1

    def test_gamut_mapping(self):
        """Test that out-of-gamut colors are properly mapped."""
        # Start with a highly saturated color
        r, g, b = 1.0, 0.0, 0.0  # Pure red

        # Attractor that would push out of gamut
        # Approximate values for a very saturated cyan in Oklab
        attractors_lab = np.array([[0.7, -0.3, -0.2]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([100.0], dtype=np.float32)

        r_out, g_out, b_out = transform_pixel_fused(r, g, b, attractors_lab, tolerances, strengths, True, True, True)

        # Result must be in valid sRGB range
        assert 0 <= r_out <= 1
        assert 0 <= g_out <= 1
        assert 0 <= b_out <= 1


class TestImageTransformation:
    """Test batch image transformation."""

    def test_small_image_transformation(self):
        """Test transformation of small test image."""
        # Create 4x4 test image
        image = np.random.rand(4, 4, 3).astype(np.float32)

        # Single attractor
        attractors_lab = np.array([[0.6, 0.1, 0.05]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([50.0], dtype=np.float32)

        result = transform_image_fused(image, attractors_lab, tolerances, strengths, True, True, True)

        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_image_with_no_attractors(self):
        """Test that image unchanged with empty attractors."""
        image = np.random.rand(5, 5, 3).astype(np.float32)

        attractors_lab = np.empty((0, 3), dtype=np.float32)
        tolerances = np.empty(0, dtype=np.float32)
        strengths = np.empty(0, dtype=np.float32)

        result = transform_image_fused(image, attractors_lab, tolerances, strengths, True, True, True)

        np.testing.assert_array_almost_equal(result, image, decimal=6)

    def test_uniform_image_transformation(self):
        """Test transformation of uniform color image."""
        # Uniform gray image
        gray_value = 0.5
        image = np.full((10, 10, 3), gray_value, dtype=np.float32)

        attractors_lab = np.array([[0.7, 0.0, 0.0]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([75.0], dtype=np.float32)

        result = transform_image_fused(
            image,
            attractors_lab,
            tolerances,
            strengths,
            True,
            False,
            False,  # Only luminance
        )

        # All pixels should transform identically
        first_pixel = result[0, 0]
        assert np.all(np.abs(result - first_pixel) < 1e-6)

    def test_parallel_consistency(self):
        """Test that parallel processing produces consistent results."""
        # Test image that should stress parallel processing
        image = np.random.rand(32, 32, 3).astype(np.float32)

        attractors_lab = np.array([[0.5, 0.1, 0.1], [0.7, -0.1, 0.05]], dtype=np.float32)
        tolerances = np.array([80.0, 60.0], dtype=np.float32)
        strengths = np.array([60.0, 40.0], dtype=np.float32)

        # Transform same image multiple times
        result1 = transform_image_fused(image.copy(), attractors_lab, tolerances, strengths, True, True, True)

        result2 = transform_image_fused(image.copy(), attractors_lab, tolerances, strengths, True, True, True)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2, decimal=6)

    def test_edge_preservation(self):
        """Test that sharp edges are preserved."""
        # Create image with sharp edge
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[:, :5] = 0.3  # Left half dark
        image[:, 5:] = 0.7  # Right half light

        attractors_lab = np.array([[0.5, 0.05, 0.05]], dtype=np.float32)
        tolerances = np.array([50.0], dtype=np.float32)
        strengths = np.array([30.0], dtype=np.float32)

        result = transform_image_fused(image, attractors_lab, tolerances, strengths, True, True, True)

        # Check that edge is still present
        left_mean = np.mean(result[:, :5])
        right_mean = np.mean(result[:, 5:])
        assert right_mean > left_mean  # Edge preserved

    def test_color_variety_preservation(self):
        """Test that color variety is maintained."""
        # Create image with different colors
        image = np.zeros((3, 3, 3), dtype=np.float32)
        image[0, 0] = [1.0, 0.0, 0.0]  # Red
        image[0, 1] = [0.0, 1.0, 0.0]  # Green
        image[0, 2] = [0.0, 0.0, 1.0]  # Blue
        image[1, 0] = [1.0, 1.0, 0.0]  # Yellow
        image[1, 1] = [1.0, 0.0, 1.0]  # Magenta
        image[1, 2] = [0.0, 1.0, 1.0]  # Cyan
        image[2, :] = 0.5  # Gray

        attractors_lab = np.array([[0.6, 0.0, 0.0]], dtype=np.float32)
        tolerances = np.array([50.0], dtype=np.float32)
        strengths = np.array([30.0], dtype=np.float32)

        result = transform_image_fused(image, attractors_lab, tolerances, strengths, True, True, True)

        # Check that colors are still different
        unique_colors_before = len(np.unique(image.reshape(-1, 3), axis=0))
        unique_colors_after = len(np.unique(np.round(result, 3).reshape(-1, 3), axis=0))

        # Should maintain some color variety
        assert unique_colors_after >= unique_colors_before * 0.5


class TestPerformanceCharacteristics:
    """Test performance-related aspects of the kernel."""

    def test_dtype_preservation(self):
        """Test that float32 dtype is preserved for performance."""
        image = np.random.rand(10, 10, 3).astype(np.float32)
        attractors_lab = np.array([[0.5, 0.1, 0.1]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([100.0], dtype=np.float32)

        result = transform_image_fused(image, attractors_lab, tolerances, strengths, True, True, True)

        assert result.dtype == np.float32

    def test_memory_layout(self):
        """Test that output has same memory layout as input."""
        # C-contiguous input
        image_c = np.ascontiguousarray(np.random.rand(10, 10, 3).astype(np.float32))
        attractors_lab = np.array([[0.5, 0.1, 0.1]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([100.0], dtype=np.float32)

        result = transform_image_fused(image_c, attractors_lab, tolerances, strengths, True, True, True)

        assert result.flags.c_contiguous

    def test_large_attractor_count(self):
        """Test performance with many attractors."""
        image = np.random.rand(20, 20, 3).astype(np.float32)

        # Create 10 random attractors
        n_attractors = 10
        attractors_lab = np.random.rand(n_attractors, 3).astype(np.float32)
        attractors_lab[:, 0] *= 0.8  # L in [0, 0.8]
        attractors_lab[:, 1] = attractors_lab[:, 1] * 0.4 - 0.2  # a in [-0.2, 0.2]
        attractors_lab[:, 2] = attractors_lab[:, 2] * 0.4 - 0.2  # b in [-0.2, 0.2]

        tolerances = np.full(n_attractors, 50.0, dtype=np.float32)
        strengths = np.full(n_attractors, 10.0, dtype=np.float32)  # Low strengths

        result = transform_image_fused(image, attractors_lab, tolerances, strengths, True, True, True)

        assert result.shape == image.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)
