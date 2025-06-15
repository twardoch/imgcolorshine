#!/usr/bin/env python
"""
Test suite for 3D Color Look-Up Table (LUT) implementation.

Tests LUT building, caching, interpolation, and application to images.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from imgcolorshine.lut import ColorLUT, apply_lut_trilinear, create_identity_lut, transform_pixel_for_lut


class TestColorLUT:
    """Test ColorLUT class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary cache directory
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.cache_dir.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test LUT initialization."""
        lut = ColorLUT(size=33, cache_dir=self.cache_dir)

        assert lut.size == 33
        assert lut.cache_dir == self.cache_dir
        assert lut.lut is None  # Not built yet

    def test_cache_key_generation(self):
        """Test cache key generation for different parameters."""
        lut = ColorLUT(size=17, cache_dir=self.cache_dir)

        # Test parameters
        attractors1 = np.array([[0.5, 0.1, 0.1]], dtype=np.float32)
        attractors2 = np.array([[0.5, 0.1, 0.2]], dtype=np.float32)  # Different
        tolerances = np.array([50.0], dtype=np.float32)
        strengths = np.array([75.0], dtype=np.float32)
        channels = (True, True, True)

        # Generate keys
        key1 = lut._get_cache_key(attractors1, tolerances, strengths, channels)
        key2 = lut._get_cache_key(attractors2, tolerances, strengths, channels)
        key3 = lut._get_cache_key(attractors1, tolerances, strengths, channels)

        # Same parameters should give same key
        assert key1 == key3
        # Different parameters should give different key
        assert key1 != key2
        # Key should be 16 characters
        assert len(key1) == 16

    def test_cache_path_generation(self):
        """Test cache file path generation."""
        lut = ColorLUT(size=33, cache_dir=self.cache_dir)

        cache_key = "test1234abcd5678"
        path = lut._get_cache_path(cache_key)

        assert path.parent == self.cache_dir
        assert path.name == f"lut_{cache_key}_33.pkl"

    def test_identity_lut_creation(self):
        """Test identity LUT creation."""
        size = 17
        identity_lut = create_identity_lut(size)

        assert identity_lut.shape == (size, size, size, 3)
        assert identity_lut.dtype == np.float32

        # Check some values
        assert np.allclose(identity_lut[0, 0, 0], [0, 0, 0])
        assert np.allclose(identity_lut[-1, -1, -1], [1, 1, 1])

        # Check middle value
        mid = size // 2
        expected = mid / (size - 1)
        assert np.allclose(identity_lut[mid, mid, mid], [expected, expected, expected])

    def test_build_lut_basic(self):
        """Test basic LUT building."""
        lut = ColorLUT(size=9, cache_dir=self.cache_dir)  # Small for testing

        # Mock transform function that slightly shifts colors
        def mock_transform(rgb, attractors_lab, tolerances, strengths, l, s, h):
            return rgb * 0.9  # Simple darkening

        attractors = np.array([[0.5, 0.1, 0.1]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([100.0], dtype=np.float32)
        channels = (True, True, True)

        # Build LUT
        lut.build_lut(mock_transform, attractors, tolerances, strengths, channels)

        # Check LUT was built
        assert lut.lut is not None
        assert lut.lut.shape == (9, 9, 9, 3)

        # Check some transformations
        assert np.allclose(lut.lut[0, 0, 0], [0, 0, 0])  # Black stays black
        assert np.allclose(lut.lut[-1, -1, -1], [0.9, 0.9, 0.9])  # White darkened

    def test_lut_caching(self):
        """Test LUT caching functionality."""
        lut = ColorLUT(size=9, cache_dir=self.cache_dir)

        # Mock transform
        call_count = 0

        def counting_transform(rgb, *args):
            nonlocal call_count
            call_count += 1
            return rgb * 0.8

        attractors = np.array([[0.5, 0.1, 0.1]], dtype=np.float32)
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([100.0], dtype=np.float32)
        channels = (True, True, True)

        # First build - should call transform
        lut.build_lut(counting_transform, attractors, tolerances, strengths, channels)
        first_call_count = call_count
        assert first_call_count == 9 * 9 * 9  # All grid points

        # Second build with same parameters - should load from cache
        lut2 = ColorLUT(size=9, cache_dir=self.cache_dir)
        lut2.build_lut(counting_transform, attractors, tolerances, strengths, channels)

        # Transform shouldn't be called again
        assert call_count == first_call_count
        # LUTs should be identical
        np.testing.assert_array_equal(lut.lut, lut2.lut)

    def test_apply_lut_error_handling(self):
        """Test error handling when LUT not built."""
        lut = ColorLUT(size=17)
        image = np.random.rand(10, 10, 3).astype(np.float32)

        with pytest.raises(ValueError, match="LUT not built"):
            lut.apply_lut(image)

    def test_trilinear_interpolation(self):
        """Test trilinear interpolation accuracy."""
        # Create simple gradient LUT
        size = 5
        lut_array = create_identity_lut(size)

        # Test image with exact grid points
        test_image = np.array(
            [[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]], dtype=np.float32
        )

        result = apply_lut_trilinear(test_image, lut_array, size)

        # Should be identical for identity LUT
        np.testing.assert_allclose(result, test_image, atol=1e-6)

    def test_trilinear_interpolation_between_points(self):
        """Test interpolation between grid points."""
        size = 3
        # Create LUT that doubles the red channel
        lut_array = np.empty((size, size, size, 3), dtype=np.float32)
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    r_val = r / (size - 1)
                    g_val = g / (size - 1)
                    b_val = b / (size - 1)
                    lut_array[r, g, b] = [min(1.0, r_val * 2), g_val, b_val]

        # Test interpolation at midpoint
        test_image = np.array([[[0.25, 0.25, 0.25]]], dtype=np.float32)
        result = apply_lut_trilinear(test_image, lut_array, size)

        # Red should be doubled (0.25 * 2 = 0.5)
        expected = np.array([[[0.5, 0.25, 0.25]]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_edge_case_handling(self):
        """Test edge cases in LUT application."""
        size = 9
        lut_array = create_identity_lut(size)

        # Test with edge values
        test_image = np.array(
            [
                [[0.0, 0.0, 0.0]],  # Pure black
                [[1.0, 1.0, 1.0]],  # Pure white
                [[0.999, 0.999, 0.999]],  # Near white
                [[0.001, 0.001, 0.001]],  # Near black
            ],
            dtype=np.float32,
        )

        result = apply_lut_trilinear(test_image, lut_array, size)

        # Should handle edge cases without errors
        assert result.shape == test_image.shape
        np.testing.assert_allclose(result, test_image, atol=1e-5)

    def test_full_pipeline(self):
        """Test complete LUT pipeline with real transformation."""
        lut = ColorLUT(size=9, cache_dir=self.cache_dir)

        # Use the actual transform function for LUT
        from imgcolorshine.kernel import transform_pixel_fused

        def wrapped_transform(rgb, attractors_lab, tolerances, strengths, l, s, h):
            r_out, g_out, b_out = transform_pixel_fused(
                rgb[0], rgb[1], rgb[2], attractors_lab, tolerances, strengths, l, s, h
            )
            return np.array([r_out, g_out, b_out], dtype=np.float32)

        # Set up transformation
        attractors = np.array([[0.7, 0.0, 0.0]], dtype=np.float32)  # Bright neutral
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([50.0], dtype=np.float32)
        channels = (True, False, False)  # Only luminance

        # Build LUT
        lut.build_lut(wrapped_transform, attractors, tolerances, strengths, channels)

        # Create test image
        test_image = np.random.rand(20, 20, 3).astype(np.float32) * 0.5  # Dark image

        # Apply LUT
        result = lut.apply_lut(test_image)

        # Check result
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype
        # Should be brighter on average (pulled toward bright attractor)
        assert np.mean(result) > np.mean(test_image)


class TestLUTPerformance:
    """Test performance-related aspects of LUT."""

    def test_different_lut_sizes(self):
        """Test LUT with different sizes."""
        sizes = [9, 17, 33, 65]

        for size in sizes:
            lut = ColorLUT(size=size)
            lut_array = create_identity_lut(size)
            lut.lut = lut_array

            # Apply to test image
            test_image = np.random.rand(50, 50, 3).astype(np.float32)
            result = lut.apply_lut(test_image)

            assert result.shape == test_image.shape
            # Identity LUT should preserve values
            np.testing.assert_allclose(result, test_image, atol=1e-3)

    def test_large_image_handling(self):
        """Test LUT application to large images."""
        lut = ColorLUT(size=17)
        lut.lut = create_identity_lut(17)

        # Create large test image
        large_image = np.random.rand(500, 500, 3).astype(np.float32)

        # Apply LUT
        result = lut.apply_lut(large_image)

        assert result.shape == large_image.shape
        assert result.dtype == large_image.dtype

    def test_memory_efficiency(self):
        """Test memory efficiency of LUT operations."""
        # Small LUT should have small memory footprint
        ColorLUT(size=9)
        small_array = create_identity_lut(9)
        expected_size = 9 * 9 * 9 * 3 * 4  # float32 = 4 bytes
        assert small_array.nbytes == expected_size

        # Larger LUT
        ColorLUT(size=65)
        large_array = create_identity_lut(65)
        expected_size = 65 * 65 * 65 * 3 * 4
        assert large_array.nbytes == expected_size

    def test_parallel_consistency(self):
        """Test that parallel processing is consistent."""
        lut = ColorLUT(size=17)
        lut.lut = create_identity_lut(17)

        # Same image processed multiple times
        test_image = np.random.rand(100, 100, 3).astype(np.float32)

        results = []
        for _ in range(3):
            result = lut.apply_lut(test_image.copy())
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i], decimal=6)


class TestTransformPixelForLUT:
    """Test the transform pixel wrapper for LUT building."""

    def test_transform_pixel_wrapper_integration(self):
        """Test that the wrapper works correctly with the real kernel."""
        # Test inputs - gray pixel with gray attractor
        rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        attractors = np.array([[0.7, 0.0, 0.0]], dtype=np.float32)  # Brighter gray
        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([50.0], dtype=np.float32)

        # Call wrapper
        result = transform_pixel_for_lut(
            rgb,
            attractors,
            tolerances,
            strengths,
            True,
            False,
            False,  # Only luminance
        )

        # Check result
        assert result.shape == (3,)
        assert result.dtype == np.float32
        # Should be brighter than original
        assert np.mean(result) > 0.5
        # Should still be gray (all channels similar)
        assert np.allclose(result[0], result[1], atol=1e-4)
        assert np.allclose(result[1], result[2], atol=1e-4)
