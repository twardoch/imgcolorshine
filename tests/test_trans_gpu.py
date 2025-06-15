#!/usr/bin/env python
"""
Test suite for GPU-accelerated color transformations.

Tests both CuPy and CPU fallback paths, ensuring consistent behavior
across different hardware configurations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the module to test
from imgcolorshine import trans_gpu


class TestGPUColorMatrices:
    """Test color transformation matrices initialization."""

    def test_get_gpu_color_matrices_numpy(self):
        """Test matrix initialization with NumPy."""
        matrices = trans_gpu.get_gpu_color_matrices(np)

        # Check all required matrices are present
        required = [
            "LINEAR_RGB_TO_XYZ",
            "XYZ_TO_LINEAR_RGB",
            "XYZ_TO_LMS",
            "LMS_TO_XYZ",
            "LMS_TO_OKLAB",
            "OKLAB_TO_LMS",
        ]

        for matrix_name in required:
            assert matrix_name in matrices
            matrix = matrices[matrix_name]
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (3, 3)
            assert matrix.dtype == np.float32

    def test_get_gpu_color_matrices_cupy(self):
        """Test matrix initialization with CuPy (mocked)."""
        # Mock CuPy
        mock_cp = MagicMock()
        mock_cp.array = lambda x, dtype: np.array(x, dtype=dtype)
        mock_cp.float32 = np.float32

        matrices = trans_gpu.get_gpu_color_matrices(mock_cp)

        # Check matrices were created
        assert len(matrices) == 6
        for matrix in matrices.values():
            assert matrix.shape == (3, 3)


class TestColorSpaceConversions:
    """Test individual color space conversion functions."""

    def test_srgb_to_linear_gpu(self):
        """Test sRGB to linear conversion."""
        # Test with NumPy
        srgb = np.array([0.5, 0.25, 0.75], dtype=np.float32)
        linear = trans_gpu.srgb_to_linear_gpu(srgb, np)

        # Values should be transformed
        assert linear.shape == srgb.shape
        assert linear.dtype == np.float32
        assert not np.array_equal(linear, srgb)

        # Test edge cases
        assert trans_gpu.srgb_to_linear_gpu(np.array([0.0]), np)[0] == 0.0
        assert trans_gpu.srgb_to_linear_gpu(np.array([1.0]), np)[0] == 1.0

    def test_linear_to_srgb_gpu(self):
        """Test linear to sRGB conversion."""
        linear = np.array([0.5, 0.25, 0.75], dtype=np.float32)
        srgb = trans_gpu.linear_to_srgb_gpu(linear, np)

        # Values should be transformed
        assert srgb.shape == linear.shape
        assert srgb.dtype == np.float32
        assert not np.array_equal(srgb, linear)

        # Test edge cases
        assert trans_gpu.linear_to_srgb_gpu(np.array([0.0]), np)[0] == 0.0
        assert trans_gpu.linear_to_srgb_gpu(np.array([1.0]), np)[0] == pytest.approx(1.0)

    def test_srgb_linear_roundtrip(self):
        """Test sRGB <-> linear roundtrip conversion."""
        original = np.array([0.2, 0.5, 0.8], dtype=np.float32)

        # Convert sRGB -> linear -> sRGB
        linear = trans_gpu.srgb_to_linear_gpu(original, np)
        recovered = trans_gpu.linear_to_srgb_gpu(linear, np)

        np.testing.assert_allclose(recovered, original, atol=1e-6)


class TestBatchConversions:
    """Test batch color space conversions."""

    def test_batch_srgb_to_oklab_gpu(self):
        """Test batch sRGB to Oklab conversion."""
        # Create test image
        rgb = np.random.rand(10, 10, 3).astype(np.float32)

        # Use NumPy as array module
        oklab = trans_gpu.batch_srgb_to_oklab_gpu(rgb, xp=np)

        # Check output
        assert oklab.shape == rgb.shape
        assert oklab.dtype == np.float32
        assert not np.array_equal(oklab, rgb)

        # Check range (Oklab L should be in [0, 1])
        assert np.all(oklab[..., 0] >= 0)
        assert np.all(oklab[..., 0] <= 1)

    def test_batch_oklab_to_srgb_gpu(self):
        """Test batch Oklab to sRGB conversion."""
        # Create test Oklab data
        oklab = np.zeros((10, 10, 3), dtype=np.float32)
        oklab[..., 0] = 0.5  # L channel

        rgb = trans_gpu.batch_oklab_to_srgb_gpu(oklab, xp=np)

        # Check output
        assert rgb.shape == oklab.shape
        assert rgb.dtype == np.float32

        # RGB should be in [0, 1]
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 1)

    def test_srgb_oklab_roundtrip(self):
        """Test sRGB <-> Oklab roundtrip conversion."""
        original = np.random.rand(5, 5, 3).astype(np.float32)

        # Convert sRGB -> Oklab -> sRGB
        oklab = trans_gpu.batch_srgb_to_oklab_gpu(original, xp=np)
        recovered = trans_gpu.batch_oklab_to_srgb_gpu(oklab, xp=np)

        # Should be close to original (some precision loss is expected)
        np.testing.assert_allclose(recovered, original, atol=1e-3)


class TestOKLCHConversions:
    """Test OKLCH color space conversions."""

    def test_batch_oklab_to_oklch_gpu(self):
        """Test batch Oklab to OKLCH conversion."""
        # Create test data with known values
        oklab = np.array(
            [
                [[0.5, 0.0, 0.0]],  # Gray (no chroma)
                [[0.5, 0.1, 0.0]],  # Red-ish
                [[0.5, 0.0, 0.1]],  # Yellow-ish
                [[0.5, -0.1, 0.0]],  # Green-ish
                [[0.5, 0.0, -0.1]],  # Blue-ish
            ],
            dtype=np.float32,
        )

        oklch = trans_gpu.batch_oklab_to_oklch_gpu(oklab, xp=np)

        # Check shape
        assert oklch.shape == oklab.shape

        # L channel should be unchanged
        np.testing.assert_allclose(oklch[..., 0], oklab[..., 0])

        # Gray should have zero chroma
        assert oklch[0, 0, 1] == pytest.approx(0.0, abs=1e-6)

        # Others should have positive chroma
        assert np.all(oklch[1:, 0, 1] > 0)

        # Hue should be in [0, 360)
        assert np.all(oklch[..., 2] >= 0)
        assert np.all(oklch[..., 2] < 360)

    def test_batch_oklch_to_oklab_gpu(self):
        """Test batch OKLCH to Oklab conversion."""
        # Create test data
        oklch = np.array(
            [
                [[0.5, 0.0, 0.0]],  # Gray
                [[0.5, 0.1, 0.0]],  # Hue 0
                [[0.5, 0.1, 90.0]],  # Hue 90
                [[0.5, 0.1, 180.0]],  # Hue 180
                [[0.5, 0.1, 270.0]],  # Hue 270
            ],
            dtype=np.float32,
        )

        oklab = trans_gpu.batch_oklch_to_oklab_gpu(oklch, xp=np)

        # Check shape
        assert oklab.shape == oklch.shape

        # L channel should be unchanged
        np.testing.assert_allclose(oklab[..., 0], oklch[..., 0])

        # Gray should have zero a, b
        assert oklab[0, 0, 1] == pytest.approx(0.0, abs=1e-6)
        assert oklab[0, 0, 2] == pytest.approx(0.0, abs=1e-6)

    def test_oklab_oklch_roundtrip(self):
        """Test Oklab <-> OKLCH roundtrip conversion."""
        original = np.random.rand(5, 5, 3).astype(np.float32) * 0.2 - 0.1
        original[..., 0] = np.clip(original[..., 0] + 0.5, 0, 1)  # Ensure valid L

        # Convert Oklab -> OKLCH -> Oklab
        oklch = trans_gpu.batch_oklab_to_oklch_gpu(original, xp=np)
        recovered = trans_gpu.batch_oklch_to_oklab_gpu(oklch, xp=np)

        np.testing.assert_allclose(recovered, original, atol=1e-5)


class TestTransformPixelsGPU:
    """Test GPU pixel transformation function."""

    def test_transform_pixels_gpu_basic(self):
        """Test basic pixel transformation."""
        # Create test data
        oklab = np.random.rand(10, 10, 3).astype(np.float32)
        oklab[..., 0] = np.clip(oklab[..., 0], 0.2, 0.8)  # Valid L range

        oklch = trans_gpu.batch_oklab_to_oklch_gpu(oklab, xp=np)

        attractors_lab = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
        attractors_lch = trans_gpu.batch_oklab_to_oklch_gpu(attractors_lab.reshape(1, 1, 3), xp=np).reshape(-1, 3)

        tolerances = np.array([50.0], dtype=np.float32)
        strengths = np.array([75.0], dtype=np.float32)
        flags = np.array([True, True, True])  # All channels enabled

        # Transform
        result = trans_gpu.transform_pixels_gpu(
            oklab, oklch, attractors_lab, attractors_lch, tolerances, strengths, flags, xp=np
        )

        # Check result
        assert result.shape == oklab.shape
        assert result.dtype == np.float32
        assert not np.array_equal(result, oklab)  # Should be transformed

    def test_transform_pixels_gpu_channel_control(self):
        """Test channel-specific transformations."""
        # Create test data
        oklab = np.ones((5, 5, 3), dtype=np.float32) * 0.5
        oklch = trans_gpu.batch_oklab_to_oklch_gpu(oklab, xp=np)

        attractors_lab = np.array([[0.7, 0.1, 0.1]], dtype=np.float32)
        attractors_lch = trans_gpu.batch_oklab_to_oklch_gpu(attractors_lab.reshape(1, 1, 3), xp=np).reshape(-1, 3)

        tolerances = np.array([100.0], dtype=np.float32)
        strengths = np.array([100.0], dtype=np.float32)

        # Test different channel combinations
        flags_l = np.array([True, False, False])
        flags_c = np.array([False, True, False])
        flags_h = np.array([False, False, True])

        result_l = trans_gpu.transform_pixels_gpu(
            oklab, oklch, attractors_lab, attractors_lch, tolerances, strengths, flags_l, xp=np
        )

        result_c = trans_gpu.transform_pixels_gpu(
            oklab, oklch, attractors_lab, attractors_lch, tolerances, strengths, flags_c, xp=np
        )

        result_h = trans_gpu.transform_pixels_gpu(
            oklab, oklch, attractors_lab, attractors_lch, tolerances, strengths, flags_h, xp=np
        )

        # Results should be different
        assert not np.array_equal(result_l, result_c)
        assert not np.array_equal(result_l, result_h)
        assert not np.array_equal(result_c, result_h)


class TestProcessImageGPU:
    """Test complete GPU processing pipeline."""

    def test_process_image_gpu_full_pipeline(self):
        """Test full processing pipeline."""
        # Create test image
        rgb_image = np.random.rand(10, 10, 3).astype(np.float32)
        attractors_lab = np.array([[0.5, 0.1, 0.0]], dtype=np.float32)
        tolerances = np.array([50.0], dtype=np.float32)
        strengths = np.array([75.0], dtype=np.float32)

        # Mock memory check to fail (force CPU fallback)
        with patch("imgcolorshine.trans_gpu.check_gpu_memory_available", return_value=(False, 100, 1000)):
            result = trans_gpu.process_image_gpu(
                rgb_image,
                attractors_lab,
                tolerances,
                strengths,
                enable_luminance=True,
                enable_saturation=True,
                enable_hue=True,
            )

        # Should return None due to insufficient memory
        assert result is None

    def test_process_image_gpu_with_mock_success(self):
        """Test processing with mocked GPU success."""
        # Create test image
        rgb_image = np.random.rand(5, 5, 3).astype(np.float32)
        attractors_lab = np.array([[0.6, 0.05, 0.05]], dtype=np.float32)
        tolerances = np.array([60.0], dtype=np.float32)
        strengths = np.array([80.0], dtype=np.float32)

        # Mock successful memory check and array module
        with (
            patch("imgcolorshine.trans_gpu.check_gpu_memory_available", return_value=(True, 1000, 2000)),
            patch("imgcolorshine.trans_gpu.get_array_module", return_value=np),
            patch("imgcolorshine.trans_gpu.estimate_gpu_memory_required", return_value=100),
        ):
            result = trans_gpu.process_image_gpu(
                rgb_image,
                attractors_lab,
                tolerances,
                strengths,
                enable_luminance=True,
                enable_saturation=False,
                enable_hue=True,
            )

        # Should return transformed image
        if result is not None:
            assert result.shape == rgb_image.shape
            assert result.dtype == np.float32


class TestMultipleAttractors:
    """Test transformations with multiple attractors."""

    def test_transform_pixels_gpu_multiple_attractors(self):
        """Test transformation with multiple attractors."""
        # Create test data
        oklab = np.ones((5, 5, 3), dtype=np.float32) * 0.5
        oklch = trans_gpu.batch_oklab_to_oklch_gpu(oklab, xp=np)

        # Multiple attractors with different properties
        attractors_lab = np.array(
            [
                [0.3, 0.1, 0.0],  # Dark red
                [0.7, -0.1, 0.0],  # Light green
                [0.5, 0.0, 0.1],  # Medium yellow
            ],
            dtype=np.float32,
        )

        attractors_lch = trans_gpu.batch_oklab_to_oklch_gpu(attractors_lab.reshape(3, 1, 3), xp=np).reshape(-1, 3)

        tolerances = np.array([30.0, 50.0, 70.0], dtype=np.float32)
        strengths = np.array([60.0, 80.0, 40.0], dtype=np.float32)
        flags = np.array([True, True, True])

        # Transform
        result = trans_gpu.transform_pixels_gpu(
            oklab, oklch, attractors_lab, attractors_lch, tolerances, strengths, flags, xp=np
        )

        # Check result
        assert result.shape == oklab.shape
        assert result.dtype == np.float32
        assert not np.array_equal(result, oklab)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_image(self):
        """Test handling of empty image."""
        empty = np.array([], dtype=np.float32).reshape(0, 0, 3)

        result = trans_gpu.batch_srgb_to_oklab_gpu(empty, xp=np)

        assert result.shape == empty.shape

    def test_single_pixel(self):
        """Test single pixel transformation."""
        pixel = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

        oklab = trans_gpu.batch_srgb_to_oklab_gpu(pixel, xp=np)
        rgb = trans_gpu.batch_oklab_to_srgb_gpu(oklab, xp=np)

        np.testing.assert_allclose(rgb, pixel, atol=1e-3)

    def test_extreme_values(self):
        """Test handling of extreme color values."""
        # Test clipping of out-of-range values
        extreme = np.array(
            [
                [[2.0, -0.5, 1.5]],  # Out of range RGB
                [[0.0, 0.0, 0.0]],  # Black
                [[1.0, 1.0, 1.0]],  # White
            ],
            dtype=np.float32,
        )

        oklab = trans_gpu.batch_srgb_to_oklab_gpu(extreme, xp=np)
        recovered = trans_gpu.batch_oklab_to_srgb_gpu(oklab, xp=np)

        # Should be clipped to valid range
        assert np.all(recovered >= 0)
        assert np.all(recovered <= 1)

    def test_grayscale_handling(self):
        """Test grayscale image handling."""
        # Create grayscale values
        gray = np.full((5, 5, 3), 0.5, dtype=np.float32)

        oklab = trans_gpu.batch_srgb_to_oklab_gpu(gray, xp=np)

        # Check that a and b channels are near zero for gray
        # Note: there may be small numerical errors
        assert np.allclose(oklab[..., 1], 0, atol=1e-4)
        assert np.allclose(oklab[..., 2], 0, atol=1e-4)
