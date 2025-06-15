#!/usr/bin/env python
"""
Additional tests to improve coverage for utils.py module.

Tests utility functions for memory management, image processing,
validation, and batch operations.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from imgcolorshine.utils import (
    batch_process_images,
    clamp_to_gamut,
    create_progress_bar,
    estimate_optimal_tile_size,
    process_large_image,
    validate_image,
)


class TestMemoryManagement:
    """Test memory management functions."""

    def test_estimate_optimal_tile_size(self):
        """Test optimal tile size estimation."""
        # Small image - should suggest full image
        small_shape = (256, 256, 3)
        tile_size = estimate_optimal_tile_size(small_shape, available_memory_mb=1024)
        assert tile_size >= 256  # Should process whole image

        # Large image with limited memory
        large_shape = (4096, 4096, 3)
        tile_size = estimate_optimal_tile_size(large_shape, available_memory_mb=100)
        assert tile_size < 4096  # Should tile
        assert tile_size > 0
        assert tile_size % 64 == 0  # Should be multiple of 64

        # Custom bytes per pixel
        shape = (2048, 2048, 3)
        tile_size = estimate_optimal_tile_size(shape, available_memory_mb=200, bytes_per_pixel=24)
        assert tile_size < 2048  # More memory per pixel means smaller tiles


class TestImageProcessing:
    """Test image processing utilities."""

    def test_process_large_image_basic(self):
        """Test basic tiled processing."""
        # Create test image
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        # Simple transform function
        def transform(tile):
            return tile * 2.0

        # Process with small tiles
        result = process_large_image(image, transform, tile_size=32)

        # Verify transformation applied
        expected = np.ones((100, 100, 3), dtype=np.float32) * 1.0
        np.testing.assert_allclose(result, expected)

    def test_process_large_image_with_overlap(self):
        """Test tiled processing with overlap."""
        # Create gradient image to test overlap handling
        image = np.zeros((100, 100, 1), dtype=np.float32)
        for i in range(100):
            image[i, :] = i / 100.0

        # Identity transform
        def transform(tile):
            return tile

        # Process with overlap
        result = process_large_image(image, transform, tile_size=32, overlap=8)

        # Result should be identical to input
        np.testing.assert_allclose(result, image)

    def test_process_large_image_progress_callback(self):
        """Test progress callback functionality."""
        image = np.ones((64, 64, 3), dtype=np.float32)

        # Track progress calls
        progress_values = []

        def progress_callback(progress):
            progress_values.append(progress)

        def transform(tile):
            return tile

        # Process with progress tracking
        process_large_image(image, transform, tile_size=32, progress_callback=progress_callback)

        # Should have received progress updates
        assert len(progress_values) > 0
        assert progress_values[-1] == 1.0  # Final progress
        # Progress should be monotonically increasing
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]


class TestValidation:
    """Test validation functions."""

    def test_validate_image(self):
        """Test image validation."""
        # Valid image
        valid_image = np.random.rand(10, 10, 3).astype(np.float32)
        validate_image(valid_image)  # Should not raise

        # Invalid - wrong number of dimensions
        with pytest.raises(ValueError, match="dimensions"):
            invalid_2d = np.zeros((10, 10))
            validate_image(invalid_2d)

        # Invalid - wrong number of channels
        with pytest.raises(ValueError, match="channels"):
            invalid_channels = np.zeros((10, 10, 4))
            validate_image(invalid_channels)

        # Invalid - wrong dtype
        with pytest.raises(ValueError, match="dtype"):
            invalid_dtype = np.zeros((10, 10, 3), dtype=np.int32)
            validate_image(invalid_dtype)


class TestColorOperations:
    """Test color-related utilities."""

    def test_clamp_to_gamut(self):
        """Test gamut clamping."""
        # In-gamut colors should be unchanged
        in_gamut = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        clamped = clamp_to_gamut(in_gamut)
        np.testing.assert_array_equal(clamped, in_gamut)

        # Out-of-gamut colors should be clamped
        out_of_gamut = np.array(
            [
                [-0.1, 0.5, 0.5],  # Negative
                [0.5, 1.2, 0.5],  # Over 1
                [1.5, -0.5, 2.0],  # Multiple issues
            ],
            dtype=np.float32,
        )

        clamped = clamp_to_gamut(out_of_gamut)

        # Check all values in [0, 1]
        assert np.all(clamped >= 0.0)
        assert np.all(clamped <= 1.0)

        # Check specific clamps
        assert clamped[0, 0] == 0.0  # Negative clamped to 0
        assert clamped[1, 1] == 1.0  # Over 1 clamped to 1
        assert clamped[2, 0] == 1.0
        assert clamped[2, 1] == 0.0
        assert clamped[2, 2] == 1.0


class TestBatchOperations:
    """Test batch processing utilities."""

    def test_batch_process_images(self):
        """Test batch image processing."""
        # Create temporary test images
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test images
            image_paths = []
            for i in range(3):
                np.ones((10, 10, 3), dtype=np.float32) * (i + 1) * 0.2
                path = tmpdir / f"test_{i}.png"
                # Save using OpenCV or PIL mock
                image_paths.append(str(path))

            # Create output directory
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            # Simple transform
            def transform(image):
                return image * 0.5

            # Mock the actual file I/O
            with patch("imgcolorshine.io.ImageProcessor") as mock_processor:
                mock_io = Mock()
                mock_processor.return_value = mock_io

                # Mock load to return different images
                mock_io.load_image.side_effect = [
                    np.ones((10, 10, 3)) * 0.2,
                    np.ones((10, 10, 3)) * 0.4,
                    np.ones((10, 10, 3)) * 0.6,
                ]

                # Process batch
                batch_process_images(image_paths, str(output_dir), transform)

                # Check that images were loaded and saved
                assert mock_io.load_image.call_count == 3
                assert mock_io.save_image.call_count == 3


class TestProgressBar:
    """Test progress bar functionality."""

    def test_create_progress_bar(self):
        """Test progress bar creation."""
        # Mock Rich progress bar
        with patch("imgcolorshine.utils.Progress") as mock_progress:
            mock_instance = Mock()
            mock_progress.return_value.__enter__.return_value = mock_instance

            # Create progress bar
            progress = create_progress_bar(100, "Testing")

            # Should return a callable
            assert callable(progress)

            # Test updating progress
            progress(0.5)  # 50%
            progress(1.0)  # 100%

            # If the function returns None for the progress callable,
            # test that it handles gracefully
            none_progress = create_progress_bar(0, "Empty")
            if none_progress is not None:
                none_progress(0.5)  # Should not crash
