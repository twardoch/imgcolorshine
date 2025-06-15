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
        # Check it's a power of 2
        assert (tile_size & (tile_size - 1)) == 0

        # Custom bytes per pixel
        shape = (2048, 2048, 3)
        tile_size = estimate_optimal_tile_size(shape, available_memory_mb=200, bytes_per_pixel=24)
        assert tile_size < 2048  # More memory per pixel means smaller tiles


class TestImageProcessing:
    """Test image processing utilities."""

    def test_process_large_image_basic(self):
        """Test basic tiled processing."""
        # Create test image - use 96x96 to evenly divide by 32
        image = np.ones((96, 96, 3), dtype=np.float32) * 0.5

        # Simple transform function
        def transform(tile):
            return tile * 2.0

        # Process with tiles that evenly divide the image
        result = process_large_image(image, transform, tile_size=48, overlap=16)

        # Verify transformation applied
        expected = np.ones((96, 96, 3), dtype=np.float32) * 1.0
        # Due to overlap handling, allow some tolerance
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_process_large_image_with_overlap(self):
        """Test tiled processing with overlap."""
        # Create simple uniform image for easier testing
        image = np.ones((64, 64, 3), dtype=np.float32) * 0.7

        # Transform that adds a small value
        def transform(tile):
            return tile + 0.1

        # Process with overlap
        result = process_large_image(image, transform, tile_size=32, overlap=8)

        # Result should have transform applied
        expected = np.ones((64, 64, 3), dtype=np.float32) * 0.8
        # With overlap there might be slight variations at tile boundaries
        np.testing.assert_allclose(result, expected, atol=0.01)

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
        with pytest.raises(ValueError, match="must be 3D"):
            invalid_2d = np.zeros((10, 10))
            validate_image(invalid_2d)

        # Invalid - wrong number of channels
        with pytest.raises(ValueError, match="must have 3 channels"):
            invalid_channels = np.zeros((10, 10, 4))
            validate_image(invalid_channels)

        # Invalid - wrong dtype
        with pytest.raises(ValueError, match="must be float32 or float64"):
            invalid_dtype = np.zeros((10, 10, 3), dtype=np.int32)
            validate_image(invalid_dtype)

        # Invalid - values out of range
        with pytest.raises(ValueError, match="must be in range"):
            invalid_range = np.ones((10, 10, 3), dtype=np.float32) * 2.0
            validate_image(invalid_range)


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
        # Create progress bar
        progress = create_progress_bar(100, "Testing")

        # Should return a SimpleProgress instance
        assert hasattr(progress, "update")
        assert hasattr(progress, "__enter__")
        assert hasattr(progress, "__exit__")

        # Test using as context manager
        with progress as p:
            p.update(10)  # Update by 10
            p.update(40)  # Update by 40
            assert p.current == 50
            assert p.total == 100
