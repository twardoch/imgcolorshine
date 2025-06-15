#!/usr/bin/env python
"""
Test suite for image I/O operations.

Tests image loading, saving, format support, and memory management.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imgcolorshine.io import ImageProcessor


class TestImageIO:
    """Test image I/O functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_load_save_cycle_preserves_data(self):
        """Test that loading and saving preserves image data."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            # Save test image
            self.processor.save_image(self.test_image, tmp_path)

            # Load it back
            loaded_image = self.processor.load_image(tmp_path)

            # Verify data is preserved
            np.testing.assert_array_equal(self.test_image, loaded_image)

            # Cleanup
            tmp_path.unlink()

    def test_png_format_support(self):
        """Test PNG format loading and saving."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            self.processor.save_image(self.test_image, tmp_path)
            loaded = self.processor.load_image(tmp_path)

            assert loaded.shape == self.test_image.shape
            assert loaded.dtype == self.test_image.dtype

            tmp_path.unlink()

    @pytest.mark.skipif(not hasattr(ImageProcessor, "HAS_OPENCV"), reason="OpenCV not available")
    def test_jpeg_format_support(self):
        """Test JPEG format loading and saving."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            self.processor.save_image(self.test_image, tmp_path)
            loaded = self.processor.load_image(tmp_path)

            # JPEG is lossy, so we can't expect exact equality
            assert loaded.shape == self.test_image.shape
            assert loaded.dtype == self.test_image.dtype

            tmp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_image("nonexistent_file.png")

    def test_save_to_invalid_path(self):
        """Test saving to an invalid path."""
        invalid_path = Path("/invalid/path/image.png")

        with pytest.raises(OSError):
            self.processor.save_image(self.test_image, invalid_path)

    def test_memory_usage_estimation(self):
        """Test memory usage estimation for images."""
        # Test small image
        small_image = np.zeros((100, 100, 3), dtype=np.uint8)
        memory = self.processor.estimate_memory_usage(small_image)

        # Should be approximately 100*100*3 bytes plus overhead
        assert memory > 30000  # At least the raw size
        assert memory < 1000000  # Less than 1MB for small image

        # Test large image
        large_shape = (2048, 2048, 3)
        memory = self.processor.estimate_memory_usage_from_shape(large_shape)

        # Should be approximately 2048*2048*3 bytes plus overhead
        assert memory > 12_000_000  # At least 12MB
        assert memory < 100_000_000  # Less than 100MB

    def test_should_tile_large_image(self):
        """Test decision logic for tiling large images."""
        # Small image should not need tiling
        small_image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not self.processor.should_tile(small_image)

        # Mock available memory
        with patch("imgcolorshine.io.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value.available = 100 * 1024 * 1024  # 100MB

            # Large image should need tiling if it uses too much memory
            large_shape = (4096, 4096, 3)
            assert self.processor.should_tile_from_shape(large_shape)

    def test_normalize_image_data(self):
        """Test image data normalization."""
        # Test uint8 to float32 normalization
        uint8_image = np.array([[[0, 128, 255]]], dtype=np.uint8)
        normalized = self.processor.normalize_to_float32(uint8_image)

        assert normalized.dtype == np.float32
        np.testing.assert_allclose(normalized[0, 0], [0.0, 128 / 255, 1.0])

        # Test float32 passthrough
        float_image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        normalized = self.processor.normalize_to_float32(float_image)

        assert normalized is float_image  # Should be the same object

    def test_denormalize_image_data(self):
        """Test image data denormalization."""
        # Test float32 to uint8 conversion
        float_image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        uint8_image = self.processor.denormalize_to_uint8(float_image)

        assert uint8_image.dtype == np.uint8
        np.testing.assert_array_equal(uint8_image[0, 0], [0, 128, 255])

    def test_handle_grayscale_image(self):
        """Test handling of grayscale images."""
        # Create grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            # Save as grayscale
            import cv2

            cv2.imwrite(str(tmp_path), gray_image)

            # Load and verify it's converted to RGB
            loaded = self.processor.load_image(tmp_path)

            assert loaded.shape == (100, 100, 3)
            assert loaded.dtype == np.uint8

            tmp_path.unlink()

    def test_handle_rgba_image(self):
        """Test handling of RGBA images."""
        # Create RGBA image
        rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            # Save RGBA
            import cv2

            cv2.imwrite(str(tmp_path), rgba_image)

            # Load and verify alpha is handled
            loaded = self.processor.load_image(tmp_path)

            # Should be converted to RGB (3 channels)
            assert loaded.shape == (100, 100, 3)

            tmp_path.unlink()

    def test_image_metadata_preservation(self):
        """Test that basic image metadata is preserved."""
        # This is a placeholder - actual implementation would depend
        # on the specific metadata handling in the ImageProcessor
