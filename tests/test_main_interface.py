#!/usr/bin/env python
"""
Test suite for main interface functions.

Tests the primary user-facing functions in colorshine module.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imgcolorshine.colorshine import (
    generate_output_path,
    parse_attractor,
    process_image,
)


class TestMainInterface:
    """Test main interface functionality."""

    def test_parse_attractor_valid(self):
        """Test parsing valid attractor strings."""
        # Basic color with parameters
        color, tolerance, strength = parse_attractor("red;50;75")
        assert color == "red"
        assert tolerance == 50.0
        assert strength == 75.0

        # Hex color
        color, tolerance, strength = parse_attractor("#ff0000;30;60")
        assert color == "#ff0000"
        assert tolerance == 30.0
        assert strength == 60.0

        # OKLCH color
        color, tolerance, strength = parse_attractor("oklch(70% 0.2 120);40;80")
        assert color == "oklch(70% 0.2 120)"
        assert tolerance == 40.0
        assert strength == 80.0

    def test_parse_attractor_invalid(self):
        """Test parsing invalid attractor strings."""
        # Missing parameters
        with pytest.raises(ValueError):
            parse_attractor("red;50")  # Missing strength

        # Invalid format
        with pytest.raises(ValueError):
            parse_attractor("invalid_format")

        # Non-numeric values
        with pytest.raises(ValueError):
            parse_attractor("red;fifty;seventy-five")

    def test_generate_output_path(self):
        """Test automatic output path generation."""
        # Test with simple filename
        input_path = Path("image.png")
        output_path = generate_output_path(input_path)
        assert str(output_path) == "image_colorshine.png"

        # Test with path
        input_path = Path("/path/to/image.jpg")
        output_path = generate_output_path(input_path)
        assert str(output_path) == "/path/to/image_colorshine.jpg"

    def test_process_image_basic(self):
        """Test basic image processing."""
        with (
            patch("imgcolorshine.colorshine.ImageProcessor") as mock_processor_class,
            patch("imgcolorshine.colorshine.OKLCHEngine") as mock_engine_class,
            patch("imgcolorshine.colorshine.ColorTransformer") as mock_transformer_class,
            patch("imgcolorshine.colorshine.logger"),
        ):
            # Setup mocks
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            test_image = np.random.rand(100, 100, 3).astype(np.float32)
            mock_processor.load_image.return_value = test_image

            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            mock_transformer = Mock()
            mock_transformer_class.return_value = mock_transformer
            mock_transformer.process_with_attractors.return_value = test_image

            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                output_path = tmp.name

            # Process image
            process_image("test.png", ("red;50;75",), output_image=output_path)

            # Verify calls
            mock_processor.load_image.assert_called_once_with("test.png")
            mock_processor.save_image.assert_called_once()

            # Cleanup
            Path(output_path).unlink(missing_ok=True)

    def test_process_image_multiple_attractors(self):
        """Test processing with multiple attractors."""
        with (
            patch("imgcolorshine.colorshine.ImageProcessor") as mock_processor_class,
            patch("imgcolorshine.colorshine.OKLCHEngine") as mock_engine_class,
            patch("imgcolorshine.colorshine.ColorTransformer") as mock_transformer_class,
            patch("imgcolorshine.colorshine.logger"),
        ):
            # Setup mocks
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            test_image = np.random.rand(100, 100, 3).astype(np.float32)
            mock_processor.load_image.return_value = test_image

            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            mock_transformer = Mock()
            mock_transformer_class.return_value = mock_transformer
            mock_transformer.process_with_attractors.return_value = test_image

            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                output_path = tmp.name

            # Process with multiple attractors
            attractors = ("red;50;75", "blue;30;60", "green;40;50")
            process_image("test.png", attractors, output_image=output_path)

            # Verify that OKLCHEngine was called to create 3 attractors
            assert mock_engine.create_attractor.call_count == 3

            # Cleanup
            Path(output_path).unlink(missing_ok=True)

    def test_process_image_channel_control(self):
        """Test channel-specific transformations."""
        with (
            patch("imgcolorshine.colorshine.ImageProcessor") as mock_processor_class,
            patch("imgcolorshine.colorshine.OKLCHEngine") as mock_engine_class,
            patch("imgcolorshine.colorshine.ColorTransformer") as mock_transformer_class,
            patch("imgcolorshine.colorshine.logger"),
        ):
            # Setup mocks
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            test_image = np.random.rand(100, 100, 3).astype(np.float32)
            mock_processor.load_image.return_value = test_image

            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            mock_transformer = Mock()
            mock_transformer_class.return_value = mock_transformer
            mock_transformer.process_with_attractors.return_value = test_image

            # Test with specific channel settings
            process_image(
                "test.png",
                ("red;50;75",),
                luminance=True,
                saturation=False,
                chroma=True,
            )

            # Verify ColorTransformer was created with correct settings
            mock_transformer_class.assert_called_once()
            call_kwargs = mock_transformer_class.call_args[1]
            assert call_kwargs["transform_lightness"] is True
            assert call_kwargs["transform_chroma"] is True

    def test_process_image_custom_output(self):
        """Test custom output path specification."""
        with (
            patch("imgcolorshine.colorshine.ImageProcessor") as mock_processor_class,
            patch("imgcolorshine.colorshine.OKLCHEngine") as mock_engine_class,
            patch("imgcolorshine.colorshine.ColorTransformer") as mock_transformer_class,
            patch("imgcolorshine.colorshine.logger"),
        ):
            # Setup mocks
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            test_image = np.random.rand(100, 100, 3).astype(np.float32)
            mock_processor.load_image.return_value = test_image

            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            mock_transformer = Mock()
            mock_transformer_class.return_value = mock_transformer
            mock_transformer.process_with_attractors.return_value = test_image

            # Process with custom output
            custom_output = "/tmp/custom_output.png"
            process_image("test.png", ("red;50;75",), output_image=custom_output)

            # Verify save was called with custom path
            save_call = mock_processor.save_image.call_args
            assert str(save_call[0][1]) == custom_output
