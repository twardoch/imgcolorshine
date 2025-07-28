# tests/test_colorshine.py

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from imgcolorshine.colorshine import (
    generate_output_path,
    parse_attractor,
    process_image,
    setup_logging,
)


def test_shine_function():
    """Test the main shine() function."""
    # Since process_image is the main function, let's test it with mocks
    with (
        patch("imgcolorshine.colorshine.ImageProcessor") as mock_processor,
        patch("imgcolorshine.colorshine.OKLCHEngine") as mock_engine,
        patch("imgcolorshine.colorshine.ColorTransformer") as mock_transformer,
        patch("imgcolorshine.colorshine.process_with_optimizations") as mock_process_optimized,
        patch("imgcolorshine.colorshine.logger"),
    ):
        # Setup mocks
        mock_img_proc = Mock()
        mock_processor.return_value = mock_img_proc
        test_image = np.ones((10, 10, 3), dtype=np.float32)
        mock_img_proc.load_image.return_value = test_image

        # Mock the engine and transformer
        mock_color_engine = Mock()
        mock_engine.return_value = mock_color_engine
        mock_attractor = Mock()
        mock_color_engine.create_attractor.return_value = mock_attractor

        mock_color_transformer = Mock()
        mock_transformer.return_value = mock_color_transformer

        # Mock the optimization function to return transformed image
        mock_process_optimized.return_value = test_image * 0.9

        # Call process_image
        process_image(
            input_image="test.png",
            attractors=("red;50;75",),
            output_image="output.png",
            luminance=True,
            saturation=True,
            chroma=False,
            verbose=False,
        )

        # Verify the processing pipeline was called
        mock_processor.assert_called_once()
        mock_img_proc.load_image.assert_called_once_with(Path("test.png"))
        mock_img_proc.save_image.assert_called_once()
        mock_color_engine.create_attractor.assert_called_once_with("red", 50.0, 75.0)


def test_attractor_parsing():
    """Test attractor string parsing."""
    # Test valid attractor parsing
    color, tolerance, strength = parse_attractor("red;50;75")
    assert color == "red"
    assert tolerance == 50.0
    assert strength == 75.0

    # Test with spaces
    color, tolerance, strength = parse_attractor(" blue ; 30 ; 60 ")
    assert color == "blue"
    assert tolerance == 30.0
    assert strength == 60.0

    # Test with OKLCH color
    color, tolerance, strength = parse_attractor("oklch(70% 0.2 120);40;80")
    assert color == "oklch(70% 0.2 120)"
    assert tolerance == 40.0
    assert strength == 80.0

    # Test invalid format
    with pytest.raises(ValueError, match="Invalid attractor format"):
        parse_attractor("red;50")

    with pytest.raises(ValueError, match="Invalid attractor format"):
        parse_attractor("red")

    # Test invalid tolerance
    with pytest.raises(ValueError, match="Tolerance must be 0-100"):
        parse_attractor("red;-10;50")

    with pytest.raises(ValueError, match="Tolerance must be 0-100"):
        parse_attractor("red;150;50")

    # Test invalid strength
    with pytest.raises(ValueError, match="Strength must be 0-100"):
        parse_attractor("red;50;-10")

    with pytest.raises(ValueError, match="Strength must be 0-100"):
        parse_attractor("red;50;150")

    # Test non-numeric values
    with pytest.raises(ValueError):
        parse_attractor("red;abc;50")


def test_output_path_generation():
    """Test automatic output path generation."""
    # Test basic path generation
    input_path = Path("/tmp/test.png")
    output_path = generate_output_path(input_path)
    assert output_path == Path("/tmp/test_colorshine.png")

    # Test with different extension
    input_path = Path("/home/user/image.jpg")
    output_path = generate_output_path(input_path)
    assert output_path == Path("/home/user/image_colorshine.jpg")

    # Test with complex filename
    input_path = Path("./my.complex.image.name.png")
    output_path = generate_output_path(input_path)
    assert output_path == Path("./my.complex.image.name_colorshine.png")

    # Test with no extension
    input_path = Path("/tmp/image")
    output_path = generate_output_path(input_path)
    assert output_path == Path("/tmp/image_colorshine")


def test_setup_logging():
    """Test logging configuration."""
    with patch("imgcolorshine.colorshine.logger") as mock_logger:
        # Test verbose mode
        setup_logging(verbose=True)
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        add_call = mock_logger.add.call_args
        assert add_call[1]["level"] == "DEBUG"

        # Reset mock
        mock_logger.reset_mock()

        # Test non-verbose mode
        setup_logging(verbose=False)
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        add_call = mock_logger.add.call_args
        assert add_call[1]["level"] == "INFO"


def test_process_image_channel_defaults():
    """Test that process_image uses correct channel defaults."""
    with (
        patch("imgcolorshine.colorshine.ImageProcessor") as mock_processor,
        patch("imgcolorshine.colorshine.OKLCHEngine") as mock_engine,
        patch("imgcolorshine.colorshine.ColorTransformer") as mock_transformer,
        patch("imgcolorshine.colorshine.process_with_optimizations") as mock_process_optimized,
        patch("imgcolorshine.colorshine.logger"),
    ):
        # Setup mocks
        mock_img_proc = Mock()
        mock_processor.return_value = mock_img_proc
        test_image = np.ones((10, 10, 3), dtype=np.float32)
        mock_img_proc.load_image.return_value = test_image

        mock_color_engine = Mock()
        mock_engine.return_value = mock_color_engine
        mock_attractor = Mock()
        mock_color_engine.create_attractor.return_value = mock_attractor

        mock_color_transformer = Mock()
        mock_transformer.return_value = mock_color_transformer

        # Mock the optimization function
        mock_process_optimized.return_value = test_image

        # Call with defaults
        process_image(
            input_image="test.png",
            attractors=("red;50;75",),
        )

        # Check that transformer was created with engine only
        mock_transformer.assert_called_once_with(mock_color_engine)

        # Check that process_with_optimizations was called with correct channel defaults
        mock_process_optimized.assert_called_once()
        call_args = mock_process_optimized.call_args[0]
        # Arguments: image, attractor_objects, luminance, saturation, chroma, fast_hierar, fast_spatial, transformer, engine
        assert call_args[2] is True  # luminance
        assert call_args[3] is True  # saturation
        assert call_args[4] is True  # chroma
