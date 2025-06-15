# tests/test_cli.py

from unittest.mock import patch

import numpy as np
import pytest

from imgcolorshine.cli import ImgColorShineCLI


@pytest.fixture
def cli():
    """Create CLI instance for testing."""
    return ImgColorShineCLI()


@pytest.fixture
def test_image_path(tmp_path):
    """Create a temporary test image."""
    # Create a simple 10x10 RGB image
    np.ones((10, 10, 3), dtype=np.uint8) * 128
    img_path = tmp_path / "test.png"

    # Mock the image save
    return str(img_path)


def test_basic_transformation(cli, test_image_path):
    """Test basic CLI transformation command."""
    with patch("imgcolorshine.cli.process_image") as mock_process:
        # Test with single attractor
        cli.shine(
            test_image_path,
            "red;50;75",
            output_image="output.jpg",
            luminance=True,
            saturation=True,
            hue=True,
            verbose=True,
        )

        # Verify process_image was called with correct arguments
        mock_process.assert_called_once_with(
            input_image=test_image_path,
            attractors=("red;50;75",),
            output_image="output.jpg",
            luminance=True,
            saturation=True,
            hue=True,
            verbose=True,
            tile_size=1024,
            gpu=True,
            lut_size=0,
            fast_hierar=False,
            fast_spatial=True,
        )


def test_multiple_attractors(cli, test_image_path):
    """Test CLI with multiple attractors."""
    with patch("imgcolorshine.cli.process_image") as mock_process:
        cli.shine(
            test_image_path,
            "red;50;75",
            "blue;30;60",
            luminance=False,
            saturation=True,
            hue=False,
        )

        mock_process.assert_called_once_with(
            input_image=test_image_path,
            attractors=("red;50;75", "blue;30;60"),
            output_image=None,
            luminance=False,
            saturation=True,
            hue=False,
            verbose=False,
            tile_size=1024,
            gpu=True,
            lut_size=0,
            fast_hierar=False,
            fast_spatial=True,
        )


def test_channel_flags(cli, test_image_path):
    """Test luminance/saturation/hue channel controls."""
    with patch("imgcolorshine.cli.process_image") as mock_process:
        # Test with channel flags
        cli.shine(
            test_image_path,
            "red;50;75",
            luminance=True,
            saturation=True,
            hue=False,
        )

        # Verify channel flags were passed correctly
        mock_process.assert_called_once()
        call_args = mock_process.call_args[1]
        assert call_args["luminance"] is True
        assert call_args["saturation"] is True
        assert call_args["hue"] is False


def test_optimization_flags(cli, test_image_path):
    """Test GPU, LUT, hierarchical flags."""
    with patch("imgcolorshine.cli.process_image") as mock_process:
        # Test with optimization flags
        cli.shine(
            test_image_path,
            "red;50;75",
            gpu=False,
            LUT_size=65,
            fast_hierar=True,
            Fast_spatial=False,
        )

        # Verify optimization flags were passed correctly
        mock_process.assert_called_once()
        call_args = mock_process.call_args[1]
        assert call_args["gpu"] is False
        assert call_args["lut_size"] == 65
        assert call_args["fast_hierar"] is True
        assert call_args["fast_spatial"] is False


def test_output_path_specification(cli, test_image_path):
    """Test custom output path specification."""
    with patch("imgcolorshine.cli.process_image") as mock_process:
        output_path = "/tmp/output.png"

        cli.shine(
            test_image_path,
            "red;50;75",
            output_image=output_path,
        )

        # Verify output path was passed correctly
        mock_process.assert_called_once()
        call_args = mock_process.call_args[1]
        assert call_args["output_image"] == output_path


def test_verbose_flag(cli, test_image_path):
    """Test verbose logging flag."""
    with patch("imgcolorshine.cli.process_image") as mock_process:
        cli.shine(
            test_image_path,
            "red;50;75",
            verbose=True,
        )

        # Verify verbose flag was passed correctly
        mock_process.assert_called_once()
        call_args = mock_process.call_args[1]
        assert call_args["verbose"] is True


def test_tile_size_parameter(cli, test_image_path):
    """Test tile size parameter."""
    with patch("imgcolorshine.cli.process_image") as mock_process:
        cli.shine(
            test_image_path,
            "red;50;75",
            tile_size=2048,
        )

        # Verify tile size was passed correctly
        mock_process.assert_called_once()
        call_args = mock_process.call_args[1]
        assert call_args["tile_size"] == 2048
