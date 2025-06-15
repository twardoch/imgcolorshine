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

from imgcolorshine.colorshine import parse_attractor, generate_output_path, process_image


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
        assert "image" in str(output_path)
        assert output_path.suffix == ".png"
        
        # Test with path
        input_path = Path("/path/to/image.jpg")
        output_path = generate_output_path(input_path)
        assert "image" in str(output_path)
        assert output_path.suffix == ".jpg"
    
    @patch("imgcolorshine.io.ImageProcessor")
    def test_process_image_basic(self, mock_io):
        """Test basic image processing."""
        # Setup mock
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_processor = Mock()
        mock_processor.load_image.return_value = test_image
        mock_processor.save_image.return_value = None
        mock_io.return_value = mock_processor
        
        # Process image
        output = process_image("test.png", ("red;50;75",))
        
        # Verify
        assert output is not None
        mock_processor.load_image.assert_called_once()
        mock_processor.save_image.assert_called_once()
    
    @patch("imgcolorshine.io.ImageProcessor")
    def test_process_image_multiple_attractors(self, mock_io):
        """Test processing with multiple attractors."""
        # Setup mock
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_processor = Mock()
        mock_processor.load_image.return_value = test_image
        mock_processor.save_image.return_value = None
        mock_io.return_value = mock_processor
        
        # Process with multiple attractors
        attractors = ("red;50;75", "blue;30;60", "#00ff00;40;80")
        output = process_image("test.png", attractors)
        
        assert output is not None
    
    @patch("imgcolorshine.io.ImageProcessor")
    def test_process_image_channel_control(self, mock_io):
        """Test channel-specific processing."""
        # Setup mock
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_processor = Mock()
        mock_processor.load_image.return_value = test_image
        mock_processor.save_image.return_value = None
        mock_io.return_value = mock_processor
        
        # Test with only hue enabled
        output = process_image(
            "test.png",
            ("green;60;90",),
            luminance=False,
            saturation=False,
            hue=True
        )
        
        assert output is not None
    
    @patch("imgcolorshine.io.ImageProcessor")
    def test_process_image_custom_output(self, mock_io):
        """Test custom output path."""
        # Setup mock
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_processor = Mock()
        mock_processor.load_image.return_value = test_image
        mock_processor.save_image.return_value = None
        mock_io.return_value = mock_processor
        
        # Process with custom output
        output = process_image(
            "test.png",
            ("red;50;75",),
            output_image="custom_output.png"
        )
        
        # Verify custom path was used
        save_call = mock_processor.save_image.call_args
        output_path = save_call[0][1]
        assert str(output_path) == "custom_output.png"