#!/usr/bin/env python
"""
Simple test suite for CLI interface.

Tests the CLI class and its methods.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imgcolorshine.cli import ImgColorShineCLI


class TestCLISimple:
    """Test CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = ImgColorShineCLI()
    
    @patch("imgcolorshine.cli.process_image")
    def test_shine_basic(self, mock_process):
        """Test basic shine command."""
        self.cli.shine("input.jpg", "red;50;75")
        
        # Verify process_image was called
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        
        assert call_kwargs["input_image"] == "input.jpg"
        assert "red;50;75" in call_kwargs["attractors"]
    
    @patch("imgcolorshine.cli.process_image")
    def test_shine_multiple_attractors(self, mock_process):
        """Test shine with multiple attractors."""
        self.cli.shine("input.jpg", "red;50;75", "blue;30;60", "#00ff00;40;80")
        
        call_kwargs = mock_process.call_args[1]
        attractors = call_kwargs["attractors"]
        
        assert len(attractors) == 3
        assert "red;50;75" in attractors
        assert "blue;30;60" in attractors
        assert "#00ff00;40;80" in attractors
    
    @patch("imgcolorshine.cli.process_image")
    def test_shine_with_options(self, mock_process):
        """Test shine with various options."""
        self.cli.shine(
            "input.jpg",
            "red;50;75",
            output_image="output.png",
            luminance=True,
            saturation=False,
            chroma=True,
            verbose=True,
            tile_size=2048,
            gpu=False,
            LUT_size=65,
            fast_hierar=True,
            Fast_spatial=False
        )
        
        call_kwargs = mock_process.call_args[1]
        
        assert call_kwargs["output_image"] == "output.png"
        assert call_kwargs["luminance"] is True
        assert call_kwargs["saturation"] is False
        assert call_kwargs["chroma"] is True
        assert call_kwargs["verbose"] is True
        assert call_kwargs["tile_size"] == 2048
        assert call_kwargs["gpu"] is False
        assert call_kwargs["lut_size"] == 65
        assert call_kwargs["fast_hierar"] is True
        assert call_kwargs["fast_spatial"] is False