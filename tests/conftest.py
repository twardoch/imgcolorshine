# this_file: tests/conftest.py

"""Shared test fixtures and utilities for imgcolorshine tests."""

from pathlib import Path

import numpy as np
import pytest
from coloraide import Color

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "testdata"


@pytest.fixture
def test_image_path():
    """Provide path to test image."""
    return TEST_DATA_DIR / "louis.jpg"


@pytest.fixture
def sample_rgb_array():
    """Create a small sample RGB array for testing."""
    # 4x4 RGB image with various colors
    return np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],  # Red, Green, Blue, Yellow
            [[255, 0, 255], [0, 255, 255], [128, 128, 128], [255, 255, 255]],  # Magenta, Cyan, Gray, White
            [[0, 0, 0], [64, 64, 64], [192, 192, 192], [128, 0, 0]],  # Black, Dark gray, Light gray, Dark red
            [
                [0, 128, 0],
                [0, 0, 128],
                [128, 128, 0],
                [128, 0, 128],
            ],  # Dark green, Dark blue, Dark yellow, Dark magenta
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def sample_oklch_array():
    """Create a sample OKLCH array for testing."""
    # 2x2 OKLCH values
    return np.array([[[0.7, 0.2, 30], [0.5, 0.1, 120]], [[0.3, 0.15, 240], [0.9, 0.05, 0]]], dtype=np.float32)


@pytest.fixture
def sample_colors():
    """Provide sample Color objects for testing."""
    return {
        "red": Color("red"),
        "green": Color("green"),
        "blue": Color("blue"),
        "white": Color("white"),
        "black": Color("black"),
        "gray": Color("gray"),
        "oklch_bright": Color("oklch(80% 0.2 60)"),
        "oklch_muted": Color("oklch(50% 0.1 180)"),
    }


@pytest.fixture
def attractor_params():
    """Sample attractor parameters for testing."""
    return [("red", 50, 75), ("oklch(70% 0.2 120)", 30, 60), ("#0066cc", 40, 80)]


def assert_image_shape(image: np.ndarray, expected_shape: tuple[int, ...]):
    """Assert that an image has the expected shape."""
    assert image.shape == expected_shape, f"Expected shape {expected_shape}, got {image.shape}"


def assert_image_dtype(image: np.ndarray, expected_dtype: np.dtype):
    """Assert that an image has the expected data type."""
    assert image.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {image.dtype}"


def assert_color_close(color1: Color, color2: Color, tolerance: float = 0.01):
    """Assert that two colors are close in OKLCH space."""
    c1_oklch = color1.convert("oklch")
    c2_oklch = color2.convert("oklch")

    diff_l = abs(c1_oklch["lightness"] - c2_oklch["lightness"])
    diff_c = abs(c1_oklch["chroma"] - c2_oklch["chroma"])
    # Handle hue wraparound
    diff_h = abs(c1_oklch["hue"] - c2_oklch["hue"])
    if diff_h > 180:
        diff_h = 360 - diff_h

    assert diff_l <= tolerance, f"Lightness difference {diff_l} exceeds tolerance {tolerance}"
    assert diff_c <= tolerance, f"Chroma difference {diff_c} exceeds tolerance {tolerance}"
    assert diff_h <= tolerance * 360, f"Hue difference {diff_h} exceeds tolerance {tolerance * 360}"


def create_test_image(width: int = 100, height: int = 100, pattern: str = "gradient") -> np.ndarray:
    """Create a test image with a specific pattern."""
    if pattern == "gradient":
        # Create a gradient from black to white
        x = np.linspace(0, 255, width)
        y = np.linspace(0, 255, height)
        xx, yy = np.meshgrid(x, y)
        gray = ((xx + yy) / 2).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    if pattern == "rainbow":
        # Create a rainbow pattern
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            hue = int(360 * x / width)
            color = Color(f"hsl({hue} 100% 50%)").convert("srgb")
            rgb = [int(color[ch] * 255) for ch in ["red", "green", "blue"]]
            image[:, x] = rgb
        return image

    if pattern == "checkerboard":
        # Create a checkerboard pattern
        block_size = 10
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                if ((x // block_size) + (y // block_size)) % 2 == 0:
                    image[y : y + block_size, x : x + block_size] = [255, 255, 255]
        return image

    msg = f"Unknown pattern: {pattern}"
    raise ValueError(msg)


# Performance benchmarking utilities
@pytest.fixture
def benchmark_image():
    """Create a larger image for benchmarking."""
    return create_test_image(1920, 1080, "rainbow")
