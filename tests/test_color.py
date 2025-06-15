# this_file: tests/test_color.py

import numpy as np
import pytest
from coloraide import Color

from imgcolorshine.color import Attractor, OKLCHEngine


# Helper function to compare colors with tolerance
def assert_colors_equal(color1: Color, color2: Color, tolerance: float = 1e-6):
    """"""
    # Convert to Oklab and get coordinates
    c1_oklab = color1.convert("oklab")
    c2_oklab = color2.convert("oklab")

    coords1 = np.array([c1_oklab["lightness"], c1_oklab["a"], c1_oklab["b"]])
    coords2 = np.array([c2_oklab["lightness"], c2_oklab["a"], c2_oklab["b"]])

    # Calculate Euclidean distance in Oklab space
    delta_e_val = np.sqrt(np.sum((coords1 - coords2) ** 2))
    assert delta_e_val == pytest.approx(0, abs=tolerance)


# Helper function to compare numpy arrays with tolerance
def assert_np_arrays_equal(arr1: np.ndarray, arr2: np.ndarray, tolerance: float = 1e-6):
    """"""
    assert np.allclose(arr1, arr2, atol=tolerance)


@pytest.fixture
def engine() -> OKLCHEngine:
    """"""
    return OKLCHEngine()


# Tests for Attractor dataclass
def test_attractor_initialization(engine: OKLCHEngine):
    """"""
    color_obj = engine.parse_color("oklch(70% 0.2 120)")
    attractor = Attractor(color=color_obj, tolerance=50.0, strength=75.0)
    assert attractor.color is color_obj
    assert attractor.tolerance == 50.0
    assert attractor.strength == 75.0
    # __post_init__ conversions
    assert attractor.oklch_values == (
        pytest.approx(0.7),
        0.2,
        120.0,
    )  # coloraide stores lightness 0-1

    oklab_color = color_obj.convert("oklab")
    expected_oklab = (oklab_color["lightness"], oklab_color["a"], oklab_color["b"])
    assert_np_arrays_equal(np.array(attractor.oklab_values), np.array(expected_oklab))


def test_attractor_post_init_chroma_handling(engine: OKLCHEngine):
    """"""
    # Test with a color that might have a different chroma value after conversion
    # For OKLCH, the 'chroma' attribute is directly used.
    # The original test had self.color["chroma"] repeated for oklch_values[2] which was hue
    # Corrected oklch_values to (L, C, H)
    color_obj = engine.parse_color("oklch(50% 0.1 270)")  # Example: a less vibrant blue
    attractor = Attractor(color=color_obj, tolerance=30.0, strength=60.0)
    assert attractor.oklch_values == (pytest.approx(0.5), 0.1, 270.0)  # L, C, H

    oklab_color = color_obj.convert("oklab")
    expected_oklab = (oklab_color["lightness"], oklab_color["a"], oklab_color["b"])
    assert_np_arrays_equal(np.array(attractor.oklab_values), np.array(expected_oklab))


# Tests for OKLCHEngine
def test_parse_color_valid(engine: OKLCHEngine):
    """"""
    color_str = "color(srgb 1 0.5 0)"  # Orange
    color_obj = engine.parse_color(color_str)
    assert isinstance(color_obj, Color)
    assert_colors_equal(color_obj, Color("color(srgb 1 0.5 0)"))

    # Test caching
    color_obj_cached = engine.parse_color(color_str)
    assert color_obj_cached is not color_obj  # Should be a clone
    assert_colors_equal(color_obj_cached, color_obj)
    assert color_str in engine.cache


def test_parse_color_invalid(engine: OKLCHEngine):
    """"""
    with pytest.raises(ValueError, match="Invalid color specification: notacolor"):
        engine.parse_color("notacolor")


def test_create_attractor(engine: OKLCHEngine):
    """"""
    attractor = engine.create_attractor("red", 40.0, 60.0)
    assert isinstance(attractor, Attractor)
    assert attractor.tolerance == 40.0
    assert attractor.strength == 60.0
    assert attractor.color.space() == "oklch"
    # Check if 'red' in oklch is correct. ColorAide default for 'red' is srgb(1 0 0)
    # srgb(1 0 0) -> oklch(0.62796 0.25768 29.234) approx (Values from ColorAide latest)
    assert_np_arrays_equal(
        np.array(attractor.oklch_values),
        np.array([0.62796, 0.25768, 29.234]),
        tolerance=1e-4,
    )


def test_calculate_delta_e(engine: OKLCHEngine):
    """"""
    # Using Oklab values for delta E calculation
    # Oklab for "red" (sRGB 1 0 0): L=0.62796, a=0.22486, b=0.12518
    # Oklab for "blue" (sRGB 0 0 1): L=0.45030, a=-0.03901, b=-0.31189
    color1_oklab = np.array([0.62796, 0.22486, 0.12518])
    color2_oklab = np.array([0.45030, -0.03901, -0.31189])
    # Expected delta E: sqrt((0.62796-0.45030)^2 + (0.22486 - (-0.03901))^2 + (0.12518 - (-0.31189))^2)
    # = sqrt(0.17766^2 + 0.26387^2 + 0.43707^2)
    # = sqrt(0.031564 + 0.069627 + 0.191029) = sqrt(0.29222) = 0.54057
    delta_e = engine.calculate_delta_e(color1_oklab, color2_oklab)
    assert isinstance(delta_e, float)
    assert np.isclose(delta_e, 0.54057, atol=1e-4)


def test_oklch_to_oklab_conversion(engine: OKLCHEngine):
    """"""
    # Red: oklch(0.62796 0.22486 29.233)
    l, c, h = 0.62796, 0.22486, 29.233
    oklab_l, oklab_a, oklab_b = engine.oklch_to_oklab(l, c, h)
    # Expected Oklab: L=0.62796, a=0.1963, b=0.1095 (approx from ColorAide for this specific OKLCH)
    # Using ColorAide to verify the direct conversion logic
    ca_color = Color("oklch", [l, c, h]).convert("oklab")
    assert np.isclose(oklab_l, ca_color["lightness"])
    assert np.isclose(oklab_a, ca_color["a"])
    assert np.isclose(oklab_b, ca_color["b"])


def test_oklab_to_oklch_conversion(engine: OKLCHEngine):
    """"""
    # Oklab for "red": L=0.62796, a=0.22486, b=0.12518
    l, a, b = 0.62796, 0.22486, 0.12518
    oklch_l, oklch_c, oklch_h = engine.oklab_to_oklch(l, a, b)
    # Expected OKLCH: L=0.62796, C=0.2572, H=29.233 (approx from ColorAide)
    ca_color = Color("oklab", [l, a, b]).convert("oklch")
    assert np.isclose(oklch_l, ca_color["lightness"])
    assert np.isclose(oklch_c, ca_color["chroma"])
    assert np.isclose(oklch_h, ca_color["hue"])

    # Test negative hue case
    l, a, b = 0.5, -0.1, -0.1  # Should result in hue > 180
    _, _, oklch_h_neg = engine.oklab_to_oklch(l, a, b)
    assert oklch_h_neg > 180  # Specifically, 225 for (-0.1, -0.1)


def test_srgb_to_linear_and_back(engine: OKLCHEngine):
    """"""
    srgb_color = np.array([0.5, 0.2, 0.8])
    linear_color = engine.srgb_to_linear(
        srgb_color.copy()
    )  # Use copy to avoid in-place modification issues if any
    srgb_restored = engine.linear_to_srgb(linear_color.copy())
    assert_np_arrays_equal(srgb_restored, srgb_color, tolerance=1e-6)

    # Test edge cases
    srgb_black = np.array([0.0, 0.0, 0.0])
    linear_black = engine.srgb_to_linear(srgb_black.copy())
    assert_np_arrays_equal(linear_black, srgb_black)
    srgb_white = np.array([1.0, 1.0, 1.0])
    linear_white = engine.srgb_to_linear(srgb_white.copy())
    assert_np_arrays_equal(linear_white, srgb_white)

    # Test specific value based on formula
    srgb_val = 0.04045
    linear_val = engine.srgb_to_linear(np.array([srgb_val]))
    assert np.isclose(linear_val[0], srgb_val / 12.92)

    srgb_val_high = 0.5
    linear_val_high = engine.srgb_to_linear(np.array([srgb_val_high]))
    assert np.isclose(linear_val_high[0], ((srgb_val_high + 0.055) / 1.055) ** 2.4)


def test_gamut_map_oklch_already_in_gamut(engine: OKLCHEngine):
    """"""
    # A color known to be in sRGB gamut
    l, c, h = 0.7, 0.1, 120  # A mild green
    mapped_l, mapped_c, mapped_h = engine.gamut_map_oklch(l, c, h)
    assert mapped_l == l
    assert mapped_c == c
    assert mapped_h == h


def test_gamut_map_oklch_out_of_gamut(engine: OKLCHEngine):
    """"""
    # A very vibrant color likely out of sRGB gamut
    l, c, h = 0.8, 0.5, 240  # A very bright and saturated blue
    original_color = Color("oklch", [l, c, h])
    assert not original_color.in_gamut("srgb")

    mapped_l, mapped_c, mapped_h = engine.gamut_map_oklch(l, c, h)
    assert mapped_l == l
    assert mapped_h == h
    assert mapped_c < c
    assert mapped_c >= 0  # Chroma should not be negative

    # Check if the mapped color is now in gamut
    mapped_color = Color("oklch", [mapped_l, mapped_c, mapped_h])
    assert mapped_color.in_gamut(
        "srgb", tolerance=0.0001
    )  # Use small tolerance due to binary search precision


# The following tests for batch operations are simplified as they call Numba functions.
# We primarily test if they can be called and return expected shapes/types.
# More detailed testing of Numba functions would be in their own test files or integration tests.


@pytest.mark.parametrize("rgb_shape", [(10, 10, 3), (1, 1, 3)])
def test_batch_rgb_to_oklab(engine: OKLCHEngine, rgb_shape: tuple):
    """"""
    rgb_image = np.random.rand(*rgb_shape).astype(np.float32)
    oklab_image = engine.batch_rgb_to_oklab(rgb_image)
    assert oklab_image.shape == rgb_shape
    assert oklab_image.dtype == np.float32  # Numba functions are set to use float32


@pytest.mark.parametrize("oklab_shape", [(10, 10, 3), (1, 1, 3)])
def test_batch_oklab_to_rgb(engine: OKLCHEngine, oklab_shape: tuple):
    """"""
    # Create Oklab values within a typical range
    oklab_image = np.random.rand(*oklab_shape).astype(np.float32)
    oklab_image[..., 0] = oklab_image[..., 0]  # L: 0-1
    oklab_image[..., 1:] = (
        oklab_image[..., 1:] * 0.5 - 0.25
    )  # a, b: approx -0.25 to 0.25

    rgb_image = engine.batch_oklab_to_rgb(oklab_image)
    assert rgb_image.shape == oklab_shape
    assert rgb_image.dtype == np.float32  # Numba functions output float32
    assert np.all(rgb_image >= -0.001)
    assert np.all(rgb_image <= 1.001)


# Test rgb_to_oklab and oklab_to_rgb (single version, which use ColorAide)
def test_rgb_to_oklab_single(engine: OKLCHEngine):
    """"""
    rgb_np = np.array([1.0, 0.0, 0.0])  # Red
    oklab_np = engine.rgb_to_oklab(rgb_np)

    ca_color = Color("srgb", [1, 0, 0]).convert("oklab")
    expected_oklab = np.array([ca_color["lightness"], ca_color["a"], ca_color["b"]])
    assert_np_arrays_equal(oklab_np, expected_oklab)


def test_oklab_to_rgb_single(engine: OKLCHEngine):
    """"""
    # Oklab for red: L=0.62796, a=0.22486, b=0.12518
    oklab_np = np.array([0.62796, 0.22486, 0.12518])
    rgb_np = engine.oklab_to_rgb(oklab_np)

    ca_color = Color("oklab", list(oklab_np)).convert("srgb")
    expected_rgb = np.array([ca_color["r"], ca_color["g"], ca_color["b"]])
    assert_np_arrays_equal(rgb_np, expected_rgb)
    assert np.all(rgb_np >= 0)
    assert np.all(rgb_np <= 1)
