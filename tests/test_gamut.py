# this_file: tests/test_gamut.py

import numpy as np
import pytest
from coloraide import Color

from imgcolorshine.gamut import (
    GamutMapper,
    batch_map_oklch_numba,
    binary_search_chroma,
    create_gamut_boundary_lut,
)

# Import Numba helpers from trans_numba that are used by gamut.py for context,
# though their direct testing might be elsewhere or implicit.
from imgcolorshine.trans_numba import (
    is_in_gamut_srgb,
    oklab_to_srgb_single,
    oklch_to_oklab_single,
)


# Helper for comparing OKLCH tuples
def assert_oklch_equal(oklch1, oklch2, tol=1e-4):
    """"""
    assert np.isclose(oklch1[0], oklch2[0], atol=tol)  # L
    assert np.isclose(oklch1[1], oklch2[1], atol=tol)  # C
    assert np.isclose(oklch1[2], oklch2[2], atol=tol)  # H


@pytest.fixture
def srgb_mapper() -> GamutMapper:
    """"""
    return GamutMapper(target_space="srgb")


@pytest.fixture
def p3_mapper() -> GamutMapper:
    """"""
    # Using display-p3 as another common color space
    return GamutMapper(target_space="display-p3")


# Test Numba-optimized binary_search_chroma
# These tests depend on the correctness of oklch_to_oklab_single, oklab_to_srgb_single, is_in_gamut_srgb
# which are part of trans_numba
def test_binary_search_chroma_in_gamut():
    """"""
    # sRGB Red: oklch(0.62796, 0.22486, 29.233) is in sRGB gamut
    l, c, h = 0.62796, 0.22486, 29.233
    # Check with ColorAide first to confirm it's in gamut
    assert Color("oklch", [l, c, h]).in_gamut("srgb")
    mapped_c = binary_search_chroma(l, c, h)
    assert np.isclose(mapped_c, c)


def test_binary_search_chroma_out_of_gamut():
    """"""
    # A very saturated P3 green, likely out of sRGB gamut
    # P3 Green: oklch(0.90325, 0.25721, 134.09) -> ColorAide says this is in sRGB.
    # Let's pick a more extreme color: oklch(0.8, 0.4, 150) # Bright, very saturated green
    l, c, h = 0.8, 0.4, 150.0
    assert not Color("oklch", [l, c, h]).in_gamut("srgb")

    mapped_c = binary_search_chroma(l, c, h)
    assert mapped_c < c
    assert mapped_c >= 0
    # Check if the new color is in gamut
    mapped_oklch = np.array([l, mapped_c, h], dtype=np.float32)
    mapped_oklab = oklch_to_oklab_single(mapped_oklch)
    mapped_rgb = oklab_to_srgb_single(mapped_oklab)
    assert is_in_gamut_srgb(mapped_rgb)


def test_binary_search_chroma_black_white_gray():
    """"""
    # Black (L=0, C=0, H=any)
    assert np.isclose(binary_search_chroma(0.0, 0.0, 0.0), 0.0)
    # White (L=1, C=0, H=any)
    assert np.isclose(binary_search_chroma(1.0, 0.0, 0.0), 0.0)
    # Gray (L=0.5, C=0, H=any)
    assert np.isclose(binary_search_chroma(0.5, 0.0, 180.0), 0.0)
    # A gray with some chroma that is in gamut
    assert np.isclose(binary_search_chroma(0.5, 0.01, 180.0), 0.01)


# Test GamutMapper class
def test_gamut_mapper_init(srgb_mapper: GamutMapper, p3_mapper: GamutMapper):
    """"""
    assert srgb_mapper.target_space == "srgb"
    assert p3_mapper.target_space == "display-p3"


def test_gamut_mapper_is_in_gamut(srgb_mapper: GamutMapper):
    """"""
    in_gamut_color = Color("oklch(0.7 0.1 120)")  # Mild green, in sRGB
    out_of_gamut_color = Color("oklch(0.8 0.5 240)")  # Bright blue, out of sRGB
    assert srgb_mapper.is_in_gamut(in_gamut_color)
    assert not srgb_mapper.is_in_gamut(out_of_gamut_color)


# Test map_oklch_to_gamut (sRGB path - uses Numba binary_search_chroma)
def test_map_oklch_to_gamut_srgb_in_gamut(srgb_mapper: GamutMapper):
    """"""
    l, c, h = 0.7, 0.1, 120.0  # Mild green
    assert Color("oklch", [l, c, h]).in_gamut("srgb")
    mapped_l, mapped_c, mapped_h = srgb_mapper.map_oklch_to_gamut(l, c, h)
    assert mapped_l == l
    assert np.isclose(mapped_c, c)
    assert mapped_h == h


def test_map_oklch_to_gamut_srgb_out_of_gamut(srgb_mapper: GamutMapper):
    """"""
    l, c, h = 0.8, 0.5, 240.0  # Bright blue, out of sRGB
    assert not Color("oklch", [l, c, h]).in_gamut("srgb")
    mapped_l, mapped_c, mapped_h = srgb_mapper.map_oklch_to_gamut(l, c, h)
    assert mapped_l == l
    assert mapped_c < c
    assert mapped_h == h
    assert Color("oklch", [mapped_l, mapped_c, mapped_h]).in_gamut(
        "srgb", tolerance=0.0015
    )  # Allow small tolerance


# Test map_oklch_to_gamut (non-sRGB path - uses ColorAide)
def test_map_oklch_to_gamut_p3_in_gamut(p3_mapper: GamutMapper):
    """"""
    # A color safely in P3 gamut
    l, c, h = 0.6, 0.2, 40.0
    assert Color("oklch", [l, c, h]).in_gamut("display-p3")
    mapped_l, mapped_c, mapped_h = p3_mapper.map_oklch_to_gamut(l, c, h)
    assert mapped_l == l
    assert np.isclose(mapped_c, c)
    assert mapped_h == h


def test_map_oklch_to_gamut_p3_out_of_gamut(p3_mapper: GamutMapper):
    """"""
    # A color outside P3 (e.g., Rec2020 green: oklch(0.94706 0.33313 141.34))
    l, c, h = 0.94706, 0.33313, 141.34
    assert not Color("oklch", [l, c, h]).in_gamut("display-p3")
    mapped_l, mapped_c, mapped_h = p3_mapper.map_oklch_to_gamut(l, c, h)
    assert mapped_l == l
    assert mapped_c < c
    assert mapped_h == h
    assert Color("oklch", [mapped_l, mapped_c, mapped_h]).in_gamut(
        "display-p3", tolerance=0.0001
    )


def test_map_oklab_to_gamut(srgb_mapper: GamutMapper):
    """"""
    # Oklab bright blue (out of sRGB): L=0.8, a=-0.25, b=-0.433 (corresponds to oklch(0.8, 0.5, 240))
    l, a, b = 0.8, -0.25, -0.4330127  # approx for c=0.5, h=240
    Color("oklab", [l, a, b]).convert("oklch")
    assert not Color("oklab", [l, a, b]).in_gamut("srgb")

    mapped_l, mapped_a, mapped_b = srgb_mapper.map_oklab_to_gamut(l, a, b)

    # Check that the mapped Oklab color is now in sRGB gamut
    mapped_color_oklab = Color("oklab", [mapped_l, mapped_a, mapped_b])
    assert mapped_color_oklab.in_gamut("srgb", tolerance=0.0015)

    # Chroma should be reduced for the mapped color
    original_chroma = np.sqrt(a**2 + b**2)
    mapped_chroma = np.sqrt(mapped_a**2 + mapped_b**2)
    assert mapped_chroma < original_chroma


def test_map_rgb_to_gamut(srgb_mapper: GamutMapper):
    """"""
    r, g, b_val = 1.5, -0.5, 0.5
    mapped_r, mapped_g, mapped_b_val = srgb_mapper.map_rgb_to_gamut(r, g, b_val)
    assert mapped_r == 1.0
    assert mapped_g == 0.0
    assert mapped_b_val == 0.5


# Test batch_map_oklch
@pytest.mark.parametrize("mapper_fixture_name", ["srgb_mapper", "p3_mapper"])
def test_batch_map_oklch(mapper_fixture_name, request):
    """"""
    mapper = request.getfixturevalue(mapper_fixture_name)
    colors = np.array(
        [
            [0.7, 0.1, 120.0],  # In sRGB and P3
            [0.8, 0.5, 240.0],  # Out of sRGB, possibly out of P3 (very vibrant blue)
            [
                0.95,
                0.4,
                150.0,
            ],  # Very bright, very saturated green (out of sRGB, maybe P3)
        ],
        dtype=np.float32,
    )

    mapped_colors = mapper.batch_map_oklch(colors)
    assert mapped_colors.shape == colors.shape

    for i in range(colors.shape[0]):
        original_l, original_c, original_h = colors[i]
        mapped_l, mapped_c, mapped_h = mapped_colors[i]

        assert mapped_l == original_l
        assert mapped_h == original_h
        assert mapped_c <= original_c
        assert Color("oklch", [mapped_l, mapped_c, mapped_h]).in_gamut(
            mapper.target_space, tolerance=0.0015
        )


def test_batch_map_oklch_numba_direct():  # Testing the Numba fn directly
    """"""
    colors_flat = np.array(
        [
            [0.7, 0.1, 120.0],  # In sRGB
            [0.8, 0.5, 240.0],  # Out of sRGB
        ],
        dtype=np.float32,
    )

    mapped_colors = batch_map_oklch_numba(colors_flat)
    assert mapped_colors.shape == colors_flat.shape

    # First color (in gamut)
    assert np.isclose(mapped_colors[0, 1], colors_flat[0, 1])
    # Second color (out of gamut)
    assert mapped_colors[1, 1] < colors_flat[1, 1]
    assert Color("oklch", list(mapped_colors[1])).in_gamut("srgb", tolerance=0.0015)


def test_analyze_gamut_coverage(srgb_mapper: GamutMapper):
    """"""
    colors = np.array(
        [
            [0.7, 0.1, 120.0],  # In sRGB
            [0.8, 0.5, 240.0],  # Out of sRGB
            [0.5, 0.05, 30.0],  # In sRGB
        ]
    )
    stats = srgb_mapper.analyze_gamut_coverage(colors)
    assert stats["total"] == 3
    assert stats["in_gamut"] == 2
    assert stats["out_of_gamut"] == 1
    assert np.isclose(stats["percentage_in_gamut"], (2 / 3) * 100)


def test_analyze_gamut_coverage_empty(srgb_mapper: GamutMapper):
    """"""
    colors = np.array([])
    # Reshape to (0,3) if needed by the function's Color iteration logic
    stats = srgb_mapper.analyze_gamut_coverage(
        colors.reshape(0, 3) if colors.ndim == 1 else colors
    )
    assert stats["total"] == 0
    assert stats["in_gamut"] == 0
    assert stats["out_of_gamut"] == 0
    assert (
        stats["percentage_in_gamut"] == 100
    )  # Or 0, depending on interpretation. Code says 100.


def test_create_gamut_boundary_lut(
    srgb_mapper: GamutMapper,
):  # Pass mapper for is_in_gamut usage
    """"""
    hue_steps = 12  # Smaller for faster test
    lightness_steps = 10
    lut = create_gamut_boundary_lut(
        hue_steps=hue_steps, lightness_steps=lightness_steps
    )
    assert lut.shape == (lightness_steps, hue_steps)
    assert lut.dtype == np.float32
    assert np.all(lut >= 0)
    # Max chroma in oklch is around 0.3-0.4 typically, but can be higher for some L/H
    # Let's use a loose upper bound of 0.5 as per code's c_max in create_gamut_boundary_lut
    assert np.all(lut <= 0.5 + 1e-3)  # Add epsilon for float comparisons

    # Check a known in-gamut gray value (L=0.5, C=0). Chroma should be small.
    # For L=0.5 (mid lightness_idx), H=any, the max chroma should be > 0 if not pure gray
    # This test is a bit tricky as it depends on the LUT indexing.
    # For L=0 (l_idx=0), C should be 0.
    assert np.all(lut[0, :] == 0)  # For L=0, max chroma must be 0
    # For L=1 (l_idx=lightness_steps-1), C should be 0.
    assert np.all(lut[-1, :] == 0)  # For L=1, max chroma must be 0
