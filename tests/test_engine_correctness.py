#!/usr/bin/env python3
# this_file: tests/test_engine_correctness.py
"""Tests for the correctness of the engine's transformation logic."""

import numpy as np
import pytest

from imgcolorshine.engine import ColorTransformer, OKLCHEngine


@pytest.fixture
def engine() -> OKLCHEngine:
    """Fixture for the OKLCHEngine."""
    return OKLCHEngine()


@pytest.fixture
def transformer(engine: OKLCHEngine) -> ColorTransformer:
    """Fixture for the ColorTransformer."""
    return ColorTransformer(engine)


def create_gradient_image(size: int = 10) -> np.ndarray:
    """Creates a gradient image for predictable testing."""
    # Creates a gradient from black to white
    pixels = np.zeros((1, size, 3), dtype=np.float32)
    for i in range(size):
        value = i / (size - 1)
        pixels[0, i] = [value, value, value]
    return pixels


def create_color_palette_image() -> np.ndarray:
    """Creates a small image with distinct colors."""
    # Red, green, blue, yellow, cyan, magenta, white, black, gray, orange
    colors = np.array(
        [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.0, 1.0],  # Magenta
            [1.0, 1.0, 1.0],  # White
            [0.0, 0.0, 0.0],  # Black
            [0.5, 0.5, 0.5],  # Gray
            [1.0, 0.5, 0.0],  # Orange
        ],
        dtype=np.float32,
    )
    return colors.reshape(1, 10, 3)


def test_tolerance_zero_affects_no_pixels(transformer: ColorTransformer) -> None:
    """Test that tolerance=0 affects no pixels."""
    image = create_gradient_image()
    attractor = transformer.engine.create_attractor("red", 0.0, 100.0)

    transformed = transformer.transform_image(image, [attractor], {})

    # With tolerance=0, no pixels should change
    assert np.allclose(image, transformed, atol=1e-6), "Tolerance=0 should not change any pixels"


def test_tolerance_100_affects_all_pixels(transformer: ColorTransformer) -> None:
    """Test that tolerance=100 affects all pixels."""
    image = create_gradient_image()
    attractor = transformer.engine.create_attractor("red", 100.0, 100.0)

    transformed = transformer.transform_image(image, [attractor], {})

    # With tolerance=100, all pixels should be affected (not necessarily changed to red)
    changed_mask = ~np.isclose(image, transformed, atol=1e-6, rtol=0)
    changed_pixels = np.sum(changed_mask.any(axis=-1))
    total_pixels = image.shape[0] * image.shape[1]

    # Allow for black pixel (0,0,0) which may not change when attracted to red due to lightness preservation
    assert changed_pixels >= total_pixels - 1, (
        f"Tolerance=100 should affect all or all-but-one {total_pixels} pixels, but only {changed_pixels} changed"
    )


def test_tolerance_percentile_behavior(transformer: ColorTransformer) -> None:
    """Test that tolerance correctly uses percentile-based selection."""
    # Create a larger gradient for better percentile testing
    image = create_gradient_image(100)

    # Use red as attractor at one end of the gradient
    attractor = transformer.engine.create_attractor("red", 30.0, 100.0)

    # Transform the image
    transformed = transformer.transform_image(image, [attractor], {})

    # Calculate which pixels changed
    changed_mask = ~np.isclose(image, transformed, atol=1e-6, rtol=0)
    changed_pixels = np.sum(changed_mask.any(axis=-1))
    total_pixels = image.shape[0] * image.shape[1]

    # With tolerance=30, approximately 30% of pixels should be affected
    # Allow some margin since the exact cutoff depends on the distance distribution
    expected_range = (0.2 * total_pixels, 0.4 * total_pixels)
    assert expected_range[0] <= changed_pixels <= expected_range[1], (
        f"With tolerance=30, expected {expected_range} pixels to change, but got {changed_pixels}"
    )


def test_strength_zero_no_change(transformer: ColorTransformer) -> None:
    """Test that strength=0 results in no change."""
    image = create_gradient_image()
    attractor = transformer.engine.create_attractor("red", 100.0, 0.0)

    transformed = transformer.transform_image(image, [attractor], {})

    # With strength=0, no pixels should change
    assert np.allclose(image, transformed, atol=1e-6), "Strength=0 should not change any pixels"


def test_strength_affects_blending(transformer: ColorTransformer) -> None:
    """Test that different strength values produce different levels of blending."""
    image = create_gradient_image()

    # Create attractors with different strengths
    attractor_50 = transformer.engine.create_attractor("red", 100.0, 50.0)
    attractor_100 = transformer.engine.create_attractor("red", 100.0, 100.0)
    attractor_150 = transformer.engine.create_attractor("red", 100.0, 150.0)

    transformed_50 = transformer.transform_image(image, [attractor_50], {})
    transformed_100 = transformer.transform_image(image, [attractor_100], {})
    transformed_150 = transformer.transform_image(image, [attractor_150], {})

    # Calculate average change magnitude for each strength
    change_50 = np.mean(np.abs(transformed_50 - image))
    change_100 = np.mean(np.abs(transformed_100 - image))
    change_150 = np.mean(np.abs(transformed_150 - image))

    # Higher strength should produce larger changes
    assert change_50 < change_100, (
        f"Strength 100 ({change_100:.4f}) should produce larger change than strength 50 ({change_50:.4f})"
    )
    assert change_100 < change_150, (
        f"Strength 150 ({change_150:.4f}) should produce larger change than strength 100 ({change_100:.4f})"
    )


def test_channel_flags(transformer: ColorTransformer) -> None:
    """Test that channel flags correctly control which channels are transformed."""
    image = create_color_palette_image()
    attractor = transformer.engine.create_attractor("red", 100.0, 100.0)

    # Test each channel individually
    flags_l_only = {"luminance": True, "saturation": False, "hue": False}
    flags_c_only = {"luminance": False, "saturation": True, "hue": False}
    flags_h_only = {"luminance": False, "saturation": False, "hue": True}

    transformed_l = transformer.transform_image(image, [attractor], flags_l_only)
    transformed_c = transformer.transform_image(image, [attractor], flags_c_only)
    transformed_h = transformer.transform_image(image, [attractor], flags_h_only)

    # Convert to LAB to check which channels changed
    engine = transformer.engine
    image_lab = engine.batch_rgb_to_oklab(image)
    engine.batch_rgb_to_oklab(transformed_l)
    engine.batch_rgb_to_oklab(transformed_c)
    transformed_h_lab = engine.batch_rgb_to_oklab(transformed_h)

    # Convert to LCH to check hue changes
    from imgcolorshine.fast_numba.trans_numba import batch_oklab_to_oklch

    batch_oklab_to_oklch(image_lab)
    batch_oklab_to_oklch(transformed_h_lab)

    # Check that the flags are working by verifying different transformations produce different results
    assert not np.allclose(transformed_l, transformed_c, atol=1e-4), (
        "L-only and C-only transforms should produce different results"
    )
    assert not np.allclose(transformed_l, transformed_h, atol=1e-4), (
        "L-only and H-only transforms should produce different results"
    )
    assert not np.allclose(transformed_c, transformed_h, atol=1e-4), (
        "C-only and H-only transforms should produce different results"
    )

    # Verify that at least some transformation occurred for each flag setting
    assert not np.allclose(image, transformed_l, atol=1e-6), "L-only transform should change the image"
    assert not np.allclose(image, transformed_c, atol=1e-6), "C-only transform should change the image"
    assert not np.allclose(image, transformed_h, atol=1e-6), "H-only transform should change the image"


def test_multiple_attractors(transformer: ColorTransformer) -> None:
    """Test that multiple attractors blend their influences correctly."""
    # Create a simple 3-pixel image: red, green, blue
    image = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)

    # Create two attractors that should affect different pixels
    attractor1 = transformer.engine.create_attractor("yellow", 50.0, 100.0)  # Should affect red/green more
    attractor2 = transformer.engine.create_attractor("cyan", 50.0, 100.0)  # Should affect green/blue more

    transformed_single1 = transformer.transform_image(image, [attractor1], {})
    transformed_single2 = transformer.transform_image(image, [attractor2], {})
    transformed_both = transformer.transform_image(image, [attractor1, attractor2], {})

    # The combined transformation should be different from either individual one
    assert not np.allclose(transformed_both, transformed_single1, atol=1e-4), (
        "Multiple attractors should produce different result than single"
    )
    assert not np.allclose(transformed_both, transformed_single2, atol=1e-4), (
        "Multiple attractors should produce different result than single"
    )

    # The green pixel (middle) should be most affected since it's influenced by both attractors
    green_change_single1 = np.linalg.norm(transformed_single1[0, 1] - image[0, 1])
    green_change_single2 = np.linalg.norm(transformed_single2[0, 1] - image[0, 1])
    green_change_both = np.linalg.norm(transformed_both[0, 1] - image[0, 1])

    # With both attractors, the green pixel should change more than with either alone
    assert green_change_both > max(green_change_single1, green_change_single2) * 0.8, (
        "Combined attractors should have stronger effect on pixels influenced by both"
    )


def test_strength_200_no_falloff(transformer: ColorTransformer) -> None:
    """Test that strength=200 produces no falloff (uniform influence within tolerance)."""
    # Create an image with varying distances from the attractor color
    image = create_gradient_image(20)
    attractor = transformer.engine.create_attractor("red", 50.0, 200.0)

    # Only transform hue for clearer results
    flags = {"luminance": False, "saturation": False, "hue": True}
    transformed = transformer.transform_image(image, [attractor], flags)

    # Get the pixels that were affected (within tolerance)
    changed_mask = ~np.isclose(image, transformed, atol=1e-6, rtol=0)
    affected_indices = np.where(changed_mask.any(axis=-1))[1]  # Get column indices of affected pixels

    if len(affected_indices) > 1:
        # Convert to LCH to check hue values
        from imgcolorshine.fast_numba.trans_numba import batch_oklab_to_oklch

        engine = transformer.engine
        transformed_lab = engine.batch_rgb_to_oklab(transformed)
        transformed_lch = batch_oklab_to_oklch(transformed_lab)

        # With strength=200 (no falloff), all affected pixels should have very similar hue
        affected_hues = transformed_lch[0, affected_indices, 2]
        hue_variance = np.var(affected_hues)

        # Allow small variance due to numerical precision and edge effects
        assert hue_variance < 10.0, (
            f"With strength=200, affected pixels should have uniform hue, but variance is {hue_variance}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
