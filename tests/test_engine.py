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


def create_test_image() -> np.ndarray:
    """Creates a 10x1 a gradient image for predictable testing."""
    # Creates a gradient from red to blue
    pixels = np.zeros((1, 10, 3), dtype=np.float32)
    for i in range(10):
        ratio = i / 9.0
        pixels[0, i, 0] = 1.0 - ratio  # Red component
        pixels[0, i, 2] = ratio  # Blue component
    return pixels


def test_tolerance_percentile(transformer: ColorTransformer) -> None:
    """
    Tests if the tolerance=30 affects exactly 30% of the pixels.
    """
    image = create_test_image()
    # Attractor is pure red, which is the first pixel.
    attractor = transformer.engine.create_attractor("red", 30.0, 100.0)

    # We only care about hue for this test.
    flags = {"luminance": False, "saturation": False, "hue": True}

    transformed_image = transformer.transform_image(image, [attractor], flags)

    # Compare original and transformed images pixel by pixel
    changed_pixels = np.sum(np.any(image != transformed_image, axis=-1))

    # With a 10-pixel image and tolerance=30, exactly 3 pixels should change.
    # The first pixel (pure red) is the attractor itself.
    # The next two closest pixels should be affected by the falloff.
    assert changed_pixels == 3, f"Expected 3 changed pixels, but got {changed_pixels}"


def test_strength(transformer: ColorTransformer) -> None:
    """
    Tests if strength=50 results in a 50% blend for the most similar pixel.
    """
    image = create_test_image()
    # Attractor is magenta, halfway between red and blue.
    attractor_color_str = "magenta"
    attractor = transformer.engine.create_attractor(attractor_color_str, 100.0, 50.0)

    flags = {"luminance": True, "saturation": True, "hue": True}

    transformed_image = transformer.transform_image(image, [attractor], flags)

    # The most similar pixel to magenta is the one in the middle (pixel 5)
    original_pixel_5 = image[0, 5]
    transformed_pixel_5 = transformed_image[0, 5]

    # Get the attractor color in sRGB [0,1] format
    attractor_srgb = np.array(transformer.engine.parse_color(attractor_color_str).convert("srgb").coords())

    # Expected result is a 50/50 blend of original and attractor color
    expected_pixel_5 = 0.5 * original_pixel_5 + 0.5 * attractor_srgb

    assert np.allclose(transformed_pixel_5, expected_pixel_5, atol=1e-3), (
        f"Expected ~{expected_pixel_5}, but got {transformed_pixel_5}"
    )
