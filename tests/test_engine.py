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

    # The most similar pixel to magenta is the one closest to the attractor
    # in OKLab space.  For the red→blue gradient, this is approximately pixel 4 or 5.
    # We find it dynamically to make the assertion robust.
    engine = transformer.engine
    image_lab = engine.batch_rgb_to_oklab(image)  # (1, 10, 3)
    attractor_lab = np.array(attractor.oklab_values, dtype=np.float32)
    dists = np.linalg.norm(image_lab[0] - attractor_lab, axis=-1)  # (10,)
    closest_idx = int(np.argmin(dists))

    original_dist = dists[closest_idx]
    transformed_lab = engine.batch_rgb_to_oklab(transformed_image)
    new_dist = float(np.linalg.norm(transformed_lab[0, closest_idx] - attractor_lab))

    # With strength=50 the most-similar pixel (d_norm=0, falloff=1) receives
    # weight=0.5, so its OKLab distance to the attractor should drop by ~50 %.
    # Allow 15 % tolerance to account for the LCH circular hue interpolation.
    assert new_dist < original_dist, (
        f"Pixel {closest_idx} should have moved toward attractor; "
        f"original_dist={original_dist:.4f}, new_dist={new_dist:.4f}"
    )
    ratio = new_dist / original_dist
    assert ratio < 0.65, (
        f"Expected ~50 % distance reduction (strength=50), "
        f"but ratio={ratio:.3f} (new_dist={new_dist:.4f}, orig={original_dist:.4f})"
    )
