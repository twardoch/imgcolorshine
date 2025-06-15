# this_file: tests/test_falloff.py

import numpy as np
import pytest
from imgcolorshine.falloff import (
    FalloffType,
    falloff_cosine,
    falloff_linear,
    falloff_quadratic,
    falloff_gaussian,
    falloff_cubic,
    calculate_falloff,
    get_falloff_function,
    visualize_falloff,
    precompute_falloff_lut,
    apply_falloff_lut,
)

# Test individual falloff functions
@pytest.mark.parametrize(
    "func, d_norm, expected",
    [
        (falloff_cosine, 0.0, 1.0),
        (falloff_cosine, 0.5, 0.5),
        (falloff_cosine, 1.0, 0.0),
        (falloff_linear, 0.0, 1.0),
        (falloff_linear, 0.5, 0.5),
        (falloff_linear, 1.0, 0.0),
        (falloff_quadratic, 0.0, 1.0),
        (falloff_quadratic, 0.5, 0.75),
        (falloff_quadratic, 1.0, 0.0),
        (falloff_gaussian, 0.0, 1.0), # e^0 = 1
        (falloff_gaussian, 1.0, np.exp(-1/(2*0.4*0.4))), # d_norm=1, sigma=0.4 -> exp(-1 / 0.32)
        (falloff_cubic, 0.0, 1.0),
        (falloff_cubic, 0.5, 0.125), # (1-0.5)^3 = 0.5^3 = 0.125
        (falloff_cubic, 1.0, 0.0),
    ],
)
def test_individual_falloff_functions(func, d_norm, expected):
    assert np.isclose(func(d_norm), expected)

def test_falloff_gaussian_mid_value():
    # Check a specific mid-value for Gaussian
    # For d_norm = sigma = 0.4, value should be exp(-0.5)
    sigma = 0.4
    expected_val = np.exp(- (sigma**2) / (2 * sigma**2) ) # exp(-0.5)
    assert np.isclose(falloff_gaussian(sigma), expected_val)


# Test calculate_falloff dispatcher
@pytest.mark.parametrize(
    "falloff_type_enum, falloff_type_int, d_norm, expected_func",
    [
        (FalloffType.COSINE, 0, 0.25, falloff_cosine),
        (FalloffType.LINEAR, 1, 0.25, falloff_linear),
        (FalloffType.QUADRATIC, 2, 0.25, falloff_quadratic),
        (FalloffType.GAUSSIAN, 3, 0.25, falloff_gaussian),
        (FalloffType.CUBIC, 4, 0.25, falloff_cubic),
    ],
)
def test_calculate_falloff(falloff_type_enum, falloff_type_int, d_norm, expected_func):
    # Test direct call with integer type
    assert np.isclose(calculate_falloff(d_norm, falloff_type_int), expected_func(d_norm))
    # Test that default (no type given or invalid type) is cosine
    if falloff_type_enum == FalloffType.COSINE:
        assert np.isclose(calculate_falloff(d_norm), expected_func(d_norm)) # Default call
        assert np.isclose(calculate_falloff(d_norm, 99), expected_func(d_norm)) # Invalid type


# Test get_falloff_function
@pytest.mark.parametrize(
    "falloff_type, expected_func",
    [
        (FalloffType.COSINE, falloff_cosine),
        (FalloffType.LINEAR, falloff_linear),
        (FalloffType.QUADRATIC, falloff_quadratic),
        (FalloffType.GAUSSIAN, falloff_gaussian),
        (FalloffType.CUBIC, falloff_cubic),
    ],
)
def test_get_falloff_function(falloff_type, expected_func):
    assert get_falloff_function(falloff_type) is expected_func

def test_get_falloff_function_default():
    # Test that an unknown FalloffType (if it could be created) defaults to cosine
    # This is more of a conceptual test as Enum prevents arbitrary values.
    # The .get(falloff_type, falloff_cosine) handles this.
    # We can test by passing a non-enum value, though it's not type-safe.
    assert get_falloff_function(None) is falloff_cosine # type: ignore

# Test visualize_falloff
@pytest.mark.parametrize("falloff_type", list(FalloffType))
def test_visualize_falloff(falloff_type):
    samples = 50
    vis_data = visualize_falloff(falloff_type, samples=samples)
    assert isinstance(vis_data, np.ndarray)
    assert vis_data.shape == (samples, 2)
    assert np.isclose(vis_data[0, 0], 0.0)  # First distance is 0
    assert np.isclose(vis_data[-1, 0], 1.0) # Last distance is 1

    # Check if the falloff values match the expected function for first and last points
    expected_func = get_falloff_function(falloff_type)
    assert np.isclose(vis_data[0, 1], expected_func(0.0))
    assert np.isclose(vis_data[-1, 1], expected_func(1.0))

# Test precompute_falloff_lut
@pytest.mark.parametrize("falloff_type", list(FalloffType))
def test_precompute_falloff_lut(falloff_type):
    resolution = 64
    lut = precompute_falloff_lut(falloff_type, resolution=resolution)
    assert isinstance(lut, np.ndarray)
    assert lut.shape == (resolution,)
    assert lut.dtype == np.float32

    # Check boundary values
    expected_func = get_falloff_function(falloff_type)
    assert np.isclose(lut[0], expected_func(0.0))
    # For d_norm = 1, index is resolution - 1
    assert np.isclose(lut[-1], expected_func(1.0))

    # Check a mid value if possible (e.g. for linear)
    if falloff_type == FalloffType.LINEAR:
        mid_index = (resolution -1) // 2
        d_norm_mid = mid_index / (resolution -1)
        assert np.isclose(lut[mid_index], expected_func(d_norm_mid))


# Test apply_falloff_lut
def test_apply_falloff_lut():
    # Use a simple linear falloff LUT for easy verification
    resolution = 11 # Results in d_norm steps of 0.1
    lut = precompute_falloff_lut(FalloffType.LINEAR, resolution=resolution)
    # lut should be [1.0, 0.9, 0.8, ..., 0.1, 0.0]

    # Test exact points
    assert np.isclose(apply_falloff_lut(0.0, lut), 1.0)
    assert np.isclose(apply_falloff_lut(1.0, lut), 0.0)
    assert np.isclose(apply_falloff_lut(0.5, lut), 0.5) # (lut[5]*(1-0) + lut[6]*0) -> lut[5] = 0.5

    # Test interpolated points
    # d_norm = 0.25 -> idx_float = 0.25 * 10 = 2.5. idx = 2, frac = 0.5
    # lut[2]*(1-0.5) + lut[3]*0.5 = 0.8*0.5 + 0.7*0.5 = 0.4 + 0.35 = 0.75
    assert np.isclose(apply_falloff_lut(0.25, lut), 0.75)
    assert np.isclose(apply_falloff_lut(0.75, lut), 0.25)

    # Test edge cases for d_norm
    assert np.isclose(apply_falloff_lut(-0.5, lut), 1.0) # Should clamp to lut[0]
    assert np.isclose(apply_falloff_lut(1.5, lut), 0.0)  # Should clamp to lut[-1]

def test_apply_falloff_lut_cosine():
    # Test with cosine LUT as it's the default
    resolution = 1024
    lut = precompute_falloff_lut(FalloffType.COSINE, resolution=resolution)

    # Test some points
    assert np.isclose(apply_falloff_lut(0.0, lut), falloff_cosine(0.0))
    assert np.isclose(apply_falloff_lut(1.0, lut), falloff_cosine(1.0))

    # Test an interpolated value against direct calculation
    # This will have some minor precision difference due to LUT resolution and interpolation
    d_norm_test = 0.333
    assert np.isclose(apply_falloff_lut(d_norm_test, lut), falloff_cosine(d_norm_test), atol=1e-3) # Looser tolerance for LUT

    d_norm_test_2 = 0.666
    assert np.isclose(apply_falloff_lut(d_norm_test_2, lut), falloff_cosine(d_norm_test_2), atol=1e-3)

def test_calculate_falloff_invalid_integer_type():
    d_norm = 0.3
    # Test that an invalid integer type defaults to cosine
    assert np.isclose(calculate_falloff(d_norm, -1), falloff_cosine(d_norm))
    assert np.isclose(calculate_falloff(d_norm, 100), falloff_cosine(d_norm))

def test_apply_falloff_lut_near_boundary():
    resolution = 11
    lut = precompute_falloff_lut(FalloffType.LINEAR, resolution=resolution)
    # d_norm = 0.99 -> idx_float = 0.99 * 10 = 9.9. idx = 9, frac = 0.9
    # Expected: lut[9]*(1-0.9) + lut[10]*0.9
    # lut for linear: [1.0, 0.9, ..., 0.1, 0.0]
    # So, lut[9] = 0.1, lut[10] = 0.0
    # Expected: 0.1 * 0.1 + 0.0 * 0.9 = 0.01
    assert np.isclose(apply_falloff_lut(0.99, lut), 0.01, atol=1e-5)
    # Compare with direct calculation too
    assert np.isclose(apply_falloff_lut(0.99, lut), falloff_linear(0.99), atol=1e-5)
