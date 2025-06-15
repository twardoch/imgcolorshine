# this_file: tests/test_hierar.py

from unittest import mock

import cv2
import numpy as np
import pytest

from imgcolorshine.hierar import (
    HierarchicalProcessor,
    PyramidLevel,
    compute_gradient_magnitude,
    compute_perceptual_distance_mask,
)

# For _compute_perceptual_distance which is a static njit method in HierarchicalProcessor
# We can access it via an instance or the class if it's truly static in behavior.
# Let's assume we'll test it via an instance for simplicity or call it directly if possible.


# --- Fixtures ---
@pytest.fixture
def processor_default() -> HierarchicalProcessor:
    """"""
    return HierarchicalProcessor()


@pytest.fixture
def processor_no_adaptive() -> HierarchicalProcessor:
    """"""
    return HierarchicalProcessor(use_adaptive_subdivision=False)


@pytest.fixture
def sample_image_rgb_uint8() -> np.ndarray:
    """"""
    # Simple 128x128 RGB image
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[32:96, 32:96, :] = 255  # White square in the middle
    img[0:10, 0:10, 0] = 200  # Reddish patch
    return img


@pytest.fixture
def sample_image_lab_float32() -> np.ndarray:
    """"""
    # Oklab: L ranges 0-1, a,b roughly -0.5 to 0.5
    img = np.random.rand(64, 64, 3).astype(np.float32)
    img[..., 0] = np.clip(img[..., 0], 0, 1)
    img[..., 1:] = (img[..., 1:] - 0.5) * 0.8  # Scale a,b to approx -0.4 to 0.4
    return img


# --- Numba function tests ---
def test_compute_perceptual_distance_mask(sample_image_lab_float32: np.ndarray):
    """"""
    lab1 = sample_image_lab_float32.copy()
    lab2 = sample_image_lab_float32.copy()

    # Identical images, threshold > 0
    mask_identical = compute_perceptual_distance_mask(lab1, lab2, threshold=0.1)
    assert not np.any(mask_identical)

    # Different images
    lab2[0, 0, :] += 0.5  # Introduce a large difference at one pixel
    mask_different_large_thresh = compute_perceptual_distance_mask(lab1, lab2, threshold=1.0)  # High threshold
    assert not np.any(mask_different_large_thresh)  # Difference might be < 1.0

    mask_different_small_thresh = compute_perceptual_distance_mask(lab1, lab2, threshold=0.01)  # Low threshold
    assert mask_different_small_thresh[0, 0]  # Pixel with large diff should be True
    assert np.sum(mask_different_small_thresh) == 1  # Only that pixel


def test_compute_gradient_magnitude():
    """"""
    gray_flat = np.full((64, 64), 128, dtype=np.float32)
    grad_flat = compute_gradient_magnitude(gray_flat)
    # Gradients are computed skipping 1-pixel border, so border is 0. Center should be 0 for flat.
    assert np.all(grad_flat == 0)

    gray_edge = np.zeros((64, 64), dtype=np.float32)
    gray_edge[:, 32:] = 255  # Vertical edge
    grad_edge = compute_gradient_magnitude(gray_edge)
    assert np.any(grad_edge[:, 31:33] > 0)  # Expect non-zero gradient around the edge line
    # Check that borders are zero
    assert np.all(grad_edge[0, :] == 0)
    assert np.all(grad_edge[-1, :] == 0)
    assert np.all(grad_edge[:, 0] == 0)
    assert np.all(grad_edge[:, -1] == 0)

    # Max value of Sobel for 0 to 255 step is sqrt((4*255)^2 + 0) = 1020
    # Or for diagonal it could be sqrt((3*255)^2 + (3*255)^2) approx
    # Check that values are within a reasonable range, e.g. not excessively large.
    assert np.max(grad_edge) < 255 * 5  # Generous upper bound for Sobel on 0-255 range data


def test_hierarchical_processor_compute_perceptual_distance_static_method():
    """"""
    # Test the static Numba method _compute_perceptual_distance
    # This method is on HierarchicalProcessor, but njit decorated.
    lab1 = np.array([[[0.5, 0.1, 0.1]]], dtype=np.float32)  # H, W, C
    lab2 = np.array([[[0.6, 0.2, 0.0]]], dtype=np.float32)
    expected_dist_sq = (0.1**2) + (0.1**2) + (0.1**2)
    expected_dist = np.sqrt(expected_dist_sq)

    # Access static method via class or instance
    dist_map = HierarchicalProcessor._compute_perceptual_distance(lab1, lab2)
    assert dist_map.shape == (1, 1)
    assert np.isclose(dist_map[0, 0], expected_dist)


# --- HierarchicalProcessor tests ---
def test_build_pyramid_default(processor_default: HierarchicalProcessor, sample_image_rgb_uint8: np.ndarray):
    """"""
    img_h, img_w = sample_image_rgb_uint8.shape[:2]  # 128, 128
    pyramid = processor_default.build_pyramid(sample_image_rgb_uint8)

    assert len(pyramid) > 1
    assert pyramid[0].image.shape == (img_h, img_w, 3)
    assert pyramid[0].level == 0
    assert pyramid[0].scale == 1.0

    # Default min_size=64, pyramid_factor=0.5
    # Level 0: 128x128
    # Level 1: 64x64 (pyrDown)
    # Min(64,64) is not > 64, so loop terminates. Add coarsest 64x64.
    # Expected: 128x128, 64x64. So 2 levels.
    assert len(pyramid) == 2
    assert pyramid[1].shape == (img_h // 2, img_w // 2)
    assert pyramid[1].level == 1
    assert np.isclose(pyramid[1].scale, 0.5)

    # Test with smaller image where min_size condition hits earlier
    small_img = cv2.resize(sample_image_rgb_uint8, (processor_default.min_size, processor_default.min_size))
    pyramid_small = processor_default.build_pyramid(small_img)
    assert len(pyramid_small) == 1  # Only the original image
    assert pyramid_small[0].shape == (processor_default.min_size, processor_default.min_size)


@mock.patch("imgcolorshine.hierar.batch_srgb_to_oklab", return_value=np.zeros((64, 64, 3), dtype=np.float32))
@mock.patch("imgcolorshine.hierar.compute_perceptual_distance_mask")
def test_compute_difference_mask(mock_compute_mask, mock_batch_srgb_to_oklab, processor_default: HierarchicalProcessor):
    """"""
    fine_level_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    coarse_upsampled_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    threshold = 0.2  # This is user input 0-1 range

    processor_default.compute_difference_mask(fine_level_rgb, coarse_upsampled_rgb, threshold)

    # Check batch_srgb_to_oklab calls
    assert mock_batch_srgb_to_oklab.call_count == 2
    # Check first call args
    np.testing.assert_array_almost_equal(
        mock_batch_srgb_to_oklab.call_args_list[0][0][0], fine_level_rgb.astype(np.float32) / 255.0
    )
    np.testing.assert_array_almost_equal(
        mock_batch_srgb_to_oklab.call_args_list[1][0][0], coarse_upsampled_rgb.astype(np.float32) / 255.0
    )

    # Check compute_perceptual_distance_mask call
    mock_compute_mask.assert_called_once()
    call_args = mock_compute_mask.call_args[0]
    assert call_args[0] is mock_batch_srgb_to_oklab.return_value  # fine_lab
    assert call_args[1] is mock_batch_srgb_to_oklab.return_value  # coarse_lab
    assert call_args[2] == threshold * 2.5  # oklab_threshold


@mock.patch("cv2.cvtColor")
@mock.patch("imgcolorshine.hierar.compute_gradient_magnitude")
@mock.patch("cv2.getStructuringElement")
@mock.patch("cv2.dilate")
def test_detect_gradient_regions(
    mock_dilate, mock_getse, mock_compute_grad, mock_cvtcolor, processor_default: HierarchicalProcessor
):
    """"""
    image_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    gradient_threshold = 0.1

    # Mock return values
    mock_cvtcolor.return_value = np.zeros((64, 64), dtype=np.float32)  # Grayscale image
    mock_grad_mag = np.random.rand(64, 64).astype(np.float32)
    mock_compute_grad.return_value = mock_grad_mag
    mock_getse.return_value = np.ones((5, 5), dtype=np.uint8)  # Dummy kernel
    mock_dilate.return_value = np.zeros((64, 64), dtype=np.uint8)  # Dilated mask

    processor_default.detect_gradient_regions(image_rgb, gradient_threshold)

    mock_cvtcolor.assert_called_once_with(image_rgb, cv2.COLOR_RGB2GRAY)
    # mock_compute_grad.assert_called_once_with(mock_cvtcolor.return_value.astype(np.float32))
    assert mock_compute_grad.call_count == 1
    called_arg_compute_grad = mock_compute_grad.call_args[0][0]
    np.testing.assert_array_equal(called_arg_compute_grad, mock_cvtcolor.return_value.astype(np.float32))

    # Check normalization and thresholding logic implicitly by checking mock_dilate input
    # gradient_mask = gradient_magnitude > gradient_threshold
    # Dilate is called with gradient_mask.astype(np.uint8)
    normalized_grad = mock_grad_mag / mock_grad_mag.max() if mock_grad_mag.max() > 0 else mock_grad_mag
    expected_grad_mask_input_to_dilate = (normalized_grad > gradient_threshold).astype(np.uint8)

    mock_getse.assert_called_once_with(cv2.MORPH_ELLIPSE, (5, 5))
    mock_dilate.assert_called_once()
    np.testing.assert_array_equal(mock_dilate.call_args[0][0], expected_grad_mask_input_to_dilate)
    assert mock_dilate.call_args[0][1] is mock_getse.return_value


@mock.patch("cv2.cvtColor")
@mock.patch("imgcolorshine.hierar.compute_gradient_magnitude")
@mock.patch("cv2.getStructuringElement")
@mock.patch("cv2.dilate")
def test_detect_gradient_regions_no_gradient(
    mock_dilate, mock_getse, mock_compute_grad, mock_cvtcolor, processor_default: HierarchicalProcessor
):
    """"""
    image_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    gradient_threshold = 0.1

    # Mock return values
    mock_cvtcolor.return_value = np.zeros((64, 64), dtype=np.float32)  # Grayscale image
    # Key change: gradient magnitude is all zeros
    mock_compute_grad.return_value = np.zeros((64, 64), dtype=np.float32)
    mock_getse.return_value = np.ones((5, 5), dtype=np.uint8)  # Dummy kernel
    # Expect dilated mask to also be all zeros if input mask is all zeros
    mock_dilate.return_value = np.zeros((64, 64), dtype=np.uint8)

    processor_default.detect_gradient_regions(image_rgb, gradient_threshold)

    mock_cvtcolor.assert_called_once_with(image_rgb, cv2.COLOR_RGB2GRAY)
    assert mock_compute_grad.call_count == 1
    called_arg_compute_grad = mock_compute_grad.call_args[0][0]
    np.testing.assert_array_equal(called_arg_compute_grad, mock_cvtcolor.return_value.astype(np.float32))

    # Check normalization and thresholding logic implicitly by checking mock_dilate input
    # If mock_grad_mag is all zeros, normalized_grad is all zeros.
    # expected_grad_mask_input_to_dilate should be all zeros.
    expected_grad_mask_input_to_dilate = np.zeros((64, 64), dtype=np.uint8)

    mock_getse.assert_called_once_with(cv2.MORPH_ELLIPSE, (5, 5))
    mock_dilate.assert_called_once()
    np.testing.assert_array_equal(mock_dilate.call_args[0][0], expected_grad_mask_input_to_dilate)
    assert mock_dilate.call_args[0][1] is mock_getse.return_value


# --- Test process_hierarchical (main logic) ---


# Mock transform function for testing process_hierarchical
def mock_transform_func(image, attractors, tolerances, strengths, channels):
    """"""
    # Example: Invert colors, or add a constant. Let's make it identifiable.
    # Ensure output is same dtype and range as expected by HierarchicalProcessor (RGB 0-255 uint8)
    if image.dtype == np.uint8:
        return 255 - image
    # If input is float (e.g. from resize), handle that then convert back
    return (
        (255 - (image * 255).astype(np.uint8)).astype(np.uint8) if image.max() <= 1 else (255 - image).astype(np.uint8)
    )


@mock.patch("cv2.resize")  # Removed problematic side_effect
def test_process_hierarchical_small_image_no_pyramid(
    mock_cv_resize, processor_default: HierarchicalProcessor, sample_image_rgb_uint8: np.ndarray
):
    """"""
    # Let the initial resize for test setup use the real cv2.resize
    # The mock_cv_resize will apply to calls inside processor_default.process_hierarchical
    small_img = _original_cv2_resize(sample_image_rgb_uint8, (processor_default.min_size, processor_default.min_size))

    # Mock build_pyramid to ensure it returns only one level for this test
    with mock.patch.object(
        processor_default,
        "build_pyramid",
        return_value=[PyramidLevel(image=small_img, scale=1.0, shape=small_img.shape[:2], level=0)],
    ) as mock_build_pyr:
        result = processor_default.process_hierarchical(small_img, mock_transform_func, [], [], [], [])
        mock_build_pyr.assert_called_once_with(small_img)
        # Check that transform_func was called directly on the small_img
        # This requires transform_func to be "spyable" or check its effect
        np.testing.assert_array_equal(result, mock_transform_func(small_img, [], [], [], []))
        # mock_cv_resize (the one from @mock.patch) should not be called if only one level
        mock_cv_resize.assert_not_called()  # No upsampling should occur


@mock.patch("cv2.resize")  # Removed problematic side_effect
def test_process_hierarchical_full_run(
    mock_cv_resize, processor_default: HierarchicalProcessor, sample_image_rgb_uint8: np.ndarray
):
    """"""
    # Mock sub-functions to control their behavior and verify calls
    with (
        mock.patch.object(processor_default, "compute_difference_mask") as mock_comp_diff,
        mock.patch.object(processor_default, "detect_gradient_regions") as mock_detect_grad,
    ):
        # Setup mocks:
        # Coarsest level (64x64 for 128x128 input)
        # Fine level (128x128)
        # Let compute_difference_mask return a mask that refines some pixels
        mock_comp_diff.return_value = np.zeros(sample_image_rgb_uint8.shape[:2], dtype=bool)  # Default no diff
        mock_comp_diff.return_value[0:16, 0:16] = True  # Refine top-left corner

        # Let detect_gradient_regions return no gradients to isolate diff_mask effect
        mock_detect_grad.return_value = np.zeros(sample_image_rgb_uint8.shape[:2], dtype=bool)

        # Configure mock_cv_resize to return an appropriately shaped array
        # This will be the 'upsampled' variable in the code.
        mock_upsampled_array = np.zeros_like(sample_image_rgb_uint8)
        mock_cv_resize.return_value = mock_upsampled_array

        # Dummy args for transform_func
        att, tol, stren, chan = np.array([]), np.array([]), np.array([]), []

        result = processor_default.process_hierarchical(
            sample_image_rgb_uint8, mock_transform_func, att, tol, stren, chan
        )

        # --- Verification ---
        # 1. Pyramid building (implicitly tested by levels processed)
        #    For 128x128 -> levels: 128 (lvl 0), 64 (lvl 1, coarsest)
        # 2. Coarsest level (64x64) is transformed
        #    (transform_func is called with pyramid[-1].image)
        # 3. Upsampling: cv2.resize called to upsample from 64x64 to 128x128
        assert mock_cv_resize.call_count >= 1
        # First call to resize should be from 64x64 to 128x128
        resized_from_shape = mock_cv_resize.call_args_list[0][0][0].shape
        resized_to_dsize = mock_cv_resize.call_args_list[0][0][1]
        assert resized_from_shape[0] == sample_image_rgb_uint8.shape[0] // 2  # 64
        assert resized_to_dsize == (sample_image_rgb_uint8.shape[1], sample_image_rgb_uint8.shape[0])  # (128, 128)

        # 4. compute_difference_mask called for level 0 (128x128)
        mock_comp_diff.assert_called_once()
        # Check args: fine_level (pyramid[0].image), upsampled, threshold
        assert mock_comp_diff.call_args[0][0].shape == sample_image_rgb_uint8.shape
        assert mock_comp_diff.call_args[0][1].shape == sample_image_rgb_uint8.shape  # upsampled result
        assert mock_comp_diff.call_args[0][2] == processor_default.difference_threshold

        # 5. detect_gradient_regions called if use_adaptive_subdivision is True
        if processor_default.use_adaptive_subdivision:
            mock_detect_grad.assert_called_once()
            assert mock_detect_grad.call_args[0][0].shape == sample_image_rgb_uint8.shape
            assert mock_detect_grad.call_args[0][1] == processor_default.gradient_threshold
        else:
            mock_detect_grad.assert_not_called()

        # 6. Result assembly:
        #    - Pixels in the True part of refinement_mask should come from mock_transform_func(level.image)
        #    - Pixels in False part should come from upsampled image

        # Coarsest image (64x64)
        coarsest_img = cv2.pyrDown(sample_image_rgb_uint8)  # Uses real cv2.pyrDown
        transformed_coarsest = mock_transform_func(coarsest_img, att, tol, stren, chan)
        # This call to cv2.resize in the test logic will use the mock
        upsampled_from_coarsest = cv2.resize(
            transformed_coarsest,
            (sample_image_rgb_uint8.shape[1], sample_image_rgb_uint8.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        # Assert that upsampled_from_coarsest is actually mock_upsampled_array due to the mock
        assert upsampled_from_coarsest is mock_upsampled_array

        # Fine level image (128x128)
        transformed_fine_level = mock_transform_func(sample_image_rgb_uint8, att, tol, stren, chan)

        expected_result = upsampled_from_coarsest.copy()
        refinement_mask = mock_comp_diff.return_value  # In this test, grad_mask is all False
        expected_result[refinement_mask] = transformed_fine_level[refinement_mask]

        np.testing.assert_array_equal(result, expected_result)


@mock.patch("cv2.resize")
def test_process_hierarchical_no_adaptive_subdivision(
    mock_cv_resize, processor_no_adaptive: HierarchicalProcessor, sample_image_rgb_uint8: np.ndarray
):
    """"""
    # This test uses processor_no_adaptive fixture (use_adaptive_subdivision=False)
    with (
        mock.patch.object(processor_no_adaptive, "compute_difference_mask") as mock_comp_diff,
        mock.patch.object(processor_no_adaptive, "detect_gradient_regions") as mock_detect_grad,
    ):
        # Setup mocks for this specific test
        mock_comp_diff.return_value = np.zeros(sample_image_rgb_uint8.shape[:2], dtype=bool)  # No diffs
        mock_upsampled_array = np.zeros_like(sample_image_rgb_uint8)
        mock_cv_resize.return_value = mock_upsampled_array

        # Dummy args for transform_func
        att, tol, stren, chan = np.array([]), np.array([]), np.array([]), []

        result = processor_no_adaptive.process_hierarchical(
            sample_image_rgb_uint8, mock_transform_func, att, tol, stren, chan
        )

        # Key assertion: detect_gradient_regions should not be called
        mock_detect_grad.assert_not_called()

        # Other assertions to ensure flow is correct (e.g., result is based on upsampled)
        mock_comp_diff.assert_called_once()  # compute_difference_mask is always called

        # Expected result: upsampled from coarsest transformed level, as refinement_mask will be only diff_mask (all False)
        coarsest_img = cv2.pyrDown(sample_image_rgb_uint8)
        transformed_coarsest = mock_transform_func(coarsest_img, att, tol, stren, chan)
        # This call to cv2.resize in the test logic will use the mock
        upsampled_from_coarsest = cv2.resize(
            transformed_coarsest,
            (sample_image_rgb_uint8.shape[1], sample_image_rgb_uint8.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        assert upsampled_from_coarsest is mock_upsampled_array  # Verifying mock was used as expected

        np.testing.assert_array_equal(result, mock_upsampled_array)


def test_process_hierarchical_no_refinement_needed(
    processor_default: HierarchicalProcessor, sample_image_rgb_uint8: np.ndarray
):
    """"""
    # Test case where difference_mask and gradient_mask are all False
    # Here, processor_default has use_adaptive_subdivision=True, so detect_gradient_regions will be called.
    with (
        mock.patch.object(
            processor_default,
            "compute_difference_mask",
            return_value=np.zeros(sample_image_rgb_uint8.shape[:2], dtype=bool),
        ),
        mock.patch.object(
            processor_default,
            "detect_gradient_regions",
            return_value=np.zeros(sample_image_rgb_uint8.shape[:2], dtype=bool),
        ),
    ):
        att, tol, stren, chan = [], [], [], []
        result = processor_default.process_hierarchical(
            sample_image_rgb_uint8, mock_transform_func, att, tol, stren, chan
        )

        # Expected: result should be purely the upsampled version of the transformed coarsest level
        coarsest_img = cv2.pyrDown(sample_image_rgb_uint8)
        transformed_coarsest = mock_transform_func(coarsest_img, att, tol, stren, chan)
        expected_result = cv2.resize(
            transformed_coarsest,
            (sample_image_rgb_uint8.shape[1], sample_image_rgb_uint8.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )  # As per code

        np.testing.assert_array_equal(result, expected_result)


# --- Test process_hierarchical_tiled ---

# Store original cv2.resize to use it for test setup if needed, before mock applies.
_original_cv2_resize = cv2.resize


def test_process_hierarchical_tiled_small_image_no_tiling(
    processor_default: HierarchicalProcessor, sample_image_rgb_uint8: np.ndarray
):
    """"""
    # Image is 128x128, default tile_size=512. Should not tile.
    with mock.patch.object(processor_default, "process_hierarchical") as mock_proc_hier:
        mock_proc_hier.return_value = sample_image_rgb_uint8  # Dummy return

        processor_default.process_hierarchical_tiled(sample_image_rgb_uint8, mock_transform_func, [], [], [], [])

        mock_proc_hier.assert_called_once_with(sample_image_rgb_uint8, mock_transform_func, [], [], [], [])


def test_process_hierarchical_tiled_large_image(processor_default: HierarchicalProcessor):
    """"""
    # Image size chosen to ensure tiling is triggered based on condition:
    # h > tile_size * 2 or w > tile_size * 2
    # tile_size = 512, so tile_size * 2 = 1024. Let's use 1025x1025.
    img_dim = 1025
    large_img = np.zeros((img_dim, img_dim, 3), dtype=np.uint8)
    tile_size = 512  # Default in code
    overlap = tile_size // 4  # 128
    tile_size - overlap  # 384

    # Expected number of tiles:
    # N = ceil((img_dim - overlap) / step) if img_dim > tile_size else 1, but loop is range(0, D, step)
    # For y: range(0, 1025, 384) -> 0, 384, 768. (3 iterations)
    # For x: range(0, 1025, 384) -> 0, 384, 768. (3 iterations)
    # So, 3x3 = 9 calls to process_hierarchical

    # Mock process_hierarchical to just return the tile it received, for simplicity
    def simple_tile_processor(tile, *args):
        """"""
        return tile

    with mock.patch.object(
        processor_default, "process_hierarchical", side_effect=simple_tile_processor
    ) as mock_proc_hier:
        result = processor_default.process_hierarchical_tiled(
            large_img, mock_transform_func, [], [], [], [], tile_size=tile_size
        )

        assert mock_proc_hier.call_count == 9  # 3x3 tiles

        # Check some call args (e.g., shape of first tile)
        first_call_tile_arg = mock_proc_hier.call_args_list[0][0][0]
        # y_start_overlap = max(0, 0 - 64) = 0
        # x_start_overlap = max(0, 0 - 64) = 0
        # y_end_iter = 0 + 512 = 512
        # x_end_iter = 0 + 512 = 512
        # y_end_overlap = min(img_dim, 512 + 64) = min(1025, 576) = 576
        # x_end_overlap = min(img_dim, 512 + 64) = min(1025, 576) = 576
        # tile shape = (576 - 0, 576 - 0) = (576, 576, 3)
        assert first_call_tile_arg.shape == (576, 576, 3)

        # The result should be the original image because our mock_proc_hier returns the tile itself
        # and the blending logic is designed to reconstruct.
        # This is a simplified check; more complex blending would need careful verification.
        # For this test, with simple_tile_processor, the result will be mostly the input image,
        # but edges might be slightly off due to overlap handling.
        # A full check of result is complex. Let's check shape.
        assert result.shape == large_img.shape
