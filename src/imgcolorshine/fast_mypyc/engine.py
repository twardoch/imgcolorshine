#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["coloraide", "numpy", "numba", "loguru"]
# ///
# this_file: src/imgcolorshine/fast_mypyc/engine.py

"""
Core color transformation engine for imgcolorshine.

This module contains the primary logic for the percentile-based color
transformation model. It handles color parsing, attractor management,
and the main two-pass image processing algorithm.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from coloraide import Color
from loguru import logger

from imgcolorshine.fast_numba import trans_numba
from imgcolorshine.fast_numba.engine_kernels import _fused_transform_kernel
from imgcolorshine.fast_mypyc.engine_helpers import (
    blend_pixel_colors,
    _calculate_weights_percentile,
    _transform_pixels_percentile_vec,
)
from imgcolorshine.gpu import GPU_AVAILABLE, ArrayModule, get_array_module

# Constants for attractor model
TOLERANCE_MIN, TOLERANCE_MAX = 0.0, 100.0
STRENGTH_MIN, STRENGTH_MAX = 0.0, 200.0
STRENGTH_TRADITIONAL_MAX = 100.0  # Traditional strength regime boundary
FULL_CIRCLE_DEGREES = 360.0  # Degrees in a full circle


@dataclass
class Attractor:
    """
    Represents a single color attractor.

    Attributes:
        color: The target color object in OKLCH space.
        tolerance: The percentage of pixels (0-100) to be influenced.
        strength: The maximum transformation intensity (0-100).
        oklch_values: A cached tuple of the attractor's (L, C, H) values.
        oklab_values: A cached tuple of the attractor's (L, a, b) values.
    """

    color: Color
    tolerance: float
    strength: float
    oklch_values: tuple[float, float, float] = field(init=False)
    oklab_values: tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        """Cache commonly used color conversions for performance."""
        self.oklch_values = (self.color["lightness"], self.color["chroma"], self.color["hue"])
        oklab_color = self.color.convert("oklab")
        self.oklab_values = (oklab_color["lightness"], oklab_color["a"], oklab_color["b"])


class OKLCHEngine:
    """
    Handles color space operations, color parsing, and batch conversions.
    """

    def __init__(self) -> None:
        """Initializes the engine with a cache for parsed colors."""
        self.cache: dict[str, Color] = {}
        logger.debug("Initialized OKLCH color engine.")

    def parse_color(self, color_str: str) -> Color:
        """Parses a CSS color string into a ColorAide object, with caching."""
        if color_str in self.cache:
            return self.cache[color_str].clone()
        try:
            color = Color(color_str)
            self.cache[color_str] = color.clone()
            return color
        except Exception as e:
            msg = f"Invalid color specification: {color_str}"
            raise ValueError(msg) from e

    def create_attractor(self, color_str: str, tolerance: float, strength: float) -> Attractor:
        """Creates an Attractor object from a color string and parameters."""
        color = self.parse_color(color_str).convert("oklch")
        return Attractor(color=color, tolerance=tolerance, strength=strength)

    def batch_rgb_to_oklab(self, rgb_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Converts an entire sRGB image to Oklab using Numba."""
        return trans_numba.batch_srgb_to_oklab(rgb_image.astype(np.float32))

    def batch_oklab_to_rgb(self, oklab_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Converts an entire Oklab image to sRGB with gamut mapping using Numba."""
        oklch_image = trans_numba.batch_oklab_to_oklch(oklab_image.astype(np.float32))
        oklch_mapped = trans_numba.batch_gamut_map_oklch(oklch_image)
        oklab_mapped = trans_numba.batch_oklch_to_oklab(oklch_mapped)
        return trans_numba.batch_oklab_to_srgb(oklab_mapped)





def _transform_pixels_gpu(
    image_lab: np.ndarray,
    image_lch: np.ndarray,
    attractors_lab: np.ndarray,
    attractors_lch: np.ndarray,
    delta_e_maxs: np.ndarray,
    strengths: np.ndarray,
    flags: np.ndarray,
    xp: Any,
) -> np.ndarray:
    """GPU-accelerated version of the percentile transformation.

    Uses CuPy for GPU computation with the same algorithm as the CPU version.
    """
    h, w = image_lab.shape[:2]

    # Transfer data to GPU
    image_lab_gpu = xp.asarray(image_lab)
    image_lch_gpu = xp.asarray(image_lch)
    attractors_lab_gpu = xp.asarray(attractors_lab)
    attractors_lch_gpu = xp.asarray(attractors_lch)
    delta_e_maxs_gpu = xp.asarray(delta_e_maxs)
    strengths_gpu = xp.asarray(strengths)

    # Flatten images for vectorized operations
    pix_lab_flat = image_lab_gpu.reshape(-1, 3)  # (N, 3)
    pix_lch_flat = image_lch_gpu.reshape(-1, 3)  # (N, 3)

    # Compute distances
    deltas = pix_lab_flat[:, None, :] - attractors_lab_gpu[None, :, :]  # (N, A, 3)
    delta_e = xp.sqrt(xp.sum(deltas**2, axis=2))  # (N, A)

    # Compute weights
    within_tol = (delta_e < delta_e_maxs_gpu) & (delta_e_maxs_gpu > 0.0)
    d_norm = xp.where(within_tol, delta_e / delta_e_maxs_gpu, 1.0)
    falloff = 0.5 * (xp.cos(d_norm * xp.pi) + 1.0)

    base_strength = xp.minimum(strengths_gpu, STRENGTH_TRADITIONAL_MAX) / STRENGTH_TRADITIONAL_MAX
    s_extra = xp.clip(strengths_gpu - STRENGTH_TRADITIONAL_MAX, 0.0, None) / STRENGTH_TRADITIONAL_MAX
    weights = base_strength * falloff + s_extra * (1.0 - falloff)
    weights = xp.where(within_tol, weights, 0.0).astype(xp.float32)

    # Blend channels
    total_w = weights.sum(axis=1, keepdims=True)
    src_w = xp.where(total_w < 1.0, 1.0 - total_w, 0.0).astype(xp.float32)

    # Pre-compute trig for attractor hues
    attr_h_rad = xp.deg2rad(attractors_lch_gpu[:, 2])
    attr_sin = xp.sin(attr_h_rad)
    attr_cos = xp.cos(attr_h_rad)

    # Luminance
    final_l = (
        src_w[:, 0] * pix_lch_flat[:, 0] + (weights * attractors_lch_gpu[:, 0][None, :]).sum(axis=1)
        if flags[0]
        else pix_lch_flat[:, 0]
    )

    # Chroma
    final_c = (
        src_w[:, 0] * pix_lch_flat[:, 1] + (weights * attractors_lch_gpu[:, 1][None, :]).sum(axis=1)
        if flags[1]
        else pix_lch_flat[:, 1]
    )

    # Hue
    if flags[2]:
        pix_h_rad = xp.deg2rad(pix_lch_flat[:, 2])
        sin_sum = src_w[:, 0] * xp.sin(pix_h_rad) + (weights * attr_sin[None, :]).sum(axis=1)
        cos_sum = src_w[:, 0] * xp.cos(pix_h_rad) + (weights * attr_cos[None, :]).sum(axis=1)
        final_h = xp.rad2deg(xp.arctan2(sin_sum, cos_sum))
    else:
        final_h = pix_lch_flat[:, 2]

    final_h = xp.where(final_h < 0.0, final_h + FULL_CIRCLE_DEGREES, final_h)
    h_rad = xp.deg2rad(final_h)

    # Convert LCH → Lab
    out_lab_flat = xp.empty_like(pix_lab_flat)
    out_lab_flat[:, 0] = final_l
    out_lab_flat[:, 1] = final_c * xp.cos(h_rad)
    out_lab_flat[:, 2] = final_c * xp.sin(h_rad)

    # Transfer back to CPU and reshape
    result = out_lab_flat.reshape(h, w, 3)
    if xp.__name__ == "cupy":
        result = xp.asnumpy(result)

    return result


class ColorTransformer:
    """High-level color transformation interface."""

    def __init__(self, engine: OKLCHEngine, use_fused_kernel: bool = False, use_gpu: bool = False):
        self.engine = engine
        self.use_fused_kernel = use_fused_kernel
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            self.array_module = ArrayModule("cupy")
            self.xp = self.array_module.xp
        else:
            self.xp = np

        logger.debug(f"Initialized ColorTransformer (fused_kernel={use_fused_kernel}, gpu={self.use_gpu})")

    def transform_image(
        self,
        image: np.ndarray[Any, Any],
        attractors: list[Attractor],
        flags: dict[str, bool],
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Transforms an image using the percentile-based color attractor model."""
        h, w = image.shape[:2]
        logger.info(f"Transforming {w}×{h} image with {len(attractors)} attractors")

        if not attractors:
            logger.warning("No attractors provided, returning original image.")
            return image.copy()

        flags_array = np.array([flags.get("luminance", True), flags.get("saturation", True), flags.get("hue", True)])
        attractors_lab = np.array([a.oklab_values for a in attractors])
        attractors_lch = np.array([a.oklch_values for a in attractors])
        tolerances = np.array([a.tolerance for a in attractors])
        strengths = np.array([a.strength for a in attractors])

        logger.info("Analyzing image color distribution...")
        image_lab = self.engine.batch_rgb_to_oklab(image)

        delta_e_maxs = np.zeros_like(tolerances, dtype=np.float32)
        for i in range(len(attractors)):
            distances = np.sqrt(np.sum((image_lab - attractors_lab[i]) ** 2, axis=-1))
            if tolerances[i] <= TOLERANCE_MIN:
                delta_e_maxs[i] = 0.0
            elif tolerances[i] >= TOLERANCE_MAX:
                delta_e_maxs[i] = np.max(distances) + 1e-6
            else:
                delta_e_maxs[i] = np.percentile(distances, tolerances[i])

        logger.debug(f"Calculated distance thresholds (ΔE max): {delta_e_maxs}")

        logger.info("Applying color transformation...")

        if self.use_gpu:
            logger.debug("Using GPU acceleration")
            image_lch = trans_numba.batch_oklab_to_oklch(image_lab.astype(np.float32))
            transformed_lab = _transform_pixels_gpu(
                image_lab.astype(np.float32),
                image_lch,
                attractors_lab.astype(np.float32),
                attractors_lch.astype(np.float32),
                delta_e_maxs.astype(np.float32),
                strengths.astype(np.float32),
                flags_array,
                self.xp,
            )
        elif self.use_fused_kernel:
            logger.debug("Using fused transformation kernel")
            transformed_lab = _fused_transform_kernel(
                image_lab.astype(np.float32),
                attractors_lab.astype(np.float32),
                attractors_lch.astype(np.float32),
                delta_e_maxs.astype(np.float32),
                strengths.astype(np.float32),
                flags_array.astype(np.bool_),
            )
        else:
            image_lch = trans_numba.batch_oklab_to_oklch(image_lab.astype(np.float32))
            transformed_lab = _transform_pixels_percentile_vec(
                image_lab, image_lch, attractors_lab, attractors_lch, delta_e_maxs, strengths, flags_array
            )

        result = self.engine.batch_oklab_to_rgb(transformed_lab)
        if progress_callback:
            progress_callback(1.0)

        logger.info("Transformation complete.")
        return np.clip(result, 0, 1)
