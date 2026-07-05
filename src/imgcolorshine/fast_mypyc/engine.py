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

from imgcolorshine.fast_mypyc.engine_helpers import (
    _transform_pixels_percentile_vec,
)
from imgcolorshine.fast_numba import trans_numba
from imgcolorshine.fast_numba.engine_kernels import _fused_transform_kernel
from imgcolorshine.gpu import GPU_AVAILABLE, ArrayModule

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

    # ------------------------------------------------------------------
    # Single-pixel / scalar colour-space helpers
    # ------------------------------------------------------------------

    def calculate_delta_e(self, color1_oklab: np.ndarray, color2_oklab: np.ndarray) -> float:
        """Euclidean distance in OKLab space (ΔE proxy).

        Parameters
        ----------
        color1_oklab, color2_oklab:
            Arrays of shape ``(3,)`` containing ``[L, a, b]``.
        """
        diff = np.asarray(color1_oklab, dtype=np.float64) - np.asarray(color2_oklab, dtype=np.float64)
        return float(np.sqrt(np.dot(diff, diff)))

    def oklch_to_oklab(self, l: float, c: float, h: float) -> tuple[float, float, float]:
        """Convert a single OKLCH triplet to OKLab.

        Parameters
        ----------
        l, c, h:
            Lightness, chroma, hue (degrees).
        """
        h_rad = np.deg2rad(h)
        return l, float(c * np.cos(h_rad)), float(c * np.sin(h_rad))

    def oklab_to_oklch(self, l: float, a: float, b: float) -> tuple[float, float, float]:
        """Convert a single OKLab triplet to OKLCH.

        Parameters
        ----------
        l, a, b:
            Lightness and opponent channels.
        """
        c = float(np.sqrt(a * a + b * b))
        h = float(np.rad2deg(np.arctan2(b, a)))
        if h < 0:
            h += FULL_CIRCLE_DEGREES
        return l, c, h

    def srgb_to_linear(self, arr: np.ndarray) -> np.ndarray:
        """Apply sRGB gamma expansion (gamma-encoded → linear light).

        The standard IEC 61966-2-1 piecewise formula is used:
        - values ≤ 0.04045 → value / 12.92
        - values > 0.04045 → ((value + 0.055) / 1.055) ** 2.4
        """
        arr = np.asarray(arr, dtype=np.float64)
        return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(self, arr: np.ndarray) -> np.ndarray:
        """Apply sRGB gamma compression (linear light → gamma-encoded).

        Inverse of :meth:`srgb_to_linear`.
        """
        arr = np.asarray(arr, dtype=np.float64)
        return np.where(arr <= 0.0031308, arr * 12.92, 1.055 * arr ** (1.0 / 2.4) - 0.055)

    def gamut_map_oklch(self, l: float, c: float, h: float) -> tuple[float, float, float]:
        """Map an OKLCH colour to the sRGB gamut, preserving *L* and *H*.

        Uses binary search on chroma until the colour is inside sRGB gamut.
        If the colour is already in gamut it is returned unchanged.
        """
        color = Color("oklch", [l, c, h])
        if color.in_gamut("srgb"):
            return l, c, h
        lo, hi = 0.0, c
        for _ in range(32):
            mid = (lo + hi) * 0.5
            if Color("oklch", [l, mid, h]).in_gamut("srgb"):
                lo = mid
            else:
                hi = mid
        return l, lo, h

    def rgb_to_oklab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert a single sRGB pixel (array of 3) to OKLab via ColorAide."""
        rgb = np.asarray(rgb, dtype=np.float64)
        color = Color("srgb", [float(rgb[0]), float(rgb[1]), float(rgb[2])]).convert("oklab")
        return np.array([color["lightness"], color["a"], color["b"]], dtype=np.float64)

    def oklab_to_rgb(self, oklab: np.ndarray) -> np.ndarray:
        """Convert a single OKLab pixel (array of 3) to sRGB via ColorAide.

        The result is clipped to [0, 1].
        """
        oklab = np.asarray(oklab, dtype=np.float64)
        color = Color("oklab", [float(oklab[0]), float(oklab[1]), float(oklab[2])]).convert("srgb")
        rgb = np.array([color["red"], color["green"], color["blue"]], dtype=np.float64)
        return np.clip(rgb, 0.0, 1.0)


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
    result: np.ndarray = out_lab_flat.reshape(h, w, 3)
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
        result_clipped = np.clip(result, 0.0, 1.0)

        # Restore exact original sRGB values for pixels that weren't transformed.
        # Even though _transform_pixels_percentile_vec restores original Lab values
        # for zero-weight pixels, the OKLab→OKLCH→OKLab round-trip inside
        # batch_oklab_to_rgb introduces float32 noise.  Comparing the (float32)
        # Lab arrays before and after is reliable because the restore was exact.
        image_lab_f32 = image_lab.astype(np.float32)
        unchanged = np.all(transformed_lab.astype(np.float32) == image_lab_f32, axis=-1)
        result_clipped[unchanged] = image[unchanged]

        return result_clipped
