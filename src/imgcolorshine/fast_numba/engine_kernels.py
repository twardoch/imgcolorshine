"""Numba-accelerated engine kernels for color transformation.

This module contains JIT-compiled functions that perform the core
color transformation operations at the pixel level.
"""
# this_file: src/imgcolorshine/fast_numba/engine_kernels.py

from __future__ import annotations

import numba
import numpy as np

from . import trans_numba

# Constants
STRENGTH_TRADITIONAL_MAX = 100.0
FULL_CIRCLE_DEGREES = 360.0


@numba.njit(parallel=True, cache=True)
def _fused_transform_kernel(
    image_lab: np.ndarray,
    attractors_lab: np.ndarray,
    attractors_lch: np.ndarray,
    delta_e_maxs: np.ndarray,
    strengths: np.ndarray,
    flags: np.ndarray,
) -> np.ndarray:
    """Fused transformation kernel for improved performance.

    Processes one pixel at a time through the entire pipeline, keeping
    intermediate values in CPU registers and improving cache performance.
    """
    h, w, _ = image_lab.shape
    transformed_lab = np.empty_like(image_lab)
    n_attractors = len(attractors_lab)

    for i in numba.prange(h * w):  # type: ignore[attr-defined]
        y = i // w
        x = i % w
        pixel_lab = image_lab[y, x]

        # Convert pixel to LCH
        pixel_lch = trans_numba.oklab_to_oklch_single(pixel_lab)

        # Calculate distances and weights (no large intermediate arrays)
        weights = np.zeros(n_attractors, dtype=np.float32)
        for j in range(n_attractors):
            delta_e = np.sqrt(np.sum((pixel_lab - attractors_lab[j]) ** 2))
            delta_e_max = delta_e_maxs[j]

            if delta_e < delta_e_max and delta_e_max > 0:
                d_norm = delta_e / delta_e_max
                falloff = 0.5 * (np.cos(d_norm * np.pi) + 1.0)

                # Traditional vs extended strength regime
                base_strength = min(strengths[j], STRENGTH_TRADITIONAL_MAX) / STRENGTH_TRADITIONAL_MAX
                s_extra = max(strengths[j] - STRENGTH_TRADITIONAL_MAX, 0.0) / STRENGTH_TRADITIONAL_MAX
                weights[j] = base_strength * falloff + s_extra * (1.0 - falloff)
            else:
                weights[j] = 0.0

        # Blend colors
        total_w = np.sum(weights)
        src_w = 1.0 - total_w if total_w < 1.0 else 0.0

        # Luminance
        final_l = pixel_lch[0]
        if flags[0]:
            final_l = src_w * pixel_lch[0]
            for j in range(n_attractors):
                final_l += weights[j] * attractors_lch[j, 0]

        # Chroma
        final_c = pixel_lch[1]
        if flags[1]:
            final_c = src_w * pixel_lch[1]
            for j in range(n_attractors):
                final_c += weights[j] * attractors_lch[j, 1]

        # Hue (using circular average)
        final_h = pixel_lch[2]
        if flags[2]:
            pix_h_rad = np.deg2rad(pixel_lch[2])
            sin_sum = src_w * np.sin(pix_h_rad)
            cos_sum = src_w * np.cos(pix_h_rad)

            for j in range(n_attractors):
                attr_h_rad = np.deg2rad(attractors_lch[j, 2])
                sin_sum += weights[j] * np.sin(attr_h_rad)
                cos_sum += weights[j] * np.cos(attr_h_rad)

            final_h = np.rad2deg(np.arctan2(sin_sum, cos_sum))
            if final_h < 0.0:
                final_h += FULL_CIRCLE_DEGREES

        # Convert back to Lab
        final_h_rad = np.deg2rad(final_h)
        final_lab = np.array(
            [final_l, final_c * np.cos(final_h_rad), final_c * np.sin(final_h_rad)], dtype=pixel_lab.dtype
        )

        transformed_lab[y, x] = final_lab

    return transformed_lab