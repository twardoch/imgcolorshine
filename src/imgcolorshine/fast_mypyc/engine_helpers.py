"""Pure Python engine helper functions for mypyc compilation.

This module contains computationally intensive pure Python functions
extracted from engine.py that benefit from mypyc compilation.
"""
# this_file: src/imgcolorshine/fast_mypyc/engine_helpers.py

from __future__ import annotations

import numpy as np

# Constants
STRENGTH_TRADITIONAL_MAX = 100.0
FULL_CIRCLE_DEGREES = 360.0


def blend_pixel_colors(
    pixel_lab: np.ndarray,
    pixel_lch: np.ndarray,
    attractors_lab: np.ndarray,
    attractors_lch: np.ndarray,
    weights: np.ndarray,
    flags: np.ndarray,
) -> np.ndarray:
    """Blend a single pixel's color based on attractor weights."""
    total_weight = np.sum(weights)
    if total_weight == 0:
        return pixel_lab

    src_weight = 1.0 - total_weight if total_weight < 1.0 else 0.0
    final_l, final_c, final_h = pixel_lch[0], pixel_lch[1], pixel_lch[2]

    if flags[0]:  # Luminance
        weighted_l = np.sum(weights * attractors_lch[:, 0])
        final_l = src_weight * pixel_lch[0] + weighted_l
    if flags[1]:  # Saturation
        weighted_c = np.sum(weights * attractors_lch[:, 1])
        final_c = src_weight * pixel_lch[1] + weighted_c
    if flags[2]:  # Hue
        sin_sum = src_weight * np.sin(np.deg2rad(pixel_lch[2])) + np.sum(
            weights * np.sin(np.deg2rad(attractors_lch[:, 2]))
        )
        cos_sum = src_weight * np.cos(np.deg2rad(pixel_lch[2])) + np.sum(
            weights * np.cos(np.deg2rad(attractors_lch[:, 2]))
        )
        final_h = np.rad2deg(np.arctan2(sin_sum, cos_sum))

    final_h = final_h + 360 if final_h < 0 else final_h
    h_rad = np.deg2rad(final_h)
    return np.array([final_l, final_c * np.cos(h_rad), final_c * np.sin(h_rad)], dtype=pixel_lab.dtype)


def _calculate_weights_percentile(
    pixel_lab: np.ndarray,
    attractors_lab: np.ndarray,
    delta_e_maxs: np.ndarray,
    strengths: np.ndarray,
) -> np.ndarray:
    """Calculate weights based on pre-calculated percentile distance thresholds."""
    weights = np.zeros(len(attractors_lab), dtype=np.float32)
    for i in range(len(attractors_lab)):
        delta_e = np.sqrt(np.sum((pixel_lab - attractors_lab[i]) ** 2))
        delta_e_max = delta_e_maxs[i]
        if delta_e < delta_e_max and delta_e_max > 0:
            d_norm = delta_e / delta_e_max
            falloff = 0.5 * (np.cos(d_norm * np.pi) + 1.0)  # 1 → 0 as d_norm 0→1

            if strengths[i] <= 100.0:
                # Traditional regime: scale fall-off by 0-1 strength factor
                weights[i] = (strengths[i] / 100.0) * falloff
            else:
                # Extended regime 100-200: progressively override fall-off
                s_extra = (strengths[i] - 100.0) / 100.0  # 0-1
                # Blend between falloff (at s_extra=0) and full influence (1.0)
                weights[i] = falloff + s_extra * (1.0 - falloff)
    return weights


def _transform_pixels_percentile_vec(
    pixels_lab: np.ndarray,
    pixels_lch: np.ndarray,
    attractors_lab: np.ndarray,
    attractors_lch: np.ndarray,
    delta_e_maxs: np.ndarray,
    strengths: np.ndarray,
    flags: np.ndarray,
) -> np.ndarray:
    """Vectorised implementation eliminating per-pixel Python loops.

    Notes
    -----
    1.  Arrays are cast to *float32* early for speed and lower memory.
    2.  The computation is entirely NumPy-based, leveraging broadcasting
        and avoiding Python-level iteration.  This is typically faster
        than a `numba.prange` double-loop because the heavy lifting is
        performed inside highly-optimised C ‑ BLAS routines.
    """
    h, w = pixels_lab.shape[:2]
    n_pixels = h * w
    attractors_lab.shape[0]

    # Flatten pixel arrays to (N, 3)
    pix_lab_flat = pixels_lab.reshape(n_pixels, 3).astype(np.float32)
    pix_lch_flat = pixels_lch.reshape(n_pixels, 3).astype(np.float32)

    # ------------------------------------------------------------------
    # 1. Distance computation (ΔE proxy in Oklab) → (N, A)
    # ------------------------------------------------------------------
    deltas = pix_lab_flat[:, None, :] - attractors_lab[None, :, :]  # (N, A, 3)
    delta_e = np.sqrt(np.sum(deltas**2, axis=2))  # (N, A)

    # ------------------------------------------------------------------
    # 2. Weight calculation using tolerance & strength
    # ------------------------------------------------------------------
    # Broadcast delta_e_maxs/strengths to (N, A)
    delta_e_maxs_b = delta_e_maxs[None, :]
    strengths_b = strengths[None, :]

    within_tol = (delta_e < delta_e_maxs_b) & (delta_e_maxs_b > 0.0)
    d_norm = np.where(within_tol, delta_e / delta_e_maxs_b, 1.0)  # Normalised distance (0-1)
    falloff = 0.5 * (np.cos(d_norm * np.pi) + 1.0)  # Raised-cosine 1 → 0

    # Traditional (≤100) vs extended (>100) strength regime
    base_strength = np.minimum(strengths_b, STRENGTH_TRADITIONAL_MAX) / STRENGTH_TRADITIONAL_MAX  # 0-1
    s_extra = np.clip(strengths_b - STRENGTH_TRADITIONAL_MAX, 0.0, None) / STRENGTH_TRADITIONAL_MAX  # 0-1 for >100
    weights = base_strength * falloff + s_extra * (1.0 - falloff)  # (N, A)

    # Zero-out weights for pixels outside tolerance
    weights = np.where(within_tol, weights, 0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Blend channels per flags
    # ------------------------------------------------------------------
    total_w = weights.sum(axis=1, keepdims=True)  # (N,1)
    src_w = np.where(total_w < 1.0, 1.0 - total_w, 0.0).astype(np.float32)

    # Pre-compute trig for attractor hues (A,)
    attr_h_rad = np.deg2rad(attractors_lch[:, 2])
    attr_sin = np.sin(attr_h_rad)
    attr_cos = np.cos(attr_h_rad)

    # Luminance ---------------------------------------------------------
    final_l = (
        src_w[:, 0] * pix_lch_flat[:, 0] + (weights * attractors_lch[:, 0][None, :]).sum(axis=1)
        if flags[0]
        else pix_lch_flat[:, 0]
    )

    # Chroma ------------------------------------------------------------
    final_c = (
        src_w[:, 0] * pix_lch_flat[:, 1] + (weights * attractors_lch[:, 1][None, :]).sum(axis=1)
        if flags[1]
        else pix_lch_flat[:, 1]
    )

    # Hue ---------------------------------------------------------------
    if flags[2]:
        pix_h_rad = np.deg2rad(pix_lch_flat[:, 2])
        sin_sum = src_w[:, 0] * np.sin(pix_h_rad) + (weights * attr_sin[None, :]).sum(axis=1)
        cos_sum = src_w[:, 0] * np.cos(pix_h_rad) + (weights * attr_cos[None, :]).sum(axis=1)
        final_h = np.rad2deg(np.arctan2(sin_sum, cos_sum))
    else:
        final_h = pix_lch_flat[:, 2]

    final_h = np.where(final_h < 0.0, final_h + FULL_CIRCLE_DEGREES, final_h)
    h_rad = np.deg2rad(final_h)

    # Convert LCH → Lab --------------------------------------------------
    out_lab_flat = np.empty_like(pix_lab_flat)
    out_lab_flat[:, 0] = final_l
    out_lab_flat[:, 1] = final_c * np.cos(h_rad)
    out_lab_flat[:, 2] = final_c * np.sin(h_rad)

    return out_lab_flat.reshape(h, w, 3)