"""imgcolorshine.fast_numba
=================================

Internal sub-package hosting Numba-optimised kernels.  This code is *not* part
of the public API — but key symbols are re-exported in the top-level package to
preserve backwards-compatibility.
"""

# this_file: src/imgcolorshine/fast_numba/__init__.py

from __future__ import annotations

from imgcolorshine.fast_numba import engine_kernels, falloff, gamut_numba, numba_utils, trans_numba
from imgcolorshine.fast_numba.engine_kernels import _fused_transform_kernel
from imgcolorshine.fast_numba.falloff import (
    FalloffType,
    apply_falloff_lut,
    calculate_falloff,
    falloff_cosine,
    falloff_cubic,
    falloff_gaussian,
    falloff_linear,
    falloff_quadratic,
    get_falloff_function,
    precompute_falloff_lut,
    visualize_falloff,
)
from imgcolorshine.fast_numba.numba_utils import (
    apply_transformation_mask,
    compute_color_distances_batch,
    compute_edge_strength,
    compute_tile_uniformity,
    downsample_oklab,
    find_nearest_attractors,
)

__all__: list[str] = [
    "FalloffType",
    "_fused_transform_kernel",
    "apply_falloff_lut",
    "apply_transformation_mask",
    "calculate_falloff",
    "compute_color_distances_batch",
    "compute_edge_strength",
    "compute_tile_uniformity",
    "downsample_oklab",
    "engine_kernels",
    "falloff",
    "falloff_cosine",
    "falloff_cubic",
    "falloff_gaussian",
    "falloff_linear",
    "falloff_quadratic",
    "find_nearest_attractors",
    "gamut_numba",
    "get_falloff_function",
    "numba_utils",
    "precompute_falloff_lut",
    "trans_numba",
    "visualize_falloff",
]
