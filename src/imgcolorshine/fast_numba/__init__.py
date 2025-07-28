"""imgcolorshine.fast_numba
=================================

Internal sub-package hosting Numba-optimised kernels.  This code is *not* part
of the public API — but key symbols are re-exported in the top-level package to
preserve backwards-compatibility.

The pure-Python façade inside :pymod:`imgcolorshine` performs *optional*
imports from this package; if Numba is unavailable, the callers fall back to
a slower reference implementation while logging a warning.
"""

# this_file: src/imgcolorshine/fast_numba/__init__.py

from __future__ import annotations

# Re-export commonly used kernels so callers can do e.g.
#   from imgcolorshine.fast_numba import trans_numba
# The actual modules will be created/moved in follow-up commits.

from . import trans_numba
from . import gamut_numba
from . import falloff
from . import utils
from . import engine_kernels

# Re-export key functions for convenience
from .utils import (
    compute_color_distances_batch,
    find_nearest_attractors,
    compute_tile_uniformity,
    apply_transformation_mask,
    compute_edge_strength,
    downsample_oklab
)

from .engine_kernels import _fused_transform_kernel

__all__: list[str] = [
    "trans_numba",
    "gamut_numba", 
    "falloff",
    "utils",
    "engine_kernels",
    "compute_color_distances_batch",
    "find_nearest_attractors",
    "compute_tile_uniformity",
    "apply_transformation_mask",
    "compute_edge_strength",
    "downsample_oklab",
    "_fused_transform_kernel"
]
