# this_file: src/imgcolorshine/fast_numba/falloff.py
"""Public ``imgcolorshine.fast_numba.falloff`` module.

Re-exports all falloff symbols so tests and user code can import from
``imgcolorshine.fast_numba.falloff`` without caring whether they come from the
Numba-JIT layer (``falloff_numba``) or the pure-Python enum layer
(``fast_mypyc.falloff``).
"""

from __future__ import annotations

# Pure-Python enum + helpers (mypyc-compilable, no Numba required)
from imgcolorshine.fast_mypyc.falloff import (
    FalloffType,
    get_falloff_function,
    precompute_falloff_lut,
    visualize_falloff,
)

# Numba-JIT implementations (fast, work in all Python contexts too)
from imgcolorshine.fast_numba.falloff_numba import (
    apply_falloff_lut,
    calculate_falloff,
    falloff_cosine,
    falloff_cubic,
    falloff_gaussian,
    falloff_linear,
    falloff_quadratic,
)

__all__ = [
    "FalloffType",
    "apply_falloff_lut",
    "calculate_falloff",
    "falloff_cosine",
    "falloff_cubic",
    "falloff_gaussian",
    "falloff_linear",
    "falloff_quadratic",
    "get_falloff_function",
    "precompute_falloff_lut",
    "visualize_falloff",
]
