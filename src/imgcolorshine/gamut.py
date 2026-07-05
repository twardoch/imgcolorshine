# this_file: src/imgcolorshine/gamut.py
"""Top-level re-export of gamut-mapping symbols from fast_mypyc and fast_numba.

Provides ``imgcolorshine.gamut`` as the public import path so tests can write
``from imgcolorshine.gamut import GamutMapper, batch_map_oklch_numba, ...``
"""

from imgcolorshine.fast_mypyc.gamut import GamutMapper, create_gamut_boundary_lut
from imgcolorshine.fast_numba.gamut_numba import batch_map_oklch_numba, binary_search_chroma

__all__ = [
    "GamutMapper",
    "batch_map_oklch_numba",
    "binary_search_chroma",
    "create_gamut_boundary_lut",
]
