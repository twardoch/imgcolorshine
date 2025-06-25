# Performance Hot-Spots Report

This document lists modules detected with potential for acceleration.  Scan performed on $(date '+%Y-%m-%d %H:%M') using simple `ripgrep` patterns (`import numba`, `@numba`).

| Module | Hot-spot indicator | Suggested backend |
|--------|-------------------|-------------------|
| `src/imgcolorshine/gamut.py` | Pure-Python but many tight loops; good candidate for mypyc AOT | fast_mypyc |
| `src/imgcolorshine/utils.py` | Pure helpers; small loops – mypyc will remove overhead | fast_mypyc |
| `src/imgcolorshine/color.py` | Conversion math; pure-Python | fast_mypyc |
| `src/imgcolorshine/trans_numba.py` | Heavy use of `@numba.njit` loops | fast_numba |
| `src/imgcolorshine/gamut_numba.py` | Binary search & batch mapping with Numba | fast_numba |
| `src/imgcolorshine/falloff.py` | Multiple mathematical kernels with `@numba.njit` | fast_numba |
| `src/imgcolorshine/engine.py` | Contains inlined Numba kernels & loops | split: move kernels to fast_numba, keep orchestrator in core |

**Next actions**
1. For *fast_numba* modules, relocate code with `git mv` and update imports.
2. For *fast_mypyc* candidates, extract tight loops into helper modules then move.
3. Keep thin compatibility shims in original locations during deprecation window. 