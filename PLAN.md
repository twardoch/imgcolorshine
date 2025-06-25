# PLAN

## Goal of Task 1 (architecture refactor for performance)

Transform the current mixed-implementation layout into a clean three-tier structure:

1. **Pure-Python public API** (`src/imgcolorshine/…`) — remains optimisable with `mypyc`, **no** direct `numba` calls.
2. **Numba-optimised kernels** (`src/imgcolorshine/fast_numba/…`) — thin, internal helper modules compiled just-in-time by Numba.
3. **Ahead-of-time mypyc-compiled helpers** (`src/imgcolorshine/fast_mypyc/…`) — deterministic pure-Python algorithms that profit from `mypyc` AOT compilation.

## Current State Analysis

### Existing Numba Code Distribution
- **Already in `fast_numba/`**: `trans_numba.py`, `gamut_numba.py`, `falloff.py`
- **Still in root**: `numba_utils.py` (6 functions), `engine.py` (1 function + 2 commented)
- **Compatibility shims**: `trans_numba.py`, `falloff.py` in root (deprecated)

### Identified mypyc Candidates
1. **High Priority**: `_transform_pixels_percentile_vec()`, `process_large_image()`, `parse_attractor()`
2. **Medium Priority**: Gamut mapping functions, `blend_pixel_colors()`, weight calculations
3. **Low Priority**: I/O operations, validation functions

---

### High-level execution order

- [ ] **Phase 0 — Reconnaissance & safety nets**

  - [ ] Read `./llms.txt`, `cleanup.log` and re-scan the repo for `import numba`, `@numba` & heavy inner loops (regex & `grep_search`).
  - [ ] Generate a `report/perf_hotspots.md` with a table (module ▸ reason ▸ suggested backend).
  - [ ] Tag current `main` as `backup/pre-refactor-$(date +%Y-%m-%d)`.

- [ ] **Phase 1 — Create new sub-packages**

  - [ ] Create folders `src/imgcolorshine/fast_numba` and `src/imgcolorshine/fast_mypyc`, each with `__init__.py` (export convenience symbols).
  - [ ] Move or copy candidate modules (see "Allocation" below) keeping original `git` history via `git mv`.
  - [ ] Insert `this_file` headers + docstrings that explain internal-only status.

- [ ] **Phase 2 — Allocation of existing modules**

  **Numba migrations (`fast_numba/`):**
  - [ ] Move `numba_utils.py` → `fast_numba/utils.py` (6 functions)
  - [ ] Extract `_transform_pixels_numba` from `engine.py` → `fast_numba/engine_kernels.py`
  - [ ] Remove deprecated shims: root-level `trans_numba.py` and `falloff.py`
  - [ ] Update `fast_numba/__init__.py` to export all public symbols

  **Mypyc extractions (`fast_mypyc/`):**
  - [ ] Extract from `engine.py` → `fast_mypyc/engine_helpers.py`:
    - `blend_pixel_colors()` (lines 100-134)
    - `_calculate_weights_percentile()` (lines 137-161)  
    - `_transform_pixels_percentile_vec()` (lines 166-262)
  - [ ] Extract from `gamut.py` → `fast_mypyc/gamut_helpers.py`:
    - `is_in_gamut()`, `map_oklab_to_gamut()`, `analyze_gamut_coverage()`
    - `create_gamut_boundary_lut()` (lines 225-266)
  - [ ] Extract from `colorshine.py` → `fast_mypyc/colorshine_helpers.py`:
    - `parse_attractor()` (lines 39-62)
    - `generate_output_path()` (lines 64-68)
  - [ ] Move entire `utils.py` → `fast_mypyc/utils.py` (already started)
  - [ ] Extract from `io.py` → `fast_mypyc/io_helpers.py`:
    - Core `ImageProcessor` methods

- [ ] **Phase 3 — Refactor imports & maintain compatibility**

  - [ ] Search/replace root-level imports to point at new locations.
  - [ ] Add re-exports inside `fast_*.__init__` and deprecation warnings in old paths during transition.

- [ ] **Phase 4 — Update `pyproject.toml` build hooks**

  - [ ] Add optional `mypyc` build-backend:
    ```toml
    [tool.mypyc]
    packages = ["imgcolorshine.fast_mypyc"]
    options = "--verbose"
    ```
  - [ ] Expose an extra `speedups` optional-dependency group (`[project.optional-dependencies]`).
  - [ ] Document env-var `IMGCS_DISABLE_SPEEDUPS` to force pure-Python fallback.

- [ ] **Phase 5 — Continuous Integration adjustments**

  - [ ] Extend GitHub CI matrix: (CPython 3.11/3.12) × (speedups on/off).
  - [ ] Install system LLVM ≥ 14 for mypyc wheels on Linux.

- [ ] **Phase 6 — Testing & benchmarks**

  - [ ] Update tests to import via public API only.
  - [ ] Add `tests/test_speed_parity.py` verifying numerical equivalence between backends.
  - [ ] Add `scripts/bench.py` (Rich table) — compare pure vs fast_mypyc vs numba vs GPU.

- [ ] **Phase 7 — Docs & communication**

  - [ ] Update `README.md`, `docs/performance.md` and in-code docstrings with new architecture diagram.
  - [ ] Draft `CHANGELOG.md` entry under `## [Unreleased]` summarising refactor.

- [ ] **Phase 8 — Release checklist**
  - [ ] Bump version to `3.4.0` (minor, backward-compatible API).
  - [ ] Build sdist + wheels (`hatch build`); verify mypyc wheels present.
  - [ ] Publish to TestPyPI, smoke-test, then to PyPI.

---

### Detailed HOW for key items

1. **Moving Numba modules**  
   ```bash
   # Move numba_utils.py to fast_numba/utils.py
   git mv src/imgcolorshine/numba_utils.py src/imgcolorshine/fast_numba/utils.py
   
   # Remove deprecated shims
   git rm src/imgcolorshine/trans_numba.py
   git rm src/imgcolorshine/falloff.py
   ```

2. **Extracting functions to fast_mypyc**  
   For each function to extract:
   ```python
   # In fast_mypyc/engine_helpers.py
   """Pure Python engine helper functions for mypyc compilation."""
   # this_file: src/imgcolorshine/fast_mypyc/engine_helpers.py
   
   def blend_pixel_colors(pixel_colors, weights):
       """Blend multiple color values with weights."""
       # Move implementation here
   ```
   
   Then in original module:
   ```python
   # In engine.py
   from .fast_mypyc.engine_helpers import blend_pixel_colors
   ```

3. **Update imports for moved Numba code**  
   ```python
   # In engine.py
   # Old: from .numba_utils import compute_color_distances_batch
   # New:
   from .fast_numba.utils import compute_color_distances_batch
   ```

4. **Guarded optional imports with fallbacks**  
   ```python
   # In public modules
   try:
       from .fast_mypyc.engine_helpers import blend_pixel_colors as _blend_fast
   except ImportError:
       from .engine_pure import blend_pixel_colors as _blend_fast
       logger.warning("mypyc speedups not available, using pure Python")
   
   blend_pixel_colors = _blend_fast
   ```

5. **Update pyproject.toml for mypyc**  
   ```toml
   [tool.mypyc]
   packages = ["imgcolorshine.fast_mypyc"]
   exclude = ["imgcolorshine/fast_mypyc/__pycache__"]
   opt_level = "3"
   ```

6. **Other speed improvements**
   - Create SIMD-optimized versions using numpy's vectorization
   - Profile and optimize memory allocation patterns
   - Consider caching frequently computed values (color distances)
   - Implement parallel tile processing for large images

---

### Acceptance criteria

- [ ] All unit tests pass (`python -m pytest`) both with and without optional speedups.
- [ ] Benchmark shows ≥1.5 × speed gain in default configuration.
- [ ] No public import path breaks (CI fails otherwise).

---

## Implementation Order (Task 2)

### Immediate Actions (Phase 1: Numba Consolidation)
1. - [x] Move `numba_utils.py` → `fast_numba/utils.py`
2. - [x] Extract `_transform_pixels_numba` from `engine.py` → `fast_numba/engine_kernels.py`
3. - [x] Remove deprecated shims (`trans_numba.py`, `falloff.py` in root)
4. - [x] Update all imports in affected modules
5. - [ ] Verify tests still pass (NOTE: Many existing test failures unrelated to refactoring)

### Next Actions (Phase 2: Mypyc Setup)
1. - [x] Create `fast_mypyc/engine_helpers.py` and extract 3 functions from `engine.py`
2. - [x] Create `fast_mypyc/gamut_helpers.py` and extract 4 functions from `gamut.py`
3. - [x] Create `fast_mypyc/colorshine_helpers.py` and extract 2 functions from `colorshine.py`
4. - [ ] Update imports with try/except fallback patterns
5. - [x] Configure pyproject.toml for mypyc compilation (already configured)

### Final Actions (Phase 3: Testing & Polish)
1. - [ ] Run full test suite with and without speedups
2. - [ ] Create performance benchmark script
3. - [ ] Update documentation
4. - [ ] Update CHANGELOG.md

