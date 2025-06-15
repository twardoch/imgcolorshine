# PLAN: Engineering Roadmap (v2025-06-16)
# this_file: PLAN.md

Implement the tasks described below. 

Work tirelessly, without asking me any questions, until the job is completely. 

## 1. Purpose

This document provides a **deeply technical execution roadmap** for evolving `imgcolorshine` from a functional prototype into a production-grade, high-performance library. It is targeted at a senior developer familiar with Python's performance ecosystem (Numba, mypyc, packaging).

The plan is broken into atomic, idempotent work packages. Each can be actioned and merged independently to minimize integration risk.

---

## 2. Performance Optimization Strategy

The core performance drag is per-pixel processing in Python. Our strategy is two-pronged:
1.  **Numba (`@njit`)**: For all **array-oriented, numerical hot-loops**. This is our primary weapon.
2.  **Mypyc (AOT Compilation)**: For **algorithmically complex Python code** with lots of function calls, branching, and object manipulation that Numba cannot handle efficiently (or at all).

### 2.1. Numba Vectorization & Kernel Fusion

#### 2.1.1. **Objective**: Eliminate per-pixel Python-level loops inside JIT-compiled functions.

#### 2.1.2. **Target 1: Vectorize `_transform_pixels_percentile` in `engine.py`**

- [ ] **2.1.2.1. Analyze current implementation**

-   **Current State**: The function iterates over `H x W` pixels, calling helper functions for each one. This is suboptimal as it prevents SIMD vectorization across the image.

    ```python
    # src/imgcolorshine/engine.py:160 (Current)
    # @numba.njit(parallel=True)
    def _transform_pixels_percentile(...):
        h, w = pixels_lab.shape[:2]
        result = np.empty_like(pixels_lab)
        for y in numba.prange(h): # Still a Python-style loop
            for x in range(w):
                # Sub-optimal: calculates weights for one pixel at a time
                weights = _calculate_weights_percentile(pixels_lab[y, x], ...) 
                result[y, x] = blend_pixel_colors(...)
        return result
    ```

-   **Proposed Implementation**:
    1.  **Vectorize Weight Calculation**: Rewrite `_calculate_weights_percentile` to operate on the entire `pixels_lab` image `(H, W, 3)` instead of a single pixel `(3,)`. Use NumPy broadcasting to compute all distances and weights in a single, vectorized operation.

        ```python
        # Proposed new function in engine.py
        @numba.njit(parallel=True, fastmath=True)
        def _calculate_all_weights(pixels_lab, attractors_lab, delta_e_maxs, strengths):
            # pixels_lab: (H, W, 3), attractors_lab: (A, 3)
            # Reshape for broadcasting
            pixels_flat = pixels_lab.reshape(-1, 3) # (N, 3) where N=H*W
            
            # Compute distances for all pixels against all attractors at once
            # (N, 1, 3) - (1, A, 3) -> (N, A, 3)
            delta_vectors = pixels_flat[:, np.newaxis, :] - attractors_lab[np.newaxis, :, :]
            delta_e_sq = np.sum(delta_vectors**2, axis=2) # (N, A)
            delta_e = np.sqrt(delta_e_sq)

            # Mask pixels outside tolerance
            # delta_e_maxs is (A,), broadcasting to (N, A)
            mask = delta_e < delta_e_maxs 
            
            # ... rest of falloff and strength logic applied to the (N, A) weight matrix ...
            
            # Final weights array will have shape (N, A)
            return weights.reshape(pixels_lab.shape[0], pixels_lab.shape[1], -1)
        ```

    2.  **Vectorize Blending**: The `blend_pixel_colors` function also needs to be vectorized to accept the `(H, W, A)` weights matrix and the full `pixels_lch` image, performing the blend with NumPy array operations instead of a loop. The hue blend requires careful handling of the trigonometric functions on arrays.

-   **Acceptance Criteria**:
    -   The `for y in numba.prange(h):` loop inside `_transform_pixels_percentile` is completely removed.
    -   The function body consists of a handful of vectorized NumPy calls.
    -   `scalene` profiling shows time spent is now deep inside NumPy's C code, not the function itself.
    -   Benchmark shows > 5x speedup on a 4K image.

---

## 3. Build System & AOT with Mypyc

#### 3.0.1. **Objective**: Compile non-Numba-friendly Python modules into C extensions for a ~1.5-2x speedup and reduced interpreter overhead.

#### 3.0.2. **Target 1: Migrate Build System from `setup.py` to `pyproject.toml` + Hatch**

-   **Current State**: A legacy `setup.py` exists, which complicates modern packaging workflows and mypyc integration. `build_ext.py` is an unused artifact. `mypy.ini` is separate.
-   **Proposed Implementation**:
    1.  **Delete `setup.py` and `build_ext.py`**. They are obsolete.
    2.  **Consolidate Mypy Config**: Move all settings from `mypy.ini` into `pyproject.toml` under the `[tool.mypy]` section. Enable stricter checks required by mypyc.

        ```toml
        # pyproject.toml
        [tool.mypy]
        python_version = "3.10"
        warn_return_any = true
        disallow_untyped_defs = true # Crucial for mypyc
        # ... etc
        ```
    3.  **Configure Mypyc in `pyproject.toml`**: Define the modules to be compiled. `gamut.py` and `utils.py` are prime candidates because their logic (string handling, dictionary lookups, complex branching) is not suitable for Numba.

        ```toml
        # pyproject.toml
        [tool.mypyc]
        modules = [
            "imgcolorshine.gamut",
            "imgcolorshine.utils",
            # Add more modules here as identified
        ]
        opt_level = "3" # Aggressive optimization
        strip_asserts = true
        ```
    4.  **Configure Hatchling Build Hook**: Use Hatch's native capabilities to run mypyc during the build process. This is the modern replacement for `setup.py`'s `ext_modules`.

        ```toml
        # pyproject.toml
        [tool.hatch.build.hooks.mypyc]
        dependencies = ["hatch-mypyc"]
        ```

-   **Acceptance Criteria**:
    -   `uv run hatch build` successfully creates a wheel containing `.so`/`.pyd` extension modules for the specified files.
    -   The project can be installed from the built wheel and runs correctly.
    -   `setup.py`, `build_ext.py`, and `mypy.ini` are deleted.

#### 3.0.3. **Target 2: Guarded Imports for Development Mode**

-   **Current State**: In a development (editable) install, the compiled modules won't exist. Direct imports would fail.
-   **Proposed Implementation**: Use a `try...except ImportError` block to create a fallback mechanism. The compiled module is preferred, but the pure-python version is used if it's not found.

    ```python
    # Example in src/imgcolorshine/gamut.py
    try:
        # mypyc places compiled code in a parallel _compiled module
        from imgcolorshine._compiled.gamut import compiled_helper
    except ImportError:
        # This is the path taken in editable mode
        from ._pure_python_gamut import compiled_helper as python_helper
        compiled_helper = python_helper # Assign to the same name
    ```
    This requires refactoring the core logic of `gamut.py` into `_pure_python_gamut.py` so the main file can contain this import logic without circular dependencies.

-   **Acceptance Criteria**:
    -   `uv pip install -e .` works without errors.
    -   The application runs correctly in editable mode, using the pure-python code paths.
    -   When installed from a wheel, the application uses the faster, compiled C extensions.

---

## 4. Codebase Beautification & Refinement

#### 4.0.1. **Objective**: Improve code structure, readability, and maintainability.

-   **Target 1: Refactor `trans_numba.py`**
    -   **Problem**: This file is a long, flat list of functions. It's functional but lacks structure.
    -   **Proposal**: Group related functions into logical, private `numba.experimental.jitclass` instances or simply better-named internal functions. For instance, group all sRGB↔Linear functions, then XYZ↔LMS, etc. While we can't use standard Python classes with `@njit` methods in the same way, we can organize the file better. The current single-file approach is correct, but internal organization can be improved with comments and private helper functions.
    -   **Action**: Add comment blocks to delineate sections: `sRGB <> Linear RGB`, `XYZ <> LMS`, `Oklab <> LMS`, `OKLCH <> Oklab`, `Gamut Mapping`. Review `matrix_multiply_3x3` and rename to `_matmul_3x3` to signal it's an internal, unrolled helper.

-   **Target 2: Eliminate Legacy Aliases**
    -   **Problem**: `trans_numba.py` contains `srgb_to_oklab_batch = batch_srgb_to_oklab`. This is technical debt from earlier refactoring.
    -   **Action**:
        1.  Globally search for `srgb_to_oklab_batch` and `oklab_to_srgb_batch`.
        2.  Replace all usages with the canonical names (`batch_srgb_to_oklab`, `batch_oklab_to_srgb`).
        3.  Delete the alias assignments from the bottom of `trans_numba.py`.
        4.  Run the full test suite (`pytest tests/`) to ensure no breakages.

---
## 5. Timeline & Definition of Done (Revised)

| Week | Deliverable                                           | Key Result                                          |
| :--- | :---------------------------------------------------- | :-------------------------------------------------- |
| **1**  | Numba Vectorization                                   | `_transform_pixels_percentile` is loop-free.        |
| **2**  | Build System Migration & Mypy Strict Pass             | `hatch build` works; `mypy --strict` passes.        |
| **3**  | Mypyc Compilation & Guarded Imports                   | Wheels contain `.so` files; editable install works. |
| **4**  | Codebase Beautification & Alias Removal               | `trans_numba.py` is cleaner; aliases are gone.      |
| **5**  | Final Benchmarking & Documentation Update             | `CHANGELOG.md` and `README.md` reflect all changes. |

### 5.1. Definition of Done:
-   [ ] **Performance**: Numba-vectorized functions show >5x speedup. Mypyc-compiled modules show >1.5x speedup.
-   [ ] **Build**: `hatch build` is the sole mechanism for creating distributable packages.
-   [ ] **Code Quality**: `mypy --strict` passes. No legacy aliases remain.
-   [ ] **Documentation**: `README.md` and `CHANGELOG.md` are fully updated. `PLAN.md` is marked as complete.

---

## 6. Lean-Down Cleanup Candidates  🚯

_A pragmatic, performance-first audit of the repository._

The items below **provide no production value** after the refactor (Numba + mypy-strict, no legacy fallbacks) and can be **deleted outright** or **stripped from runtime paths**.  They are kept only for historical context or super-narrow debugging scenarios that are now obsolete.

### 6.1. Obsolete Build / Packaging Artifacts
- `build_ext.py` — removed already; legacy setup-py bridge for mypyc.
- `setup.py` — does not exist anymore; ensure it never re-appears.

### 6.2. Developer-Only Helper Scripts
- `testdata/example.sh`, `testdata/quicktest.sh` — ad-hoc demo scripts.
- `cleanup.sh` — CI/lint shortcut superseded by `hatch fix` and GitHub Actions.
- Everything in `tests/debug_*` (`debug_color_distances.py`, `debug_transformation.py`, etc.).
- One-shot notebooks or scratch pads that creep in the future.

### 6.3. Legacy / Compatibility Code Paths
- **GPU Fallback Layers** in `src/imgcolorshine/gpu.py`:
  - Drop JAX support (`_check_jax_available`, JAX branches) ⇒ CuPy-only path.
  - Remove `ArrayModule.backend=='cpu'` indirection — we always call NumPy directly when GPU disabled.
- **Old per-pixel kernel** `_calculate_weights_percentile()` and the commented `@numba.njit` decorator lines in `engine.py`.
- Any mention of `old/imgcolorshine/` in docstrings (pure documentation rot).
- Commented-out alias blocks (`srgb_to_oklab_batch = …`) — already excised.

### 6.4. Low-Value Tests
Unit tests that solely exercised _removed_ compatibility shims can be trimmed to speed CI:
- Tests gating JAX/cpu fallbacks in `tests/test_gpu.py` (`test_array_module_jax_*`, etc.).
- `tests/test_cli_simple.py` duplicates coverage of `tests/test_cli.py`.
- Redundant "debug" tests (`tests/simple_debug.py`, `tests/debug_*`).

### 6.5. Generated / Packed Files
- `llms.txt`  — massive concatenation for LLM ingestion; not needed in wheel/source dist.
- `.giga/`, `.cursor/` auxiliary metadata.
- Coverage artefacts (`htmlcov/`, `coverage.xml`) — keep in `.gitignore` only.

### 6.6. Documentation Stubs
- Any empty `docs/` pages or placeholder markdown (e.g. `COVERAGE_REPORT.md` once pipeline reports are automated).

> **Action**: create a one-time pruning commit that deletes the above paths and strips code branches in a single shot.  Follow with a `ruff --fix` & `mypy` pass to ensure no dangling imports.

---
