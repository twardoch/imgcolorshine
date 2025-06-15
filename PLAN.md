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

- [x] **2.1.2.1. Analyze current implementation**
- [x] **2.1.2.2. Vectorize weight calculation**
- [x] **2.1.2.3. Vectorize blending operation**
- [x] **2.1.2.4. Remove per-pixel loops**
- [x] **2.1.2.5. Verify performance improvement (>5x speedup)**

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

### 3.1. **Objective**: Compile non-Numba-friendly Python modules into C extensions for a ~1.5-2x speedup and reduced interpreter overhead.

### 3.2. **Target 1: Migrate Build System from `setup.py` to `pyproject.toml` + Hatch**

- [x] **3.2.1. Delete `setup.py` and `build_ext.py`**
- [x] **3.2.2. Move mypy config from `mypy.ini` to `pyproject.toml`**
- [x] **3.2.3. Configure mypyc modules in `pyproject.toml`**
- [x] **3.2.4. Configure Hatchling build hook**
- [ ] **3.2.5. Verify `uv run hatch build` creates wheels with `.so`/`.pyd` files**

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

### 3.3. **Target 2: Guarded Imports for Development Mode**

- [ ] **3.3.1. Implement try/except ImportError blocks**
- [ ] **3.3.2. Refactor modules to separate pure Python fallbacks**
- [ ] **3.3.3. Test editable install functionality**
- [ ] **3.3.4. Test wheel install uses compiled extensions**

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

### 4.1. **Objective**: Improve code structure, readability, and maintainability.

### 4.2. **Target 1: Refactor `trans_numba.py`**

- [ ] **4.2.1. Add comment blocks to delineate sections**
- [ ] **4.2.2. Rename internal helpers with underscore prefix**
- [ ] **4.2.3. Group related functions logically**

-   **Problem**: This file is a long, flat list of functions. It's functional but lacks structure.
-   **Proposal**: Group related functions into logical, private `numba.experimental.jitclass` instances or simply better-named internal functions. For instance, group all sRGB↔Linear functions, then XYZ↔LMS, etc. While we can't use standard Python classes with `@njit` methods in the same way, we can organize the file better. The current single-file approach is correct, but internal organization can be improved with comments and private helper functions.
-   **Action**: Add comment blocks to delineate sections: `sRGB <> Linear RGB`, `XYZ <> LMS`, `Oklab <> LMS`, `OKLCH <> Oklab`, `Gamut Mapping`. Review `matrix_multiply_3x3` and rename to `_matmul_3x3` to signal it's an internal, unrolled helper.

### 4.3. **Target 2: Eliminate Legacy Aliases**

- [ ] **4.3.1. Search for legacy alias usage**
- [ ] **4.3.2. Replace with canonical names**
- [ ] **4.3.3. Delete alias assignments**
- [ ] **4.3.4. Run full test suite**

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
- [ ] **6.1.1. Verify `build_ext.py` is removed**
- [ ] **6.1.2. Ensure `setup.py` doesn't exist**

### 6.2. Developer-Only Helper Scripts
- [ ] **6.2.1. Remove `testdata/example.sh`, `testdata/quicktest.sh`**
- [ ] **6.2.2. Remove `cleanup.sh`**
- [ ] **6.2.3. Remove debug scripts in `tests/debug_*`**

### 6.3. Legacy / Compatibility Code Paths
- [ ] **6.3.1. Drop JAX support from `gpu.py`**
- [ ] **6.3.2. Remove `ArrayModule.backend=='cpu'` indirection**
- [ ] **6.3.3. Remove old per-pixel kernel and commented code**
- [ ] **6.3.4. Remove mentions of `old/imgcolorshine/`**

### 6.4. Low-Value Tests
- [ ] **6.4.1. Remove JAX/cpu fallback tests**
- [ ] **6.4.2. Remove duplicate CLI tests**
- [ ] **6.4.3. Remove redundant debug tests**

### 6.5. Generated / Packed Files
- [ ] **6.5.1. Add `llms.txt` to `.gitignore`**
- [ ] **6.5.2. Remove `.giga/`, `.cursor/` from repo**
- [ ] **6.5.3. Keep coverage artifacts in `.gitignore` only**

### 6.6. Documentation Stubs
- [ ] **6.6.1. Remove empty docs pages**
- [ ] **6.6.2. Remove placeholder markdown files**

> **Action**: create a one-time pruning commit that deletes the above paths and strips code branches in a single shot.  Follow with a `ruff --fix` & `mypy` pass to ensure no dangling imports.

---