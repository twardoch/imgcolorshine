# Development Plan for imgcolorshine

The `cleanup.log` log is revealing. It shows that the codebase is in a partially broken state after what appears to have been a significant but incomplete refactoring. There are missing modules (`transform`, `hierar`, `spatial`), broken tests, and a multitude of type errors and linter warnings.

This plan will first stabilize the code, then clean it up, and finally implement a series of aggressive optimizations to achieve your performance goals.

## High-Level Strategy

1. [x] **Phase 0: Triage and Repair** - Fix critical errors to get the tests running.
2. [x] **Phase 1: Code Cleanup and Refactoring** - Address all linter warnings and type errors for a maintainable and readable codebase.
3. [x] **Phase 2: Core Logic Consolidation** - Unify the transformation logic and remove defunct code paths.
4. [x] **Phase 3: Aggressive Performance Optimization** - Implement a multi-pronged strategy using fused kernels, GPU acceleration, and Look-Up Tables (LUTs).
5. [x] **Phase 4: Build System and Packaging** - Enable MyPyc compilation to generate high-performance binary wheels.
6. [x] **Phase 5: Final Validation and Documentation** - Ensure all changes are tested and documented.

---

## 1. Phase 0: Triage and Repair - Stabilize the Codebase

The immediate goal is to fix the `pytest` collection errors and get a clean test run.

### 1.1. [x] Resolve Missing `imgcolorshine.transform` Module

The file `tests/test_tolerance.py` imports from a non-existent `imgcolorshine.transform`. The logic it intends to test (`calculate_weights`) seems to have been replaced by the percentile-based model in `engine.py`.

**Action:** Delete the obsolete test file. We will write a new, correct test for the current engine logic later.
```bash
rm tests/test_tolerance.py
```

### 1.2. [x] Resolve Missing `is_in_gamut_srgb` Import

The test `tests/test_gamut.py` fails because `imgcolorshine.gamut` cannot import `is_in_gamut_srgb` from `trans_numba.py`. Looking at `trans_numba.py`, the function exists but is named `_in_gamut` and is not intended for external use. The gamut checking logic is intertwined with the conversions.

**Action:** Refactor `trans_numba.py` to expose a proper gamut checking function. Rename `_in_gamut` to `is_in_gamut_srgb` and remove the leading underscore to make it public.

```diff
# In: src/imgcolorshine/trans_numba.py

@numba.njit(cache=True)
-def _in_gamut(r_lin: float, g_lin: float, b_lin: float) -> bool:
-    return bool((0 <= r_lin <= 1) and (0 <= g_lin <= 1) and (0 <= b_lin <= 1))
+def is_in_gamut_srgb(rgb: np.ndarray) -> bool:
+    """Checks if a single sRGB color is within the [0, 1] gamut."""
+    return bool((0.0 <= rgb[0] <= 1.0) and (0.0 <= rgb[1] <= 1.0) and (0.0 <= rgb[2] <= 1.0))


@numba.njit(cache=True)
def gamut_map_oklch_single(oklch: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    L, C, H = oklch
    oklab = oklch_to_oklab_single(oklch)
    rgb = oklab_to_srgb_single(oklab)
-    if _in_gamut(rgb[0], rgb[1], rgb[2]):
+    if is_in_gamut_srgb(rgb):
        return oklch
    c_lo, c_hi = 0.0, C
    while c_hi - c_lo > eps:
        c_mid = 0.5 * (c_lo + c_hi)
        test_oklch = np.array([L, c_mid, H], dtype=np.float32)
        test_rgb = oklab_to_srgb_single(oklch_to_oklab_single(test_oklch))
-        if _in_gamut(test_rgb[0], test_rgb[1], test_rgb[2]):
+        if is_in_gamut_srgb(test_rgb):
            c_lo = c_mid
        else:
            c_hi = c_mid
    return np.array([L, c_lo, H], dtype=np.float32)
```

After these changes, your test suite should collect successfully, though many tests will likely still fail.

## 2. Phase 1: Code Cleanup and Refactoring

Address the issues raised by `ruff`, `mypy`, and `ty` to improve code quality.

### 2.1. [x] Fix Boolean Positional Arguments

The `FBT001` and `FBT002` warnings indicate that boolean flags should be keyword-only to avoid ambiguity.

**Action:** Add a `*` in the function signature before the boolean arguments.
```diff
# In: src/imgcolorshine/cli.py
class ImgColorShineCLI:
    def shine(
        self,
        input_image: str,
        *attractors: str,
        output_image: str | None = None,
-        luminance: bool = True,
-        saturation: bool = True,
-        hue: bool = True,
-        verbose: bool = False,
+        *,
+        luminance: bool = True,
+        saturation: bool = True,
+        hue: bool = True,
+        verbose: bool = False,
    ) -> None:
        ...
```    
Apply this to all functions flagged by `ruff`.

### 2.2. [x] Remove Magic Numbers

The `PLR2004` warnings show hardcoded numbers that should be constants for clarity.

**Action:** Define constants at the module level.
```python
# In: src/imgcolorshine/colorshine.py

# Add at the top of the file
ATTRACTOR_PARTS = 3
TOLERANCE_MIN, TOLERANCE_MAX = 0.0, 100.0
STRENGTH_MIN, STRENGTH_MAX = 0.0, 200.0

def parse_attractor(attractor_str: str) -> tuple[str, float, float]:
    parts = attractor_str.split(";")
    if len(parts) != ATTRACTOR_PARTS:
        # ...
    
    if not TOLERANCE_MIN <= tolerance <= TOLERANCE_MAX:
        # ...
    
    if not STRENGTH_MIN <= strength <= STRENGTH_MAX:
        # ...
```

### 2.3. [x] Rename Ambiguous Variables

The `E741` warning for the variable `l` is critical. In color science, `L` is for Lightness. Using a lowercase `l` is confusing and error-prone.

**Action:** Rename all instances of `l` (when it means lightness) to `lightness` or `L`. Since your Numba code already uses `L`, standardize on that for consistency in those functions.

### 2.4. [x] Fix Type Errors

Work through the `mypy` and `ty` logs to fix all type-related issues.
- **`numba.prange`:** Type checkers don't understand that `numba.prange` is iterable. You can silence this with `# type: ignore [attr-defined]` where necessary, after confirming the logic is correct.
- **Missing Stubs:** For libraries like `fire` and `cupy`, you may need to add ` # type: ignore` to the import statement if stubs are unavailable or install them if they are (`pip install types-cupy`).
- **Add missing `-> None`** to functions that don't return anything.

## 3. Phase 2: Core Logic Consolidation

The current logic is fragmented. The `process_with_optimizations` function in `colorshine.py` is dead code since `hierar` and `spatial` modules are missing. The core vectorized transform is in `engine.py`. Let's clean this up.

### 3.1. [x] Remove Dead Code

**Action:** Delete the entire `process_with_optimizations` function from `src/imgcolorshine/colorshine.py`. It's non-functional and adds clutter.

**Action:** Update the `shine` command in `src/imgcolorshine/cli.py` to remove the defunct optimization flags (`fast_hierar`, `fast_spatial`) from the docstring.

### 3.2. [x] Consolidate Transformation Logic

The `ColorTransformer` class in `engine.py` is the correct place for the main algorithm. `colorshine.py` should only be responsible for orchestration.

**Action:** Ensure `colorshine.process_image` does the following and nothing more:
1. Sets up logging.
2. Parses and validates inputs (attractor strings, paths).
3. Initializes `ImageProcessor` and `OKLCHEngine`.
4. Calls `engine.create_attractor` to create attractor objects.
5. Loads the image using `processor.load_image`.
6. Calls `transformer.transform_image` with the image, attractors, and channel flags.
7. Saves the result using `processor.save_image`.

This ensures a clean separation of concerns.

## 4. Phase 3: Aggressive Performance Optimization

This is where we'll achieve the significant speedup. The current vectorized implementation is good, but we can do much better by reducing memory allocation and using more powerful parallelization techniques.

### 4.1. [x] Create a Fused Numba Kernel

The current `_transform_pixels_percentile_vec` in `engine.py` creates several large, intermediate `(N, A)` arrays (`deltas`, `delta_e`, `weights`). A fused kernel processes one pixel at a time through the entire pipeline, keeping intermediate values in CPU registers and improving cache performance.

**Action:** Create a new kernel function in `engine.py` and use it in `transform_image`.

```python
# In: src/imgcolorshine/engine.py

@numba.njit(parallel=True, cache=True)
def _fused_transform_kernel(
    image_lab: np.ndarray,
    attractors_lab: np.ndarray,
    attractors_lch: np.ndarray,
    delta_e_maxs: np.ndarray,
    strengths: np.ndarray,
    flags: np.ndarray,
) -> np.ndarray:
    h, w, _ = image_lab.shape
    transformed_lab = np.empty_like(image_lab)

    for i in numba.prange(h * w):
        y = i // w
        x = i % w
        pixel_lab = image_lab[y, x]

        # --- Inside this loop, all operations are on a single pixel ---

        # 1. Calculate distances and weights (no large intermediate arrays)
        weights = np.zeros(len(attractors_lab), dtype=np.float32)
        for j in range(len(attractors_lab)):
            delta_e = np.sqrt(np.sum((pixel_lab - attractors_lab[j]) ** 2))
            delta_e_max = delta_e_maxs[j]
            if delta_e < delta_e_max and delta_e_max > 0:
                d_norm = delta_e / delta_e_max
                falloff = 0.5 * (np.cos(d_norm * np.pi) + 1.0)
                # ... strength logic ...
                weights[j] = calculated_weight
        
        # 2. Blend colors
        pixel_lch = trans_numba.oklab_to_oklch_single(pixel_lab)
        # ... blending logic for L, C, H using weights ...
        # This is your existing blend_pixel_colors logic, adapted for Numba

        # 3. Convert back to Lab
        final_h_rad = np.deg2rad(final_h)
        final_lab = np.array([
            final_l,
            final_c * np.cos(final_h_rad),
            final_c * np.sin(final_h_rad)
        ], dtype=pixel_lab.dtype)
        
        transformed_lab[y, x] = final_lab
        
    return transformed_lab

# In ColorTransformer.transform_image, replace the call to
# _transform_pixels_percentile_vec with this new kernel.
```

### 4.2. [x] GPU Acceleration (Integrate `gpu.py`)

The `gpu.py` module is present but unused. We can use it for a CuPy-based implementation.

**Action:**
1. Add a `use_gpu: bool = True` flag to the `shine` CLI command and pass it down.
2. In `ColorTransformer.transform_image`, check this flag and GPU availability.
3. If GPU is used, the logic will be very similar to the NumPy vectorized version (`_transform_pixels_percentile_vec`), but using `cupy` as the array module (`xp`). All arrays (image, attractors) must first be moved to the GPU with `xp.asarray()`.

### 4.3. [x] Implement Look-Up Table (LUT) Acceleration

For a given set of attractors, the transformation is deterministic. We can pre-compute it on a 3D grid of colors and then use fast interpolation. This is extremely fast for subsequent runs with the same settings.

**Action:** Create a new `lut.py` module.
1. **`LUTManager` class:**
   - `__init__(cache_dir)`: Sets up a directory to store cached LUTs.
   - `get_lut(attractors, flags)`: Generates a cache key (e.g., SHA256 hash of attractor params). Checks if a cached LUT exists. If not, calls `_build_lut`.
   - `_build_lut(...)`:
     - Creates a 3D grid of RGB colors (e.g., 65x65x65).
     - Runs the *entire* `transform_image` logic on this small grid image.
     - Saves the resulting transformed grid to the cache.
   - `apply_lut(image, lut)`:
     - Uses `scipy.interpolate.interpn` to perform fast trilinear interpolation for every pixel in the input image against the LUT. This is the fastest path.

## 5. Phase 4: Build System and Packaging

Your `pyproject.toml` is already set up for `hatchling` and has a disabled `mypyc` hook. Let's enable it and configure it properly.

### 5.1. [x] Enable and Configure MyPyc

MyPyc compiles typed Python modules into C extensions, removing interpreter overhead for significant speed gains on Python-level logic.

**Action:** Modify `pyproject.toml`.

```toml
[build-system]
requires = [
    'hatchling>=1.27.0',
    'hatch-vcs>=0.4.0',
    'hatch-mypyc>=0.16.0', # Enable this line
]
build-backend = 'hatchling.build'

[tool.hatch.build.hooks.mypyc]
dependencies = ["hatch-mypyc"]
# Only include modules with pure Python/NumPy/ColorAide logic.
# Numba-heavy files should be excluded as MyPyc can't compile them.
files = [
    "src/imgcolorshine/colorshine.py",
    "src/imgcolorshine/engine.py", # MyPyc can compile the Python parts
    "src/imgcolorshine/gamut.py",
    "src/imgcolorshine/io.py",
    "src/imgcolorshine/utils.py",
    "src/imgcolorshine/falloff.py",
]

# Add mypyc options if needed
[tool.mypyc]
opt_level = "3"
strip_asserts = true
```

This will produce faster binary wheels (`.whl` files) that contain compiled `.so` or `.pyd` extensions, which will be used automatically when your package is installed.

## 6. Phase 5: Final Validation and Documentation

### 6.1. [x] Write Correctness Tests

**Action:** Add a new test file, `tests/test_engine.py`, to replace the deleted `test_tolerance.py`. It should specifically test the `transform_image` method with the percentile-based model.
- Create a simple gradient image where you can predict which pixels will be affected.
- Test `tolerance=30` on a 10-pixel image and assert that exactly 3 pixels have changed.
- Test `strength=50` and assert that the most affected pixel is a 50/50 blend of its original color and the attractor color. The existing `test_engine.py` is a great starting point for this.

### 6.2. [x] Update Documentation

**Action:** Update `README.md` and the CLI help text to reflect the new performance features and any new flags (`--use-gpu`, `--use-lut`). Remove references to old, defunct flags.

**Action:** Update the `CHANGELOG.md` to document this massive refactoring and optimization effort.

By following this plan, you will transform `imgcolorshine` into a stable, maintainable, and exceptionally high-performance tool that fulfills the ambitious goals laid out in its documentation.