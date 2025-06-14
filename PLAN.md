# PLAN

This document outlines a comprehensive plan to address the issues identified in `TODO.md` and the provided research documents. It synthesizes the analysis of the current codebase, proposes a revised and detailed specification, and lays out an implementation roadmap to make `imgcolorshine` a robust, intuitive, and high-performance tool.

## 0. Executive Summary

The `imgcolorshine` tool is failing to produce expected visual results due to a **critical algorithmic flaw** in its tolerance calculation, which quadratically and unintentionally shrinks the influence of color attractors. The immediate priority is to correct this flaw.

This plan proposes a **Unified Specification** that rectifies the algorithm, refines the user-facing parameters for intuitive control, and redesigns the CLI for better usability. The implementation strategy focuses on a correct, gamma-aware processing pipeline, followed by significant performance optimizations using vectorization, Numba, and optional GPU acceleration via CuPy. Finally, a phased rollout plan introduces new features like a debug visualization mode and "repulsors" to expand the tool's creative capabilities.

## 1. Problem Diagnosis & Analysis

The investigation into why `example.sh` produces underwhelming results reveals a primary bug compounded by a misunderstanding of the transformation's nature.

### 1.1. The Core Algorithmic Flaw: Quadratic Tolerance Mapping

The single greatest issue is in `src/imgcolorshine/transforms.py`:

```python
# The flawed line in calculate_weights()
delta_e_max = 1.0 * (tolerances[i] / 100.0) ** 2
```

This formula squares the normalized tolerance, which dramatically and non-intuitively reduces the attractor's effective radius.

**Impact Analysis:**
- A `tolerance` of **100** becomes `delta_e_max = 1.0`.
- A `tolerance` of **80** (used in the example) becomes `delta_e_max = 0.64`.
- A `tolerance` of **50** becomes `delta_e_max = 0.25`.
- A `tolerance` of **20** becomes a tiny `delta_e_max = 0.04`.

The perceptual distance (ΔE) in Oklab space between two distinct colors can easily be greater than 1.0. By squaring the tolerance, only colors that are *already very close* to the attractor are affected. The blue jacket in `louis.jpg` is perceptually further from pure "blue" than the `0.64` ΔE radius, so it receives zero influence.

**Solution:** The relationship must be linear. The `tolerance` parameter should directly represent a percentage of a maximum reasonable perceptual distance.

### 1.2. The Subtlety of Hue-Only Transformations

The example script uses `--luminance=False --saturation=False`. This constrains the transformation to the hue channel only. For colors with very low chroma (saturation), such as grays, blacks, whites, and muted tones, changing the hue has a negligible visual effect. A gray pixel remains gray regardless of its hue angle. This explains why large parts of the image, especially those with low saturation, show no change. This is not a bug but a correct outcome of the user's command that needs to be better communicated.

### 1.3. User Experience & Parameter Intuition

The current model is a "pull" or "attraction" model, not a "replace" model. A `strength` of 100 does not guarantee a full color replacement for all pixels within tolerance; it only does so for a pixel with the exact same color as the attractor (where distance is 0 and falloff has no effect). The effect weakens with distance. This distinction is crucial and the CLI and documentation must make it clear.

## 2. The New Unified Specification

This specification synthesizes the best approaches from the research documents, focusing on correctness, clarity, and performance.

### 2.1. High-Level Concept

`imgcolorshine` transforms image colors by treating user-defined **attractors** as "color gravity" sources. Each pixel is pulled towards influential attractors in the perceptually uniform **OKLCH** color space. The degree of this pull is a function of its initial color similarity (`tolerance`), a smooth `falloff` curve, and the user-defined `strength`.

### 2.2. Command-Line Interface (CLI) Redesign

The current `fire` CLI is functional but limited. We will migrate to `click` for better validation, help text generation, and extensibility.

```bash
imgcolorshine shine INPUT_IMAGE [OPTIONS] [ATTRACTORS]...

Options:
  --output, -o PATH        Output image path. [default: auto-generated]
  --lum / --no-lum         Enable/disable lightness channel. [default: on]
  --sat / --no-sat         Enable/disable saturation (chroma) channel. [default: on]
  --hue / --no-hue         Enable/disable hue channel. [default: on]
  --falloff [cosine|linear|gaussian]
                           Attraction falloff curve. [default: cosine]
  --idw-power FLOAT        Inverse Distance Weighting power. [default: 2.0]
  --debug-mask             Output a grayscale mask of weights instead of a
                           color image.
  --verbose, -v            Enable verbose logging.

Attractors:
  One or more attractor strings in the format:
  "css_color;tolerance;strength"
  "repulsor:css_color;tolerance;strength"
```

### 2.3. The Core Transformation Model (The Math)

#### 2.3.1. Attractor Primitive
- **Attractor:** `css_color;tolerance;strength` (e.g., `"blue;50;80"`)
- **Repulsor:** `repulsor:css_color;tolerance;strength` (e.g., `"repulsor:red;30;60"`)

#### 2.3.2. Tolerance Mapping (FIXED)
The user's `tolerance` (0-100) will be mapped **linearly** to a maximum perceptual distance (ΔEmax). A `MAX_DELTA_E` constant will be defined to represent a large but reasonable distance in Oklab space (e.g., 2.5).

`delta_e_max = MAX_DELTA_E * (tolerance / 100.0)`

A `tolerance` of 100 will now cover a very wide perceptual range, ensuring it affects most colors as intuitively expected.

#### 2.3.3. Attraction Falloff
The influence of an attractor will diminish with distance according to a selectable falloff function.
1.  Calculate normalized distance: `d_norm = delta_e / delta_e_max` (for `delta_e <= delta_e_max`)
2.  Apply falloff function `f(d_norm)` (e.g., `f_cosine(x) = 0.5 * (cos(x * pi) + 1.0)`).
3.  This will be user-configurable via the `--falloff` flag.

#### 2.3.4. Multi-Attractor/Repulsor Blending
We will use a **Normalized Weighted Average** approach.
1.  For each pixel, calculate the raw weight `w_i` for each attractor `i`:
    `w_i = (strength_i / 100.0) * f(d_norm_i)`
2.  For repulsors, the weight is negative: `w_i = -1 * (strength_i / 100.0) * f(d_norm_i)`
3.  Sum all weights: `W_total = Σ w_i`.
4.  Calculate the final blended color `P_final` as a weighted average in Oklab space. The source color's weight `w_src` is `1 - |W_total|`, clamped at 0.
    `P_final_vector = w_src * P_src_vector + Σ (w_i * (C_attri_vector - P_src_vector))`
    This formula represents pulling the source pixel vector towards (or pushing it away from) the attractor vectors.

#### 2.3.5. Channel-Specific Application
The `--lum`, `--sat`, `--hue` flags will act as masks on the final transformation vector. If a channel is disabled, its component in the final blended vector will be reverted to the source pixel's original value.

## 3. The End-to-End Implementation Pipeline

The processing pipeline must be strictly followed to ensure colorimetric accuracy.

1.  **Argument Parsing:** Parse CLI args using `click`.
2.  **Attractor Initialization:** Parse attractor strings, convert CSS colors to internal Oklab representations.
3.  **Image Loading:** Load image using OpenCV (with PIL fallback).
4.  **Gamma Decoding (sRGB -> Linear RGB):** **CRITICAL STEP.** All calculations must be performed in a linear color space.
5.  **Color Space Conversion (Linear RGB -> Oklab):** Convert the entire image buffer.
6.  **Per-Pixel Transformation Loop:** Execute the blending algorithm from Section 2.3. This loop will be JIT-compiled with Numba or run on a GPU with CuPy.
7.  **Color Space Conversion (Oklab -> Linear RGB):** Convert the transformed buffer back.
8.  **Gamut Clipping:** Map out-of-gamut colors back to the sRGB gamut. The recommended method is to preserve L and H while reducing C (Chroma) until the color fits.
9.  **Gamma Encoding (Linear RGB -> sRGB):** Apply the sRGB gamma curve to the final buffer.
10. **Image Saving:** Save the final buffer to the output file.

## 4. Technology & Performance Strategy

### 4.1. Core Libraries
- **CLI:** `click` (replaces `fire`)
- **Color Science:** `coloraide` (for parsing and reference conversions)
- **Image I/O:** `opencv-python` (primary), `Pillow` (fallback)
- **Computation:** `numpy` (base), `numba` (CPU JIT), `cupy` (optional GPU)

### 4.2. CPU Performance (Numba)
The core `transform_pixels` function will remain decorated with `@numba.njit(parallel=True)`. We will ensure that all helper functions called within it (`calculate_weights`, `blend_colors`) are also Numba-compatible to avoid performance penalties from dropping back to Python mode.

### 4.3. GPU Acceleration (CuPy)
A parallel implementation of the transformation kernel will be created using `cupy`. The application will dynamically detect if a CUDA-enabled GPU and `cupy` are available. If so, it will use the GPU backend; otherwise, it will fall back to the Numba CPU backend. This provides "radically faster" performance for users with compatible hardware.

### 4.4. Memory Management
The existing tiled processing approach in `src/imgcolorshine/utils.py` for large images is sound and will be retained.

## 5. New & Enhanced Features

### 5.1. Debug & Visualization Mode
The `--debug-mask` flag will generate a grayscale image where pixel intensity represents the total absolute weight (`abs(W_total)`) from all attractors/repulsors. This will be an invaluable tool for users to visually debug how their `tolerance` and `falloff` settings are affecting the image.

### 5.2. Negative Attractors (Repulsors)
The blending logic will be extended to support "repulsors", which push colors *away* from a target instead of pulling them closer. This doubles the creative potential of the tool.

### 5.3. Enhanced Gamut Handling
The gamut mapping will strictly follow the CSS Color Module 4 algorithm (preserve L/H, clip C), which is a perceptually superior method to simple RGB clamping.

## 6. Detailed Implementation Plan (Phased Rollout)

### Phase 1: Critical Fixes & Core Refactor (Highest Priority)
1.  **Fix Tolerance:** Implement the linear tolerance mapping in `calculate_weights`.
2.  **Refactor CLI:** Replace `fire` with `click` and implement the new CLI structure defined in Sec 2.2.
3.  **Improve Logging:** Use `loguru` to provide clear feedback on parameters, chosen backend (CPU/GPU), and processing steps.
4.  **Update Documentation:** Revise `README.md` to explain the fixed parameters and the "pull" vs. "replace" concept.
5.  **Test:** Create unit tests that specifically validate the new tolerance logic with known color distances.

### Phase 2: Performance & Pipeline Optimization
1.  **Gamma-Correct Pipeline:** Refactor the `transform_image` function to strictly follow the gamma-correct pipeline (Linearize -> Oklab -> Transform -> Gamut Clip -> Gamma Encode).
2.  **Vectorize Color Conversions:** Replace per-pixel `coloraide` conversions in the main loop with pre-calculated, vectorized NumPy operations for the entire image buffer where possible.
3.  **Optimize Numba Kernels:** Profile and optimize the Numba-jitted functions for maximum CPU performance.

### Phase 3: Advanced Features & GPU Support
1.  **Implement GPU Backend:** Create the `cupy` version of the transformation kernel and the dynamic dispatcher to choose between CPU and GPU.
2.  **Implement Debug Mask:** Add the `--debug-mask` functionality.
3.  **Implement Repulsors:** Extend the blending logic to handle negative weights.
4.  **Implement Falloff Selection:** Add the `--falloff` option and corresponding logic.

## 7. Conclusion

This plan addresses the critical flaws in `imgcolorshine` while laying a clear foundation for its evolution into a powerful, performant, and intuitive color transformation tool. By prioritizing the algorithmic fix and then systematically rolling out performance and feature enhancements, we can align the codebase with its objectives and deliver a uniquely valuable utility for artists and developers.