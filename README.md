# imgcolorshine

**Transform image colors with artistic precision using OKLCH color attractors—a physics-inspired tool operating in a perceptually uniform color space.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/imgcolorshine.svg)](https://badge.fury.io/py/imgcolorshine)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Test Coverage](https://img.shields.io/badge/coverage-50%25-yellow.svg)](#) <!-- TODO: Update coverage badge if possible -->
[![Performance](https://img.shields.io/badge/performance-optimised-brightgreen.svg)](#performance-optimizations)

## Table of Contents

- [What is imgcolorshine?](#what-is-imgcolorshine)
- [Who is it for?](#who-is-it-for)
- [Why is it useful?](#why-is-it-useful)
- [Key Features](#key-features)
- [Installation](#installation)
  - [From PyPI (Recommended)](#from-pypi-recommended)
  - [From Source](#from-source)
  - [Optional Dependencies for GPU Acceleration](#optional-dependencies-for-gpu-acceleration)
- [Quick Start](#quick-start)
  - [Basic Usage](#basic-usage)
  - [Multiple Attractors](#multiple-attractors)
  - [Channel-Specific Transformation](#channel-specific-transformation)
- [Usage Guide](#usage-guide)
  - [Command Structure](#command-structure)
  - [Attractor Format](#attractor-format)
  - [Main Command Options](#main-command-options)
- [How imgcolorshine Works: A Technical Deep Dive](#how-imgcolorshine-works-a-technical-deep-dive)
  - [1. Color Space Conversions: The Foundation](#1-color-space-conversions-the-foundation)
  - [2. The Attractor Model: Tolerance and Strength](#2-the-attractor-model-tolerance-and-strength)
    - [a. Pass 1: Distance Analysis (Tolerance Calculation)](#a-pass-1-distance-analysis-tolerance-calculation)
    - [b. Pass 2: Color Transformation (Strength Application)](#b-pass-2-color-transformation-strength-application)
  - [3. Performance Optimizations](#3-performance-optimizations)
  - [4. Code Architecture Overview](#4-code-architecture-overview)
- [Development & Contribution](#development--contribution)
  - [Setting Up Your Development Environment](#setting-up-your-development-environment)
  - [Coding Guidelines](#coding-guidelines)
  - [Development Workflow & Code Quality](#development-workflow--code-quality)
  - [Contribution Philosophy](#contribution-philosophy)

---

## What is imgcolorshine?

`imgcolorshine` is a sophisticated command-line tool designed for artists, photographers, designers, and anyone looking to creatively manipulate image colors. It uses a unique "color attractor" model: you define specific target colors, and the tool intelligently pulls the existing colors of your image towards these attractors. This process is akin to a gravitational pull, resulting in smooth, organic, and natural-looking transformations.

All color operations are performed in the **OKLCH color space**, which is perceptually uniform. This means that changes to lightness, chroma (saturation), and hue values correspond directly to how humans perceive those qualities, leading to intuitive and visually pleasing results.

## Who is it for?

*   **Photographers:** Enhance moods, correct color casts, or apply creative color grading.
*   **Digital Artists:** Achieve unique color palettes and artistic effects.
*   **Graphic Designers:** Fine-tune imagery to match brand colors or design aesthetics.
*   **Videographers/Animators:** (Via frame-by-frame processing) Apply consistent color transformations across sequences.
*   **Anyone curious about advanced color manipulation!**

## Why is it useful?

`imgcolorshine` stands out due to its:

1.  **Perceptual Accuracy:** Transformations in OKLCH look natural and intuitive.
2.  **Physics-Inspired Model:** "Pull" model creates smooth, organic transitions, not abrupt replacements.
3.  **Fine-Grained Control:** Adjust the "tolerance" (range of influence) and "strength" (intensity) of each attractor.
4.  **Selective Adjustments:** Independently transform lightness, chroma (saturation), and hue.
5.  **Universal Color Input:** Accepts any CSS color format for attractors (e.g., `red`, `#FF0000`, `rgb(255,0,0)`, `oklch(70% 0.2 30)`).
6.  **High Performance:** Optimized with Numba, Mypyc, GPU acceleration (if available), and LUT caching for speed.
7.  **Professional Output:** Includes CSS Color Module 4 compliant gamut mapping to ensure all colors are displayable.
8.  **Memory Efficiency:** Handles large images through robust library usage.

## Key Features

*   ✨ **Perceptually Uniform Color Space:** All operations in OKLCH.
*   🎨 **Universal Color Support:** Accepts any CSS color format for attractors.
*   🎯 **Multi-Attractor Blending:** Seamlessly combines the influence of multiple color attractors.
*   🎛️ **Channel Control:** Transform lightness, chroma, and hue independently.
*   🏎️ **Multiple Acceleration Modes:** CPU (Numba/Mypyc), GPU (CuPy), and LUT-based processing.
*   📊 **Professional Gamut Mapping:** CSS Color Module 4 compliant.
*   💾 **Memory Efficient:** Robust handling of images.

## Installation

### From PyPI (Recommended)

```bash
pip install imgcolorshine
```

### From Source

For the latest developments or if you intend to contribute:

```bash
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine
pip install -e .
```

### Optional Dependencies for GPU Acceleration

If you have an NVIDIA GPU and want to leverage GPU acceleration:

```bash
# For CUDA 11.x
pip install cupy-cuda11x
# For CUDA 12.x
pip install cupy-cuda12x
# Or other versions as per CuPy documentation
```
`imgcolorshine` will automatically detect and use CuPy if it's installed and a compatible GPU is available.

## Quick Start

### Basic Usage

Let's say you have a photo `landscape.jpg` and you want to give it a warmer, more golden-hour feel. You can pull its colors towards an orange hue:

```bash
imgcolorshine shine landscape.jpg "orange;50;75"
```

This command will:
1.  Load `landscape.jpg`.
2.  Define an `orange` color attractor.
3.  Set its `tolerance` to `50`. This means the attractor will influence the 50% of the image's pixels that are most similar in color to orange.
4.  Set its `strength` to `75`. This is a noticeable but natural-looking pull.
5.  Save the transformed image as `landscape_colorshine.jpg` (by default).

### Multiple Attractors

You can use multiple attractors to create more complex effects. For instance, to simulate a sunset with orange and deep red influences:

```bash
imgcolorshine shine photo.jpg \
  "oklch(70% 0.15 40);60;70" \  # A light, warm orange
  "darkred;40;50" \             # A touch of deep red
  --output_image=sunset_effect.jpg
```

### Channel-Specific Transformation

If you only want to change the hue and saturation towards teal, leaving the original lightness intact:

```bash
imgcolorshine shine portrait.jpg "teal;60;80" --luminance=False
```

## Usage Guide

### Command Structure

The basic command structure is:

```bash
imgcolorshine shine INPUT_IMAGE ATTRACTOR1 [ATTRACTOR2 ...] [OPTIONS]
```

*   `INPUT_IMAGE`: Path to the image file you want to transform.
*   `ATTRACTOR1`, `ATTRACTOR2`, ...: One or more attractor definitions.
*   `[OPTIONS]`: Additional flags to control the transformation.

### Attractor Format

Each attractor is a string with three parts, separated by semicolons: `"color;tolerance;strength"`

1.  **`color`**: The target color. Any valid CSS color string can be used:
    *   **Named colors:** `red`, `blue`, `forestgreen`
    *   **Hexadecimal:** `#ff0000`, `#00f`, `#336699cc` (with alpha)
    *   **RGB/RGBA:** `rgb(255,0,0)`, `rgba(0,255,0,0.5)`
    *   **HSL/HSLA:** `hsl(120,100%,50%)`, `hsla(240,100%,50%,0.8)`
    *   **OKLCH/OKLAB:** `oklch(70% 0.2 120)`, `oklab(58% -0.1 0.15)` (OKLCH is recommended for defining attractors as it's the internal working space)

2.  **`tolerance` (0–100)**: This crucial parameter defines the **range of influence** of the attractor.
    *   **Conceptually:** Think of it as casting a net. A small tolerance (e.g., `10`) catches only colors very similar to your attractor. A large tolerance (e.g., `80`) casts a wide net, affecting a broader range of colors.
    *   **Technically:** `tolerance` is a **percentile** of the image's color distribution, based on perceptual distance (ΔE in Oklab space) from the attractor color. `tolerance=50` means the attractor will influence the 50% of the image's pixels that are "closest" in color to the attractor. This makes the tool adaptive to each image's unique palette.

3.  **`strength` (0–200)**: This controls the **intensity** of the transformation for colors within the tolerance range.
    *   **`0–100` (Falloff Mode):** This is the typical range. Colors closest to the attractor are pulled hardest, and the pull's intensity smoothly "falls off" to zero for colors at the edge of the tolerance radius. A `strength` of `50` means that even the most similar colors will only move 50% of the way towards the attractor's color value (for the enabled L, C, H channels).
    *   **`101–200` (Extended Intensity / Duotone-like Mode):** This range progressively flattens the falloff effect. As strength increases beyond 100, colors within the tolerance radius are pulled more uniformly and aggressively. At `strength=200`, every color inside the tolerance net is pulled *completely* to the attractor's color values for the enabled channels, creating a flat, duotone-like effect within that specific color range.

### Main Command Options

| Option             | Type    | Default    | Description                                                                 |
| :----------------- | :------ | :--------- | :-------------------------------------------------------------------------- |
| `--output_image`   | `PATH`  | Auto       | Path for the output image. If not provided, it's auto-generated (e.g., `input_colorshine.jpg`). |
| `--luminance`      | `BOOL`  | `True`     | Enable/disable transformation of the Lightness (L) channel.                 |
| `--saturation`     | `BOOL`  | `True`     | Enable/disable transformation of the Chroma/Saturation (C) channel.         |
| `--hue`            | `BOOL`  | `True`     | Enable/disable transformation of the Hue (H) channel.                       |
| `--verbose`        | `BOOL`  | `False`    | Enable detailed logging for debugging and insight into the process.       |
| `--gpu`            | `BOOL`  | `True`     | Use GPU acceleration (CuPy) if available. Falls back to CPU if not.       |
| `--lut_size`       | `INT`   | `0`        | Size of the 3D Lookup Table (LUT) to build for acceleration. `0` disables LUT. A common size like `65` (meaning a 65x65x65 LUT) is recommended for significant speedups on repeated transformations with the same settings. Note: LUT mode may use a slightly different (faster, less precise for tolerance) transformation path. |
| `--fused_kernel`   | `BOOL`  | `False`    | (Advanced) Use a fused Numba kernel for potentially better CPU performance on some systems by improving cache locality. May not always be faster than the default vectorized CPU path. |

---

## How imgcolorshine Works: A Technical Deep Dive

`imgcolorshine` employs a sophisticated pipeline to achieve its unique color transformations. The core philosophy is a "pull" model rather than a "replace" model, ensuring smooth and natural transitions. All critical computations occur in the perceptually uniform OKLCH/Oklab color spaces.

### 1. Color Space Conversions: The Foundation

Perceptual uniformity is key. The tool uses the following conversion path:

1.  **Input (sRGB) to Linear sRGB:** The input image, typically in sRGB color space, first has its gamma companding undone (gamma decoding) to convert it to linear sRGB. This is crucial because color calculations should be performed in a linear space to be physically accurate.
2.  **Linear sRGB to CIE XYZ:** The linear sRGB values are transformed into the CIE XYZ color space, a device-independent representation.
3.  **CIE XYZ to Oklab:** XYZ values are then converted to Oklab. Oklab is a perceptually uniform color space where Euclidean distance (ΔE) between colors closely corresponds to perceived difference. It has three axes:
    *   `L`: Perceptual Lightness (similar to CIE L*)
    *   `a`: Green-Red axis
    *   `b`: Blue-Yellow axis
    All distance calculations for the "tolerance" mechanism happen in Oklab.
4.  **Oklab to OKLCH:** For intuitive manipulation, Oklab values are converted to OKLCH. OKLCH is the cylindrical representation of Oklab:
    *   `L`: Perceptual Lightness (same as Oklab's L)
    *   `C`: Chroma (saturation, distance from the neutral gray axis)
    *   `H`: Hue angle (e.g., red, yellow, green, blue)
    The actual "pull" transformation (adjusting L, C, H values) happens in OKLCH based on user flags.
5.  **Transformation in OKLCH:** The core algorithm (detailed below) modifies the L, C, and/or H values of pixels based on attractor influences.
6.  **OKLCH back to Oklab:** The transformed OKLCH values are converted back to Oklab.
7.  **Oklab to CIE XYZ:** Transformed Oklab values are converted back to XYZ.
8.  **CIE XYZ to Linear sRGB:** XYZ values are converted back to linear sRGB.
9.  **Gamut Mapping (in Oklab/OKLCH):** Before final conversion to sRGB, if any transformed colors fall outside the sRGB display gamut, they are brought back in. `imgcolorshine` uses a CSS Color Module 4 compliant algorithm: it typically reduces chroma (C in OKLCH) while preserving hue (H) and lightness (L) as much as possible, finding the closest in-gamut color by moving towards the L axis in Oklab.
10. **Linear sRGB to Output (sRGB):** Finally, the linear sRGB values are gamma encoded back to the standard sRGB color space for display and saving.

### 2. The Attractor Model: Tolerance and Strength

The core of `imgcolorshine` lies in its two-pass percentile-based attractor model for each defined attractor:

#### a. Pass 1: Distance Analysis (Tolerance Calculation)

For each attractor, the engine first determines its effective **radius of influence**. This is not a fixed radius but is dynamically calculated based on the image's specific color distribution and the user-provided `tolerance` (0-100) parameter:

1.  **Attractor Color in Oklab:** The attractor's defined color is converted to its Oklab coordinates (`L_attr`, `a_attr`, `b_attr`).
2.  **Per-Pixel Distance (ΔE):** The perceptual distance (ΔE) from this attractor color to *every pixel* in the image (also converted to Oklab) is calculated using the Euclidean distance formula in Oklab space:
    `ΔE_pixel = sqrt((L_pixel - L_attr)² + (a_pixel - a_attr)² + (b_pixel - b_attr)²)`.
3.  **Percentile Calculation:** The `tolerance` value (0-100) is interpreted as a **percentile**. The engine analyzes the distribution of all calculated ΔE values and finds the ΔE value that corresponds to this percentile. For example:
    *   `tolerance = 0`: The radius is effectively zero (only exact matches, though practically no pixels).
    *   `tolerance = 50`: The engine finds the median ΔE value. Half the pixels are closer to the attractor than this median ΔE, and half are further.
    *   `tolerance = 100`: The radius includes the furthest pixel (effectively all pixels are influenced to some degree, though falloff still applies).
4.  **Maximum Influence Radius (ΔE_max):** This percentile-derived ΔE value becomes the `delta_e_max` for the current attractor. Only pixels whose ΔE to the attractor is less than or equal to this `delta_e_max` will be affected by this attractor in the next pass.

This percentile-based approach makes the `tolerance` adaptive to the image content. A `tolerance=30` on a monochrome image will pick a different `delta_e_max` than on a vibrant, colorful image.

#### b. Pass 2: Color Transformation (Strength Application)

Once `delta_e_max` is known for each attractor, the engine transforms the pixel colors:

1.  **Pixel Iteration:** The engine (conceptually) iterates through each pixel again. For each pixel and each attractor:
2.  **Check Influence:** If the pixel's ΔE to the current attractor (calculated in Pass 1) is within that attractor's `delta_e_max`:
    *   **Normalized Distance (`d_norm`):** The pixel's distance to the attractor is normalized relative to `delta_e_max`:
        `d_norm = ΔE_pixel / delta_e_max` (ranges from 0 to 1).
    *   **Falloff Calculation:** A falloff factor is calculated to ensure that pixels closer to the attractor are influenced more strongly. `imgcolorshine` uses a raised cosine falloff:
        `falloff = 0.5 * (cos(d_norm * π) + 1.0)`. This yields a value of 1.0 for `d_norm=0` (pixel is identical to attractor) and 0.0 for `d_norm=1` (pixel is at the edge of `delta_e_max`).
    *   **Strength Application & Weight Calculation:** The user-defined `strength` (0-200) determines the final transformation weight for this pixel-attractor pair.
        *   **For `strength <= 100` (Traditional Falloff Mode):**
            The base interpolation factor is `strength_scaled = strength / 100.0`.
            The final weight for this attractor on this pixel is `weight = strength_scaled * falloff`.
        *   **For `strength > 100` (Extended Intensity / Duotone-like Mode):**
            The falloff effect is progressively flattened. The formula effectively transitions from the falloff-modulated pull to a more uniform pull:
            `base_strength_factor = 1.0` (since `strength` is already > 100)
            `extra_strength_factor = (strength - 100.0) / 100.0` (scales 101-200 to 0.01-1.0)
            `weight = (base_strength_factor * falloff) + (extra_strength_factor * (1.0 - falloff))`
            At `strength = 200`, this simplifies to `weight = falloff + (1.0 * (1.0 - falloff)) = 1.0`, meaning a full pull for all pixels within `delta_e_max`.
3.  **Blending Multiple Attractors:** If a pixel is influenced by multiple attractors, their effects are blended:
    *   The final L, C, H values for the pixel are a weighted average of its original L, C, H and the L, C, H values of all influencing attractors.
    *   The `weight` calculated above for each attractor is used.
    *   A `source_weight` is also calculated: `source_weight = max(0, 1 - sum_of_all_attractor_weights)`.
    *   The final channel value (e.g., Lightness `L_final`) is:
        `L_final = (source_weight * L_original) + sum(weight_i * L_attractor_i for each attractor i)`.
    *   This is done for L, C, and H channels independently, *only if the respective channel transformation flag (`--luminance`, `--saturation`, `--hue`) is enabled*.
    *   For Hue (H), which is an angle, the blending uses a weighted circular mean to correctly average hues (e.g., blending red and blue might result in purple, not by averaging their numerical hue values directly which could lead to green). This involves converting hues to Cartesian coordinates (sine/cosine), averaging, and then converting back to an angle.
4.  **Final Color:** The blended L, C, H values form the new color of the pixel in OKLCH, which then goes through the reverse conversion and gamut mapping process described earlier.

### 3. Performance Optimizations

`imgcolorshine` incorporates several strategies for high performance:

*   **Vectorization (NumPy):** Core numerical operations and image manipulations are heavily vectorized using NumPy, which executes many operations in compiled C or Fortran code, significantly faster than Python loops.
*   **Numba JIT Compilation:** Critical, performance-sensitive Python functions (especially color space conversions, gamut mapping, and parts of the transformation kernel) are Just-In-Time (JIT) compiled to highly optimized machine code by Numba. This often brings Python code to near C-level speed.
*   **Mypyc Ahead-of-Time Compilation:** Certain Python modules (primarily helpers in `fast_mypyc`) are pre-compiled into C extensions using Mypyc. This reduces Python interpreter overhead for these modules during runtime.
*   **GPU Acceleration (CuPy):** If an NVIDIA GPU and CuPy are available, `imgcolorshine` can offload the most computationally intensive parts of the transformation (distance calculations, falloff, blending for all pixels) to the GPU for massive parallel processing, offering substantial speedups on large images. The GPU implementation mirrors the logic of the CPU path.
*   **3D Lookup Table (LUT) Acceleration:**
    *   For repeated transformations with the *exact same attractor settings and channel flags*, `imgcolorshine` can pre-compute the entire color transformation into a 3D LUT (e.g., 65x65x65).
    *   Once the LUT is built, transforming an image involves simply looking up the new color for each pixel in the LUT and interpolating (trilinear interpolation) between the LUT grid points. This is extremely fast.
    *   **Note:** The LUT-based path is an optimization that approximates the full percentile-based tolerance. The LUT is generated by sampling points in the sRGB color space, transforming them using a simplified (non-percentile) distance model, and storing the results. This is faster but may produce slightly different results than the default two-pass percentile engine, especially regarding how `tolerance` behaves.
*   **Fused Kernels (Numba):** An optional fused Numba kernel (`--fused_kernel`) processes one pixel at a time through a larger portion of the transformation pipeline within a single compiled function. This can improve CPU cache locality and reduce memory bandwidth for some operations, potentially offering speed benefits on certain CPU architectures, though it's not always faster than the default vectorized Numba approach.

### 4. Code Architecture Overview

The codebase is structured into several key modules within `src/imgcolorshine/`:

*   **`fast_mypyc/`**: Contains core Python logic that is also compiled with Mypyc for speed.
    *   `cli.py`: Command-Line Interface definition using `fire`. Parses arguments and calls `process_image`.
    *   `colorshine.py`: Orchestrates the main image processing pipeline. Handles setup, attractor parsing, calling the engine, and LUT management.
    *   `colorshine_helpers.py`: Utility functions for `colorshine.py`.
    *   `engine.py`: The heart of the transformation. Defines `Attractor`, `OKLCHEngine` (for color parsing and batch conversions), and `ColorTransformer` (which implements the two-pass percentile algorithm and GPU/fused kernel paths).
    *   `engine_helpers.py`: Helper functions for `engine.py`, including blending logic.
    *   `falloff.py`: Defines falloff functions.
    *   `gamut.py`: Implements gamut mapping algorithms.
    *   `gamut_helpers.py`: Helper functions for gamut mapping.
    *   `io.py`: Handles image loading and saving via `Pillow` and `opencv-python`.
    *   `utils.py`: General utility functions.

*   **`fast_numba/`**: Contains modules heavily optimized with Numba.
    *   `engine_kernels.py`: Numba-compiled kernels for fused operations.
    *   `falloff_numba.py`: Numba-optimized falloff calculations.
    *   `gamut_numba.py`: Numba-optimized gamut mapping.
    *   `numba_utils.py`: Utility functions for Numba code.
    *   `trans_numba.py`: Crucial Numba-optimized color space conversion routines and batch gamut mapping.

*   **`gpu.py`**: Manages GPU detection (CuPy), provides an `ArrayModule` wrapper for transparent CPU/GPU array operations, and includes GPU memory utilities.
*   **`lut.py`**: (Module exists, e.g., `from imgcolorshine.lut import LUTManager`) Manages the creation and application of 3D Lookup Tables for accelerated processing.

*   **`__main__.py`**: Entry point for running the package as a script.
*   **`__init__.py`**: Package initializer.

This architecture separates concerns: CLI handling, main orchestration, the core transformation engine, optimized sub-modules (Numba, Mypyc), GPU interfacing, and I/O.

---

## Development & Contribution

We welcome contributions to `imgcolorshine`! Whether it's bug fixes, new features, or documentation improvements, please feel free to open an issue or submit a pull request.

### Setting Up Your Development Environment

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/twardoch/imgcolorshine.git
    cd imgcolorshine
    ```

2.  **Create and activate a virtual environment:**
    Using `venv` (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
    Or your preferred virtual environment tool.

3.  **Install in editable mode with development dependencies:**
    `imgcolorshine` uses `uv` for faster package management if available, but `pip` will also work.
    ```bash
    # Using uv (if installed)
    uv pip install -e ".[dev,test,speedups]"

    # Using pip
    pip install -e ".[dev,test,speedups]"
    ```
    This installs the package in a way that your changes in the source code are immediately reflected when you run the tool. The `dev`, `test`, and `speedups` extras include tools for linting, formatting, testing, and Numba/Mypyc.

### Coding Guidelines

To maintain code quality and consistency, please adhere to the following guidelines:

*   **PEP 8:** Follow the PEP 8 style guide for Python code. We use `Ruff` to format and lint the code.
*   **Type Hints:** Use type hints for all function signatures and important variables (PEP 484). Aim for simple types (`list`, `dict`, `str | int`) where possible.
*   **Docstrings:** Write clear, imperative docstrings for all modules, classes, functions, and methods, following PEP 257. Explain *what* the code does and *why*. If it's used or referred to elsewhere, mention it.
*   **f-strings:** Use f-strings for string formatting.
*   **Logging:** Implement `loguru`-based logging for CLI outputs, especially with a "verbose" mode for debugging.
*   **CLI Scripts:** For new CLI scripts, consider using `fire` and `rich`. Scripts should ideally start with:
    ```python
    #!/usr/bin/env -S uv run -s
    # /// script
    # dependencies = ["PKG1", "PKG2"]
    # ///
    # this_file: path/to/current_file.py
    ```
    (The `uv run` shebang is for standalone script execution; adjust if contributing to the main package.)
*   **Keep Track of Paths:** For standalone scripts or key modules, include a `this_file` comment near the top indicating its path relative to the project root.
*   **Constants:** Prefer constants over magic numbers or hardcoded strings.
*   **Modularity:** Encapsulate repeated logic into concise, single-purpose functions.
*   **Simplicity & Readability (PEP 20):** Keep code simple, explicit, and prioritize readability.

### Development Workflow & Code Quality

1.  **Make your changes:** Implement your feature or bug fix.
2.  **Run code quality checks and tests frequently.** After making Python changes, run the following checks (you might want to set this up as a script or pre-commit hook):
    ```bash
    # Ensure Python files are formatted and linted correctly
    # (Ruff is configured in pyproject.toml)
    ruff format .
    ruff check --fix --unsafe-fixes .

    # Run tests using pytest
    pytest
    ```
    Or, using `hatch` environments (see `pyproject.toml` for `hatch env run lint:fix` and `hatch env run test`):
    ```bash
    hatch run lint:fix
    hatch run test:test-cov # For tests with coverage
    ```

3.  **Update Documentation:** If your changes affect user-facing features or the API, please update this `README.md` and any relevant docstrings.
4.  **Commit your changes:** Use clear and descriptive commit messages.
5.  **Submit a Pull Request:** Push your changes to a fork and open a PR against the main `imgcolorshine` repository.

### Contribution Philosophy

*   **Iterate Gradually:** Avoid major, sweeping changes in a single PR if possible.
*   **Minimal Viable Increments:** Focus on delivering working, valuable changes.
*   **Handle Failures Gracefully:** Implement proper error handling, retries, or fallbacks where appropriate.
*   **Consult Existing Code:** Before implementing something new, check if similar solutions already exist in the codebase.
*   **Keep the Big Picture:** Maintain a holistic understanding of the codebase and how your changes fit in.

For more detailed internal guidelines and the project's development philosophy (including concepts like "Ideot" and "Critin" for constructive self-critique), contributors can refer to the `AGENT.md` or `CLAUDE.md` files in the repository root. These provide deeper context for agent-based or collaborative development on this project.

---
