# imgcolorshine

Transform image colors using OKLCH color attractors—a physics-inspired tool that operates in a perceptually uniform color space.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Test Coverage](https://img.shields.io/badge/coverage-50%25-yellow.svg)
![Performance](https://img.shields.io/badge/performance-100x%20faster-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Command Structure](#command-structure)
  - [Attractor Format](#attractor-format)
    - [Tolerance (0–100)](#tolerance-0100)
    - [Strength (0–200)](#strength-0200)
  - [Command Options](#command-options)
- [How It Works](#how-it-works)
  - [The "Pull" Model](#the-pull-model)
  - [The Transformation Process](#the-transformation-process)
- [Performance](#performance)
- [Architecture](#architecture)
- [Development](#development)
- [API Reference](#api-reference)

## Overview

`imgcolorshine` is a high-performance image color transformation tool that uses a physics-inspired model to transform images. It works by defining "attractor" colors that pull the image's colors toward them, similar to gravitational attraction. All operations are performed in the perceptually uniform **OKLCH color space**, ensuring that transformations look natural and intuitive to the human eye.

### What Makes `imgcolorshine` Special?

1.  **Perceptually Uniform**: All transformations happen in OKLCH, where changes to lightness, chroma, and hue values correspond directly to how we perceive those qualities.
2.  **Physics-Inspired**: The gravitational model provides smooth, organic color transitions rather than abrupt replacements.
3.  **Blazing Fast**: Multiple optimization layers deliver 100x+ speedups through Numba JIT compilation, GPU acceleration, 3D LUT caching, and fused kernels.
4.  **Production Ready**: The tool includes a comprehensive test suite, professional-grade gamut mapping, and memory-efficient processing for large images.
5.  **Flexible**: Offers fine-grained control over which color channels to transform and how strongly to influence them.

## Key Features

- ✨ **Perceptually Uniform Color Space**: All operations in OKLCH for natural results.
- 🎨 **Universal Color Support**: Accepts any CSS color format (hex, rgb, hsl, oklch, named colors, etc.).
- 🎯 **Multi-Attractor Blending**: Seamlessly combines the influence of multiple color attractors.
- 🎛️ **Channel Control**: Transforms lightness, chroma, and hue independently.
- 🏎️ **Multiple Acceleration Modes**: CPU (Numba), GPU (CuPy), and LUT-based processing.
- 📊 **Professional Gamut Mapping**: CSS Color Module 4 compliant algorithm ensures all colors are displayable.
- 💾 **Memory Efficient**: Automatic tiling for images of any size.

## Installation

### From PyPI (Recommended)

```bash
pip install imgcolorshine
```

### From Source

```bash
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine
pip install -e .
```

### Optional Dependencies

For GPU acceleration:
```bash
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

## Quick Start

### Basic Usage

Transform an image to have a warmer tone by pulling its colors toward orange:

```bash
imgcolorshine shine photo.jpg "orange;50;75"
```

This command:
- Loads `photo.jpg`.
- Creates an `orange` color attractor.
- Sets the `tolerance` to `50` (influencing the 50% of pixels most similar to orange).
- Sets the `strength` to `75` (a noticeable but natural pull).
- Saves the result as `photo_colorshine.jpg`.

### Multiple Attractors

Create a sunset effect with multiple color influences:

```bash
imgcolorshine shine landscape.png \
  "oklch(80% 0.2 60);40;60" \
  "#ff6b35;30;80" \
  --output_image=sunset.png
```

### High-Performance Processing

For maximum speed on large images or batch processing:

```bash
# Use GPU acceleration (default: on if available)
imgcolorshine shine large_photo.jpg "purple;50;70" --gpu=True

# Build and use a 3D LUT for extremely fast processing
imgcolorshine shine photo.jpg "cyan;40;60" --lut_size=65

# Combine optimizations for best performance
imgcolorshine shine huge_image.jpg "red;45;65" \
  --gpu=True --lut_size=65 --fused_kernel=True
```

## Usage Guide

### Command Structure

```bash
imgcolorshine shine INPUT_IMAGE ATTRACTOR1 [ATTRACTOR2 ...] [OPTIONS]
```

### Attractor Format

Each attractor is a string with three parts separated by semicolons: `"color;tolerance;strength"`

-   **`color`**: Any valid CSS color string.
    -   Named: `"red"`, `"blue"`, `"forestgreen"`
    -   Hex: `"#ff0000"`, `"#00f"`
    -   RGB: `"rgb(255, 0, 0)"`, `"rgba(0, 255, 0, 0.5)"`
    -   HSL: `"hsl(120, 100%, 50%)"`
    -   OKLCH: `"oklch(70% 0.2 120)"`

#### Tolerance (0–100)

This parameter determines the **range of influence** of your attractor.

-   **For Designers**: Think of `tolerance` as casting a net. A small tolerance (e.g., `10`) catches only colors that are very similar to your attractor. A large tolerance (e.g., `80`) casts a wide net, catching a broad range of colors and pulling all of them gently toward your target. A tolerance of `50` will affect the half of the image's colors that are most similar to your attractor. This makes the tool adaptive to each image's unique palette.

-   **Technical Explanation**: Tolerance is a **percentile** of an image's color distribution, calculated based on perceptual distance (Euclidean distance in Oklab space) from the attractor color. `tolerance=50` finds the median distance of all pixels to the attractor and uses that distance as the maximum radius of influence. Any pixel with a distance greater than this radius will not be affected.

#### Strength (0–200)

This parameter controls the **intensity** of the transformation.

-   **For Designers**: `strength` is how hard you pull the "net" of colors you've captured with `tolerance`.
    -   **`0–100` (Falloff Mode)**: A gentle tug. Colors closest to your attractor are pulled the hardest, and the pull's intensity smoothly "falls off" to zero for colors at the edge of the tolerance radius. A `strength` of `50` means that even the most similar colors will only move 50% of the way toward the attractor's color.
    -   **`101–200` (Duotone Mode)**: A powerful, uniform yank. In this range, the falloff effect is progressively flattened. At a `strength` of `200`, every color inside your tolerance net is pulled *completely* to the attractor's color, creating a flat, duotone-like effect within that specific color range.

-   **Technical Explanation**: Strength controls the interpolation factor between an original pixel's color and the attractor's color. The influence is modulated by a raised-cosine falloff function based on the normalized distance within the tolerance radius.
    -   For `strength <= 100`, the final weight is `(strength / 100) * falloff`.
    -   For `strength > 100`, the falloff is overridden. The weight is calculated as `falloff + (strength_extra * (1 - falloff))`, where `strength_extra` is the scaled value from 100 to 200. At 200, the weight is 1.0 for all pixels within the tolerance radius.

### Command Options

| Option             | Type | Default | Description                                        |
| ------------------ | ---- | ------- | -------------------------------------------------- |
| `--output_image`   | PATH | Auto    | Output file path. Auto-generates if not provided.  |
| `--luminance`      | BOOL | True    | Transform the lightness channel (L).               |
| `--saturation`     | BOOL | True    | Transform the chroma/saturation channel (C).       |
| `--hue`            | BOOL | True    | Transform the hue channel (H).                     |
| `--verbose`        | BOOL | False   | Enable detailed logging for debugging.             |
| `--gpu`            | BOOL | True    | Use GPU acceleration if available.                 |
| `--lut_size`       | INT  | 0       | 3D LUT size (0=disabled, 65=recommended).          |
| `--fused_kernel`   | BOOL | False   | Use fused Numba kernel for better performance.     |

## How It Works

### The "Pull" Model

`imgcolorshine` uses a **"pull" model**, not a "replace" model. Instead of finding and replacing colors, it simulates a gravitational field where every pixel is gently pulled toward the defined attractor colors. This results in smooth, natural transitions that respect the original image's complexity.

### The Transformation Process

The process is designed for both correctness and performance:

1.  **Color Space Conversion**: The input image is converted from sRGB to the **Oklab** color space. All distance calculations happen here because Euclidean distance in Oklab closely approximates human perceptual color difference. The Oklab values are also converted to **OKLCH** (Lightness, Chroma, Hue) for intuitive channel-specific transformations.

2.  **Distance Analysis (The "Tolerance" Pass)**: For each attractor, the engine calculates the perceptual distance (ΔE) from the attractor to *every pixel* in the image. It then finds the distance value that corresponds to the specified `tolerance` percentile. This value becomes the maximum radius of influence (`delta_e_max`).

3.  **Color Transformation (The "Strength" Pass)**: The engine iterates through the pixels again (via a single vectorized operation). For each pixel within an attractor's `delta_e_max` radius:
    a. It calculates a `weight` based on the pixel's distance from the attractor and the attractor's `strength`.
    b. The influence of all attractors is blended.
    c. The pixel's L, C, or H values are interpolated toward the blended attractor values.

4.  **Gamut Mapping**: After transformation, some colors may be "out of gamut" (i.e., not displayable on a standard sRGB screen). A CSS Color Module 4-compliant algorithm carefully brings these colors back into the sRGB gamut by reducing chroma while preserving hue and lightness, ensuring no color data is simply clipped.

5.  **Final Conversion**: The transformed, gamut-mapped image is converted from Oklab back to sRGB and saved.

## Performance

The core transformation engine is highly optimized for speed:

-   **Vectorization**: The main processing loop is fully vectorized with NumPy, eliminating slow, per-pixel Python loops and leveraging optimized C/Fortran routines for calculations.
-   **Numba**: Critical, hot-path functions like color space conversions and gamut mapping are Just-in-Time (JIT) compiled to native machine code by Numba, providing significant speedup.
-   **GPU Acceleration**: When available, transformations can be accelerated using CuPy for GPU processing, offering dramatic speedups for large images.
-   **LUT Acceleration**: For repeated transformations with the same settings, a 3D lookup table can be pre-computed and cached, enabling real-time performance.
-   **Fused Kernels**: An optional fused transformation kernel processes one pixel at a time through the entire pipeline, improving cache locality and reducing memory bandwidth.
-   **Mypyc**: Key modules are pre-compiled into C extensions using Mypyc to reduce Python interpreter overhead.

The refactored codebase is optimized for correctness and maintainability. Performance is enhanced through:
-   **Numba**: Critical numerical loops are JIT-compiled to C-like speed.
-   **Mypyc**: Core modules are compiled into native C extensions, removing Python interpreter overhead.
-   **GPU Acceleration**: CuPy backend for 10-100x speedups on NVIDIA GPUs.
-   **3D LUT**: Pre-computed transformations with trilinear interpolation for 5-20x speedups.
-   **Fused Kernels**: Single-pass pixel transformation keeping operations in CPU registers.

On a modern machine:
- A 2048×2048 pixel image processes in 2-4 seconds (CPU)
- With GPU acceleration: <0.5 seconds
- With cached LUT: <0.2 seconds
- Combined optimizations: <10ms for 1920×1080 images

## Architecture

### Module Overview

```
imgcolorshine/
├── Core Modules
│   ├── engine.py         # OKLCH color engine and attractor management
│   ├── colorshine.py     # High-level API and orchestration
│   └── falloff.py        # Distance-based influence functions
│
├── Performance Modules
│   ├── trans_numba.py    # Numba-optimized color conversions
│   ├── kernel.py         # Fused transformation kernels
│   ├── lut.py            # 3D lookup table implementation
│   ├── gpu.py            # GPU backend management
│   └── numba_utils.py    # Additional optimized utilities
│
├── Support Modules
│   ├── gamut.py          # CSS Color Module 4 gamut mapping
│   ├── io.py             # Optimized image I/O
│   ├── utils.py          # General utilities
│   └── cli.py            # Command-line interface
```

### Key Design Principles

1.  **Modular Architecture**: Clear separation of concerns between the color engine, API, and utilities.
2.  **Performance First**: Multiple optimization paths with automatic fallback chain (GPU → LUT → CPU Numba → Pure Python).
3.  **Type Safety**: Comprehensive type hints are used throughout the codebase and checked with Mypy.
4.  **Memory Efficiency**: Streaming and tiling for large images.
5.  **Test Coverage**: 50%+ coverage with comprehensive test suite.
6.  **Correctness First**: Transformations and color science are based on established standards.

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with all development dependencies
pip install -e ".[dev,test]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/imgcolorshine --cov-report=html
```

### Code Quality

```bash
# Format and lint code with Ruff
ruff check --fix .
ruff format .
```

## API Reference

### Simple API

```python
from imgcolorshine.colorshine import process_image

# Basic transformation with string-based attractors
process_image(
    input_image="photo.jpg",
    attractors=["red;50;75", "blue;30;60"],
    output_image="result.jpg"
)
```

### Advanced API

```python
import numpy as np
from imgcolorshine.engine import OKLCHEngine, ColorTransformer
from imgcolorshine.io import ImageProcessor

# 1. Initialize components
engine = OKLCHEngine()
transformer = ColorTransformer(engine)
processor = ImageProcessor()

# 2. Create attractor objects
attractor1 = engine.create_attractor("oklch(70% 0.2 30)", tolerance=50, strength=80)
attractor2 = engine.create_attractor("#ff6b35", tolerance=40, strength=60)
attractors = [attractor1, attractor2]

# 3. Load image
image_data = processor.load_image("input.jpg") # Returns a float32 NumPy array

# 4. Define which channels to transform
# The keys must be 'luminance', 'saturation', and 'hue'
channel_flags = {"luminance": True, "saturation": True, "hue": False}

# 5. Transform the image
transformed_image = transformer.transform_image(
    image=image_data, 
    attractors=attractors,
    flags=channel_flags
)

# 6. Save the result
processor.save_image(transformed_image, "output.jpg")
```