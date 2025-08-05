# imgcolorshine Documentation

**Transform image colors with artistic precision using OKLCH color attractors—a physics-inspired tool operating in a perceptually uniform color space.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/imgcolorshine.svg)](https://badge.fury.io/py/imgcolorshine)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## What is imgcolorshine?

`imgcolorshine` is a sophisticated command-line tool that uses a unique "color attractor" model to transform image colors. You define specific target colors, and the tool intelligently pulls the existing colors of your image towards these attractors, creating smooth, organic, and natural-looking transformations.

All color operations are performed in the **OKLCH color space**, which is perceptually uniform, ensuring that changes correspond directly to how humans perceive color qualities.

## Quick TLDR

!!! tip "Quick Start"
    ```bash
    pip install imgcolorshine
    imgcolorshine shine photo.jpg "orange;50;75"
    ```

## Documentation Overview

This documentation is organized into 9 comprehensive chapters:

### [Chapter 1: Installation](installation.md)
**TLDR:** Install from PyPI with `pip install imgcolorshine`. Optional GPU acceleration available with CuPy.

Learn how to install imgcolorshine on different platforms, set up optional dependencies for GPU acceleration, and verify your installation.

### [Chapter 2: Quick Start](quickstart.md) 
**TLDR:** Basic syntax is `imgcolorshine shine input.jpg "color;tolerance;strength"`. Start with moderate values like `"orange;50;75"`.

Get up and running quickly with practical examples and your first color transformations.

### [Chapter 3: Basic Usage](basic-usage.md)
**TLDR:** Attractors use format `"color;tolerance;strength"` where tolerance (0-100) is percentile-based influence and strength (0-200) controls transformation intensity.

Master the fundamental concepts of color attractors, tolerance, and strength parameters through detailed examples.

### [Chapter 4: Understanding Attractors](understanding-attractors.md)
**TLDR:** Attractors use a physics-inspired "pull" model in OKLCH space. Tolerance is adaptive percentile-based, not fixed radius.

Deep dive into the attractor model, how tolerance works as a percentile system, and the physics-inspired transformation approach.

### [Chapter 5: Advanced Features](advanced-features.md)
**TLDR:** Control individual L/C/H channels, use multiple attractors, batch processing, and extended strength modes for duotone effects.

Explore channel-specific transformations, multi-attractor blending, batch processing workflows, and advanced strength modes.

### [Chapter 6: Performance Optimization](performance-optimization.md)
**TLDR:** Enable GPU with CuPy, use 3D LUTs (`--lut_size=65`) for repeated transformations, and leverage hierarchical processing for large images.

Maximize performance with GPU acceleration, lookup tables, fused kernels, and optimization strategies for different use cases.

### [Chapter 7: Color Science](color-science.md)
**TLDR:** All operations in perceptually uniform OKLCH/Oklab space with professional CSS Color Module 4 compliant gamut mapping.

Understand the color science behind imgcolorshine, including OKLCH color space, perceptual uniformity, and gamut mapping.

### [Chapter 8: API Reference](api-reference.md)
**TLDR:** Python API available for programmatic use with `OKLCHEngine`, `ColorTransformer`, and batch processing functions.

Complete reference for the Python API, command-line options, and programmatic usage.

### [Chapter 9: Development](development.md)
**TLDR:** Modular architecture with fast_mypyc and fast_numba optimizations. Uses Ruff for linting, pytest for testing.

Learn about the codebase architecture, development setup, contributing guidelines, and extending imgcolorshine.

## Key Features at a Glance

- ✨ **Perceptually Uniform Color Space:** All operations in OKLCH
- 🎨 **Universal Color Support:** Any CSS color format for attractors  
- 🎯 **Multi-Attractor Blending:** Combine multiple color influences seamlessly
- 🎛️ **Channel Control:** Transform lightness, chroma, and hue independently
- 🏎️ **Multiple Acceleration Modes:** CPU, GPU, and LUT-based processing
- 📊 **Professional Gamut Mapping:** CSS Color Module 4 compliant
- 💾 **Memory Efficient:** Automatic tiling for images of any size

## Who Should Use This Documentation?

- **Photographers** seeking advanced color grading tools
- **Digital Artists** wanting unique color palette effects  
- **Graphic Designers** needing precise color control
- **Developers** integrating color transformation into applications
- **Researchers** exploring perceptual color spaces

## Getting Help

- 📖 Read through the documentation chapters
- 🐛 [Report issues on GitHub](https://github.com/twardoch/imgcolorshine/issues)
- 💬 Join discussions in the repository
- 📝 Check the [API Reference](api-reference.md) for detailed function documentation

---

**Ready to transform your images?** Start with the [Installation](installation.md) guide or jump straight to [Quick Start](quickstart.md) if you're eager to begin.