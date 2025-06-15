# imgcolorshine

Transform image colors using OKLCH color attractors - a physics-inspired tool that operates in perceptually uniform color space.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Test Coverage](https://img.shields.io/badge/coverage-50%25-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [How It Works](#how-it-works)
- [Performance](#performance)
- [Architecture](#architecture)
- [Development](#development)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Overview

`imgcolorshine` is a high-performance image color transformation tool that uses a physics-inspired model to transform images. It works by defining "attractor" colors that pull the image's colors toward them, similar to gravitational attraction. All operations are performed in the perceptually uniform OKLCH color space, ensuring natural and visually pleasing results.

### What Makes imgcolorshine Special?

1. **Perceptually Uniform**: Uses OKLCH color space for intuitive, natural-looking transformations
2. **Physics-Inspired**: Gravitational model provides smooth, organic color transitions
3. **Blazing Fast**: 100x+ faster than naive implementations through multiple optimization layers
4. **Production Ready**: Comprehensive test suite, professional gamut mapping, memory efficient
5. **Flexible**: Fine-grained control over color channels and transformation parameters

## Key Features

### Core Capabilities

- ✨ **Perceptually Uniform Color Space**: All operations in OKLCH for natural results
- 🎨 **Universal Color Support**: Any CSS color format (hex, rgb, hsl, oklch, named colors, etc.)
- 🎯 **Multi-Attractor Blending**: Combine multiple color influences seamlessly
- 🎛️ **Channel Control**: Transform lightness, chroma, and hue independently
- 🏎️ **Multiple Acceleration Modes**: CPU, GPU, and LUT-based processing
- 📊 **Professional Gamut Mapping**: CSS Color Module 4 compliant
- 💾 **Memory Efficient**: Automatic tiling for images of any size

### Performance Features

- **Numba JIT Compilation**: 77-115x faster color space conversions
- **GPU Acceleration**: 10-100x speedup with NVIDIA GPUs (CuPy)
- **3D Color LUT**: Pre-computed transformations with caching
- **Fused Kernels**: All operations in a single pass, minimizing memory traffic
- **Spatial Acceleration**: KD-tree based optimization for local color queries
- **Hierarchical Processing**: Multi-resolution pyramid for large images

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

Transform an image to have a warmer tone:

```bash
imgcolorshine shine photo.jpg "orange;50;75"
```

This command:
- Loads `photo.jpg`
- Creates an orange color attractor with 50% tolerance and 75% strength
- Saves the result as `photo_colorshine.jpg`

### Multiple Attractors

Create a sunset effect with multiple color influences:

```bash
imgcolorshine shine landscape.png \
  "oklch(80% 0.2 60);40;60" \
  "#ff6b35;30;80" \
  --output_image=sunset.png
```

## Usage Guide

### Command Structure

```bash
imgcolorshine shine INPUT_IMAGE ATTRACTOR1 [ATTRACTOR2 ...] [OPTIONS]
```

### Attractor Format

Each attractor is specified as: `"color;tolerance;strength"`

- **color**: Any CSS color
  - Named: `"red"`, `"blue"`, `"forestgreen"`
  - Hex: `"#ff0000"`, `"#00ff00"`
  - RGB: `"rgb(255, 0, 0)"`, `"rgba(0, 255, 0, 0.5)"`
  - HSL: `"hsl(120, 100%, 50%)"`
  - OKLCH: `"oklch(70% 0.2 120)"`
  
- **tolerance** (0-100): Radius of influence
  - 0-20: Only very similar colors affected
  - 30-60: Moderate range of influence
  - 70-100: Broad influence across many colors
  
- **strength** (0-200): Transformation intensity
  - 0-30: Subtle shifts, original color dominates
  - 40-70: Noticeable but natural transformations
  - 80-100: Strong pull toward attractor (fall-off still applies)
  - 100-200: Extended range – progressively flattens the fall-off curve

### Command Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output_image` | PATH | Auto | Output file path |
| `--luminance` | BOOL | True | Transform lightness channel |
| `--saturation` | BOOL | True | Transform chroma channel |
| `--hue` | BOOL | True | Transform hue channel |
| `--verbose` | BOOL | False | Enable detailed logging |
| `--tile_size` | INT | 1024 | Tile size for large images |
| `--gpu` | BOOL | True | Use GPU if available |
| `--lut_size` | INT | 0 | 3D LUT size (0=disabled, 65=recommended) |
| `--hierarchical` | BOOL | False | Multi-resolution processing |
| `--spatial_accel` | BOOL | True | Spatial acceleration structures |

### Advanced Examples

#### Channel-Specific Transformation

Transform only the hue, preserving lightness and saturation:

```bash
imgcolorshine shine portrait.jpg "teal;60;80" \
  --luminance=False --saturation=False
```

#### High-Performance Processing

For maximum speed on repeated transformations:

```bash
# Build and cache a 3D LUT for fast processing
imgcolorshine shine photo.jpg "purple;50;70" --lut_size=65

# Use GPU acceleration
imgcolorshine shine photo.jpg "cyan;40;60" --gpu=True

# Combine optimizations
imgcolorshine shine large_image.jpg "red;45;65" \
  --gpu=True --lut_size=65 --hierarchical=True
```

#### Batch Processing

Process multiple images with the same transformation:

```bash
for img in *.jpg; do
  imgcolorshine shine "$img" "seagreen;55;75" \
    --output_image="processed/${img}"
done
```

## How It Works

### The Attraction Model: "Pull" vs "Replace"
`imgcolorshine` uses a **"pull" model**, not a "replace" model. Colors are gradually pulled toward attractors, creating natural, smooth transitions.

### The Transformation Process
1.  **Color Space**: All operations happen in the perceptually uniform OKLCH color space.
2.  **Attraction Model**: Each attractor's influence is determined by:
    -   **Tolerance (0-100)**: This is a **percentile**. `tolerance=50` means the attractor will influence the 50% of the image's pixels that are most similar to it. This makes the effect adaptive to each image's unique color palette.
    -   **Strength (0-200)**: This controls the **intensity of the pull** – and, beyond 100, how much the raised-cosine fall-off is overridden.
3.  **Blending**: Influences from multiple attractors are blended using a normalized, weighted average.
4.  **Gamut Mapping**: Any resulting colors that are outside the displayable sRGB gamut are carefully mapped back in, preserving the perceived color as much as possible.

## Performance
The refactored codebase is optimized for correctness and maintainability. Performance is enhanced through:
-   **Numba**: Critical numerical loops are JIT-compiled to C-like speed.
-   **Mypyc**: Core modules are compiled into native C extensions, removing Python interpreter overhead.

A 2048x2048 image is processed in a few seconds on a modern machine.

## Architecture

### Module Overview

```
imgcolorshine/
├── Core Modules
│   ├── color.py          # OKLCH color engine and attractor management
│   ├── transform.py      # Main transformation logic
│   ├── colorshine.py     # High-level API and orchestration
│   └── falloff.py        # Distance-based influence functions
│
├── Performance Modules
│   ├── trans_numba.py    # Numba-optimized color conversions
│   ├── kernel.py         # Fused transformation kernels
│   ├── lut.py            # 3D lookup table implementation
│   ├── gpu.py            # GPU backend management
│   └── trans_gpu.py      # CuPy transformations
│
├── Optimization Modules
│   ├── spatial.py        # Spatial acceleration structures
│   ├── hierar.py         # Hierarchical processing
│   └── numba_utils.py    # Additional optimized utilities
│
├── Support Modules
│   ├── gamut.py          # CSS Color Module 4 gamut mapping
│   ├── io.py             # Optimized image I/O
│   ├── utils.py          # General utilities
│   └── cli.py            # Command-line interface
```

### Key Design Principles

1. **Modular Architecture**: Clear separation of concerns
2. **Performance First**: Multiple optimization paths
3. **Fallback Chain**: GPU → LUT → CPU Numba → Pure Python
4. **Type Safety**: Comprehensive type hints
5. **Memory Efficiency**: Streaming and tiling for large images
6. **Test Coverage**: 50%+ coverage with comprehensive test suite

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with test dependencies
pip install -e ".[dev,test]"
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src/imgcolorshine --cov-report=html

# Run specific test file
python -m pytest tests/test_color.py -v

# Run benchmarks
python -m pytest tests/test_performance.py -v
```

### Code Quality

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests

# Type checking
mypy src/imgcolorshine
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html
```

## API Reference

### Python API

```python
from imgcolorshine import process_image

# Basic transformation
process_image(
    input_image="photo.jpg",
    attractors=["red;50;75", "blue;30;60"],
    output_image="result.jpg"
)

# Advanced options
from imgcolorshine import OKLCHEngine, ColorTransformer

# Create color engine
engine = OKLCHEngine()

# Create attractors
attractor1 = engine.create_attractor("oklch(70% 0.2 30)", tolerance=50, strength=80)
attractor2 = engine.create_attractor("#ff6b35", tolerance=40, strength=60)

# Create transformer
transformer = ColorTransformer(
    engine,
    transform_lightness=True,
    transform_chroma=True,
    transform_hue=True
)

# Load and transform image
import numpy as np
from imgcolorshine.io import ImageProcessor

processor = ImageProcessor()
image = processor.load_image("input.jpg")

# Transform with specific channels
transformed = transformer.transform_image(
    image, 
    [attractor1, attractor2],
    {"luminance": True, "saturation": True, "hue": False}
)

processor.save_image(transformed, "output.jpg")
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- 🧪 **Testing**: Improve test coverage (target: 80%+)
- 📚 **Documentation**: Tutorials, examples, API docs
- 🚀 **Performance**: New optimization strategies
- 🎨 **Features**: Additional color spaces, effects
- 🐛 **Bug Fixes**: Issue resolution
- 🌐 **Localization**: Multi-language support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed by [Adam Twardoch](https://github.com/twardoch) with assistance from Anthropic's Claude
- Inspired by color science research and perceptual uniformity principles
- Built on the shoulders of giants: NumPy, Numba, ColorAide, OpenCV
- Special thanks to the OKLCH color space creators for enabling intuitive color manipulation

## Citation

If you use imgcolorshine in your research or projects, please cite:

```bibtex
@software{imgcolorshine,
  author = {Twardoch, Adam},
  title = {imgcolorshine: Perceptually Uniform Color Transformation},
  year = {2025},
  url = {https://github.com/twardoch/imgcolorshine}
}
```