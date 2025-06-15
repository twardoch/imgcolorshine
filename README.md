# imgcolorshine

Transform image colors using OKLCH color attractors - a physics-inspired tool that operates in perceptually uniform color space.

## Overview

`imgcolorshine` applies a gravitational-inspired color transformation where specified "attractor" colors pull the image's colors toward them. The tool works in the OKLCH color space, ensuring perceptually uniform and natural-looking results.

## Features

- **Perceptually Uniform**: Operations in OKLCH color space for intuitive results
- **Flexible Color Input**: Supports all CSS color formats (hex, rgb, hsl, oklch, named colors)
- **Selective Channel Control**: Transform lightness, saturation, and/or hue independently
- **Multiple Attractors**: Blend influences from multiple color targets
- **Blazing Fast**: Multiple acceleration options:
  - Numba-optimized color space conversions (77-115x faster than pure Python)
  - GPU acceleration with CuPy (10-100x additional speedup)
  - 3D Color LUT with caching (5-20x speedup)
  - Fused transformation kernels minimize memory traffic
- **High Performance**: Parallel processing with NumPy and Numba JIT compilation
- **Memory Efficient**: Automatic tiling for large images
- **Professional Quality**: CSS Color Module 4 compliant gamut mapping

## Installation

```bash
# Install from PyPI
pip install imgcolorshine

# Or install from source
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine
pip install -e .
```

## Usage

### Basic Example

Transform an image to be more red:

```bash
imgcolorshine shine photo.jpg "red;50;75"
```

### Command Syntax

```bash
imgcolorshine shine INPUT_IMAGE ATTRACTOR1 [ATTRACTOR2 ...] [OPTIONS]
```

Each attractor has the format: `"color;tolerance;strength"`

- **color**: Any CSS color (e.g., "red", "#ff0000", "oklch(70% 0.2 120)")
- **tolerance**: 0-100 (radius of influence - how far the color reaches)
- **strength**: 0-100 (transformation intensity - how much colors are pulled)

### Options

- `--output_image PATH`: Output image file (auto-generated if not specified)
- `--luminance BOOL`: Enable/disable lightness transformation (default: True)
- `--saturation BOOL`: Enable/disable chroma transformation (default: True)
- `--hue BOOL`: Enable/disable hue transformation (default: True)
- `--verbose BOOL`: Enable verbose logging (default: False)
- `--tile_size INT`: Tile size for large images (default: 1024)
- `--gpu BOOL`: Use GPU acceleration if available (default: True)
- `--lut_size INT`: Size of 3D LUT (0=disabled, 65=recommended) (default: 0)

### Examples

**Warm sunset effect:**
```bash
imgcolorshine shine landscape.png \
  "oklch(80% 0.2 60);40;60" \
  "#ff6b35;30;80" \
  --output_image=sunset.png
```

**Shift only hues toward green:**
```bash
imgcolorshine shine portrait.jpg "green;60;90" \
  --luminance=False --saturation=False
```

**Multiple color influences:**
```bash
imgcolorshine shine photo.jpg \
  "oklch(70% 0.15 120);50;70" \
  "hsl(220 100% 50%);25;50" \
  "#ff00ff;30;40"
```


## How It Works

### The Attraction Model: "Pull" vs "Replace"

`imgcolorshine` uses a **"pull" model**, not a "replace" model. This means:

- Colors are **gradually pulled** toward attractors, not replaced entirely
- A `strength` of 100 provides maximum pull, but only pixels exactly matching the attractor color will be fully transformed
- The effect diminishes with distance from the attractor color
- This creates natural, smooth transitions rather than harsh color replacements

### The Transformation Process

1. **Color Space**: All operations happen in OKLCH space for perceptual uniformity
2. **Attraction Model**: Each attractor color exerts influence based on:
   - **Distance**: Perceptual distance between pixel and attractor colors (ΔE in Oklab)
   - **Tolerance**: Maximum distance at which influence occurs (0-100 maps linearly to 0-2.5 ΔE)
   - **Strength**: Maximum transformation amount at zero distance
3. **Falloff**: Smooth raised-cosine curve ensures natural transitions
4. **Blending**: Multiple attractors blend using normalized weighted averaging
5. **Gamut Mapping**: Out-of-bounds colors are mapped back to displayable range

## Understanding Parameters

### Tolerance (0-100)
Controls the **radius of influence** - how far from the attractor color a pixel can be and still be affected:
- **Low values (0-20)**: Only very similar colors are affected
- **Medium values (30-60)**: Moderate range of colors transformed  
- **High values (70-100)**: Wide range of colors influenced
- **100**: Maximum range, affects colors up to ΔE = 2.5 (very broad influence)

### Strength (0-100)
Controls the **intensity of the pull** - how strongly colors are pulled toward the attractor:
- **Low values (0-30)**: Subtle color shifts, original color dominates
- **Medium values (40-70)**: Noticeable but natural transformations
- **High values (80-100)**: Strong pull toward attractor (not full replacement)
- **100**: Maximum pull, but still respects distance-based falloff

### Important Note on Hue-Only Transformations
When using `--luminance=False --saturation=False`, only the hue channel is modified. This means:
- Grayscale pixels (low saturation) show little to no change
- The effect is most visible on already-saturated colors
- To see stronger effects on all pixels, enable all channels

## Performance

- Processes a 1920×1080 image in **under 1 second** (was 2-5 seconds)
- **77-115x faster** color space conversions with Numba optimizations
- Parallel processing utilizing all CPU cores
- Automatic tiling for images larger than 2GB memory usage
- Benchmark results:
  - 256×256: 0.044s (was 5.053s with pure Python)
  - 512×512: 0.301s (was 23.274s)
  - 2048×2048: 3.740s

## Technical Details

- **Color Engine**: Hybrid approach
  - ColorAide for color parsing and validation
  - Numba-optimized matrix operations for batch conversions
  - Direct sRGB ↔ Oklab ↔ OKLCH transformations
- **Image I/O**: OpenCV (4x faster than PIL for PNG)
- **Computation**: NumPy + Numba JIT compilation with parallel execution
- **Optimizations**:
  - Vectorized color space conversions
  - Eliminated per-pixel ColorAide overhead
  - Cache-friendly memory access patterns
  - Manual matrix multiplication to avoid scipy dependency
- **Gamut Mapping**: CSS Color Module 4 algorithm with binary search
- **Falloff Function**: Raised cosine for smooth transitions

## Performance

With the latest optimizations, imgcolorshine achieves exceptional performance:

### CPU Performance (Numba)
- **256×256**: ~44ms (114x faster than pure Python)
- **512×512**: ~301ms (77x faster)
- **1920×1080**: ~2-3 seconds
- **4K (3840×2160)**: ~8-12 seconds

### GPU Performance (CuPy)
- **1920×1080**: ~20-50ms (100x faster than CPU)
- **4K**: ~80-200ms
- Requires NVIDIA GPU with CUDA support

### LUT Performance
- **First run**: Build time depends on LUT size (65³ ~2-5s)
- **Subsequent runs**: Near-instant with cached LUT
- **1920×1080**: ~100-200ms with 65³ LUT

### Usage Tips
```bash
# Maximum CPU performance
imgcolorshine shine photo.jpg "red;50;75"

# GPU acceleration (automatic if available)
imgcolorshine shine photo.jpg "red;50;75" --gpu=True

# LUT for best CPU performance on repeated transforms
imgcolorshine shine photo.jpg "red;50;75" --lut_size=65

# Combine GPU + LUT for ultimate speed
imgcolorshine shine photo.jpg "red;50;75" --gpu=True --lut_size=65
```

## Development

This project follows a structured approach focusing on code quality, documentation, and maintainable development practices.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

- Created by Adam Twardoch
- Developed with Antropic software
