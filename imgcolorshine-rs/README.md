# imgcolorshine-rs

Ultrafast Rust port of imgcolorshine - Transform image colors using OKLCH color attractors.

## Overview

`imgcolorshine-rs` is a high-performance image color transformation tool that uses a physics-inspired model to transform images. It works by defining "attractor" colors that pull the image's colors toward them, similar to gravitational attraction. All operations are performed in the perceptually uniform OKLCH color space, ensuring natural and visually pleasing results.

## Features

- ✨ **Perceptually Uniform Color Space**: All operations in OKLCH for natural results
- 🎨 **Universal Color Support**: Any CSS color format (hex, rgb, hsl, oklch, named colors, etc.)
- 🎯 **Multi-Attractor Blending**: Combine multiple color influences seamlessly
- 🎛️ **Channel Control**: Transform lightness, chroma, and hue independently
- 🏎️ **High Performance**: Multi-threaded processing with Rust's performance
- 📊 **Professional Gamut Mapping**: CSS Color Module 4 compliant
- 💾 **Memory Efficient**: Optimized for images of any size

## Installation

### From Source

```bash
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine/imgcolorshine-rs
cargo build --release
```

The binary will be available at `target/release/imgcolorshine`.

### From crates.io (Coming Soon)

```bash
cargo install imgcolorshine
```

## Quick Start

### Basic Usage

Transform an image to have a warmer tone:

```bash
imgcolorshine photo.jpg "orange;50;75"
```

This command:
- Loads `photo.jpg`
- Creates an orange color attractor with 50% tolerance and 75% strength
- Saves the result as `photo_colorshine.jpg`

### Multiple Attractors

Create a sunset effect with multiple color influences:

```bash
imgcolorshine landscape.png \
  "oklch(80% 0.2 60);40;60" \
  "#ff6b35;30;80" \
  --output-image sunset.png
```

## Usage Guide

### Command Structure

```bash
imgcolorshine INPUT_IMAGE ATTRACTOR1 [ATTRACTOR2 ...] [OPTIONS]
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
  - 0-30: Subtle shifts
  - 40-70: Noticeable but natural transformations
  - 80-100: Strong pull toward attractor
  - 100-200: Extended range with progressively flattened falloff

### Command Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output-image` | PATH | Auto | Output file path |
| `--luminance` | BOOL | true | Transform lightness channel |
| `--saturation` | BOOL | true | Transform chroma channel |
| `--hue` | BOOL | true | Transform hue channel |
| `--verbose` | BOOL | false | Enable detailed logging |
| `--threads` | INT | 0 (auto) | Number of threads to use |

### Advanced Examples

#### Channel-Specific Transformation

Transform only the hue, preserving lightness and saturation:

```bash
imgcolorshine portrait.jpg "teal;60;80" \
  --luminance=false --saturation=false
```

#### Batch Processing

Process multiple images with the same transformation:

```bash
for img in *.jpg; do
  imgcolorshine "$img" "seagreen;55;75" \
    --output-image "processed/${img}"
done
```

## How It Works

### The Attraction Model

`imgcolorshine-rs` uses a "pull" model where colors are gradually pulled toward attractors:

1. **Color Space**: All operations happen in OKLCH color space
2. **Tolerance**: Percentile-based - `tolerance=50` means the attractor influences the 50% of pixels most similar to it
3. **Strength**: Controls the intensity of the pull (0-200)
4. **Blending**: Multiple attractors blend using normalized weighted averages

### The Transformation Process

1. Convert image from sRGB to OKLCH
2. Calculate attractor influence radii based on image color distribution
3. Apply transformations with falloff curves
4. Map results back to sRGB gamut
5. Save the transformed image

## Performance

`imgcolorshine-rs` is optimized for speed:

- Multi-threaded processing using Rayon
- Efficient color space conversions
- Memory-efficient processing
- Typical processing time: 1-3 seconds for a 2048x2048 image

## Development

### Building from Source

```bash
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Python implementation: [imgcolorshine](https://github.com/twardoch/imgcolorshine)
- Color science based on [OKLCH color space](https://bottosson.github.io/posts/oklab/)
- Built with [palette](https://crates.io/crates/palette) for color conversions