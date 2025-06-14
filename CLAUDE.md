# When you write code

- Iterate gradually, avoiding major changes
- Minimize confirmations and checks
- Preserve existing code/structure unless necessary
- Use constants over magic numbers
- Check for existing solutions in the codebase before starting
- Check often the coherence of the code you’re writing with the rest of the code.
- Focus on minimal viable increments and ship early
- Write explanatory docstrings/comments that explain what and WHY this does, explain where and how the code is used/referred to elsewhere in the code
- Analyze code line-by-line
- Handle failures gracefully with retries, fallbacks, user guidance
- Address edge cases, validate assumptions, catch errors early
- Let the computer do the work, minimize user decisions
- Reduce cognitive load, beautify code
- Modularize repeated logic into concise, single-purpose functions
- Favor flat over nested structures
- Consistently keep, document, update and consult the holistic overview mental image of the codebase. 

## Keep track of paths

In each source file, maintain the up-to-date `this_file` record that shows the path of the current file relative to project root. Place the `this_file` record near the top of the file, as a comment after the shebangs, or in the YAML Markdown frontmatter.

## When you write Python

- Use `uv pip`, never `pip`
- Use `python -m` when running code
- PEP 8: Use consistent formatting and naming
- Write clear, descriptive names for functions and variables
- PEP 20: Keep code simple and explicit. Prioritize readability over cleverness
- Use type hints in their simplest form (list, dict, | for unions)
- PEP 257: Write clear, imperative docstrings
- Use f-strings. Use structural pattern matching where appropriate
- ALWAYS add "verbose" mode logugu-based logging, & debug-log
- For CLI Python scripts, use fire & rich, and start the script with

```
#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["PKG1", "PKG2"]
# ///
# this_file: PATH_TO_CURRENT_FILE
```

Work in rounds: 

- Create `PLAN.md` as a detailed flat plan with `[ ]` items. 
- Identify the most important TODO items, and create `TODO.md` with `[ ]` items. 
- Implement the changes. 
- Update `PLAN.md` and `TODO.md` as you go. 
- After each round of changes, update `CHANGELOG.md` with the changes.
- Update `README.md` to reflect the changes.

Ask before extending/refactoring existing code in a way that may add complexity or break things.

When you’re finished, print "Wait, but" to go back, think & reflect, revise & improvement what you’ve done (but don’t invent functionality freely). Repeat this. But stick to the goal of "minimal viable next version". Lead two experts: "Ideot" for creative, unorthodox ideas, and "Critin" to critique flawed thinking and moderate for balanced discussions. The three of you shall illuminate knowledge with concise, beautiful responses, process methodically for clear answers, collaborate step-by-step, sharing thoughts and adapting. If errors are found, step back and focus on accuracy and progress.

## After Python changes run:

```
fd -e py -x autoflake -i {}; fd -e py -x pyupgrade --py311-plus {}; fd -e py -x ruff check --output-format=github --fix --unsafe-fixes {}; fd -e py -x ruff format --respect-gitignore --target-version py311 {}; python -m pytest;
```

Be creative, diligent, critical, relentless & funny!

# imgcolorshine

Transform image colors using OKLCH color attractors - a physics-inspired tool that operates in perceptually uniform color space.

## Overview

`imgcolorshine` applies a gravitational-inspired color transformation where specified "attractor" colors pull the image's colors toward them. The tool works in the OKLCH color space, ensuring perceptually uniform and natural-looking results.

## Features

- **Perceptually Uniform**: Operations in OKLCH color space for intuitive results
- **Flexible Color Input**: Supports all CSS color formats (hex, rgb, hsl, oklch, named colors)
- **Selective Channel Control**: Transform lightness, saturation, and/or hue independently
- **Multiple Attractors**: Blend influences from multiple color targets
- **High Performance**: Optimized with NumPy and Numba for fast processing
- **Memory Efficient**: Automatic tiling for large images
- **Professional Quality**: CSS Color Module 4 compliant gamut mapping

## Installation

```bash
# Using uv (recommended)
uv run imgcolorshine.py --help

# Or install dependencies manually
pip install coloraide opencv-python numpy numba click pillow loguru rich
```

## Usage

### Basic Example

Transform an image to be more red:

```bash
./imgcolorshine.py photo.jpg "red;50;75"
```

### Command Syntax

```bash
imgcolorshine INPUT_IMAGE ATTRACTOR1 [ATTRACTOR2 ...] [OPTIONS]
```

Each attractor has the format: `"color;tolerance;strength"`

- **color**: Any CSS color (e.g., "red", "#ff0000", "oklch(70% 0.2 120)")
- **tolerance**: 0-100 (radius of influence - how far the color reaches)
- **strength**: 0-100 (transformation intensity - how much colors are pulled)

### Options

- `--output-image PATH`: Output image file (auto-generated if not specified)
- `--luminance/--no-luminance`: Enable/disable lightness transformation (default: True)
- `--saturation/--no-saturation`: Enable/disable chroma transformation (default: True)
- `--hue/--no-hue`: Enable/disable hue transformation (default: True)
- `--verbose`: Enable verbose logging (default: False)
- `--tile-size INT`: Tile size for large images (default: 1024)

### Examples

**Warm sunset effect:**
```bash
./imgcolorshine.py landscape.png \
  "oklch(80% 0.2 60);40;60" \
  "#ff6b35;30;80" \
  --output-image sunset.png
```

**Shift only hues toward green:**
```bash
./imgcolorshine.py portrait.jpg "green;60;90" \
  --no-luminance --no-saturation
```

**Multiple color influences:**
```bash
./imgcolorshine.py photo.jpg \
  "oklch(70% 0.15 120);50;70" \
  "hsl(220 100% 50%);25;50" \
  "#ff00ff;30;40"
```


## How It Works

1. **Color Space**: All operations happen in OKLCH space for perceptual uniformity
2. **Attraction Model**: Each attractor color exerts influence based on:
   - **Distance**: How similar a pixel's color is to the attractor
   - **Tolerance**: Maximum distance at which influence occurs
   - **Strength**: Maximum transformation amount
3. **Falloff**: Smooth raised-cosine curve for natural transitions
4. **Blending**: Multiple attractors blend using normalized weighted averaging
5. **Gamut Mapping**: Out-of-bounds colors are mapped back to displayable range

## Understanding Parameters

### Tolerance (0-100)
- **Low values (0-20)**: Only very similar colors are affected
- **Medium values (30-60)**: Moderate range of colors transformed
- **High values (70-100)**: Wide range of colors influenced

### Strength (0-100)
- **Low values (0-30)**: Subtle color shifts
- **Medium values (40-70)**: Noticeable but natural transformations
- **High values (80-100)**: Strong color replacement

## Performance

- Processes a 1920×1080 image in ~2-5 seconds
- Automatic tiling for images larger than 2GB memory usage
- GPU acceleration available with CuPy (10-100x speedup)

## Technical Details

- **Color Engine**: ColorAide for accurate OKLCH operations
- **Image I/O**: OpenCV (4x faster than PIL for PNG)
- **Computation**: NumPy + Numba JIT compilation
- **Gamut Mapping**: CSS Color Module 4 algorithm
- **Falloff Function**: Raised cosine for smooth transitions

## Development

This project follows a structured approach focusing on code quality, documentation, and maintainable development practices.
