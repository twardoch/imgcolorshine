# Chapter 2: Quick Start

Get up and running with imgcolorshine in minutes! This chapter provides practical examples to help you understand the basics quickly.

## Your First Transformation

Let's start with the simplest possible transformation:

```bash
imgcolorshine shine photo.jpg "orange;50;75"
```

This command:
- Loads `photo.jpg`
- Creates an orange color attractor
- Influences 50% of pixels most similar to orange (tolerance=50)
- Applies a strong but natural transformation (strength=75)
- Saves result as `photo_colorshine.jpg`

## Understanding the Basic Syntax

The command structure is:
```bash
imgcolorshine shine <INPUT_IMAGE> "<COLOR>;<TOLERANCE>;<STRENGTH>" [OPTIONS]
```

### Color Format
You can specify colors in any CSS format:

=== "Named Colors"
    ```bash
    imgcolorshine shine image.jpg "red;50;60"
    imgcolorshine shine image.jpg "forestgreen;40;70"
    imgcolorshine shine image.jpg "deepskyblue;60;80"
    ```

=== "Hex Colors"
    ```bash
    imgcolorshine shine image.jpg "#ff0000;50;60"
    imgcolorshine shine image.jpg "#00ff00;40;70"
    imgcolorshine shine image.jpg "#0080ff;60;80"
    ```

=== "RGB Colors"
    ```bash
    imgcolorshine shine image.jpg "rgb(255,0,0);50;60"
    imgcolorshine shine image.jpg "rgb(0,255,0);40;70"
    imgcolorshine shine image.jpg "rgba(0,128,255,0.8);60;80"
    ```

=== "OKLCH Colors"
    ```bash
    imgcolorshine shine image.jpg "oklch(70% 0.2 30);50;60"
    imgcolorshine shine image.jpg "oklch(60% 0.15 120);40;70"
    imgcolorshine shine image.jpg "oklch(80% 0.1 240);60;80"
    ```

### Tolerance (0-100): Range of Influence

Think of tolerance as "how picky" the attractor is:

- **Low tolerance (10-30):** Very selective, only affects very similar colors
- **Medium tolerance (40-70):** Balanced influence on similar colors  
- **High tolerance (80-100):** Broad influence across many colors

!!! tip "Tolerance is Adaptive"
    Tolerance works as a percentile. `tolerance=50` means the attractor influences the 50% of pixels most similar to it, regardless of the image's color palette.

### Strength (0-200): Transformation Intensity

Controls how much the colors are pulled toward the attractor:

- **Subtle (20-40):** Gentle color shifts
- **Natural (50-80):** Noticeable but realistic changes
- **Strong (90-100):** Dramatic transformations
- **Duotone (101-200):** Increasingly uniform effects

## Essential Examples

### Warm Up a Photo
Make a photo feel warmer and more golden:

```bash
imgcolorshine shine landscape.jpg "oklch(75% 0.15 60);60;70"
```

### Cool Down Colors  
Add a cooler, more cinematic feel:

```bash
imgcolorshine shine portrait.jpg "oklch(65% 0.12 240);50;65"
```

### Enhance Sunset Colors
Boost existing warm tones:

```bash
imgcolorshine shine sunset.jpg "orange;40;80"
```

### Create Vintage Look
Add a sepia-like vintage tone:

```bash
imgcolorshine shine old_photo.jpg "oklch(70% 0.08 80);70;60"
```

## Multiple Attractors

Use multiple color influences for complex effects:

### Sunset Effect
```bash
imgcolorshine shine photo.jpg \
  "oklch(75% 0.18 50);50;70" \
  "oklch(65% 0.15 20);40;60"
```

### Film Look
```bash
imgcolorshine shine image.jpg \
  "teal;30;40" \
  "orange;30;45"
```

### Dramatic Sky
```bash
imgcolorshine shine landscape.jpg \
  "darkblue;60;70" \
  "gold;40;50"
```

## Common Options

### Custom Output Name
```bash
imgcolorshine shine input.jpg "blue;50;70" --output_image=result.jpg
```

### Verbose Mode
See detailed processing information:
```bash
imgcolorshine shine input.jpg "red;50;70" --verbose=True
```

### Channel Control
Transform only specific color aspects:

=== "Hue Only"
    ```bash
    imgcolorshine shine image.jpg "purple;50;70" \
      --luminance=False --saturation=False
    ```

=== "Saturation Only"
    ```bash
    imgcolorshine shine image.jpg "red;60;80" \
      --luminance=False --hue=False
    ```

=== "Lightness Only"
    ```bash
    imgcolorshine shine image.jpg "white;70;60" \
      --saturation=False --hue=False
    ```

## Quick Recipe Guide

### Portrait Photography

**Warm Skin Tones:**
```bash
imgcolorshine shine portrait.jpg "oklch(80% 0.08 50);40;50"
```

**Cool Dramatic Look:**
```bash
imgcolorshine shine portrait.jpg "oklch(60% 0.1 250);60;70"
```

### Landscape Photography

**Golden Hour Enhancement:**
```bash
imgcolorshine shine landscape.jpg \
  "oklch(85% 0.15 80);50;60" \
  "oklch(70% 0.18 40);40;70"
```

**Moody Blue Hour:**
```bash
imgcolorshine shine landscape.jpg "oklch(40% 0.12 260);70;80"
```

### Street Photography

**Film Emulation:**
```bash
imgcolorshine shine street.jpg \
  "oklch(70% 0.05 60);80;50" \
  "oklch(50% 0.08 200);60;40"
```

**High Contrast:**
```bash
imgcolorshine shine street.jpg "black;30;90" --saturation=False --hue=False
```

## Performance Tips for Quick Work

### Fast Processing
Enable GPU acceleration (if available):
```bash
imgcolorshine shine large_image.jpg "blue;50;70" --gpu=True
```

### Repeated Transformations
Use lookup tables for speed:
```bash
imgcolorshine shine image.jpg "red;50;70" --lut_size=65
```

### Large Images
Use smaller tile sizes if memory is limited:
```bash
imgcolorshine shine huge_image.jpg "green;50;70" --tile_size=512
```

## Common Beginner Mistakes

### ❌ Don't: Use Extreme Values Initially
```bash
# Too aggressive for most images
imgcolorshine shine photo.jpg "red;100;200"
```

### ✅ Do: Start with Moderate Values
```bash
# Natural-looking transformation
imgcolorshine shine photo.jpg "red;50;70"
```

### ❌ Don't: Ignore Color Harmony
```bash
# Clashing colors
imgcolorshine shine warm_photo.jpg "cyan;80;90"
```

### ✅ Do: Consider Existing Color Palette
```bash
# Complement existing warm tones
imgcolorshine shine warm_photo.jpg "orange;60;70"
```

## Testing and Experimentation

### A/B Testing Approach
```bash
# Version A: Subtle
imgcolorshine shine original.jpg "blue;40;50" --output_image=subtle.jpg

# Version B: Strong  
imgcolorshine shine original.jpg "blue;60;80" --output_image=strong.jpg

# Version C: Different color
imgcolorshine shine original.jpg "purple;50;65" --output_image=purple.jpg
```

### Parameter Exploration
Try these systematic tests to understand the effects:

**Tolerance Range:**
```bash
imgcolorshine shine test.jpg "red;20;70" --output_image=tol20.jpg
imgcolorshine shine test.jpg "red;50;70" --output_image=tol50.jpg  
imgcolorshine shine test.jpg "red;80;70" --output_image=tol80.jpg
```

**Strength Range:**
```bash
imgcolorshine shine test.jpg "blue;50;30" --output_image=str30.jpg
imgcolorshine shine test.jpg "blue;50;70" --output_image=str70.jpg
imgcolorshine shine test.jpg "blue;50;120" --output_image=str120.jpg
```

## Batch Processing Basics

Process multiple images with the same transformation:

### Bash (Linux/macOS)
```bash
for img in *.jpg; do
  imgcolorshine shine "$img" "orange;50;70" \
    --output_image="processed_${img}"
done
```

### PowerShell (Windows)
```powershell
Get-ChildItem *.jpg | ForEach-Object {
  imgcolorshine shine $_.Name "orange;50;70" --output_image "processed_$($_.Name)"
}
```

### Python Script
```python
import os
import subprocess

images = [f for f in os.listdir('.') if f.endswith('.jpg')]
for img in images:
    output = f"processed_{img}"
    subprocess.run([
        'imgcolorshine', 'shine', img, 'orange;50;70',
        '--output_image', output
    ])
```

## Troubleshooting Quick Issues

### Command Not Found
```bash
# Check installation
which imgcolorshine
python -c "import imgcolorshine; print('OK')"
```

### Invalid Color Format
```bash
# ❌ This will fail
imgcolorshine shine image.jpg "not-a-color;50;70"

# ✅ Use valid CSS colors
imgcolorshine shine image.jpg "red;50;70"
```

### File Not Found
```bash
# Check file exists
ls -la your_image.jpg

# Use absolute path if needed
imgcolorshine shine /full/path/to/image.jpg "blue;50;70"
```

## What's Next?

Now that you can perform basic transformations:

1. **[Basic Usage](basic-usage.md)** - Learn the theory behind attractors
2. **[Understanding Attractors](understanding-attractors.md)** - Deep dive into the physics model
3. **[Advanced Features](advanced-features.md)** - Multi-attractor techniques and channel control

!!! success "Quick Start Complete!"
    You've learned the essentials of imgcolorshine. Try these examples with your own images and experiment with different values!