# Chapter 5: Advanced Features

This chapter explores sophisticated techniques and workflows for professional-grade color transformations using imgcolorshine's advanced capabilities.

## Multi-Attractor Mastery

Advanced multi-attractor techniques unlock complex, professional color grading effects.

### Strategic Attractor Placement

#### Zone-Based Color Grading

Emulate traditional three-way color correction:

```bash
# Shadows, midtones, highlights approach
imgcolorshine shine portrait.jpg \
  "oklch(25% 0.08 240);35;60" \   # Cool shadows
  "oklch(50% 0.04 30);25;40" \    # Neutral midtones  
  "oklch(85% 0.06 60);35;70"      # Warm highlights
```

This technique:
- Targets different luminosity ranges
- Creates depth and dimension
- Mimics professional color grading workflows

#### Complementary Color Schemes

Leverage color theory for cinematic looks:

```bash
# Orange/Teal blockbuster look
imgcolorshine shine action_scene.jpg \
  "oklch(75% 0.18 50);45;70" \    # Orange for skin/fire/warmth
  "oklch(45% 0.15 210);45;70"     # Teal for sky/metal/coolness
```

#### Split-Toning Effects

Traditional film emulation techniques:

```bash
# Warm highlights, cool shadows
imgcolorshine shine film_look.jpg \
  "oklch(80% 0.12 45);40;50" \    # Warm highlights
  "oklch(30% 0.08 220);40;50"     # Cool shadows
```

### Advanced Blending Strategies

#### Layered Color Building

Build complex looks incrementally:

```bash
# Step 1: Base color temperature
imgcolorshine shine original.jpg \
  "oklch(65% 0.05 45);70;50" \
  --output_image=step1_base.jpg

# Step 2: Enhance specific colors
imgcolorshine shine step1_base.jpg \
  "red;35;70" "blue;35;60" \
  --luminance=False --output_image=step2_enhanced.jpg

# Step 3: Final mood adjustment
imgcolorshine shine step2_enhanced.jpg \
  "oklch(40% 0.1 260);60;40" \
  --output_image=final_moody.jpg
```

#### Selective Channel Enhancement

Target specific visual aspects:

```bash
# Enhance sunset colors - hue shift only
imgcolorshine shine sunset.jpg \
  "oklch(75% 0.2 40);50;80" \
  --luminance=False --saturation=False

# Boost vibrance - chroma only
imgcolorshine shine landscape.jpg \
  "oklch(70% 0.25 0);80;60" \
  --luminance=False --hue=False

# Dodge/burn effect - lightness only
imgcolorshine shine portrait.jpg \
  "oklch(85% 0.1 60);40;70" \
  --saturation=False --hue=False
```

## Channel-Specific Workflows

Master independent channel control for precise adjustments.

### Hue Shifting Techniques

#### Color Temperature Adjustment

Warm up or cool down images naturally:

```bash
# Global warming (toward orange/yellow)
imgcolorshine shine cool_image.jpg \
  "oklch(70% 0.08 60);80;50" \
  --luminance=False --saturation=False

# Global cooling (toward blue/cyan)  
imgcolorshine shine warm_image.jpg \
  "oklch(70% 0.08 240);80;50" \
  --luminance=False --saturation=False
```

#### Creative Color Shifts

Artistic hue modifications:

```bash
# Infrared effect (shift foliage to red)
imgcolorshine shine landscape.jpg \
  "oklch(60% 0.2 0);60;90" \
  --luminance=False --saturation=False

# Alien world (shift sky to purple)
imgcolorshine shine earth_scene.jpg \
  "oklch(50% 0.15 300);70;80" \
  --luminance=False --saturation=False
```

### Chroma/Saturation Control

#### Selective Desaturation

Remove color from specific areas while preserving others:

```bash
# Desaturate everything except reds
imgcolorshine shine colorful_scene.jpg \
  "oklch(50% 0.02 120);80;90" \    # Desaturate greens
  "oklch(50% 0.02 240);80;90" \    # Desaturate blues
  --luminance=False --hue=False
```

#### Vibrance Enhancement

Boost saturation with natural falloff:

```bash
# Intelligent saturation boost
imgcolorshine shine dull_photo.jpg \
  "oklch(70% 0.25 0);60;70" \      # Enhance reds
  "oklch(70% 0.25 120);60;70" \    # Enhance greens
  "oklch(70% 0.25 240);60;70" \    # Enhance blues
  --luminance=False --hue=False
```

### Lightness Manipulation

#### Advanced Dodge and Burn

Selective lightness adjustments:

```bash
# Brighten subject, darken background
imgcolorshine shine portrait.jpg \
  "oklch(85% 0.1 30);40;60" \      # Brighten skin tones
  "oklch(25% 0.05 200);60;70" \    # Darken blue background
  --saturation=False --hue=False
```

#### HDR-like Effects

Compress dynamic range artistically:

```bash
# Lift shadows, preserve highlights
imgcolorshine shine high_contrast.jpg \
  "oklch(60% 0.1 0);30;50" \       # Lift dark areas
  "oklch(90% 0.1 0);20;30" \       # Slightly compress highlights
  --saturation=False --hue=False
```

## Extended Strength Modes

Leverage strength values above 100 for special effects.

### Understanding Extended Strength

Strength > 100 progressively flattens the falloff curve:

| Strength | Falloff Behavior | Use Case |
|----------|------------------|----------|
| 50-100 | Natural gradient | Realistic adjustments |
| 101-130 | Slight flattening | Enhanced natural look |
| 131-170 | Moderate flattening | Stylized effects |
| 171-200 | Near-uniform | Duotone/posterization |

### Duotone and Posterization Effects

#### Classic Duotone

Create two-color artistic effects:

```bash
# Sepia duotone
imgcolorshine shine bw_photo.jpg \
  "oklch(80% 0.08 60);100;180" \   # Warm highlights
  "oklch(30% 0.05 30);100;180"     # Warm shadows
```

#### Multi-Tone Posterization

Limited color palette effects:

```bash
# Three-color poster effect
imgcolorshine shine complex_image.jpg \
  "oklch(85% 0.2 60);33;200" \     # Bright yellow
  "oklch(50% 0.25 0);33;200" \     # Pure red  
  "oklch(25% 0.15 240);33;200"     # Dark blue
```

### Gradient Flattening

Create smooth, uniform color zones:

```bash
# Flatten skin tone variations
imgcolorshine shine portrait.jpg \
  "oklch(75% 0.08 45);40;150" \
  --output_image=smooth_skin.jpg

# Uniform sky coloring
imgcolorshine shine landscape.jpg \
  "oklch(70% 0.12 240);60;160" \
  --output_image=uniform_sky.jpg
```

## Batch Processing Workflows

Automate processing for consistency across image sets.

### Bash Scripting (Linux/macOS)

#### Basic Batch Processing

```bash
#!/bin/bash
# batch_process.sh

# Configuration
ATTRACTOR="oklch(70% 0.1 45);50;70"
INPUT_DIR="./input"
OUTPUT_DIR="./output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process all JPEG files
for img in "$INPUT_DIR"/*.jpg; do
    filename=$(basename "$img")
    name_without_ext="${filename%.*}"
    
    echo "Processing: $filename"
    imgcolorshine shine "$img" "$ATTRACTOR" \
        --output_image="$OUTPUT_DIR/${name_without_ext}_processed.jpg" \
        --verbose=False
done

echo "Batch processing complete!"
```

#### Advanced Batch with Variations

```bash
#!/bin/bash
# advanced_batch.sh

# Define multiple looks
declare -A LOOKS=(
    ["warm"]="oklch(75% 0.1 50);60;70"
    ["cool"]="oklch(65% 0.1 240);60;70"  
    ["vintage"]="oklch(70% 0.05 40);70;60 oklch(40% 0.08 20);50;50"
    ["dramatic"]="oklch(30% 0.15 240);40;80 oklch(85% 0.12 60);40;80"
)

INPUT_DIR="./source"
OUTPUT_DIR="./processed"

for style in "${!LOOKS[@]}"; do
    mkdir -p "$OUTPUT_DIR/$style"
    
    for img in "$INPUT_DIR"/*.jpg; do
        filename=$(basename "$img")
        name_without_ext="${filename%.*}"
        
        echo "Creating $style version of $filename"
        imgcolorshine shine "$img" ${LOOKS[$style]} \
            --output_image="$OUTPUT_DIR/$style/${name_without_ext}_${style}.jpg"
    done
done
```

### Python Scripting

#### Batch Processing with Progress

```python
#!/usr/bin/env python3
"""
Advanced batch processing with progress tracking
"""
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def batch_process(input_dir, output_dir, attractor_specs, **kwargs):
    """
    Process images in batch with progress tracking
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    images = [f for f in input_path.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    # Process with progress bar
    for img_file in tqdm(images, desc="Processing images"):
        output_file = output_path / f"{img_file.stem}_processed{img_file.suffix}"
        
        cmd = [
            'imgcolorshine', 'shine', str(img_file),
            *attractor_specs,
            '--output_image', str(output_file)
        ]
        
        # Add optional parameters
        for key, value in kwargs.items():
            cmd.extend([f'--{key}', str(value)])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {img_file.name}: {e}")

# Usage example
if __name__ == "__main__":
    batch_process(
        input_dir="./input",
        output_dir="./output", 
        attractor_specs=["oklch(70% 0.1 50);60;70"],
        gpu=True,
        verbose=False
    )
```

#### Parameter Sweep Analysis

```python
#!/usr/bin/env python3
"""
Systematic parameter exploration
"""
import subprocess
from pathlib import Path
from itertools import product

def parameter_sweep(input_image, base_color, output_dir):
    """
    Test range of tolerance and strength values
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    tolerances = [30, 50, 70, 90]
    strengths = [40, 70, 100, 130]
    
    for tolerance, strength in product(tolerances, strengths):
        attractor = f"{base_color};{tolerance};{strength}"
        output_file = output_path / f"t{tolerance}_s{strength}.jpg"
        
        cmd = [
            'imgcolorshine', 'shine', str(input_image),
            attractor, '--output_image', str(output_file)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Generated: {output_file.name}")

# Usage
parameter_sweep("test_image.jpg", "oklch(70% 0.15 240)", "./sweep_results")
```

### PowerShell Scripting (Windows)

```powershell
# batch_process.ps1

param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputDir,
    
    [Parameter(Mandatory=$true)]
    [string]$Attractor
)

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir

# Get all image files
$imageFiles = Get-ChildItem -Path $InputDir -Include "*.jpg","*.jpeg","*.png" -File

foreach ($file in $imageFiles) {
    $outputName = $file.BaseName + "_processed" + $file.Extension
    $outputPath = Join-Path $OutputDir $outputName
    
    Write-Host "Processing: $($file.Name)"
    
    & imgcolorshine shine $file.FullName $Attractor --output_image $outputPath
}

Write-Host "Batch processing complete!"
```

## Creative Artistic Techniques

### Film Emulation

#### Kodak Portra Look

```bash
imgcolorshine shine portrait.jpg \
  "oklch(82% 0.06 45);40;50" \     # Warm skin highlights
  "oklch(45% 0.08 20);35;45" \     # Warm shadows
  "oklch(70% 0.15 350);25;40"      # Magenta midtones
```

#### Fuji Film Simulation

```bash
imgcolorshine shine landscape.jpg \
  "oklch(75% 0.12 160);50;60" \    # Green enhancement
  "oklch(60% 0.18 240);40;70" \    # Blue boost
  "oklch(80% 0.08 60);30;50"       # Warm highlights
```

#### Cinematic Color Grading

```bash
# Blade Runner 2049 style
imgcolorshine shine urban_scene.jpg \
  "oklch(35% 0.2 30);60;90" \      # Orange for warmth
  "oklch(25% 0.15 280);70;80" \    # Purple/blue for cool
  "oklch(15% 0.05 200);80;70"      # Deep blue shadows
```

### Abstract and Artistic Effects

#### Color Isolation

Keep one color, desaturate others:

```bash
# Red rose effect
imgcolorshine shine colorful_garden.jpg \
  "oklch(50% 0.02 60);70;90" \     # Desaturate yellows
  "oklch(50% 0.02 120);70;90" \    # Desaturate greens  
  "oklch(50% 0.02 240);70;90" \    # Desaturate blues
  --luminance=False --hue=False
```

#### Psychedelic Color Shifts

```bash
# Alien landscape
imgcolorshine shine normal_scene.jpg \
  "oklch(70% 0.25 300);60;120" \   # Shift greens to purple
  "oklch(80% 0.2 180);50;110" \    # Shift blues to cyan
  "oklch(60% 0.3 60);40;100"       # Boost orange/yellow
```

#### Infrared Simulation

```bash
# False color infrared
imgcolorshine shine vegetation.jpg \
  "oklch(80% 0.25 0);70;150" \     # Vegetation to red
  "oklch(40% 0.15 240);80;120" \   # Sky to deep blue
  --luminance=False --saturation=False
```

## Professional Workflow Integration

### Lightroom-Style Adjustments

#### Basic Panel Emulation

```bash
# Exposure correction
imgcolorshine shine underexposed.jpg \
  "oklch(70% 0.1 0);100;60" \
  --saturation=False --hue=False

# Highlight recovery
imgcolorshine shine overexposed.jpg \
  "oklch(80% 0.1 0);20;70" \
  --saturation=False --hue=False

# Shadow lifting
imgcolorshine shine dark_shadows.jpg \
  "oklch(60% 0.1 0);30;50" \
  --saturation=False --hue=False
```

#### HSL Panel Emulation

```bash
# Orange/Yellow skin tone adjustment
imgcolorshine shine portrait.jpg \
  "oklch(78% 0.08 50);35;60" \     # Adjust orange tones
  "oklch(85% 0.06 80);25;50"       # Adjust yellow tones
```

### DaVinci Resolve Integration

#### Primary Color Correction

```bash
# Lift, Gamma, Gain equivalent
imgcolorshine shine footage_frame.jpg \
  "oklch(40% 0.05 240);40;50" \    # Lift (shadows)
  "oklch(60% 0.03 30);50;40" \     # Gamma (midtones)
  "oklch(85% 0.04 60);30;60"       # Gain (highlights)
```

#### Secondary Color Correction

```bash
# Isolate and adjust specific colors
imgcolorshine shine scene.jpg \
  "oklch(75% 0.12 120);35;80" \    # Enhance grass greens
  "oklch(65% 0.15 210);40;70"      # Adjust sky blues
```

## Performance Considerations for Complex Workflows

### Optimization Strategies

#### LUT-Based Processing

For repeated similar transformations:

```bash
# Build LUT once, reuse for speed
imgcolorshine shine reference.jpg \
  "oklch(70% 0.1 50);60;70" \
  --lut_size=65 --output_image=reference_processed.jpg

# Subsequent images use cached LUT
imgcolorshine shine image1.jpg \
  "oklch(70% 0.1 50);60;70" \
  --lut_size=65

imgcolorshine shine image2.jpg \
  "oklch(70% 0.1 50);60;70" \
  --lut_size=65
```

#### GPU Acceleration

```bash
# Enable GPU for large batches
imgcolorshine shine large_image.jpg \
  "complex;attractor;specs" \
  --gpu=True --verbose=True
```

#### Hierarchical Processing

```bash
# Multi-resolution for very large images
imgcolorshine shine 8k_image.jpg \
  "attractor;specs" \
  --hierarchical=True --tile_size=1024
```

## Quality Control and Validation

### Systematic Testing

#### Before/After Comparison Scripts

```bash
#!/bin/bash
# create_comparison.sh

INPUT="$1"
ATTRACTOR="$2"
BASENAME=$(basename "$INPUT" .jpg)

# Process image
imgcolorshine shine "$INPUT" "$ATTRACTOR" \
    --output_image="${BASENAME}_processed.jpg"

# Create side-by-side comparison using ImageMagick
montage "$INPUT" "${BASENAME}_processed.jpg" \
    -geometry +5+5 \
    "${BASENAME}_comparison.jpg"

echo "Comparison saved as ${BASENAME}_comparison.jpg"
```

#### Parameter Documentation

```python
#!/usr/bin/env python3
"""
Document processing parameters in image metadata
"""
import subprocess
from PIL import Image
from PIL.ExifTags import TAGS

def add_processing_metadata(image_path, attractors, **kwargs):
    """Add processing parameters to image metadata"""
    # Process image
    subprocess.run(['imgcolorshine', 'shine', image_path, *attractors])
    
    # Add metadata
    img = Image.open(image_path)
    
    # Create processing description
    description = f"imgcolorshine: {', '.join(attractors)}"
    for key, value in kwargs.items():
        description += f", {key}={value}"
    
    # Save with metadata
    img.save(image_path, description=description)
```

## Next Steps

With advanced features mastered:

1. **[Performance Optimization](performance-optimization.md)** - Maximize processing speed
2. **[Color Science](color-science.md)** - Understand the mathematical foundations
3. **[API Reference](api-reference.md)** - Programmatic usage and integration

!!! tip "Creative Exploration"
    The most powerful techniques often combine multiple concepts. Try layering different attractor strategies and channel controls to create unique artistic looks.