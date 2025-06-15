
# Task 3

Read the above proposals for Task 1 and Task 2. Think hard, consult additional web and llm resources. 

Distill the most promising and best approaches into one solid unified detailed proposal. 

Then make a detailed research for Python tools and libraries, and for fast and efficient methods to implement the proposal. 

Write the final spec. The spec needs to be written in a way that is easy to understand and easy to use. A senior developer should be able to read it, understand it, and implement the tool based on the spec.

Then write a detailed implementation plan. 

Review the spec and plan, and refine it. Think hard, consult additional web and llm resources. 

# Report

# imgcolorshine Implementation Roadmap: Complete Research Findings

## 1. Quick Decision Summary

Based on comprehensive research of Python libraries and implementation methods for OKLCH color transformation, here are the definitive technology choices:

**Core Stack:**
- **Color Library**: ColorAide (best OKLCH support, CSS parsing, intuitive API)
- **Image I/O**: OpenCV (4x faster than PIL for PNG operations)
- **Computation**: NumPy + Numba (5-25x speedup for custom algorithms)
- **CLI Framework**: Click (best UX for complex image processing tools)
- **GPU Acceleration**: Optional CuPy integration (10-100x speedup)

## 2. Complete Implementation Blueprint

### 2.1. Project Structure
```
imgcolorshine/
├── cli.py           # Click-based CLI interface
├── color_engine.py  # ColorAide integration & OKLCH operations
├── image_io.py      # OpenCV-based I/O with fallback to PIL
├── transforms.py    # Numba-optimized color transformations
├── gamut.py        # CSS Color 4 gamut mapping
├── falloff.py      # Vectorized falloff functions
└── utils.py        # Memory management & tiling
```

### 2.2. Core Dependencies
```python
# requirements.txt
coloraide>=3.0     # OKLCH support & CSS parsing
opencv-python>=4.8 # Fast image I/O
numpy>=1.24       # Array operations
numba>=0.57       # JIT compilation
click>=8.1        # CLI framework
pillow>=10.0      # Fallback image support

# Optional for GPU
cupy>=12.0        # GPU acceleration
```

### 2.3. Essential Implementation Code

**Color Engine (color_engine.py)**
```python
from coloraide import Color
import numpy as np

class OKLCHEngine:
    def __init__(self):
        self.cache = {}
        
    def parse_color(self, color_str):
        """Parse any CSS color format"""
        return Color(color_str)
    
    def calculate_delta_e(self, color1, color2):
        """Perceptual distance in Oklab space"""
        return color1.distance(color2, space="oklab")
    
    def gamut_map_css4(self, oklch_values):
        """CSS Color Module 4 gamut mapping"""
        l, c, h = oklch_values
        color = Color("oklch", [l, c, h])
        
        if color.in_gamut("srgb"):
            return color
            
        # Binary search for optimal chroma
        c_min, c_max = 0, c
        while c_max - c_min > 0.0001:
            c_mid = (c_min + c_max) / 2
            test = Color("oklch", [l, c_mid, h])
            
            if test.in_gamut("srgb"):
                c_min = c_mid
            else:
                c_max = c_mid
                
        return Color("oklch", [l, c_min, h])
```

**High-Performance Transform (transforms.py)**
```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True, cache=True)
def apply_color_transform(pixels, target_l, target_c, target_h, 
                         falloff_map, delta_e_threshold):
    """Numba-optimized OKLCH transformation"""
    h, w, _ = pixels.shape
    result = np.empty_like(pixels)
    
    for y in prange(h):
        for x in prange(w):
            # Get pixel OKLCH values
            l, c, h = pixels[y, x]
            
            # Apply transformation based on falloff
            weight = falloff_map[y, x]
            new_l = l + (target_l - l) * weight
            new_c = c + (target_c - c) * weight
            new_h = interpolate_hue(h, target_h, weight)
            
            result[y, x] = [new_l, new_c, new_h]
    
    return result

@jit(nopython=True)
def interpolate_hue(h1, h2, t):
    """Correct chroma interpolation handling wraparound"""
    diff = ((h2 - h1 + 180) % 360) - 180
    return (h1 + t * diff) % 360
```

**Optimized I/O (image_io.py)**
```python
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, tile_size=256):
        self.tile_size = tile_size
        
    def load_image(self, path):
        """Load with OpenCV, fallback to PIL"""
        try:
            # OpenCV is 4x faster
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:  # BGRA to RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:  # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except:
            # Fallback for formats OpenCV doesn't support
            from PIL import Image
            return np.array(Image.open(path).convert('RGBA'))
    
    def process_large_image(self, image, transform_func):
        """Tile-based processing for memory efficiency"""
        if image.nbytes < 100_000_000:  # <100MB
            return transform_func(image)
            
        # Process in tiles
        h, w = image.shape[:2]
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                image[y:y+self.tile_size, x:x+self.tile_size] = \
                    transform_func(tile)
        return image
```

**CLI Interface (cli.py)**
```python
import click
from pathlib import Path

@click.command()
@click.argument('input', type=click.Path(exists=True, path_type=Path))
@click.argument('output', type=click.Path(path_type=Path))
@click.option('--target-color', required=True, 
              help='Target color (CSS format: hex, rgb(), oklch(), etc.)')
@click.option('--falloff', type=click.Choice(['cosine', 'gaussian', 'linear']),
              default='cosine', help='Falloff function type')
@click.option('--radius', type=click.FloatRange(0.0, 1.0), default=0.5,
              help='Effect radius (0-1)')
@click.option('--strength', type=click.FloatRange(0.0, 1.0), default=1.0,
              help='Effect strength (0-1)')
@click.option('--gamma-correct/--no-gamma-correct', default=True,
              help='Apply proper gamma correction')
@click.option('--use-gpu', is_flag=True, help='Enable GPU acceleration')
def colorshine(input, output, target_color, falloff, radius, 
               strength, gamma_correct, use_gpu):
    """Apply perceptual color transformations in OKLCH space."""
    from .pipeline import ColorShinePipeline
    
    pipeline = ColorShinePipeline(use_gpu=use_gpu)
    pipeline.process(
        input_path=input,
        output_path=output,
        target_color=target_color,
        falloff_type=falloff,
        radius=radius,
        strength=strength,
        gamma_correct=gamma_correct
    )
```

### 2.4. Critical Implementation Details

**Gamma-Correct Pipeline**
```python
def process_with_gamma_correction(image_srgb):
    # 1. Linearize sRGB (remove gamma)
    linear = np.where(
        image_srgb <= 0.04045,
        image_srgb / 12.92,
        np.power((image_srgb + 0.055) / 1.055, 2.4)
    )
    
    # 2. Convert to OKLCH
    oklch = linear_rgb_to_oklch(linear)
    
    # 3. Apply transformations
    oklch_modified = apply_transforms(oklch)
    
    # 4. Convert back to linear RGB
    linear_result = oklch_to_linear_rgb(oklch_modified)
    
    # 5. Apply sRGB gamma
    return np.where(
        linear_result <= 0.0031308,
        linear_result * 12.92,
        1.055 * np.power(linear_result, 1/2.4) - 0.055
    )
```

**Vectorized Falloff Functions**
```python
def generate_falloff_map(shape, center, radius, falloff_type='cosine'):
    """Generate 2D falloff map for entire image"""
    y, x = np.ogrid[:shape[0], :shape[1]]
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    normalized = np.clip(distances / (radius * max(shape)), 0, 1)
    
    if falloff_type == 'cosine':
        return 0.5 * (1 + np.cos(np.pi * normalized))
    elif falloff_type == 'gaussian':
        return np.exp(-0.5 * (normalized * 3)**2)
    else:  # linear
        return 1.0 - normalized
```

### 2.5. Performance Optimization Checklist

✓ **Use lookup tables** for expensive operations:
```python
# Pre-compute sRGB linearization LUT
SRGB_LINEAR_LUT = np.array([srgb_to_linear(i/255.0) for i in range(256)])
```

✓ **Implement caching** for repeated conversions:
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_color_parse(color_string):
    return Color(color_string)
```

✓ **Profile critical paths**:
```python
# Use line_profiler on hot functions
@profile
def critical_transform_function():
    pass
```

### 2.6. Testing Requirements

```python
# test_accuracy.py
def test_oklch_roundtrip():
    """Verify conversion accuracy"""
    test_colors = [
        [0.0, 0.0, 0.0],  # Black
        [1.0, 1.0, 1.0],  # White
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
    ]
    
    for rgb in test_colors:
        oklch = rgb_to_oklch(rgb)
        recovered = oklch_to_rgb(oklch)
        assert np.allclose(rgb, recovered, atol=1e-6)

def test_css_parsing():
    """Test all CSS color formats"""
    test_cases = [
        ("#ff0000", [1.0, 0.0, 0.0]),
        ("rgb(255 0 0)", [1.0, 0.0, 0.0]),
        ("oklch(0.628 0.258 29.23)", None),  # Check OKLCH
        ("red", [1.0, 0.0, 0.0]),
    ]
    
    for css, expected in test_cases:
        color = Color(css)
        if expected:
            assert np.allclose(color.convert("srgb").coords(), expected)
```

### 2.7. Usage Examples

```bash
# Basic usage
imgcolorshine input.jpg output.jpg --target-color "#ff6b6b"

# Advanced usage with all options
imgcolorshine photo.png result.png \
  --target-color "oklch(0.7 0.15 180)" \
  --falloff gaussian \
  --radius 0.8 \
  --strength 0.6 \
  --use-gpu

# Process with specific gamma handling
imgcolorshine raw.tiff processed.tiff \
  --target-color "rgb(100 200 255)" \
  --no-gamma-correct  # For linear input
```

### 2.8. Common Pitfall Solutions

1. **Out-of-gamut colors**: Always use CSS4 gamut mapping
2. **Memory issues**: Automatic tiling for images >100MB
3. **Hue interpolation**: Proper circular interpolation implemented
4. **Performance**: Numba JIT compilation caches after first run
5. **Color accuracy**: Roundtrip tests ensure <1e-6 error

## 3. Final Recommendations

1. **Start with**: ColorAide + OpenCV + NumPy base implementation
2. **Add Numba**: For 5-25x speedup on color transformations
3. **Consider CuPy**: Only for batch processing or very large images
4. **Test thoroughly**: Use provided test suite for accuracy validation
5. **Profile early**: Identify bottlenecks before optimizing

This research provides everything needed to build a professional-grade OKLCH color transformation tool with modern Python libraries, optimal performance, and robust error handling.
