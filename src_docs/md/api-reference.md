# Chapter 8: API Reference

This chapter provides comprehensive documentation for imgcolorshine's Python API, command-line interface, and programmatic usage.

## Python API Overview

The Python API provides full programmatic access to imgcolorshine's color transformation capabilities.

### Core Modules

```python
from imgcolorshine import (
    process_image,           # High-level processing function
    OKLCHEngine,            # Color engine and conversions
    ColorTransformer,       # Transformation pipeline
    Attractor,              # Attractor class
)

from imgcolorshine.io import ImageProcessor
from imgcolorshine.gpu import ArrayModule
from imgcolorshine.gamut import GamutMapper
```

## High-Level API

### process_image()

The simplest way to transform images programmatically.

```python
def process_image(
    input_image: str | Path | np.ndarray,
    attractors: list[str],
    output_image: str | Path | None = None,
    luminance: bool = True,
    saturation: bool = True,
    hue: bool = True,
    gpu: bool = True,
    lut_size: int = 0,
    tile_size: int = 1024,
    hierarchical: bool = False,
    spatial_accel: bool = True,
    fused_kernel: bool = False,
    verbose: bool = False
) -> np.ndarray | None:
    """
    Transform image colors using OKLCH attractors
    
    Args:
        input_image: Path to image file or numpy array
        attractors: List of attractor specifications ["color;tolerance;strength"]
        output_image: Output path (if None, returns array)
        luminance: Transform lightness channel
        saturation: Transform chroma channel  
        hue: Transform hue channel
        gpu: Use GPU acceleration if available
        lut_size: 3D LUT size (0=disabled, 65=recommended)
        tile_size: Tile size for large images
        hierarchical: Multi-resolution processing
        spatial_accel: Spatial acceleration structures
        fused_kernel: Use fused Numba kernels
        verbose: Enable detailed logging
        
    Returns:
        Transformed image as numpy array (if output_image is None)
        
    Raises:
        ValueError: Invalid attractor specification
        FileNotFoundError: Input image not found
        RuntimeError: Processing error
    """
```

#### Basic Usage

```python
import imgcolorshine

# Transform image with single attractor
result = imgcolorshine.process_image(
    "photo.jpg",
    ["orange;50;70"],
    output_image="warm_photo.jpg"
)

# Multiple attractors
imgcolorshine.process_image(
    "landscape.jpg", 
    ["oklch(70% 0.15 50);60;70", "oklch(40% 0.12 240);50;60"],
    output_image="cinematic.jpg",
    gpu=True,
    lut_size=65
)

# Return array instead of saving
import numpy as np

image_array = imgcolorshine.process_image(
    "input.jpg",
    ["blue;50;70"],
    output_image=None  # Return array
)

print(f"Result shape: {image_array.shape}")
print(f"Data type: {image_array.dtype}")
```

#### Channel Control

```python
# Hue shift only
imgcolorshine.process_image(
    "photo.jpg",
    ["oklch(70% 0.15 240);60;80"],
    luminance=False,
    saturation=False,
    hue=True,
    output_image="hue_shifted.jpg"
)

# Saturation boost only
imgcolorshine.process_image(
    "dull_image.jpg",
    ["red;70;60"],
    luminance=False,
    saturation=True,
    hue=False,
    output_image="vibrant.jpg"
)
```

## Core Classes

### OKLCHEngine

The color engine handles color space conversions and attractor management.

```python
class OKLCHEngine:
    """OKLCH color space engine for conversions and attractor management"""
    
    def __init__(self, gpu_module: ArrayModule | None = None):
        """Initialize engine with optional GPU support"""
        
    def parse_color(self, color_string: str) -> tuple[float, float, float]:
        """
        Parse CSS color string to OKLCH values
        
        Args:
            color_string: CSS color ("red", "#ff0000", "rgb(255,0,0)", etc.)
            
        Returns:
            Tuple of (L, C, H) in OKLCH space
            
        Raises:
            ValueError: Invalid color format
        """
        
    def create_attractor(
        self, 
        color: str, 
        tolerance: float, 
        strength: float
    ) -> 'Attractor':
        """
        Create an attractor from specification
        
        Args:
            color: CSS color string
            tolerance: Influence range (0-100)
            strength: Transformation intensity (0-200)
            
        Returns:
            Configured Attractor instance
        """
        
    def srgb_to_oklch(self, rgb_array: np.ndarray) -> np.ndarray:
        """Convert sRGB array to OKLCH"""
        
    def oklch_to_srgb(self, oklch_array: np.ndarray) -> np.ndarray:
        """Convert OKLCH array to sRGB"""
```

#### Usage Examples

```python
from imgcolorshine import OKLCHEngine

# Create engine
engine = OKLCHEngine()

# Parse different color formats
red_oklch = engine.parse_color("red")
print(f"Red in OKLCH: L={red_oklch[0]:.1f}% C={red_oklch[1]:.2f} H={red_oklch[2]:.0f}°")

blue_oklch = engine.parse_color("#0080ff")
custom_oklch = engine.parse_color("oklch(70% 0.15 240)")

# Create attractors
warm_attractor = engine.create_attractor("orange", tolerance=50, strength=70)
cool_attractor = engine.create_attractor("oklch(60% 0.12 240)", 40, 60)

# Color space conversions
import numpy as np

# Create test image data
rgb_image = np.random.rand(100, 100, 3).astype(np.float32)

# Convert to OKLCH
oklch_image = engine.srgb_to_oklch(rgb_image)
print(f"OKLCH shape: {oklch_image.shape}")

# Convert back to sRGB
rgb_back = engine.oklch_to_srgb(oklch_image)
```

### Attractor

Represents a single color attractor with all its parameters.

```python
class Attractor:
    """Color attractor with influence parameters"""
    
    def __init__(
        self,
        color_oklch: tuple[float, float, float],
        tolerance: float,
        strength: float,
        color_string: str | None = None
    ):
        """
        Initialize attractor
        
        Args:
            color_oklch: Target color in OKLCH space
            tolerance: Influence range percentage (0-100)
            strength: Transformation intensity (0-200)
            color_string: Original color specification for reference
        """
        
    @property
    def lightness(self) -> float:
        """Target lightness (L) value"""
        
    @property 
    def chroma(self) -> float:
        """Target chroma (C) value"""
        
    @property
    def hue(self) -> float:
        """Target hue (H) value in degrees"""
        
    def calculate_influence(
        self, 
        pixel_oklch: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Calculate influence weights for pixels
        
        Args:
            pixel_oklch: Pixel colors in OKLCH space
            threshold: Distance threshold from tolerance analysis
            
        Returns:
            Influence weights (0.0 to 1.0)
        """
```

#### Usage Examples

```python
from imgcolorshine import OKLCHEngine

engine = OKLCHEngine()

# Create attractor manually
attractor = engine.create_attractor("teal", 60, 80)

# Access properties
print(f"Target color:")
print(f"  Lightness: {attractor.lightness:.1f}%")
print(f"  Chroma: {attractor.chroma:.2f}")
print(f"  Hue: {attractor.hue:.0f}°")

# Check influence on specific colors
test_colors = np.array([
    [60.0, 0.15, 180.0],  # Cyan-ish
    [40.0, 0.20, 180.0],  # Dark cyan
    [80.0, 0.10, 180.0]   # Light cyan
])

# Would need threshold from tolerance analysis
# influences = attractor.calculate_influence(test_colors, threshold=0.15)
```

### ColorTransformer

The main transformation pipeline that applies attractors to images.

```python
class ColorTransformer:
    """Main color transformation pipeline"""
    
    def __init__(
        self,
        engine: OKLCHEngine,
        transform_lightness: bool = True,
        transform_chroma: bool = True,
        transform_hue: bool = True,
        gpu_module: ArrayModule | None = None
    ):
        """
        Initialize transformer
        
        Args:
            engine: OKLCH engine for color operations
            transform_lightness: Enable lightness transformation
            transform_chroma: Enable chroma transformation
            transform_hue: Enable hue transformation
            gpu_module: GPU acceleration module
        """
        
    def transform_image(
        self,
        image_rgb: np.ndarray,
        attractors: list[Attractor],
        processing_options: dict | None = None
    ) -> np.ndarray:
        """
        Transform image using attractors
        
        Args:
            image_rgb: Input image in sRGB space
            attractors: List of configured attractors
            processing_options: Additional processing configuration
            
        Returns:
            Transformed image in sRGB space
        """
        
    def analyze_tolerance_thresholds(
        self,
        image_oklch: np.ndarray,
        attractors: list[Attractor]
    ) -> dict[int, float]:
        """
        Analyze tolerance thresholds for attractors
        
        Args:
            image_oklch: Image in OKLCH space
            attractors: List of attractors
            
        Returns:
            Dictionary mapping attractor index to threshold
        """
```

#### Usage Examples

```python
from imgcolorshine import OKLCHEngine, ColorTransformer
from imgcolorshine.io import ImageProcessor
import numpy as np

# Setup
engine = OKLCHEngine()
transformer = ColorTransformer(
    engine,
    transform_lightness=True,
    transform_chroma=True,
    transform_hue=False  # Preserve original hues
)

# Load image
processor = ImageProcessor()
image = processor.load_image("photo.jpg")

# Create attractors
attractors = [
    engine.create_attractor("orange", 50, 70),
    engine.create_attractor("blue", 40, 60)
]

# Transform
result = transformer.transform_image(image, attractors)

# Save result
processor.save_image(result, "transformed.jpg")

# Analyze what happened
image_oklch = engine.srgb_to_oklch(image)
thresholds = transformer.analyze_tolerance_thresholds(image_oklch, attractors)

for i, threshold in thresholds.items():
    print(f"Attractor {i}: threshold = {threshold:.3f}")
```

## Utility Classes

### ImageProcessor

Handles image loading, saving, and format conversions.

```python
class ImageProcessor:
    """Image I/O and format handling"""
    
    def __init__(self, target_bit_depth: int = 8):
        """
        Initialize processor
        
        Args:
            target_bit_depth: Output bit depth (8 or 16)
        """
        
    def load_image(self, image_path: str | Path) -> np.ndarray:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as float32 numpy array (0.0-1.0 range)
            
        Raises:
            FileNotFoundError: Image file not found
            ValueError: Unsupported image format
        """
        
    def save_image(
        self,
        image_array: np.ndarray,
        output_path: str | Path,
        quality: int = 95
    ) -> None:
        """
        Save image to file
        
        Args:
            image_array: Image data (0.0-1.0 range)
            output_path: Output file path
            quality: JPEG quality (1-100, ignored for PNG/TIFF)
        """
        
    @staticmethod
    def supported_formats() -> list[str]:
        """Return list of supported image formats"""
```

#### Usage Examples

```python
from imgcolorshine.io import ImageProcessor
import numpy as np

# Create processor
processor = ImageProcessor(target_bit_depth=8)

# Check supported formats
formats = processor.supported_formats()
print(f"Supported formats: {formats}")

# Load image
try:
    image = processor.load_image("input.jpg")
    print(f"Loaded image: {image.shape}, dtype: {image.dtype}")
    print(f"Value range: {image.min():.3f} to {image.max():.3f}")
except FileNotFoundError:
    print("Image file not found")

# Create test image
test_image = np.random.rand(256, 256, 3).astype(np.float32)

# Save in different formats
processor.save_image(test_image, "output.jpg", quality=90)
processor.save_image(test_image, "output.png")
processor.save_image(test_image, "output.tiff")
```

### ArrayModule

Provides unified interface for CPU/GPU array operations.

```python
class ArrayModule:
    """Unified CPU/GPU array operations"""
    
    def __init__(self, prefer_gpu: bool = True):
        """
        Initialize array module
        
        Args:
            prefer_gpu: Use GPU if available
        """
        
    @property
    def gpu_available(self) -> bool:
        """Check if GPU acceleration is available"""
        
    @property
    def device_name(self) -> str:
        """Get device name (CPU or GPU model)"""
        
    def asarray(self, data: np.ndarray) -> np.ndarray:
        """Convert to appropriate array type (CPU/GPU)"""
        
    def to_cpu(self, array: np.ndarray) -> np.ndarray:
        """Transfer array to CPU memory"""
        
    def to_gpu(self, array: np.ndarray) -> np.ndarray:
        """Transfer array to GPU memory"""
```

#### Usage Examples

```python
from imgcolorshine.gpu import ArrayModule
import numpy as np

# Create array module
am = ArrayModule(prefer_gpu=True)

print(f"GPU available: {am.gpu_available}")
print(f"Device: {am.device_name}")

# Create test data
cpu_data = np.random.rand(1000, 1000, 3).astype(np.float32)

if am.gpu_available:
    # Transfer to GPU
    gpu_data = am.to_gpu(cpu_data)
    print(f"GPU data type: {type(gpu_data)}")
    
    # Perform operations on GPU
    # (actual operations would be done by imgcolorshine internals)
    
    # Transfer back to CPU
    result = am.to_cpu(gpu_data)
    print(f"Result shape: {result.shape}")
else:
    print("GPU not available, using CPU")
    result = am.asarray(cpu_data)
```

## Command-Line Interface

### Main Command Structure

```bash
imgcolorshine shine INPUT_IMAGE ATTRACTOR1 [ATTRACTOR2 ...] [OPTIONS]
```

### Command-Line Options

#### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output_image` | PATH | Auto-generated | Output file path |
| `--luminance` | BOOL | True | Transform lightness channel |
| `--saturation` | BOOL | True | Transform chroma channel |
| `--hue` | BOOL | True | Transform hue channel |
| `--verbose` | BOOL | False | Enable detailed logging |

#### Performance Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gpu` | BOOL | True | Use GPU acceleration |
| `--lut_size` | INT | 0 | 3D LUT size (0=disabled) |
| `--tile_size` | INT | 1024 | Tile size for large images |
| `--hierarchical` | BOOL | False | Multi-resolution processing |
| `--spatial_accel` | BOOL | True | Spatial acceleration |
| `--fused_kernel` | BOOL | False | Use fused kernels |

#### Examples

```bash
# Basic transformation
imgcolorshine shine photo.jpg "orange;50;70"

# Multiple attractors with custom output
imgcolorshine shine landscape.jpg \
  "oklch(70% 0.15 50);60;70" \
  "oklch(40% 0.12 240);50;60" \
  --output_image=cinematic.jpg

# Channel-specific transformation
imgcolorshine shine portrait.jpg "blue;60;80" \
  --luminance=False --saturation=False

# Performance optimization
imgcolorshine shine large_image.jpg "red;50;70" \
  --gpu=True --lut_size=65 --tile_size=2048

# Verbose output
imgcolorshine shine test.jpg "green;40;60" --verbose=True
```

### Error Handling

The CLI provides detailed error messages:

```bash
# Invalid color format
$ imgcolorshine shine image.jpg "not-a-color;50;70"
Error: Invalid color specification: 'not-a-color'

# File not found
$ imgcolorshine shine missing.jpg "red;50;70"
Error: Input image not found: 'missing.jpg'

# Invalid parameter range
$ imgcolorshine shine image.jpg "red;150;70"
Error: Tolerance must be between 0 and 100, got 150
```

## Advanced Usage Patterns

### Batch Processing

```python
#!/usr/bin/env python3
"""
Professional batch processing script
"""
import imgcolorshine
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging

def process_single_image(args):
    """Process a single image with error handling"""
    input_path, output_path, attractors, options = args
    
    try:
        imgcolorshine.process_image(
            str(input_path),
            attractors,
            output_image=str(output_path),
            **options
        )
        return f"Success: {input_path.name}"
    except Exception as e:
        return f"Error processing {input_path.name}: {e}"

def batch_process(
    input_dir: Path,
    output_dir: Path,
    attractors: list[str],
    max_workers: int = 4,
    **processing_options
):
    """Process directory of images in parallel"""
    
    # Setup
    output_dir.mkdir(exist_ok=True)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    # Prepare arguments
    tasks = []
    for img_file in image_files:
        output_file = output_dir / f"{img_file.stem}_processed{img_file.suffix}"
        tasks.append((img_file, output_file, attractors, processing_options))
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_image, tasks))
    
    # Report results
    for result in results:
        print(result)

# Usage
if __name__ == "__main__":
    batch_process(
        input_dir=Path("./input"),
        output_dir=Path("./output"),
        attractors=["oklch(70% 0.1 50);60;70"],
        gpu=True,
        lut_size=65,
        verbose=False
    )
```

### Custom Color Analysis

```python
#!/usr/bin/env python3
"""
Analyze image colors and suggest attractors
"""
import imgcolorshine
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def analyze_image_colors(image_path: str, n_colors: int = 5):
    """Analyze dominant colors in image"""
    
    # Load and convert image
    from imgcolorshine.io import ImageProcessor
    from imgcolorshine import OKLCHEngine
    
    processor = ImageProcessor()
    engine = OKLCHEngine()
    
    image_rgb = processor.load_image(image_path)
    image_oklch = engine.srgb_to_oklch(image_rgb)
    
    # Reshape for clustering
    pixels = image_oklch.reshape(-1, 3)
    
    # Find dominant colors using K-means
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_
    
    # Count pixels in each cluster
    labels = kmeans.labels_
    counts = Counter(labels)
    
    # Sort by frequency
    sorted_colors = []
    for label in sorted(counts.keys(), key=lambda x: counts[x], reverse=True):
        color = dominant_colors[label]
        percentage = counts[label] / len(labels) * 100
        sorted_colors.append((color, percentage))
    
    return sorted_colors

def suggest_attractors(dominant_colors, style="complementary"):
    """Suggest attractors based on color analysis"""
    
    attractors = []
    
    if style == "complementary":
        # Use complementary colors
        for color, percentage in dominant_colors[:2]:  # Top 2 colors
            L, C, H = color
            comp_H = (H + 180) % 360
            
            # Create attractor for complementary hue
            tolerance = min(70, max(30, percentage))  # Scale with dominance
            strength = 60
            
            attractor_spec = f"oklch({L:.0f}% {C:.2f} {comp_H:.0f});{tolerance:.0f};{strength}"
            attractors.append(attractor_spec)
    
    elif style == "enhance":
        # Enhance existing colors
        for color, percentage in dominant_colors[:3]:
            L, C, H = color
            
            # Boost chroma slightly
            enhanced_C = min(0.4, C * 1.2)
            
            tolerance = min(60, max(20, percentage))
            strength = 50
            
            attractor_spec = f"oklch({L:.0f}% {enhanced_C:.2f} {H:.0f});{tolerance:.0f};{strength}"
            attractors.append(attractor_spec)
    
    return attractors

# Usage example
if __name__ == "__main__":
    # Analyze image
    colors = analyze_image_colors("photo.jpg", n_colors=5)
    
    print("Dominant colors:")
    for i, (color, pct) in enumerate(colors):
        L, C, H = color
        print(f"  {i+1}: L={L:.0f}% C={C:.2f} H={H:.0f}° ({pct:.1f}%)")
    
    # Suggest attractors
    attractors = suggest_attractors(colors, style="complementary")
    
    print("\nSuggested attractors:")
    for attractor in attractors:
        print(f"  {attractor}")
    
    # Apply transformation
    imgcolorshine.process_image(
        "photo.jpg",
        attractors,
        output_image="auto_graded.jpg",
        verbose=True
    )
```

### Integration with Image Processing Pipelines

```python
#!/usr/bin/env python3
"""
Integration with existing image processing workflows
"""
import imgcolorshine
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def advanced_image_pipeline(
    input_path: str,
    output_path: str,
    attractors: list[str],
    enhance_contrast: float = 1.0,
    enhance_sharpness: float = 1.0,
    resize_factor: float = 1.0
):
    """
    Complete image processing pipeline with color transformation
    """
    
    # Load image
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if resize_factor != 1.0:
        new_height = int(image.shape[0] * resize_factor)
        new_width = int(image.shape[1] * resize_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to PIL for enhancements
    pil_image = Image.fromarray(image)
    
    # Enhance contrast
    if enhance_contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(enhance_contrast)
    
    # Enhance sharpness
    if enhance_sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(enhance_sharpness)
    
    # Convert back to numpy for imgcolorshine
    image = np.array(pil_image).astype(np.float32) / 255.0
    
    # Apply color transformation
    result = imgcolorshine.process_image(
        image,  # Pass array directly
        attractors,
        output_image=None,  # Get array back
        gpu=True,
        lut_size=65
    )
    
    # Final processing
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    final_image = Image.fromarray(result)
    
    # Save with high quality
    final_image.save(output_path, quality=95, optimize=True)
    
    return result

# Usage
if __name__ == "__main__":
    result = advanced_image_pipeline(
        "input.jpg",
        "processed.jpg",
        attractors=["oklch(70% 0.12 50);60;70"],
        enhance_contrast=1.1,
        enhance_sharpness=1.05,
        resize_factor=0.8
    )
    
    print(f"Processing complete: {result.shape}")
```

## Error Reference

### Common Exceptions

#### ValueError
- Invalid color format
- Parameter out of range
- Unsupported image format

#### FileNotFoundError
- Input image not found
- Output directory doesn't exist

#### RuntimeError
- GPU memory error
- Processing pipeline failure
- Numerical computation error

#### MemoryError
- Image too large for available RAM
- GPU memory exhausted

### Error Handling Best Practices

```python
import imgcolorshine
import logging

def robust_processing(input_path, attractors, output_path):
    """Process image with comprehensive error handling"""
    
    try:
        result = imgcolorshine.process_image(
            input_path,
            attractors,
            output_image=output_path,
            gpu=True,
            verbose=True
        )
        
        logging.info(f"Successfully processed {input_path}")
        return True
        
    except ValueError as e:
        logging.error(f"Invalid parameters: {e}")
        return False
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return False
        
    except MemoryError as e:
        logging.warning(f"Memory error, trying with smaller tiles: {e}")
        
        # Retry with smaller tiles
        try:
            result = imgcolorshine.process_image(
                input_path,
                attractors,
                output_image=output_path,
                gpu=False,  # Disable GPU
                tile_size=512,  # Smaller tiles
                hierarchical=True  # Multi-resolution
            )
            logging.info(f"Successfully processed with fallback settings")
            return True
            
        except Exception as e2:
            logging.error(f"Failed even with fallback: {e2}")
            return False
            
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False
```

## Next Steps

With the API reference mastered:

1. **[Development](development.md)** - Contribute to imgcolorshine
2. **Explore Examples** - Study the `/examples` directory in the repository
3. **Build Applications** - Create custom tools using the API

!!! tip "API Evolution"
    The Python API follows semantic versioning. Minor version updates may add new features while maintaining backward compatibility. Check the changelog for API changes between versions.