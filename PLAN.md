# Hierarchical Processing & Spatial Acceleration Implementation Plan

## Executive Summary
This plan details the implementation of two major algorithmic optimizations for imgcolorshine:
1. **Hierarchical Processing**: Multi-resolution processing with adaptive refinement (2-5x speedup)
2. **Spatial Acceleration**: KD-tree based early termination and coherence optimization (3-10x speedup)

Combined, these optimizations target 5-20x performance improvement for large images with localized color transformations.

## 1. Hierarchical Processing Implementation

### 1.1 Concept
Process images at multiple resolutions, starting with coarse levels to create an "influence map" that guides fine-level processing. This reduces computation by only processing pixels that differ significantly from the coarse approximation.

### 1.2 Architecture Design

**New Module: `src/imgcolorshine/hierarchical.py`**

```python
import cv2
import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class PyramidLevel:
    """Represents one level in the image pyramid."""
    image: np.ndarray
    scale: float
    shape: Tuple[int, int]
    level: int

class HierarchicalProcessor:
    """Multi-resolution image processing with adaptive refinement."""
    
    def __init__(self, 
                 min_size: int = 64,
                 difference_threshold: float = 0.1,
                 pyramid_factor: float = 0.5,
                 use_adaptive_subdivision: bool = True):
        """
        Args:
            min_size: Minimum dimension for coarsest level
            difference_threshold: Threshold for refinement in OKLCH space
            pyramid_factor: Downsampling factor between levels
            use_adaptive_subdivision: Enable gradient-based subdivision
        """
        self.min_size = min_size
        self.difference_threshold = difference_threshold
        self.pyramid_factor = pyramid_factor
        self.use_adaptive_subdivision = use_adaptive_subdivision
        self.pyramid_levels: List[PyramidLevel] = []
```

### 1.3 Core Algorithms

#### 1.3.1 Image Pyramid Construction
```python
def build_pyramid(self, image: np.ndarray) -> List[PyramidLevel]:
    """Build Gaussian pyramid from input image."""
    levels = []
    current = image.copy()
    level = 0
    
    while min(current.shape[:2]) > self.min_size:
        levels.append(PyramidLevel(
            image=current,
            scale=self.pyramid_factor ** level,
            shape=current.shape[:2],
            level=level
        ))
        
        # Downsample for next level
        current = cv2.pyrDown(current)
        level += 1
    
    # Add final level
    levels.append(PyramidLevel(
        image=current,
        scale=self.pyramid_factor ** level,
        shape=current.shape[:2],
        level=level
    ))
    
    return levels
```

#### 1.3.2 Difference Mask Generation
```python
def compute_difference_mask(self,
                          fine_level: np.ndarray,
                          coarse_upsampled: np.ndarray,
                          threshold: float) -> np.ndarray:
    """
    Create mask of pixels that need refinement.
    
    Uses perceptual distance in OKLCH space to determine
    which pixels differ significantly from coarse approximation.
    """
    # Convert to OKLCH for perceptual comparison
    fine_oklch = self.rgb_to_oklch(fine_level)
    coarse_oklch = self.rgb_to_oklch(coarse_upsampled)
    
    # Calculate perceptual distance
    delta_l = fine_oklch[..., 0] - coarse_oklch[..., 0]
    delta_c = fine_oklch[..., 1] - coarse_oklch[..., 1]
    delta_h = self.circular_distance(fine_oklch[..., 2], coarse_oklch[..., 2])
    
    # Combined perceptual distance
    distance = np.sqrt(delta_l**2 + delta_c**2 + (delta_h/360)**2)
    
    # Create binary mask
    return distance > threshold
```

#### 1.3.3 Adaptive Subdivision
```python
def detect_gradient_regions(self, 
                          image: np.ndarray,
                          gradient_threshold: float = 0.05) -> np.ndarray:
    """
    Detect regions with high color gradients that need finer processing.
    """
    # Convert to OKLCH
    oklch = self.rgb_to_oklch(image)
    
    # Calculate gradients for each channel
    grad_l = np.abs(cv2.Sobel(oklch[..., 0], cv2.CV_64F, 1, 0, ksize=3)) + \
             np.abs(cv2.Sobel(oklch[..., 0], cv2.CV_64F, 0, 1, ksize=3))
    
    grad_c = np.abs(cv2.Sobel(oklch[..., 1], cv2.CV_64F, 1, 0, ksize=3)) + \
             np.abs(cv2.Sobel(oklch[..., 1], cv2.CV_64F, 0, 1, ksize=3))
    
    # Combine gradients
    gradient_magnitude = np.sqrt(grad_l**2 + grad_c**2)
    
    # Dilate to ensure coverage of gradient regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient_mask = gradient_magnitude > gradient_threshold
    gradient_mask = cv2.dilate(gradient_mask.astype(np.uint8), kernel)
    
    return gradient_mask.astype(bool)
```

### 1.4 Main Processing Algorithm

```python
def process_hierarchical(self,
                        image: np.ndarray,
                        transform_func: Callable,
                        attractors: np.ndarray,
                        tolerances: np.ndarray,
                        strengths: np.ndarray,
                        channels: List[bool]) -> np.ndarray:
    """
    Hierarchical multi-resolution processing.
    
    Algorithm:
    1. Build image pyramid
    2. Process coarsest level completely
    3. For each finer level:
       a. Upsample previous result as influence map
       b. Compute difference mask
       c. Add gradient regions if adaptive
       d. Process only masked pixels
       e. Blend with influence map
    """
    # Build pyramid
    pyramid = self.build_pyramid(image)
    logger.info(f"Built pyramid with {len(pyramid)} levels")
    
    # Process coarsest level completely
    coarsest = pyramid[-1]
    logger.debug(f"Processing coarsest level: {coarsest.shape}")
    result = transform_func(coarsest.image, attractors, tolerances, strengths, channels)
    
    # Process from coarse to fine
    for i in range(len(pyramid) - 2, -1, -1):
        level = pyramid[i]
        logger.debug(f"Processing level {i}: {level.shape}")
        
        # Upsample previous result
        h, w = level.shape[:2]
        upsampled = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Compute refinement mask
        diff_mask = self.compute_difference_mask(level.image, upsampled, self.difference_threshold)
        
        # Add gradient regions if enabled
        if self.use_adaptive_subdivision:
            gradient_mask = self.detect_gradient_regions(level.image)
            refinement_mask = diff_mask | gradient_mask
        else:
            refinement_mask = diff_mask
        
        # Process only masked pixels
        if np.any(refinement_mask):
            # Create copy for selective processing
            refined = upsampled.copy()
            
            # Process masked regions
            masked_pixels = level.image[refinement_mask]
            transformed = transform_func(masked_pixels, attractors, tolerances, strengths, channels)
            refined[refinement_mask] = transformed
            
            result = refined
        else:
            # No refinement needed, use upsampled result
            result = upsampled
        
        logger.debug(f"Refined {np.sum(refinement_mask)} pixels ({np.mean(refinement_mask)*100:.1f}%)")
    
    return result
```

## 2. Spatial Acceleration Implementation

### 2.1 Concept
Use spatial data structures (KD-tree) to quickly identify which pixels can be affected by which attractors, enabling early termination for pixels outside all influence radii.

### 2.2 Architecture Design

**New Module: `src/imgcolorshine/spatial_accel.py`**

```python
from scipy.spatial import KDTree
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class InfluenceRegion:
    """Represents an attractor's region of influence."""
    center: np.ndarray  # Oklab coordinates
    radius: float      # Max perceptual distance
    attractor_idx: int

class SpatialAccelerator:
    """Spatial acceleration for color transformation queries."""
    
    def __init__(self, 
                 uniformity_threshold: float = 0.01,
                 tile_size: int = 64):
        """
        Args:
            uniformity_threshold: Variance threshold for uniform tiles
            tile_size: Size for tile coherence analysis
        """
        self.uniformity_threshold = uniformity_threshold
        self.tile_size = tile_size
        self.color_tree: Optional[KDTree] = None
        self.influence_regions: List[InfluenceRegion] = []
        self.tile_cache: Dict[Tuple, dict] = {}
```

### 2.3 Core Algorithms

#### 2.3.1 Spatial Index Construction
```python
def build_spatial_index(self, 
                       attractors_oklab: np.ndarray,
                       tolerances: np.ndarray) -> None:
    """
    Build KD-tree and influence regions from attractors.
    """
    # Map tolerances to perceptual distances
    from imgcolorshine.transforms import MAX_DELTA_E
    max_distances = MAX_DELTA_E * (tolerances / 100.0)
    
    # Build KD-tree from attractor coordinates
    self.color_tree = KDTree(attractors_oklab)
    
    # Store influence regions
    self.influence_regions = [
        InfluenceRegion(
            center=attractors_oklab[i],
            radius=max_distances[i],
            attractor_idx=i
        )
        for i in range(len(attractors_oklab))
    ]
    
    logger.debug(f"Built spatial index with {len(self.influence_regions)} attractors")
```

#### 2.3.2 Pixel Influence Detection
```python
def get_influenced_pixels_mask(self, 
                              pixels_oklab: np.ndarray) -> np.ndarray:
    """
    Create mask of pixels within any attractor's influence.
    
    Returns:
        Boolean mask where True = pixel needs processing
    """
    h, w = pixels_oklab.shape[:2]
    pixels_flat = pixels_oklab.reshape(-1, 3)
    
    # Initialize mask
    mask_flat = np.zeros(len(pixels_flat), dtype=bool)
    
    # Check each influence region
    for region in self.influence_regions:
        # Find pixels within this region's radius
        distances, indices = self.color_tree.query(
            pixels_flat,
            k=1,
            distance_upper_bound=region.radius
        )
        
        # Mark influenced pixels
        valid_indices = indices < len(self.color_tree.data)
        mask_flat[valid_indices] = True
    
    return mask_flat.reshape(h, w)
```

#### 2.3.3 Per-Pixel Attractor Query
```python
def query_pixel_attractors(self, 
                          pixel_oklab: np.ndarray) -> List[int]:
    """
    Find which attractors influence a specific pixel.
    
    Returns:
        List of attractor indices that influence this pixel
    """
    influencing_attractors = []
    
    for region in self.influence_regions:
        distance = np.linalg.norm(pixel_oklab - region.center)
        if distance <= region.radius:
            influencing_attractors.append(region.attractor_idx)
    
    return influencing_attractors
```

#### 2.3.4 Tile Coherence Analysis
```python
def analyze_tile_coherence(self,
                          tile_oklab: np.ndarray,
                          tile_coords: Tuple[int, int, int, int]) -> dict:
    """
    Analyze spatial coherence within a tile for optimization.
    """
    # Check cache first
    if tile_coords in self.tile_cache:
        return self.tile_cache[tile_coords]
    
    # Reshape tile for analysis
    pixels = tile_oklab.reshape(-1, 3)
    
    # Calculate statistics
    mean_color = np.mean(pixels, axis=0)
    variance = np.var(pixels, axis=0).sum()
    
    # Check uniformity
    is_uniform = variance < self.uniformity_threshold
    
    # Find dominant attractors for tile
    if is_uniform:
        # For uniform tiles, check which attractors affect the mean color
        dominant_attractors = self.query_pixel_attractors(mean_color)
    else:
        # Sample corners and center
        h, w = tile_oklab.shape[:2]
        sample_points = [
            tile_oklab[0, 0],      # Top-left
            tile_oklab[0, w-1],    # Top-right
            tile_oklab[h-1, 0],    # Bottom-left
            tile_oklab[h-1, w-1],  # Bottom-right
            tile_oklab[h//2, w//2] # Center
        ]
        
        # Find common attractors
        attractor_sets = [set(self.query_pixel_attractors(p)) for p in sample_points]
        dominant_attractors = list(set.intersection(*attractor_sets))
    
    result = {
        'uniform': is_uniform,
        'mean_color': mean_color,
        'variance': variance,
        'dominant_attractors': dominant_attractors,
        'coords': tile_coords
    }
    
    # Cache result
    self.tile_cache[tile_coords] = result
    
    return result
```

### 2.4 Optimized Transformation

```python
def transform_with_spatial_accel(self,
                                pixels_oklab: np.ndarray,
                                attractors_oklab: np.ndarray,
                                tolerances: np.ndarray,
                                strengths: np.ndarray,
                                transform_func: Callable,
                                channels: List[bool]) -> np.ndarray:
    """
    Transform pixels using spatial acceleration.
    """
    # Build spatial index
    self.build_spatial_index(attractors_oklab, tolerances)
    
    # Get influenced pixels mask
    influence_mask = self.get_influenced_pixels_mask(pixels_oklab)
    
    # Early exit if no pixels are influenced
    if not np.any(influence_mask):
        logger.info("No pixels within attractor influence, skipping transformation")
        return pixels_oklab
    
    logger.info(f"Processing {np.sum(influence_mask)} influenced pixels ({np.mean(influence_mask)*100:.1f}%)")
    
    # Process only influenced pixels
    result = pixels_oklab.copy()
    influenced_pixels = pixels_oklab[influence_mask]
    
    # Transform influenced pixels
    transformed = transform_func(influenced_pixels, attractors_oklab, tolerances, strengths, channels)
    result[influence_mask] = transformed
    
    return result
```

## 3. Integration Strategy

### 3.1 Modified Processing Pipeline

**Update `src/imgcolorshine/imgcolorshine.py`:**

```python
def process_image_optimized(
    input_image: str,
    attractors: tuple[str, ...],
    output_image: str | None = None,
    luminance: bool = True,
    saturation: bool = True,
    hue: bool = True,
    verbose: bool = False,
    tile_size: int = 1024,
    gpu: bool = True,
    lut_size: int = 0,
    hierarchical: bool = False,
    spatial_accel: bool = True,
) -> None:
    """Enhanced process_image with optimization flags."""
    
    # ... existing setup code ...
    
    # Choose processing path based on optimizations
    if hierarchical and spatial_accel:
        # Combined optimization path
        result = process_hierarchical_spatial(
            image, attractor_objects, channels, transformer
        )
    elif hierarchical:
        # Hierarchical only
        processor = HierarchicalProcessor()
        result = processor.process_hierarchical(
            image, transformer.transform_batch, 
            attractors_lab, tolerances, strengths, channels
        )
    elif spatial_accel:
        # Spatial acceleration only
        accelerator = SpatialAccelerator()
        result = accelerator.transform_with_spatial_accel(
            image_oklab, attractors_lab, tolerances, strengths,
            transformer.transform_batch, channels
        )
    else:
        # Standard processing
        result = transformer.transform_image(image, attractor_objects, channels)
```

### 3.2 Combined Optimization

```python
def process_hierarchical_spatial(image: np.ndarray,
                                attractors: List[Attractor],
                                channels: List[bool],
                                transformer: ColorTransformer) -> np.ndarray:
    """
    Combine hierarchical and spatial optimizations.
    
    Strategy:
    1. Build spatial index once
    2. Use it at each pyramid level
    3. Skip entire pyramid levels if no influence
    """
    hierarchical = HierarchicalProcessor()
    spatial = SpatialAccelerator()
    
    # Build spatial index
    attractors_oklab = np.array([a.oklab_values for a in attractors])
    tolerances = np.array([a.tolerance for a in attractors])
    spatial.build_spatial_index(attractors_oklab, tolerances)
    
    # Custom transform function that uses spatial acceleration
    def spatial_transform(pixels, *args):
        return spatial.transform_with_spatial_accel(
            pixels, attractors_oklab, tolerances, 
            np.array([a.strength for a in attractors]),
            transformer.transform_batch, channels
        )
    
    # Process hierarchically with spatial optimization
    return hierarchical.process_hierarchical(
        image, spatial_transform, attractors_oklab, 
        tolerances, None, channels
    )
```

### 3.3 CLI Updates

**Modify `src/imgcolorshine/cli.py`:**

```python
def shine(
    self,
    input_image: str,
    *attractors: str,
    output_image: str | None = None,
    luminance: bool = False,
    saturation: bool = False,
    chroma: bool = False,
    verbose: bool = False,
    tile_size: int = 1024,
    gpu: bool = True,
    LUT_size: int = 0,
    hierarchical: bool = False,
    spatial_accel: bool = True,
) -> None:
    """
    Transform image colors using OKLCH color attractors.
    
    New parameters:
        hierarchical: Enable multi-resolution processing (2-5x speedup)
        spatial_accel: Enable spatial acceleration (3-10x speedup)
    """
```

## 4. Performance Characteristics

### 4.1 Expected Performance Gains

| Image Type | Hierarchical | Spatial | Combined |
|------------|-------------|---------|----------|
| Large uniform regions | 3-5x | 5-10x | 10-20x |
| Complex textures | 1.5-2x | 3-5x | 4-8x |
| Few small attractors | 2-3x | 8-15x | 15-30x |
| Many attractors | 2-3x | 1.5-2x | 3-5x |

### 4.2 Memory Overhead

- Hierarchical: ~33% additional memory for pyramid
- Spatial: ~O(n) for KD-tree where n = number of attractors
- Combined: Sum of both overheads

### 4.3 Best Use Cases

**Hierarchical Processing:**
- Large images (>2K resolution)
- Smooth gradients
- Global color grading

**Spatial Acceleration:**
- Localized color corrections
- Small tolerance values
- Few attractors

## 5. Testing Strategy

### 5.1 Unit Tests

**New file: `tests/test_hierarchical.py`**
- Test pyramid construction
- Test difference mask generation
- Test adaptive subdivision
- Verify output quality

**New file: `tests/test_spatial_accel.py`**
- Test KD-tree construction
- Test influence detection
- Test tile coherence
- Verify correctness

### 5.2 Performance Benchmarks

**Update: `tests/test_performance.py`**
- Add hierarchical benchmarks
- Add spatial acceleration benchmarks
- Compare against baseline
- Test various image sizes and attractor counts

### 5.3 Integration Tests

- Test combined optimizations
- Verify visual quality (SSIM > 0.99)
- Test edge cases (single pixel, huge images)

## 6. Implementation Timeline

### Week 1: Hierarchical Processing
- Day 1-2: Core pyramid and difference mask algorithms
- Day 3: Adaptive subdivision
- Day 4: Integration with existing pipeline
- Day 5: Testing and benchmarking

### Week 2: Spatial Acceleration
- Day 1-2: KD-tree and influence detection
- Day 3: Tile coherence analysis
- Day 4: Integration and optimization
- Day 5: Testing and benchmarking

### Week 3: Polish and Documentation
- Day 1-2: Combined optimization tuning
- Day 3: Performance profiling
- Day 4: Documentation update
- Day 5: Final benchmarks and release prep