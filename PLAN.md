# imgcolorshine - Implementation Plan

## Project Overview

`imgcolorshine` is a physics-inspired color transformation tool that operates in the perceptually uniform OKLCH color space. It transforms image colors by "attracting" them toward user-defined target colors with configurable tolerance and strength parameters.

## Current Status

- Basic project structure created with modern Python packaging (PEP 621)
- Complete implementation exists in `old/imgcolorshine/` directory
- Need to port and modernize the existing implementation

## Architecture

### Core Modules

1. **color_engine.py** - OKLCH color space operations
   - Color parsing (CSS formats)
   - OKLCH/Oklab conversions
   - Delta E calculations
   - Gamut mapping

2. **image_io.py** - High-performance image I/O
   - OpenCV-based loading/saving
   - PIL fallback support
   - Format auto-detection

3. **transforms.py** - Color transformation logic
   - Numba-optimized pixel operations
   - Multi-attractor blending
   - Channel selection support

4. **falloff.py** - Attraction falloff functions
   - Raised cosine falloff
   - Vectorized operations

5. **gamut.py** - CSS Color 4 gamut mapping
   - Binary search for valid chroma
   - Preserve lightness/hue

6. **utils.py** - Utility functions
   - Memory management
   - Tiled processing for large images
   - Progress tracking

### CLI Interface

- Click-based command-line interface
- Support for multiple attractors
- Channel selection flags (--luminance, --saturation, --hue)
- Verbose logging with loguru

## Implementation Steps

1. **Phase 1: Core Module Porting**
   - Port color_engine.py with OKLCHEngine class
   - Port image_io.py with OpenCV operations
   - Port transforms.py with Numba optimizations
   - Port supporting modules (falloff.py, gamut.py, utils.py)

2. **Phase 2: CLI Implementation**
   - Create click-based CLI in main module
   - Add argument parsing for attractors
   - Implement channel selection flags
   - Add verbose logging support

3. **Phase 3: Testing & Documentation**
   - Port and update test suite
   - Add integration tests
   - Update README with usage examples
   - Add API documentation

4. **Phase 4: Performance Optimization**
   - Verify Numba JIT compilation
   - Implement tiled processing
   - Add GPU support (optional)

## Technical Specifications

### Attractor Format
```
"color;tolerance;strength"
```
- color: Any CSS color format (hex, rgb, hsl, oklch, named)
- tolerance: 0-100 (perceptual distance threshold)
- strength: 0-100 (transformation intensity)

### Color Space Pipeline
1. Load image (sRGB)
2. Convert to Oklab for calculations
3. Apply attractors with falloff
4. Convert back to sRGB
5. Apply gamut mapping if needed
6. Save optimized output

### Performance Goals
- Handle 4K images in < 2 seconds
- Support batch processing
- Memory-efficient tiled processing
- Optional GPU acceleration

## Dependencies

Core:
- coloraide: OKLCH operations
- opencv-python: Image I/O
- numpy: Array operations
- numba: JIT compilation
- click: CLI interface
- loguru: Logging

Optional:
- cupy: GPU acceleration

## Success Criteria

- [x] All core functionality ported from old implementation
- [x] CLI matches specification from PLAN.md
- [ ] Tests passing with >90% coverage
- [ ] Performance meets or exceeds old implementation
- [x] Documentation complete and accurate

## Implementation Status

### Completed
1. **Core Module Porting** - All modules successfully ported:
   - color_engine.py - OKLCH color space operations
   - image_io.py - High-performance image I/O
   - transforms.py - Numba-optimized transformations
   - falloff.py - Attraction falloff functions
   - gamut.py - CSS Color 4 gamut mapping
   - utils.py - Memory management utilities

2. **CLI Implementation** - Click-based CLI with all features:
   - Multiple attractors support
   - Channel selection flags
   - Verbose logging
   - Progress tracking
   - Auto-generated output paths

3. **Documentation** - Complete documentation:
   - Updated README.md with usage examples
   - Created CHANGELOG.md
   - Comprehensive docstrings in all modules

### Pending
- Unit tests implementation
- Integration testing
- Performance benchmarking
- Optional GPU acceleration