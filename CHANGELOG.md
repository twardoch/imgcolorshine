# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-06-15

### Fixed
- **NumPy 2.x Compatibility** 
  - Fixed JAX import errors when using NumPy 2.x with JAX compiled for NumPy 1.x
  - Made JAX imports lazy in `gpu_backend.py` to prevent module-level import failures
  - JAX availability is now checked only when needed, allowing graceful fallback to CPU

### Added
- **Major Performance Optimizations** targeting 100x additional speedup:
  - **Fused Color Transformation Kernel** (`fused_kernels.py`)
    - Single-pass pixel transformation keeping all operations in CPU registers
    - Eliminates intermediate array allocations
    - Inline color space conversions (sRGB → Oklab → OKLCH → transform → sRGB)
    - Integrated gamut mapping with binary search
    - Parallel image processing with `numba.prange`
  - **GPU Acceleration Support** with automatic fallback
    - CuPy backend for NVIDIA GPUs (`gpu_backend.py`, `gpu_transforms.py`)
    - JAX backend support (experimental)
    - Automatic memory management and device selection
    - GPU memory estimation and pooling
    - Efficient matrix operations using `einsum`
    - Broadcasting for parallel attractor calculations
  - **3D Color Look-Up Table (LUT)** for dramatic speedup (`lut.py`)
    - Pre-computed transformations on 3D RGB grid
    - Trilinear interpolation for arbitrary colors
    - Disk caching with SHA256-based keys
    - Configurable resolution (default 65³)
    - Progress logging during LUT construction
    - Integration with fused kernel for optimal performance
  - **Memory Optimizations**
    - Ensured C-contiguous arrays in image I/O
    - Added `cache=True` to all Numba JIT functions
    - Pre-allocation with `np.empty()` instead of `np.zeros()`

### Changed (2025-01-15)
- **CLI Enhancements**
  - Added `--gpu` flag for GPU acceleration control (default: True)
  - Added `--lut_size` parameter for LUT resolution (0=disabled, 65=recommended)
  - Automatic backend selection: LUT → GPU → CPU fallback chain
- **Processing Pipeline**
  - Integrated LUT processing as first priority when enabled
  - GPU processing with automatic fallback to CPU
  - Improved error handling and logging for each backend
- **Code Quality**
  - Fixed imports and module dependencies
  - Consistent code formatting with ruff
  - Updated type hints and documentation

### Performance Improvements (2025-01-15)
- Fused kernel reduces memory traffic by ~80%
- GPU acceleration provides 10-100x speedup on compatible hardware
- 3D LUT provides 5-20x speedup with near-instant cached lookups
- Combined optimizations target <10ms for 1920×1080 on modern hardware

## [0.1.1] - 2025-01-14

### Added
- Numba-optimized color space transformations (77-115x faster)
  - Direct matrix multiplication for sRGB ↔ Oklab conversions
  - Vectorized OKLCH ↔ Oklab batch conversions
  - Parallel processing with `numba.prange`
  - Optimized gamut mapping with binary search
- New module `color_transforms_numba.py` with all performance-critical color operations
- Performance benchmark script (`test_performance.py`)
- Correctness test suite for validating optimizations

### Changed
- `color_engine.py` now uses Numba-optimized functions for batch RGB ↔ Oklab conversions
- `transforms.py` uses vectorized OKLCH conversions instead of pixel-by-pixel loops
- Eliminated ColorAide bottleneck in performance-critical paths
- Matrix multiplication now uses manual implementation to avoid scipy dependency

### Performance Improvements
- 256×256 images: 5.053s → 0.044s (114.6x faster)
- 512×512 images: 23.274s → 0.301s (77.3x faster)
- 2048×2048 images now process in under 4 seconds

## [0.1.0] - 2025-01-14

### Added
- Initial release of imgcolorshine
- Core color transformation engine with OKLCH color space support
- High-performance image I/O with OpenCV and PIL fallback
- Numba-optimized pixel transformations with parallel processing
- CSS Color Module 4 compliant gamut mapping
- Multiple falloff functions (cosine, linear, quadratic, gaussian, cubic)
- Tiled processing for large images with memory management
- Click-based CLI interface with progress tracking
- Support for all CSS color formats (hex, rgb, hsl, oklch, named colors)
- Channel-specific transformations (luminance, saturation, hue)
- Multi-attractor blending with configurable tolerance and strength
- Comprehensive logging with loguru
- Rich console output with progress indicators

### Changed
- Migrated from Fire to Click for CLI implementation
- Restructured codebase to use modern Python packaging (src layout)
- Updated all modules to include proper type hints
- Enhanced documentation with detailed docstrings

### Technical Details
- Python 3.11+ required
- Dependencies: click, coloraide, opencv-python, numpy, numba, pillow, loguru, rich
- Modular architecture with separate modules for each concern
- JIT compilation for performance-critical code paths