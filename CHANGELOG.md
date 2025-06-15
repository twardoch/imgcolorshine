# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-06-15

### Changed
- **Development Workflow Improvements**
  - Updated `cleanup.sh` to use `python -m uv run` instead of `uvx` for better compatibility
  - Removed unused dependencies (`scipy-stubs`, `types-pillow`) from `uv.lock`
  - Cleaned up repository by removing `ACCOMPLISHMENTS.md`
  - Updated `llms.txt` to reflect current test files including `test_cli.py` and `test_colorshine.py`

### Added
- **Major Test Suite Expansion - Coverage Improved from 41% to 50%**
  - Created comprehensive test suite for `kernel.py` (17 tests)
    - Tests for pixel transformation, channel control, multiple attractors
    - Tests for tolerance/strength effects, gamut mapping
    - Tests for image transformation, parallel consistency
  - Created comprehensive test suite for `lut.py` (16 tests)
    - Tests for LUT building, caching, interpolation
    - Tests for identity LUT, trilinear interpolation
    - Tests for performance characteristics and memory efficiency
  - Created additional test coverage for `transform.py` (improved from 18% to 40%)
    - Tests for delta E calculation, weight computation, color blending
    - Tests for channel-specific transformations
  - Created additional test coverage for `utils.py` (improved from 8% to 79%)
    - Tests for memory management, image processing utilities
    - Tests for validation functions, batch operations
  - Fixed existing test implementations in CLI, colorshine, I/O, and main interface tests
  - Overall test coverage improved from 41% to 50% (9 percentage points increase)
  - Total of 199 passing tests with 11 tests still requiring fixes

- **New Numba utility module** (`numba_utils.py`)
  - Added optimized batch color distance computation
  - Added nearest attractor finding with parallel processing
  - Added tile uniformity computation for spatial coherence
  - Added masked transformation application
  - Added edge strength detection for hierarchical processing
  - Added perceptually-correct downsampling in Oklab space

- **Numba optimizations for performance-critical functions**
  - Hierarchical processing (`hierar.py`)
    - `compute_difference_mask` now uses perceptual color distance in Oklab space with parallel processing (~10-50x speedup)
    - `detect_gradient_regions` uses Numba-optimized Sobel operators for gradient detection
  - Spatial acceleration (`spatial.py`)
    - `_get_mask_direct` uses parallel processing for influence mask computation (massive speedup)
    - `query_pixel_attractors` uses optimized distance calculations
  - Gamut mapping (`gamut.py`)
    - `map_oklch_to_gamut` uses optimized binary search for sRGB gamut mapping
    - `batch_map_oklch` uses parallel processing for batch gamut mapping (2-5x speedup)

- **Mypyc compilation support**
  - Added mypyc configuration in `pyproject.toml`
  - Created `build_ext.py` for custom build process
  - Configured modules for compilation: `color`, `transform`, `io`, `falloff`

- **Development Infrastructure**
  - Created `COVERAGE_REPORT.md` for tracking test coverage
  - Created `TESTING_WORKFLOW.md` with development best practices
  - Established TDD workflow and continuous improvement process

### Documentation
- **Comprehensive Development Plan** (`PLAN.md`)
  - Created detailed optimization roadmap for Numba and mypyc
  - Analyzed full codebase structure and optimization opportunities
  - Identified performance bottlenecks in `hierar.py`, `spatial.py`, and `gamut.py`
  - Documented test coverage gaps (20% overall, 0% for critical modules)
  - Created test implementation strategy for missing coverage
  - Added clear testing and iteration instructions
  - Established 4-phase execution plan with success metrics

### Fixed
- **NumPy 2.x Compatibility** 
  - Fixed JAX import errors when using NumPy 2.x with JAX compiled for NumPy 1.x
  - Made JAX imports lazy in `gpu.py` to prevent module-level import failures
  - JAX availability is now checked only when needed, allowing graceful fallback to CPU

### Added
- **Hierarchical Processing Optimization** (`hierarchical.py`)
  - Multi-resolution pyramid processing (2-5x speedup)
  - Adaptive refinement based on color differences
  - Gradient detection for smart refinement
  - Coarse-to-fine processing strategy
  - Configurable pyramid levels and thresholds
- **Spatial Acceleration Structures** (`spatial_accel.py`)
  - KD-tree based color space indexing (3-10x speedup)
  - Early pixel culling outside influence radii
  - Tile coherence optimization
  - Uniform tile detection and caching
  - Spatial queries for efficient processing
- **Combined Optimization Support**
  - Hierarchical + spatial acceleration for maximum performance
  - Smart integration in `process_with_optimizations()`
  - Automatic optimization selection based on image characteristics

### Changed
- **CLI Enhancements**
  - Added `--hierarchical` flag for multi-resolution processing
  - Added `--spatial_accel` flag for spatial acceleration (default: True)
  - Updated help documentation with optimization examples
- **Processing Pipeline**
  - Integrated optimization framework in main processing flow
  - Automatic selection of optimal processing path
  - Improved memory efficiency with tile-based spatial queries
- **Major Performance Optimizations** targeting 100x additional speedup:
  - **Fused Color Transformation Kernel** (`fused_kernels.py`)
    - Single-pass pixel transformation keeping all operations in CPU registers
    - Eliminates intermediate array allocations
    - Inline color space conversions (sRGB → Oklab → OKLCH → transform → sRGB)
    - Integrated gamut mapping with binary search
    - Parallel image processing with `numba.prange`
  - **GPU Acceleration Support** with automatic fallback
    - CuPy backend for NVIDIA GPUs (`gpu.py`, `gpu_transforms.py`)
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
- New module `trans_numba.py` with all performance-critical color operations
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

### Added / Behavioural Changes

- **Extended Strength Range (0-200)**  
  The `strength` parameter for each attractor now accepts values up to **200**.  
  
  • **0-100** – behaves exactly as before; weight = strength × raised-cosine fall-off.  
  • **100-200** – gradually flattens the fall-off curve. At 200 every pixel *within the tolerance radius* is pulled with full weight (duotone effect).

  Implementation details:
  - Weight computation moved to `engine._calculate_weights_percentile()`.
  - For `strength > 100` an extra factor `s_extra = (strength-100)/100` blends the fall-off value with 1.0.
  - CLI validator in `colorshine.parse_attractor()` now allows 0-200.

  This enables one-knob transition from subtle grading to complete duotone without changing tolerance.