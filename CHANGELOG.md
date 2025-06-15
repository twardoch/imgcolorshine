# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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