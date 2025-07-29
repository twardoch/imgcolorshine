# Changelog

All notable changes to imgcolorshine-rs will be documented in this file.

## [0.1.0] - 2025-07-29

### Added
- Initial Rust port of imgcolorshine with core modules
- Core color engine with OKLCH color space support
- Color attractor system for physics-inspired transformations
- Transform pipeline for image processing
- Command-line interface with clap
- Error handling system
- Basic I/O module for image processing
- Gamut mapping for color space conversions
- Utility functions for color operations
- Complete color conversion implementations (srgb_to_oklch, oklch_to_srgb, oklch_to_oklab)
- Attractor influence calculations with raised cosine falloff
- Image loading and saving functionality
- Multi-threaded support with rayon
- CSS color parsing support
- README.md with comprehensive usage documentation
- JPEG image format support

### Fixed
- Resolved compilation error with jpeg-decoder and --check-cfg flag
- Fixed import errors in transform module
- Fixed TransformSettings visibility issues
- Corrected module structure for proper re-exports
- Fixed global cargo config conflict
- Added missing JPEG support in image crate dependencies

### Technical
- Set up Rust project structure with proper module organization
- Configured dependencies including image, palette, clap, and others
- Added rust-toolchain.toml for stable Rust version
- Created local .cargo/config.toml to override global settings
- All modules are fully implemented and functional
- Application successfully processes images with color attractors

### Current Capabilities
- Load images in multiple formats (JPEG, PNG, WebP, TIFF, etc.)
- Transform colors using OKLCH color attractors
- Support multiple attractors with blending
- Control individual color channels (lightness, chroma, hue)
- Process images with multi-threading
- Generate output filenames automatically
- Verbose logging support