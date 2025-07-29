# TODO List

## Completed Tasks ✅
- [x] Fix all compilation warnings (unused imports, unused variables)
- [x] Implement srgb_to_oklch conversion function
- [x] Implement oklch_to_srgb conversion function
- [x] Implement oklch_to_oklab conversion function
- [x] Complete ColorAttractor influence calculation
- [x] Implement falloff curve calculations
- [x] Add actual image transformation logic in transform_pixel
- [x] Implement image loading in io module
- [x] Implement image saving in io module
- [x] Add basic error handling for file operations
- [x] Wire up the CLI to process images
- [x] Enable JPEG support
- [x] Create README.md
- [x] Create CHANGELOG.md
- [x] Create TODO.md
- [x] Create PLAN.md

## Core Functionality (Medium Priority)
- [ ] Implement gamut mapping algorithm
- [ ] Add better error messages for invalid colors
- [ ] Add progress reporting for large images
- [ ] Implement batch processing mode in CLI
- [ ] Add configuration file support

## Testing (High Priority)
- [ ] Add unit tests for color conversions
- [ ] Add unit tests for attractor creation
- [ ] Add unit tests for transform operations
- [ ] Create integration tests for CLI
- [ ] Add benchmark tests for performance
- [ ] Create test images and expected outputs

## Performance Optimizations (Medium Priority)
- [ ] Add SIMD implementations for color math
- [ ] Optimize parallel processing with rayon
- [ ] Create GPU acceleration module (optional feature)
- [ ] Build LUT caching system
- [ ] Add hierarchical processing
- [ ] Implement tile-based processing for very large images

## Documentation (Low Priority)
- [x] Write README.md with usage examples
- [ ] Add inline documentation for all public APIs
- [ ] Create architecture documentation
- [ ] Write performance tuning guide
- [ ] Add contributing guidelines
- [ ] Create example gallery with before/after images

## Release Preparation
- [ ] Add CI/CD with GitHub Actions
- [ ] Create release binaries for major platforms
- [ ] Publish to crates.io
- [ ] Create Docker image
- [ ] Write announcement blog post

## Future Enhancements
- [ ] Add support for HDR images
- [ ] Implement more falloff curve types
- [ ] Add interactive mode with preview
- [ ] Create GUI version
- [ ] Add plugin system for custom transformations
- [ ] Support video processing