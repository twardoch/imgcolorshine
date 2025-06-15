# TODO

TASK: Work through the steps below. Once you've completed a step, mark it with `[x]`. Always use `uv` or `hatch` or `python`, not `python3`.

## 1. Current Session Tasks

- [x] Update `CHANGELOG.md` based on recent changes incl. recent git changes
- [x] Update `PLAN.md` removing things that are done
- [x] Update `TODO.md` removing things that are done; then add things to be done there.
- [x] Add an extensive detailed plan of things to be done into `PLAN.md`
- [x] Thoroughly revise `README.md` to comprehensively describe the entire package
- [x] Run `./cleanup.sh`, review `./cleanup.log` and fix issues
- [x] **CRITICAL FIX: Fixed luminance overexposure issue** - CLI parameter defaults were incorrect (False instead of True), causing luminance-only transformations to actually apply all channels

## 2. Immediate Priority Tasks

### 2.1. Fix Failing Tests (2 tests remaining)
- [x] ~~Fix `test_colorshine.py` (2 tests)~~ - **All tests now passing**
  - [x] ~~`test_shine_function` - mock setup issues~~
  - [x] ~~`test_process_image_channel_defaults` - KeyError issues~~
- [x] ~~Fix `test_main_interface.py` (4 tests)~~ - **All tests now passing**
  - [x] ~~Test process_image_basic~~
  - [x] ~~Test process_image_multiple_attractors~~
  - [x] ~~Test process_image_channel_control~~
  - [x] ~~Test process_image_custom_output~~
- [x] ~~Fix `test_transform_coverage.py` (1 test)~~ - **All tests now passing**
  - [x] ~~TestColorTransformer::test_empty_attractors~~
- [x] ~~Fix `test_utils_coverage.py` (4 tests)~~ - **All tests now passing**
  - [x] ~~TestImageProcessing::test_process_large_image_basic~~
  - [x] ~~TestImageProcessing::test_process_large_image_with_overlap~~
  - [x] ~~TestValidation::test_validate_image~~
  - [x] ~~TestProgressBar::test_create_progress_bar~~
- [x] ~~Fix `test_cli.py` (2 tests)~~ - **Fixed by correcting CLI parameter defaults**
  - [x] ~~test_basic_transformation~~
  - [x] ~~test_multiple_attractors~~

### 2.2. Improve Test Coverage to >60%
- [ ] Create tests for `trans_gpu.py` (0% coverage)
- [ ] Create tests for `trans_numba.py` (24% → 50%+)
- [ ] Create tests for `numba_utils.py` (0% coverage)
- [ ] Improve `kernel.py` coverage (4% → 50%+)
- [ ] Improve `transform.py` coverage (40% → 60%+)
- [ ] Improve `spatial.py` coverage (36% → 60%+)

### 2.3. Performance Optimizations
- [ ] Integrate `numba_utils.py` functions into main modules
  - [ ] Use `compute_color_distances_batch` in spatial.py
  - [ ] Use `find_nearest_attractors` in transform.py
  - [ ] Use `compute_tile_uniformity` in spatial.py
  - [ ] Use `apply_transformation_mask` in hierar.py
  - [ ] Use `compute_edge_strength` in hierar.py
  - [ ] Use `downsample_oklab` in hierar.py
- [ ] Profile and benchmark the improvements
- [ ] Document performance gains

### 2.4. Mypyc Compilation
- [ ] Complete mypyc setup and testing
- [ ] Compile `color.py` module with mypyc
- [ ] Compile `transform.py` module with mypyc
- [ ] Test compatibility on multiple platforms
- [ ] Benchmark performance improvements
- [ ] Document build process

## 3. Long-term Goals

### 3.1. Documentation
- [ ] Create comprehensive API documentation
- [ ] Add more usage examples to README
- [ ] Create tutorial notebooks
- [ ] Document performance characteristics
- [ ] Create architecture diagrams

### 3.2. Features
- [ ] Add batch processing CLI command
- [ ] Add configuration file support
- [ ] Add more color space support (LAB, HSV, etc.)
- [ ] Add interactive mode for parameter tuning
- [ ] Add preview mode for large images

### 3.3. Quality Improvements
- [ ] Achieve 80%+ test coverage on all critical modules
- [ ] Add property-based testing with Hypothesis
- [ ] Add integration tests with real images
- [ ] Set up continuous integration (CI)
- [ ] Add pre-commit hooks

### 3.4. Performance
- [ ] Optimize memory usage for very large images
- [ ] Add multi-threading support for CPU processing
- [ ] Implement streaming processing for video
- [ ] Add WebAssembly build for browser usage
- [ ] Create benchmarking suite

## 4. Completed Tasks ✅

### 4.1. Critical Bug Fixes
- **Fixed luminance overexposure issue**: CLI parameter defaults for `chroma`, `saturation`, and `luminance` were incorrectly set to `False` instead of `True`, causing channel-specific transformations to behave unexpectedly
- **All tests now passing**: 210 tests passing, 1 skipped (was 11 tests failing)

### 4.2. From Previous Session
- Created comprehensive test suites for `kernel.py` (17 tests)
- Created comprehensive test suites for `lut.py` (16 tests)
- Improved test coverage from 41% → 50%
- Created `numba_utils.py` with 6 optimized utility functions
- Updated CHANGELOG.md and PLAN.md with accomplishments
- Fixed multiple test implementation issues