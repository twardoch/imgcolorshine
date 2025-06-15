# PLAN

## Overview

This plan outlines optimization strategies for `imgcolorshine` focusing on:
1. Further `numba` optimizations for performance-critical functions
2. `mypyc` compilation for well-typed modules
3. Comprehensive test coverage improvements
4. Systematic testing and iteration process

## Phase 1. Numba Optimizations

### 1.1 Immediate Opportunities

#### `hierar.py` - High Priority
- [ ] **`compute_difference_mask`** (lines 225-252)
  - Currently uses simple RGB distance calculation
  - Convert to use perceptual distance in Lab/OKLCH space
  - Apply `@numba.njit` decorator for ~10-50x speedup
  - Benefits: More accurate refinement decisions, faster processing

- [ ] **`detect_gradient_regions`** (lines 253-288)
  - Heavy mathematical operations on gradients
  - Convert Sobel operations to Numba-optimized loops
  - Benefits: Faster gradient detection for hierarchical processing

#### `spatial.py` - Very High Priority  
- [ ] **`_get_mask_direct`** (lines 475-496)
  - Computes distances for every pixel against each attractor
  - Perfect candidate for `@numba.njit(parallel=True)`
  - Benefits: Massive speedup for spatial acceleration

- [ ] **`query_pixel_attractors`** (lines 497-515)
  - Iterates through all influence regions
  - Simple distance calculations ideal for Numba
  - Benefits: Faster per-pixel queries

#### `gamut.py` - Medium Priority
- [ ] **`map_oklch_to_gamut`** (lines 633-681)
  - Binary search loop for gamut boundary
  - Apply `@numba.njit` to the search logic
  - Benefits: Faster gamut mapping, especially for out-of-gamut colors

- [ ] **`batch_map_oklch`** (lines 727-745)
  - Currently iterates over flattened array
  - Convert to `@numba.njit(parallel=True)` with `prange`
  - Benefits: Parallel gamut mapping for entire images

### 1.2 Implementation Strategy

1. **Profile First**: Use line_profiler to identify actual bottlenecks
2. **Test Coverage**: Ensure comprehensive tests before optimization
3. **Incremental Changes**: Optimize one function at a time
4. **Benchmark**: Compare performance before/after each change

### 1.3 Code Example - `compute_difference_mask` Optimization

```python
@numba.njit(cache=True)
def compute_difference_mask_optimized(
    fine_lab: np.ndarray, 
    coarse_lab: np.ndarray, 
    threshold: float
) -> np.ndarray:
    """Numba-optimized perceptual difference mask."""
    h, w = fine_lab.shape[:2]
    mask = np.empty((h, w), dtype=np.bool_)
    
    for i in range(h):
        for j in range(w):
            # Perceptual distance in Lab space
            dl = fine_lab[i, j, 0] - coarse_lab[i, j, 0]
            da = fine_lab[i, j, 1] - coarse_lab[i, j, 1]
            db = fine_lab[i, j, 2] - coarse_lab[i, j, 2]
            distance = np.sqrt(dl*dl + da*da + db*db)
            mask[i, j] = distance > threshold
    
    return mask
```

## Phase 2. Mypyc Optimizations

### 2.1 Target Modules

#### Tier 1 - Core Modules (High Impact)
- [ ] **`color.py`** - Central to all operations
  - Well-typed with dataclasses
  - Heavy use throughout the application
  - Expected speedup: 2-5x

- [ ] **`transform.py`** - Main transformation logic
  - Core of the image processing pipeline
  - Well-defined interfaces
  - Expected speedup: 2-4x

#### Tier 2 - Utility Modules (Medium Impact)
- [ ] **`io.py`** - All image I/O operations
  - Frequent use for loading/saving
  - Clear type annotations
  - Expected speedup: 1.5-3x

- [ ] **`gamut.py`** - Mathematical operations
  - Self-contained functions
  - Heavy computation
  - Expected speedup: 2-3x

- [ ] **`falloff.py`** - Pure mathematical functions
  - Already has Numba versions
  - Mypyc for non-Numba parts
  - Expected speedup: 1.5-2x

### 2.2 Implementation Plan

1. **Setup Build System**
   ```toml
   # pyproject.toml additions
   [tool.mypyc]
   modules = ["imgcolorshine.color", "imgcolorshine.transform"]
   strict_optional = true
   ```

2. **Type Annotation Review**
   - Ensure all functions have complete type hints
   - Replace `Any` with specific types where possible
   - Add `TypedDict` for complex dictionaries

3. **Build Process**
   - Integrate mypyc into the build pipeline
   - Create fallback for pure Python when needed
   - Test on multiple platforms

### 2.3 Expected Performance Gains

Combined Numba + Mypyc optimizations should yield:
- **Small images (256×256)**: 5-10x overall speedup
- **Medium images (1920×1080)**: 3-7x overall speedup  
- **Large images (4K+)**: 2-5x overall speedup

## Phase 3. Test Coverage Analysis

### 3.1 Current Coverage Summary

Total coverage: **20%** (1623 statements, 1243 missing)

#### Modules with NO Coverage (0%)
- `__main__.py` - Entry point (low priority)
- `__version__.py` - Version info (low priority)
- `cli.py` - CLI interface (HIGH PRIORITY)
- `gpu.py` - GPU acceleration (medium priority)
- `kernel.py` - Fused kernels (HIGH PRIORITY)
- `lut.py` - LUT acceleration (medium priority)
- `trans_gpu.py` - GPU transforms (low priority)

#### Modules with Poor Coverage (<20%)
- `colorshine.py` - 19% (HIGH PRIORITY - main interface)
- `gamut.py` - 13% (medium priority)
- `io.py` - 16% (HIGH PRIORITY - critical functionality)
- `transform.py` - 18% (HIGH PRIORITY - core logic)
- `utils.py` - 8% (HIGH PRIORITY - utilities)

#### Modules with Moderate Coverage (20-70%)
- `color.py` - 54% (needs improvement)
- `falloff.py` - 33% (needs improvement)
- `spatial.py` - 43% (needs improvement)
- `trans_numba.py` - 23% (needs improvement)
- `hierar.py` - 68% (good start)

### 3.2 Missing Test Categories

1. **CLI Tests** (`cli.py`, `colorshine.py`)
   - Command parsing
   - Argument validation
   - End-to-end transformations
   - Error handling

2. **GPU Tests** (`gpu.py`, `trans_gpu.py`)
   - GPU availability detection
   - Fallback to CPU
   - Memory management
   - Performance comparison

3. **Kernel Tests** (`kernel.py`)
   - Fused transformation accuracy
   - Edge cases (gamut boundaries)
   - Performance benchmarks

4. **LUT Tests** (`lut.py`)
   - LUT building accuracy
   - Interpolation correctness
   - Cache management
   - Performance gains

5. **I/O Tests** (`io.py`)
   - Various image formats
   - Large image handling
   - Error conditions
   - Memory efficiency

## Phase 4. Test Implementation Plan

### 4.1 High Priority Tests

#### Test CLI Interface (`test_cli.py`)
```python
def test_basic_transformation():
    """Test basic CLI transformation command."""
    # Test with test image
    # Verify output created
    # Check transformation applied

def test_multiple_attractors():
    """Test CLI with multiple color attractors."""
    
def test_channel_flags():
    """Test luminance/saturation/hue channel controls."""
    
def test_optimization_flags():
    """Test GPU, LUT, hierarchical flags."""
```

#### Test Main Interface (`test_colorshine.py`)
```python
def test_shine_function():
    """Test the main shine() function."""
    
def test_attractor_parsing():
    """Test attractor string parsing."""
    
def test_output_path_generation():
    """Test automatic output path generation."""
```

#### Test I/O Operations (`test_io.py`)
```python
def test_load_save_cycle():
    """Test loading and saving preserves data."""
    
def test_format_support():
    """Test PNG, JPEG, other formats."""
    
def test_large_image_tiling():
    """Test tiling for large images."""
    
def test_memory_estimation():
    """Test memory usage estimation."""
```

### 4.2 Integration Tests

#### End-to-End Tests (`test_integration.py`)
```python
def test_full_pipeline():
    """Test complete transformation pipeline."""
    # Load image
    # Apply transformation
    # Save result
    # Verify output
    
def test_optimization_combinations():
    """Test different optimization flag combinations."""
    
def test_real_world_scenarios():
    """Test with various real images and transformations."""
```

### 4.3 Performance Tests

#### Benchmark Suite (`test_benchmarks.py`)
```python
def test_numba_speedup():
    """Verify Numba optimizations provide speedup."""
    
def test_gpu_speedup():
    """Verify GPU acceleration when available."""
    
def test_lut_speedup():
    """Verify LUT provides performance gains."""
    
def test_scaling_performance():
    """Test performance with different image sizes."""
```

## Phase 5. Testing & Iteration Instructions

### 5.1 Development Workflow

1. **Before Making Changes**
   ```bash
   # Run full test suite
   python -m pytest -v
   
   # Check coverage
   python -m pytest --cov=src/imgcolorshine --cov-report=html
   
   # Run specific test file
   python -m pytest tests/test_transform.py -v
   ```

2. **After Making Changes**
   ```bash
   # Run cleanup script
   ./cleanup.sh
   
   # Run tests again
   python -m pytest -v
   
   # Check for regressions
   python -m pytest tests/test_correctness.py -v
   ```

3. **Performance Testing**
   ```bash
   # Run performance benchmarks
   python -m pytest tests/test_performance.py -v
   
   # Profile specific functions
   python -m line_profiler script_to_profile.py
   ```

### 5.2 Test-Driven Development Process

1. **Write Test First**
   - Define expected behavior
   - Create minimal test case
   - Run test (should fail)

2. **Implement Feature**
   - Write minimal code to pass test
   - Focus on correctness first
   - Optimize later if needed

3. **Refactor**
   - Clean up implementation
   - Ensure tests still pass
   - Add edge case tests

4. **Document**
   - Update docstrings
   - Add usage examples
   - Update README if needed

### 5.3 Continuous Improvement

1. **Regular Coverage Checks**
   - Aim for >80% coverage on critical modules
   - Focus on untested code paths
   - Add tests for bug fixes

2. **Performance Monitoring**
   - Track performance metrics over time
   - Benchmark before/after optimizations
   - Document performance characteristics

3. **Code Quality**
   - Run linters regularly
   - Keep type hints up to date
   - Refactor complex functions

### 5.4 Bug Fix Process

1. **Reproduce Issue**
   - Create minimal test case
   - Verify bug exists
   - Add failing test

2. **Fix Bug**
   - Make minimal change
   - Ensure test passes
   - Check for side effects

3. **Prevent Regression**
   - Keep test in suite
   - Document fix
   - Consider related issues

## Phase 6. Execution Priority

### Phase 1: Foundation (Week 1)
1. [ ] Implement critical missing tests (CLI, I/O, main interface)
2. [ ] Fix any failing tests discovered
3. [ ] Achieve >50% coverage on core modules

### Phase 2: Numba Optimizations (Week 2)
1. [ ] Profile and identify bottlenecks
2. [ ] Implement Numba optimizations for `spatial.py`
3. [ ] Implement Numba optimizations for `hierar.py`
4. [ ] Benchmark improvements

### Phase 3: Mypyc Integration (Week 3)
1. [ ] Set up mypyc build system
2. [ ] Compile `color.py` and `transform.py`
3. [ ] Test compatibility and performance
4. [ ] Document build process

### Phase 4: Polish (Week 4)
1. [ ] Complete test coverage to >80%
2. [ ] Update documentation
3. [ ] Create performance comparison report
4. [ ] Prepare for release

## Phase 7. Success Metrics

- **Test Coverage**: >80% for critical modules
- **Performance**: 3-5x speedup on typical workloads
- **Reliability**: Zero failing tests, comprehensive edge case coverage
- **Maintainability**: Clear documentation, type hints, modular design

## Phase 8. Notes

- Always run `./cleanup.sh` after making changes
- Use `uv` for all Python operations, not `pip`
- Commit frequently with descriptive messages
- Keep backwards compatibility where possible
- Document breaking changes clearly