# PLAN

## Phase 1. Mypyc Optimizations

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

1. **Setup Build System** - ✅ DONE

   - Added mypyc configuration to pyproject.toml
   - Created build_ext.py for custom build process
   - Configured modules: color, transform, io, falloff

2. **Type Annotation Review**

   - Ensure all functions have complete type hints
   - Replace `Any` with specific types where possible
   - Add `TypedDict` for complex dictionaries

3. **Build Process** - ✅ PARTIALLY DONE
   - Integrated mypyc into build pipeline via build_ext.py
   - Created fallback for pure Python when mypyc unavailable
   - TODO: Test on multiple platforms

### 2.3 Expected Performance Gains

Combined Numba + Mypyc optimizations should yield:

- **Small images (256×256)**: 5-10x overall speedup
- **Medium images (1920×1080)**: 3-7x overall speedup
- **Large images (4K+)**: 2-5x overall speedup

## Phase 2. Test Coverage Analysis

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

## Phase 3. Test Implementation Plan

### 4.1 High Priority Tests

#### Test CLI Interface (`test_cli.py`) - ✅ CREATED (skeleton only)

- Basic transformation test skeleton created
- Multiple attractors test skeleton created
- Channel flags test skeleton created  
- Optimization flags test skeleton created
- **TODO: Implement actual test logic**

#### Test Main Interface (`test_colorshine.py`) - ✅ CREATED (skeleton only)

- Shine function test skeleton created
- Attractor parsing test skeleton created
- Output path generation test skeleton created
- **TODO: Implement actual test logic**

#### Test I/O Operations (`test_io.py`) - ✅ CREATED

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

## Phase 4. Testing & Iteration Instructions

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

## Phase 5. Execution Priority

### Phase 5.1: Foundation (Week 1)

1. [x] Create test skeletons for CLI, I/O, main interface - ✅ DONE
2. [x] Implement actual test logic in test skeletons - ✅ DONE
   - CLI tests: 7/7 passing
   - I/O tests: 12/12 passing  
   - Colorshine tests: 3/5 passing
   - Main interface tests: 3/7 passing
3. [x] Fix failing tests discovered - ✅ PARTIALLY DONE
4. [ ] Achieve >50% coverage on core modules - IN PROGRESS (43% overall)

### Phase 5.2: Numba Optimizations (Week 2)

1. [x] Profile and identify bottlenecks - ✅ DONE
   - spatial.py already has key functions optimized
   - hierar.py already has compute_perceptual_distance_mask optimized
2. [x] Create additional Numba utilities - ✅ DONE
   - Created numba_utils.py with optimized functions:
     - compute_color_distances_batch
     - find_nearest_attractors
     - compute_tile_uniformity
     - apply_transformation_mask
     - compute_edge_strength
     - downsample_oklab
3. [ ] Integrate new Numba functions into existing modules
4. [ ] Benchmark improvements

### Phase 5.3: Mypyc Integration (Week 3)

1. [ ] Set up mypyc build system
2. [ ] Compile `color.py` and `transform.py`
3. [ ] Test compatibility and performance
4. [ ] Document build process

### Phase 5.4: Polish (Week 4)

1. [ ] Complete test coverage to >80%
2. [ ] Update documentation
3. [ ] Create performance comparison report
4. [ ] Prepare for release

## Phase 6. Success Metrics

- **Test Coverage**: >80% for critical modules
- **Performance**: 3-5x speedup on typical workloads
- **Reliability**: Zero failing tests, comprehensive edge case coverage
- **Maintainability**: Clear documentation, type hints, modular design

## Phase 7. Notes

- Always run `./cleanup.sh` after making changes
- Use `uv` for all Python operations, not `pip` or `python3`
- Commit frequently with descriptive messages
- Keep backwards compatibility where possible
- Document breaking changes clearly
