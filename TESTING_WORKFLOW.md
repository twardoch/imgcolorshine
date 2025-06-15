# Testing & Iteration Workflow

## Development Workflow

### 1. Before Making Changes

```bash
# Run full test suite
python -m pytest -v

# Check coverage
python -m pytest --cov=src/imgcolorshine --cov-report=html

# Run specific test file
python -m pytest tests/test_correctness.py -v
```

### 2. After Making Changes

```bash
# Run cleanup script
./cleanup.sh

# Run tests again
python -m pytest -v

# Check for regressions
python -m pytest tests/test_correctness.py -v
```

### 3. Performance Testing

```bash
# Run performance benchmarks
python -m pytest tests/test_performance.py -v

# Profile specific functions (if needed)
python -m line_profiler script_to_profile.py
```

## Test-Driven Development Process

### 1. Write Test First
- Define expected behavior
- Create minimal test case
- Run test (should fail)

### 2. Implement Feature
- Write minimal code to pass test
- Focus on correctness first
- Optimize later if needed

### 3. Refactor
- Clean up implementation
- Ensure tests still pass
- Add edge case tests

### 4. Document
- Update docstrings
- Add usage examples
- Update README if needed

## Continuous Improvement

### 1. Regular Coverage Checks
- Aim for >80% coverage on critical modules
- Focus on untested code paths
- Add tests for bug fixes

### 2. Performance Monitoring
- Track performance metrics over time
- Benchmark before/after optimizations
- Document performance characteristics

### 3. Code Quality
- Run linters regularly: `./cleanup.sh`
- Keep type hints up to date
- Refactor complex functions

## Bug Fix Process

### 1. Reproduce Issue
- Create minimal test case
- Verify bug exists
- Add failing test

### 2. Fix Bug
- Make minimal change
- Ensure test passes
- Check for side effects

### 3. Prevent Regression
- Keep test in suite
- Document fix
- Consider related issues

## Quick Commands

```bash
# Full test with coverage
python -m pytest --cov=src/imgcolorshine --cov-report=term-missing

# Fast test run (no coverage)
python -m pytest -x

# Run only fast tests
python -m pytest -m "not slow"

# Run with verbose output
python -m pytest -vv

# Generate HTML coverage report
python -m pytest --cov=src/imgcolorshine --cov-report=html
# Open htmlcov/index.html in browser

# Run cleanup and tests
./cleanup.sh && python -m pytest
```

## Test Organization

### Test Categories
1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test module interactions
3. **Performance Tests**: Benchmark critical paths
4. **End-to-End Tests**: Test complete workflows

### Test Naming Convention
- `test_<feature>_<scenario>_<expected_result>`
- Example: `test_attractor_parsing_invalid_format_raises_error`

### Test Structure
```python
def test_feature_scenario():
    """Test description."""
    # Arrange
    input_data = prepare_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_result
```

## Coverage Goals

### Current Status (24%)
- CLI: Partial coverage
- Main interface: Partial coverage
- I/O: Partial coverage
- Transform: 18%
- Color: 54%

### Target Coverage
- Critical modules: >80%
- Utility modules: >60%
- Overall: >50%

## Performance Benchmarks

### Baseline Performance
- 256×256: ~44ms
- 512×512: ~301ms
- 1920×1080: ~2-3s
- 4K: ~8-12s

### Performance Tests Should Verify
- Numba optimizations provide speedup
- GPU acceleration when available
- LUT provides performance gains
- No performance regressions