# Chapter 9: Development

This chapter covers imgcolorshine's development setup, architecture, contribution guidelines, and extension points for developers who want to contribute or build upon the project.

## Development Environment Setup

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **Git** for version control
- **uv** (recommended) or pip for package management
- **NVIDIA GPU** (optional) for GPU development

### Clone and Setup

```bash
# Clone repository
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install in development mode with all dependencies
uv pip install -e ".[dev,test,speedups]"
# or with pip: pip install -e ".[dev,test,speedups]"
```

### Development Dependencies

The development installation includes:

- **Core:** imgcolorshine package in editable mode
- **Testing:** pytest, coverage, hypothesis for property testing
- **Linting:** ruff for formatting and linting
- **Type Checking:** mypy for static type analysis
- **Documentation:** sphinx, mkdocs-material for docs
- **Performance:** numba, mypyc for compilation
- **GPU:** cupy (optional) for GPU acceleration

### Verify Installation

```bash
# Run basic tests
python -m pytest tests/ -v

# Check linting
ruff check src/

# Verify CLI works
imgcolorshine shine testdata/louis.jpg "blue;50;70" --output_image=/tmp/test.jpg

# Check GPU availability (if applicable)
python -c "
from imgcolorshine.gpu import ArrayModule
am = ArrayModule()
print(f'GPU available: {am.gpu_available}')
"
```

## Project Architecture

### Directory Structure

```
imgcolorshine/
├── src/imgcolorshine/           # Main package
│   ├── fast_mypyc/             # Mypyc-compiled modules
│   │   ├── cli.py              # Command-line interface
│   │   ├── colorshine.py       # Main orchestration
│   │   ├── engine.py           # Core transformation engine
│   │   ├── gamut.py            # Gamut mapping
│   │   └── io.py               # Image I/O
│   ├── fast_numba/             # Numba-optimized modules
│   │   ├── engine_numba.py     # Numba transformation kernels
│   │   ├── trans_numba.py      # Color space conversions
│   │   └── gamut_numba.py      # Numba gamut mapping
│   ├── gpu.py                  # GPU abstraction layer
│   └── __init__.py             # Package interface
├── tests/                      # Test suite
│   ├── test_engine.py          # Core engine tests
│   ├── test_color.py           # Color space tests
│   ├── test_performance.py     # Performance benchmarks
│   └── conftest.py             # Test configuration
├── docs/                       # Documentation output
├── src_docs/                   # Documentation source
├── testdata/                   # Test images and scripts
├── pyproject.toml              # Project configuration
└── README.md                   # Project overview
```

### Module Overview

#### Core Modules (fast_mypyc/)

**engine.py**
- `OKLCHEngine`: Color space conversions and parsing
- `Attractor`: Color attractor representation
- `ColorTransformer`: Main transformation pipeline

**colorshine.py**
- `process_image()`: High-level API function
- LUT management and caching
- Processing orchestration

**gamut.py**
- `GamutMapper`: CSS Color Module 4 compliant mapping
- Perceptual gamut boundary detection

**io.py**
- `ImageProcessor`: Image loading/saving
- Format detection and conversion
- Bit depth management

#### Performance Modules (fast_numba/)

**trans_numba.py**
- JIT-compiled color space conversions
- Batch processing optimizations
- Memory-efficient algorithms

**engine_numba.py**
- Numba transformation kernels
- Distance calculation optimizations
- Falloff function computations

**gamut_numba.py**
- High-performance gamut mapping
- Binary search algorithms
- Vectorized boundary testing

#### GPU Support

**gpu.py**
- `ArrayModule`: CPU/GPU abstraction
- Memory management utilities
- Device detection and selection

### Design Principles

#### 1. Modular Architecture
Clear separation of concerns:
- Color science ↔ Performance optimization
- Core algorithms ↔ I/O and UI
- CPU ↔ GPU implementations

#### 2. Multiple Optimization Paths
Graceful fallback chain:
```
GPU (CuPy) → LUT → Numba → Pure Python
```

#### 3. Type Safety
Comprehensive type hints for:
- API clarity
- IDE support
- Static analysis

#### 4. Performance First
Optimization strategies:
- Vectorized operations
- JIT compilation
- Memory efficiency
- Spatial acceleration

#### 5. Extensibility
Extension points for:
- New color spaces
- Custom attractors
- Alternative algorithms
- Additional optimizations

## Testing Framework

### Test Organization

#### Unit Tests
```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_engine.py -v

# Run with coverage
python -m pytest tests/ --cov=src/imgcolorshine --cov-report=html
```

#### Performance Tests
```bash
# Benchmark processing times
python -m pytest tests/test_performance.py -v

# Profile memory usage
python -m pytest tests/test_performance.py::test_memory_usage -v
```

#### Property-Based Tests
Using Hypothesis for robust testing:

```python
from hypothesis import given, strategies as st
import numpy as np

@given(
    lightness=st.floats(min_value=0, max_value=100),
    chroma=st.floats(min_value=0, max_value=0.4),
    hue=st.floats(min_value=0, max_value=360)
)
def test_oklch_roundtrip_conversion(lightness, chroma, hue):
    """Test that OKLCH conversions are invertible"""
    from imgcolorshine import OKLCHEngine
    
    engine = OKLCHEngine()
    
    # Create OKLCH color
    oklch = np.array([lightness, chroma, hue])
    
    # Convert to sRGB and back
    rgb = engine.oklch_to_srgb(oklch.reshape(1, 1, 3))
    oklch_back = engine.srgb_to_oklch(rgb)
    
    # Should be approximately equal (within gamut mapping tolerance)
    np.testing.assert_allclose(oklch, oklch_back.reshape(3), atol=1e-3)
```

### Writing Tests

#### Test Structure Template

```python
#!/usr/bin/env python3
"""
Test module for [component]
"""
import pytest
import numpy as np
from imgcolorshine import [ComponentUnderTest]

class Test[Component]:
    """Test suite for [Component]"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.component = ComponentUnderTest()
        
    def test_basic_functionality(self):
        """Test basic operation"""
        # Arrange
        input_data = np.array([...])
        expected = np.array([...])
        
        # Act
        result = self.component.process(input_data)
        
        # Assert
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
    def test_edge_cases(self):
        """Test boundary conditions"""
        # Test empty input
        with pytest.raises(ValueError):
            self.component.process(np.array([]))
            
        # Test invalid parameters
        with pytest.raises(ValueError):
            self.component.process(invalid_param=-1)
            
    def test_performance(self):
        """Test performance characteristics"""
        import time
        
        large_input = np.random.rand(1000, 1000, 3)
        
        start = time.time()
        result = self.component.process(large_input)
        duration = time.time() - start
        
        # Should complete within reasonable time
        assert duration < 10.0  # seconds
        assert result.shape == large_input.shape
```

#### Testing Guidelines

1. **Test at multiple levels**: unit, integration, system
2. **Use property-based testing** for mathematical functions
3. **Test error conditions** and edge cases
4. **Benchmark performance** regressions
5. **Test GPU and CPU paths** separately
6. **Mock external dependencies** when needed

## Code Quality Tools

### Linting and Formatting

imgcolorshine uses **Ruff** for both linting and formatting:

```bash
# Format code
ruff format src/ tests/

# Check for issues
ruff check src/ tests/

# Fix automatically where possible
ruff check --fix src/ tests/

# Show specific rule violations
ruff check --show-source src/
```

#### Ruff Configuration

In `pyproject.toml`:
```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # Line too long (handled by formatter)
    "B008",  # Do not perform function calls in argument defaults
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]  # Allow assert in tests
```

### Type Checking

Use **mypy** for static type analysis:

```bash
# Type check entire codebase
mypy src/imgcolorshine/

# Check specific file
mypy src/imgcolorshine/engine.py

# Generate detailed report
mypy src/imgcolorshine/ --html-report mypy_report/
```

#### Type Checking Guidelines

1. **Use precise types**: Prefer `np.ndarray` over `Any`
2. **Annotate function signatures**: All public functions must have types
3. **Use generics** where appropriate: `list[str]` vs `list`
4. **Handle optional types**: Use `Union` or `|` for multiple types
5. **Document complex types**: Use `TypeAlias` for readability

```python
from typing import TypeAlias
import numpy as np

# Type aliases for clarity
RGBArray: TypeAlias = np.ndarray  # Shape: (..., 3), dtype: float32
OKLCHArray: TypeAlias = np.ndarray  # Shape: (..., 3), dtype: float32
AttractorSpec: TypeAlias = str  # Format: "color;tolerance;strength"

def transform_image(
    image: RGBArray,
    attractors: list[AttractorSpec],
    gpu: bool = True
) -> RGBArray:
    """Transform image with type-safe interface"""
    ...
```

### Pre-commit Hooks

Set up automated code quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Configuration in `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        additional_dependencies: [numpy, types-Pillow]
```

## Performance Development

### Profiling Tools

#### Built-in Profiling

```bash
# Enable verbose timing
imgcolorshine shine large_image.jpg "blue;50;70" --verbose=True

# Profile with cProfile
python -m cProfile -o profile.stats -c "
import imgcolorshine
imgcolorshine.process_image('test.jpg', ['blue;50;70'])
"

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

#### Advanced Profiling

```python
#!/usr/bin/env python3
"""
Advanced performance profiling script
"""
import cProfile
import pstats
import time
import psutil
import tracemalloc
from imgcolorshine import process_image

def profile_memory_usage():
    """Profile memory usage during processing"""
    tracemalloc.start()
    
    # Get baseline memory
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process image
    result = process_image(
        "testdata/louis.jpg",
        ["blue;50;70"],
        output_image=None
    )
    
    # Get peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    current, peak_trace = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    print(f"Peak memory: {peak_memory:.1f} MB")
    print(f"Memory increase: {peak_memory - baseline_memory:.1f} MB")
    print(f"Peak traced memory: {peak_trace / 1024 / 1024:.1f} MB")

def profile_cpu_performance():
    """Profile CPU performance hotspots"""
    profiler = cProfile.Profile()
    
    profiler.enable()
    result = process_image(
        "testdata/louis.jpg", 
        ["blue;50;70"],
        output_image=None,
        gpu=False  # Force CPU
    )
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

if __name__ == "__main__":
    print("=== Memory Profiling ===")
    profile_memory_usage()
    
    print("\n=== CPU Profiling ===")
    profile_cpu_performance()
```

### Numba Development

#### Writing Numba Functions

```python
import numba as nb
import numpy as np

@nb.jit(nopython=True, parallel=True)
def optimized_function(data: np.ndarray) -> np.ndarray:
    """
    High-performance Numba function
    
    Guidelines:
    - Use nopython=True for best performance
    - Parallel=True for CPU parallelization
    - Avoid Python objects (lists, dicts)
    - Use NumPy arrays exclusively
    """
    result = np.empty_like(data)
    
    # Parallel loop
    for i in nb.prange(data.shape[0]):
        for j in range(data.shape[1]):
            # Computation here
            result[i, j] = data[i, j] * 2.0
            
    return result

# Test compilation and performance
@nb.jit(nopython=True)
def test_numba_compilation():
    """Ensure function compiles correctly"""
    test_data = np.random.rand(100, 100).astype(np.float32)
    result = optimized_function(test_data)
    return result.shape == test_data.shape
```

#### Debugging Numba Code

```python
# Disable JIT for debugging
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Enable debug mode
@nb.jit(nopython=True, debug=True)
def debug_function(data):
    # Can use print statements in debug mode
    print("Debug: processing data shape", data.shape)
    return data * 2
```

### GPU Development

#### CuPy Development Guidelines

```python
import cupy as cp
import numpy as np

def gpu_kernel_example(data: cp.ndarray) -> cp.ndarray:
    """
    GPU kernel development example
    
    Guidelines:
    - Use CuPy arrays for GPU operations
    - Minimize CPU-GPU transfers
    - Vectorize operations when possible
    - Use custom kernels for complex operations
    """
    
    # Simple vectorized operation
    result = cp.sqrt(data ** 2 + 1.0)
    
    return result

# Custom CUDA kernel for complex operations
custom_kernel = cp.RawKernel(r'''
extern "C" __global__
void complex_operation(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Custom computation here
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}
''', 'complex_operation')

def use_custom_kernel(data: cp.ndarray) -> cp.ndarray:
    """Use custom CUDA kernel"""
    output = cp.empty_like(data)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (data.size + threads_per_block - 1) // threads_per_block
    
    custom_kernel(
        (blocks,), (threads_per_block,),
        (data, output, data.size)
    )
    
    return output
```

## Contributing Guidelines

### Contribution Workflow

1. **Fork the repository** on GitHub
2. **Create feature branch** from main
3. **Make changes** following coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit pull request** with clear description

### Pull Request Process

#### Before Submitting

```bash
# Ensure code quality
ruff format src/ tests/
ruff check src/ tests/
mypy src/imgcolorshine/

# Run full test suite
python -m pytest tests/ --cov=src/imgcolorshine

# Test on sample images
python -m pytest tests/test_integration.py -v

# Update documentation if needed
cd src_docs
mkdocs build
```

#### PR Description Template

```markdown
## Description
Brief description of the changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Added unit tests for new functionality
- [ ] All existing tests pass
- [ ] Tested on multiple image types
- [ ] Performance benchmarks included (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Development Areas

#### High-Priority Areas

1. **Performance Optimization**
   - New Numba kernels
   - GPU algorithm improvements
   - Memory usage reduction

2. **Color Science**
   - Additional color spaces
   - Improved gamut mapping
   - Perceptual metrics

3. **Usability**
   - Better error messages
   - Progress reporting
   - Configuration management

4. **Platform Support**
   - macOS Metal acceleration
   - AMD GPU support
   - ARM optimization

#### Feature Requests

Common requests from users:
- HDR image support
- Video processing capabilities
- Real-time preview modes
- Batch processing GUI
- Plugin system for custom attractors

### Code Style Guidelines

#### Python Style

Follow PEP 8 with these additions:

```python
# Good: Descriptive names
def calculate_perceptual_distance(color1: np.ndarray, color2: np.ndarray) -> float:
    """Calculate perceptual distance between colors in Oklab space"""
    delta = color1 - color2
    return np.sqrt(np.sum(delta ** 2))

# Bad: Unclear names
def calc_dist(c1, c2):
    d = c1 - c2
    return np.sqrt(np.sum(d ** 2))

# Good: Type hints and docstrings
def process_attractor(
    attractor: Attractor,
    image_data: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Process image data with single attractor
    
    Args:
        attractor: Configured color attractor
        image_data: Image in OKLCH space
        threshold: Distance threshold for influence
        
    Returns:
        Transformed image data
        
    Raises:
        ValueError: If threshold is negative
    """
    if threshold < 0:
        raise ValueError("Threshold must be non-negative")
    
    # Implementation here
    return transformed_data
```

#### Performance Code Style

```python
# Good: Numba-optimized function
@nb.jit(nopython=True, parallel=True)
def fast_color_transform(
    image: np.ndarray,
    attractor_color: np.ndarray,
    threshold: float,
    strength: float
) -> np.ndarray:
    """Fast color transformation kernel"""
    result = np.empty_like(image)
    
    for i in nb.prange(image.shape[0]):
        for j in range(image.shape[1]):
            # Pixel processing
            pixel = image[i, j]
            distance = calculate_distance(pixel, attractor_color)
            
            if distance <= threshold:
                weight = calculate_falloff(distance / threshold) * strength
                result[i, j] = blend_colors(pixel, attractor_color, weight)
            else:
                result[i, j] = pixel
                
    return result

# Bad: Python loops without optimization
def slow_color_transform(image, attractor_color, threshold, strength):
    result = []
    for row in image:
        result_row = []
        for pixel in row:
            # Inefficient processing
            result_row.append(process_pixel(pixel))
        result.append(result_row)
    return np.array(result)
```

## Extension Points

### Custom Color Spaces

Add support for new color spaces:

```python
# src/imgcolorshine/colorspaces/custom.py

import numpy as np
from abc import ABC, abstractmethod

class ColorSpace(ABC):
    """Abstract base class for color spaces"""
    
    @abstractmethod
    def from_srgb(self, srgb: np.ndarray) -> np.ndarray:
        """Convert from sRGB to this color space"""
        pass
    
    @abstractmethod
    def to_srgb(self, color: np.ndarray) -> np.ndarray:
        """Convert from this color space to sRGB"""
        pass
    
    @abstractmethod
    def calculate_distance(self, color1: np.ndarray, color2: np.ndarray) -> np.ndarray:
        """Calculate perceptual distance in this color space"""
        pass

class CustomColorSpace(ColorSpace):
    """Example custom color space implementation"""
    
    def from_srgb(self, srgb: np.ndarray) -> np.ndarray:
        # Implement conversion
        return converted_array
    
    def to_srgb(self, color: np.ndarray) -> np.ndarray:
        # Implement reverse conversion
        return srgb_array
    
    def calculate_distance(self, color1: np.ndarray, color2: np.ndarray) -> np.ndarray:
        # Implement distance metric
        return distance_array
```

### Custom Attractors

Extend the attractor system:

```python
# src/imgcolorshine/attractors/custom.py

from imgcolorshine import Attractor
import numpy as np

class CustomAttractor(Attractor):
    """Custom attractor with special behavior"""
    
    def __init__(self, color_oklch, tolerance, strength, custom_param=1.0):
        super().__init__(color_oklch, tolerance, strength)
        self.custom_param = custom_param
    
    def calculate_influence(self, pixel_oklch, threshold):
        """Custom influence calculation"""
        # Standard distance calculation
        distances = self._calculate_distances(pixel_oklch)
        
        # Custom falloff function
        normalized_distances = distances / threshold
        falloff = self._custom_falloff(normalized_distances)
        
        # Apply strength
        return self._apply_strength(falloff)
    
    def _custom_falloff(self, normalized_distances):
        """Custom falloff function"""
        # Example: exponential falloff
        return np.exp(-normalized_distances * self.custom_param)

# Register custom attractor
from imgcolorshine.engine import AttractorFactory

AttractorFactory.register("custom", CustomAttractor)
```

### Performance Backends

Add new performance backends:

```python
# src/imgcolorshine/backends/custom.py

from imgcolorshine.gpu import ArrayModule
import numpy as np

class CustomBackend(ArrayModule):
    """Custom performance backend"""
    
    def __init__(self):
        super().__init__()
        self.backend_name = "custom"
    
    @property
    def available(self) -> bool:
        """Check if backend is available"""
        try:
            import custom_library
            return True
        except ImportError:
            return False
    
    def asarray(self, data: np.ndarray) -> np.ndarray:
        """Convert to backend-specific array type"""
        if self.available:
            import custom_library
            return custom_library.asarray(data)
        return data
    
    def process_transform(self, image, attractors, options):
        """Backend-specific processing"""
        # Implement high-performance transformation
        return result

# Register backend
from imgcolorshine.backends import BackendRegistry

BackendRegistry.register("custom", CustomBackend)
```

## Documentation Development

### Building Documentation

```bash
# Navigate to docs source
cd src_docs

# Build documentation
mkdocs build

# Serve locally for development
mkdocs serve --dev-addr=127.0.0.1:8000

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

### Documentation Guidelines

1. **Write for users first**: Prioritize user needs over technical details
2. **Include examples**: Every concept should have working examples
3. **Test code snippets**: Ensure all examples actually work
4. **Use consistent formatting**: Follow established patterns
5. **Update with changes**: Keep docs in sync with code

#### Writing Style

```markdown
# Good: Clear, actionable heading
## How to Transform Image Colors

Transform images using attractors with this simple pattern:

```bash
imgcolorshine shine input.jpg "color;tolerance;strength"
```

This command creates a color attractor that pulls existing colors toward the specified target.

# Bad: Vague, technical heading
## Color Transformation API Usage Patterns

The color transformation subsystem utilizes the attractor paradigm to implement perceptual color space modifications via the OKLCH color model.
```

## Release Process

### Version Management

imgcolorshine uses semantic versioning (SemVer):

- **MAJOR.MINOR.PATCH** (e.g., 3.2.1)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** on multiple platforms
4. **Update documentation** for new features
5. **Create release tag** in Git
6. **Build and upload** to PyPI
7. **Update GitHub release** with notes

### Automated Releases

GitHub Actions handles automated releases:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Build package
        run: |
          pip install build
          python -m build
      
      - name: Upload to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

## Community and Support

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community chat
- **Documentation**: Comprehensive guides and examples
- **Code**: Well-commented source code

### Contributing Areas

#### For Beginners
- Documentation improvements
- Test case additions
- Bug reports with minimal examples
- Performance benchmarking

#### For Experienced Developers
- New algorithm implementations
- Performance optimizations
- Platform-specific enhancements
- API design improvements

#### For Researchers
- Color science improvements
- Perceptual metrics
- Novel transformation algorithms
- Academic collaborations

## Future Roadmap

### Short Term (Next Release)
- HDR image support
- Apple Silicon optimization
- Enhanced error reporting
- Configuration file support

### Medium Term (6-12 months)
- Real-time processing mode
- Video frame processing
- Advanced color harmony detection
- Machine learning integration

### Long Term (1+ years)
- Complete rewrite in Rust (imgcolorshine-rs)
- WebAssembly support for browsers
- Professional GUI application
- Plugin ecosystem

!!! success "Ready to Contribute!"
    You're now equipped with everything needed to contribute to imgcolorshine. Whether fixing bugs, optimizing performance, or adding features, your contributions help make color transformation better for everyone.

Start with the [good first issue](https://github.com/twardoch/imgcolorshine/labels/good%20first%20issue) label on GitHub to find beginner-friendly tasks.