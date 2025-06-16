# Development Plan for imgcolorshine - Remaining Tasks

## Critical Issue: Build Process Missing Source Files

### Problem
After running `uvx hatch clean; uvx hatch build`, the generated tarball `./dist/imgcolorshine-3.3.2.tar.gz` only contains:
- LICENSE, PKG-INFO, pyproject.toml, README.md
- src/imgcolorshine/__version__.py and py.typed

**All Python source files are missing from the distribution!**

### Root Cause
The `[tool.hatch.build]` section in `pyproject.toml` only includes:
```toml
include = [
    'src/imgcolorshine/py.typed',
    'src/imgcolorshine/data/**/*',
]
```

This explicitly excludes all `.py` files except what's auto-included by hatch.

### Solution
Update the build configuration to include all Python source files:

```toml
[tool.hatch.build]
include = [
    'src/imgcolorshine/**/*.py',  # Include all Python files
    'src/imgcolorshine/py.typed',
    'src/imgcolorshine/data/**/*',
]
```

## Task: Fix MyPyc Compilation Issues

### Current Status
MyPyc compilation is temporarily disabled in `pyproject.toml` due to:

1. **Code Issues:**
   - `gpu.py:24`: Duplicate `cp` definition
   - `colorshine.py:169`: Duplicate `transformed` variable
   - Missing type stubs for dependencies (numpy, loguru, etc.)

2. **Build Environment Issues:**
   - Type stubs not available during build
   - MyPyc can't compile Numba-decorated functions

### Implementation Plan

#### Step 1: Fix Code Issues
1. Fix duplicate `cp` in `gpu.py`
2. Fix duplicate `transformed` in `colorshine.py`
3. Add type ignores where needed

#### Step 2: Configure MyPyc Properly
```toml
[build-system]
requires = [
    'hatchling>=1.27.0',
    'hatch-vcs>=0.4.0',
    'mypy>=1.15.0',
    'hatch-mypyc>=0.16.0',
    # Add type stubs
    'numpy',
    'types-Pillow',
    'scipy-stubs',
]

[tool.hatch.build.hooks.mypyc]
dependencies = ["hatch-mypyc"]
mypy-args = ["--ignore-missing-imports"]
# Only compile pure Python modules (exclude Numba-heavy files)
files = [
    "src/imgcolorshine/colorshine.py",
    "src/imgcolorshine/engine.py",
    "src/imgcolorshine/gamut.py",
    "src/imgcolorshine/io.py",
    "src/imgcolorshine/utils.py",
    "src/imgcolorshine/falloff.py",
    "src/imgcolorshine/cli.py",
]
```

#### Step 3: Build and Test
```bash
# Clean and build
uvx hatch clean
uvx hatch build

# Verify the distribution contains compiled binaries
tar -tzf dist/imgcolorshine-*.tar.gz | grep -E '\.(so|pyd)$'

# Test installation
pip install dist/imgcolorshine-*.whl
python -c "import imgcolorshine; print(imgcolorshine.__version__)"
```

## Task: Integration and Testing

### After fixing the build:
1. Run full test suite
2. Verify MyPyc binaries are included in wheel
3. Benchmark performance improvements
4. Update documentation about build process

### Commands to run:
```bash
# Full test suite
python -m pytest

# Performance benchmarks
python -m pytest tests/test_benchmark.py -v

# Type checking
mypy src/imgcolorshine

# Linting
ruff check src/imgcolorshine tests
```

## Summary of Remaining Tasks

1. **[CRITICAL]** Fix build configuration to include source files
2. **[HIGH]** Fix code issues blocking MyPyc compilation
3. **[HIGH]** Re-enable and configure MyPyc properly
4. **[MEDIUM]** Verify build produces working distribution with binaries
5. **[MEDIUM]** Update documentation about the build process