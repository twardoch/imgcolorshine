# Chapter 1: Installation

This chapter covers everything you need to install and set up imgcolorshine on your system.

## System Requirements

### Python Version
- **Python 3.10 or higher** is required
- Python 3.11+ recommended for optimal performance

### Operating Systems
- **Linux** (Ubuntu 20.04+, CentOS 8+, etc.)
- **macOS** (10.15+ Catalina)
- **Windows** (10/11)

### Hardware Requirements
- **Minimum RAM:** 4GB (8GB+ recommended for large images)
- **Storage:** 100MB for base installation
- **GPU (Optional):** NVIDIA GPU with CUDA support for acceleration

## Installation Methods

### Method 1: PyPI Installation (Recommended)

The simplest way to install imgcolorshine:

```bash
pip install imgcolorshine
```

This installs the core package with all essential dependencies.

### Method 2: Development Installation

For the latest features or if you plan to contribute:

```bash
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine
pip install -e .
```

The `-e` flag creates an "editable" installation where changes to the source code are immediately reflected.

### Method 3: UV Package Manager (Faster)

If you have [uv](https://github.com/astral-sh/uv) installed:

```bash
uv pip install imgcolorshine
```

UV is significantly faster than pip for package resolution and installation.

## Optional Dependencies

### GPU Acceleration (NVIDIA Only)

For **CUDA 11.x** systems:
```bash
pip install cupy-cuda11x
```

For **CUDA 12.x** systems:
```bash
pip install cupy-cuda12x
```

!!! warning "GPU Requirements"
    - NVIDIA GPU with compute capability 6.0+
    - CUDA Toolkit installed on your system
    - Compatible GPU drivers

!!! tip "Automatic Detection"
    imgcolorshine automatically detects CuPy and uses GPU acceleration when available. No configuration needed!

### Development Dependencies

For code development and testing:

```bash
# With standard pip
pip install -e ".[dev,test,speedups]"

# With uv
uv pip install -e ".[dev,test,speedups]"
```

This includes:
- **dev:** Ruff (linting/formatting), mypy (type checking)
- **test:** pytest, coverage tools
- **speedups:** Numba, mypyc compilation tools

## Verification

### Basic Installation Check

Verify imgcolorshine is installed correctly:

```bash
imgcolorshine --help
```

Expected output:
```
Usage: imgcolorshine COMMAND

Commands:
  shine    Transform image colors using OKLCH attractors

For detailed help: imgcolorshine shine --help
```

### Version Check

Check your installed version:

```bash
python -c "import imgcolorshine; print(imgcolorshine.__version__)"
```

### Test Transform

Create a simple test transformation:

```bash
# Assuming you have a test image
imgcolorshine shine test_image.jpg "blue;50;60" --output_image=test_output.jpg
```

### GPU Detection Test

Check if GPU acceleration is available:

```bash
python -c "
from imgcolorshine.gpu import ArrayModule
am = ArrayModule()
print(f'GPU available: {am.gpu_available}')
if am.gpu_available:
    print(f'GPU name: {am.device_name}')
"
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'imgcolorshine'

**Solution:** Ensure you're in the correct Python environment
```bash
which python
python -m pip list | grep imgcolorshine
```

#### CUDA/GPU Issues

**Problem:** CuPy not found or GPU not detected

**Solutions:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Install correct CuPy version for your CUDA

#### Permission Errors (Linux/macOS)

**Problem:** Permission denied during installation

**Solutions:**
```bash
# Use user installation
pip install --user imgcolorshine

# Or virtual environment (recommended)
python -m venv imgcolorshine_env
source imgcolorshine_env/bin/activate  # Linux/macOS
# imgcolorshine_env\Scripts\activate  # Windows
pip install imgcolorshine
```

#### Memory Issues with Large Images

**Problem:** Out of memory errors

**Solutions:**
- Increase system RAM
- Use smaller tile sizes: `--tile_size=512`
- Process images in smaller batches

### Performance Issues

#### Slow Processing

**Diagnostics:**
```bash
# Check if Numba compilation is working
python -c "
from imgcolorshine.fast_numba import trans_numba
print('Numba available:', hasattr(trans_numba, 'srgb_to_oklab_batch'))
"
```

**Solutions:**
- Enable GPU: ensure CuPy is installed
- Use LUT acceleration: `--lut_size=65`
- Try fused kernels: `--fused_kernel=True`

#### First Run Slow

This is normal! Numba compiles functions on first use.

## Virtual Environment Setup

### Using venv (Standard)

```bash
python -m venv imgcolorshine_env
source imgcolorshine_env/bin/activate  # Linux/macOS
# imgcolorshine_env\Scripts\activate  # Windows
pip install imgcolorshine
```

### Using conda

```bash
conda create -n imgcolorshine python=3.11
conda activate imgcolorshine
pip install imgcolorshine
```

### Using uv (Modern)

```bash
uv venv imgcolorshine_env
source imgcolorshine_env/bin/activate  # Linux/macOS
# imgcolorshine_env\Scripts\activate  # Windows
uv pip install imgcolorshine
```

## Platform-Specific Notes

### Linux

Most straightforward installation. Package managers like apt/yum may have older Python versions.

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv
python3.11 -m pip install imgcolorshine
```

### macOS

**Homebrew Python (Recommended):**
```bash
brew install python@3.11
python3.11 -m pip install imgcolorshine
```

**System Python:** Works but may require `--user` flag

### Windows

**Command Prompt/PowerShell:**
```cmd
pip install imgcolorshine
```

**WSL (Windows Subsystem for Linux):** Follow Linux instructions

## Next Steps

With imgcolorshine installed, you're ready for:

1. **[Quick Start](quickstart.md)** - Your first color transformation
2. **[Basic Usage](basic-usage.md)** - Understanding core concepts
3. **[Performance Optimization](performance-optimization.md)** - Maximizing speed

## Advanced Installation Options

### Compile from Source with Optimizations

For maximum performance on your specific hardware:

```bash
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine

# Install with compilation optimizations
pip install -e ".[speedups]"

# Force recompilation of mypyc modules
pip install --force-reinstall --no-deps .
```

### Docker Installation

A Dockerfile is available for containerized environments:

```bash
git clone https://github.com/twardoch/imgcolorshine.git
cd imgcolorshine
docker build -t imgcolorshine .
docker run -v $(pwd):/workspace imgcolorshine shine input.jpg "blue;50;60"
```

!!! success "Installation Complete"
    You're now ready to transform images with imgcolorshine! Continue to [Quick Start](quickstart.md) for your first transformation.