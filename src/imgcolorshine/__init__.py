#!/usr/bin/env -S uv run -s
# /// script
# dependencies = []
# ///
# this_file: src/imgcolorshine/__init__.py

"""imgcolorshine: Perceptual color transformations."""

__version__ = "0.1.0"

from .fast_mypyc.cli import ImgColorShineCLI
from .fast_mypyc.colorshine import process_image
from .fast_mypyc.engine import Attractor, ColorTransformer, OKLCHEngine
from .fast_mypyc.io import ImageProcessor
from .lut import LUTManager

__all__ = [
    "Attractor",
    "ColorTransformer",
    "ImageProcessor",
    "ImgColorShineCLI",
    "LUTManager",
    "OKLCHEngine",
    "__version__",
    "process_image",
]
