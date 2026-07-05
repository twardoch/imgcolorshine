#!/usr/bin/env -S uv run -s
# /// script
# dependencies = []
# ///
# this_file: src/imgcolorshine/__init__.py

"""imgcolorshine: Perceptual color transformations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("imgcolorshine")
except PackageNotFoundError:
    __version__ = "0.0.0"

from imgcolorshine.cli import ImgColorShineCLI
from imgcolorshine.colorshine import process_image
from imgcolorshine.engine import Attractor, ColorTransformer, OKLCHEngine
from imgcolorshine.io import ImageProcessor
from imgcolorshine.lut import LUTManager

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
