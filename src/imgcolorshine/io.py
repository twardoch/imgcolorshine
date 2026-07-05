# this_file: src/imgcolorshine/io.py
"""Top-level re-export of image I/O symbols from fast_mypyc sub-package.

Provides ``imgcolorshine.io`` as the public import path.
"""

from imgcolorshine.fast_mypyc.io import ImageProcessor

__all__ = ["ImageProcessor"]
