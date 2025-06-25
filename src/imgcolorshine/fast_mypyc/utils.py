"""Proxy module for AOT-compiled utilities.

This module simply re-exports names from :pymod:`imgcolorshine.utils` so that
`mypyc` can compile this *thin* wrapper while leaving the original source file
untouched.  At runtime, the C-extension shadows this wrapper and still refers
to the original Python implementation if not compiled.
"""

# ruff: noqa
# this_file: src/imgcolorshine/fast_mypyc/utils.py

from imgcolorshine.utils import *  # type: ignore[import]
