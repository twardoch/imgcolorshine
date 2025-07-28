"""imgcolorshine.fast_mypyc
=================================

Sub-package with deterministic, pure-Python helpers that are **ahead-of-time
compiled** via `mypyc` during wheel builds.  Importing these modules from a
source checkout works in plain Python; the compiled C-extensions seamlessly
replace them once built.

Public modules in :pymod:`imgcolorshine` re-export selected symbols so existing
user code continues to work unchanged.
"""

# this_file: src/imgcolorshine/fast_mypyc/__init__.py

from __future__ import annotations
from importlib import import_module

__all__: list[str] = [
    "utils",
    "engine_helpers",
    "gamut_helpers", 
    "colorshine_helpers",
]

# Lazily import to avoid overhead during startup if speedups not loaded.
utils = import_module("imgcolorshine.fast_mypyc.utils")
engine_helpers = import_module("imgcolorshine.fast_mypyc.engine_helpers")
gamut_helpers = import_module("imgcolorshine.fast_mypyc.gamut_helpers")
colorshine_helpers = import_module("imgcolorshine.fast_mypyc.colorshine_helpers")
