# this_file: src/imgcolorshine/engine.py
"""Top-level re-export of core engine symbols from fast_mypyc sub-package.

Provides ``imgcolorshine.engine`` as the public import path so tests and
user code can write ``from imgcolorshine.engine import OKLCHEngine`` etc.
"""

from imgcolorshine.fast_mypyc.engine import Attractor, ColorTransformer, OKLCHEngine

__all__ = ["Attractor", "ColorTransformer", "OKLCHEngine"]
