#!/usr/bin/env python
"""
Build extension for mypyc compilation.

This module integrates mypyc compilation into the build process,
allowing selected modules to be compiled to C extensions for better
performance.
"""

import os
import sys
from pathlib import Path
from typing import Any

try:
    from mypyc.build import mypycify
except ImportError:
    mypycify = None


def build_mypyc_extensions() -> list[Any] | None:
    """
    Build mypyc extensions for performance-critical modules.
    
    Returns compiled extension modules or None if mypyc is not available.

    """
    if mypycify is None:
        print("mypyc not available, skipping compilation")
        return None
    
    # Get the source directory
    src_dir = Path(__file__).parent / "src"
    
    # Modules to compile (from pyproject.toml configuration)
    modules_to_compile = [
        "imgcolorshine.color",
        "imgcolorshine.transform",
        "imgcolorshine.io",
        "imgcolorshine.falloff",
    ]
    
    # Convert module names to file paths
    paths = []
    for module in modules_to_compile:
        module_path = module.replace(".", "/") + ".py"
        full_path = src_dir / module_path
        if full_path.exists():
            paths.append(str(full_path))
        else:
            print(f"Warning: Module file not found: {full_path}")
    
    if not paths:
        print("No modules found to compile")
        return None
    
    # Compile with mypyc
    print(f"Compiling {len(paths)} modules with mypyc...")
    
    # Mypyc compilation options
    options = [
        "--strict-optional",
        "--warn-return-any",
        "--warn-unused-configs",
        "--python-version", f"{sys.version_info.major}.{sys.version_info.minor}",
    ]
    
    try:
        ext_modules = mypycify(paths, options)
        print(f"Successfully compiled {len(ext_modules)} modules")
        return ext_modules
    except Exception as e:
        print(f"mypyc compilation failed: {e}")
        print("Falling back to pure Python")
        return None


if __name__ == "__main__":
    # Test the build function
    extensions = build_mypyc_extensions()
    if extensions:
        print(f"Built {len(extensions)} extensions")
    else:
        print("No extensions built")