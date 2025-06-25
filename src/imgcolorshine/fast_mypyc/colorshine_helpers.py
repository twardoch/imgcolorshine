"""Pure Python helper functions from colorshine module for mypyc compilation.

This module contains helper functions that benefit from mypyc compilation.
"""
# this_file: src/imgcolorshine/fast_mypyc/colorshine_helpers.py

from __future__ import annotations

from pathlib import Path

# Constants
ATTRACTOR_PARTS = 3
TOLERANCE_MIN, TOLERANCE_MAX = 0.0, 100.0
STRENGTH_MIN, STRENGTH_MAX = 0.0, 200.0


def parse_attractor(attractor_str: str) -> tuple[str, float, float]:
    """Parse attractor string format: 'color;tolerance;strength'."""
    try:
        parts = attractor_str.split(";")
        if len(parts) != ATTRACTOR_PARTS:
            msg = f"Invalid attractor format: {attractor_str}"
            raise ValueError(msg)

        color = parts[0].strip()
        tolerance = float(parts[1])
        strength = float(parts[2])

        if not TOLERANCE_MIN <= tolerance <= TOLERANCE_MAX:
            msg = f"Tolerance must be {TOLERANCE_MIN}-{TOLERANCE_MAX}, got {tolerance}"
            raise ValueError(msg)
        if not STRENGTH_MIN <= strength <= STRENGTH_MAX:
            msg = f"Strength must be {STRENGTH_MIN}-{STRENGTH_MAX}, got {strength}"
            raise ValueError(msg)

        return color, tolerance, strength
    except Exception as e:
        msg = f"Invalid attractor '{attractor_str}': {e}"
        raise ValueError(msg) from e


def generate_output_path(input_path: Path) -> Path:
    """Generate output filename if not provided."""
    stem = input_path.stem
    suffix = input_path.suffix
    return input_path.parent / f"{stem}_colorshine{suffix}"