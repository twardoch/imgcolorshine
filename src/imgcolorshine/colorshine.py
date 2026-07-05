# this_file: src/imgcolorshine/colorshine.py
"""Core image-processing pipeline for imgcolorshine.

Public API consumed by the CLI and directly by library users:

- :func:`parse_attractor` — parse ``"color;tolerance;strength"`` strings.
- :func:`generate_output_path` — auto-generate output file name.
- :func:`setup_logging` — configure loguru verbosity.
- :func:`process_with_optimizations` — low-level transform dispatcher.
- :func:`process_image` — high-level entry point (load → transform → save).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from imgcolorshine.engine import ColorTransformer, OKLCHEngine
from imgcolorshine.io import ImageProcessor

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ATTRACTOR_PARTS = 3
_TOLERANCE_MIN, _TOLERANCE_MAX = 0.0, 100.0
_STRENGTH_MIN, _STRENGTH_MAX = 0.0, 100.0  # public API cap; internal model allows 200


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(*, verbose: bool = False) -> None:
    """Configure loguru output level.

    Parameters
    ----------
    verbose:
        When *True* log at DEBUG level with timestamps; otherwise INFO only.
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")
    else:
        logger.add(sys.stderr, level="INFO", format="{message}")


# ---------------------------------------------------------------------------
# Attractor parsing
# ---------------------------------------------------------------------------


def parse_attractor(attractor_str: str) -> tuple[str, float, float]:
    """Parse an attractor specification string.

    Format: ``"color;tolerance;strength"``

    Parameters
    ----------
    attractor_str:
        The raw attractor string, e.g. ``"red;50;75"`` or
        ``"oklch(70% 0.2 120);40;80"``.

    Returns
    -------
    tuple[str, float, float]
        ``(color_str, tolerance, strength)``

    Raises
    ------
    ValueError
        On malformed input, out-of-range tolerance/strength, or non-numeric
        values.
    """
    parts = attractor_str.split(";")
    if len(parts) != _ATTRACTOR_PARTS:
        msg = f"Invalid attractor format: expected 'color;tolerance;strength', got {attractor_str!r}"
        raise ValueError(msg)

    color = parts[0].strip()

    try:
        tolerance = float(parts[1])
    except ValueError as exc:
        msg = f"Invalid attractor format: tolerance must be numeric, got {parts[1]!r}"
        raise ValueError(msg) from exc

    try:
        strength = float(parts[2])
    except ValueError as exc:
        msg = f"Invalid attractor format: strength must be numeric, got {parts[2]!r}"
        raise ValueError(msg) from exc

    if not _TOLERANCE_MIN <= tolerance <= _TOLERANCE_MAX:
        msg = f"Tolerance must be 0-100, got {tolerance}"
        raise ValueError(msg)

    if not _STRENGTH_MIN <= strength <= _STRENGTH_MAX:
        msg = f"Strength must be 0-100, got {strength}"
        raise ValueError(msg)

    return color, tolerance, strength


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------


def generate_output_path(input_path: Path) -> Path:
    """Append ``_colorshine`` to the input file stem.

    Parameters
    ----------
    input_path:
        Source image path.

    Returns
    -------
    Path
        E.g. ``/tmp/photo.jpg`` → ``/tmp/photo_colorshine.jpg``.
    """
    stem = input_path.stem
    suffix = input_path.suffix
    return input_path.parent / f"{stem}_colorshine{suffix}"


# ---------------------------------------------------------------------------
# Low-level transform dispatcher (patchable in tests)
# ---------------------------------------------------------------------------


def process_with_optimizations(
    image: np.ndarray,
    attractor_objects: list[Any],
    luminance: bool,
    saturation: bool,
    chroma: bool,
    fast_hierar: bool,
    fast_spatial: bool,
    transformer: ColorTransformer,
    engine: OKLCHEngine,
) -> np.ndarray:
    """Apply attractors to *image* and return the transformed result.

    All boolean flags are forwarded to :meth:`ColorTransformer.transform_image`.
    ``fast_hierar`` and ``fast_spatial`` are currently reserved for future
    acceleration paths and are accepted but not yet acted upon.

    Parameters
    ----------
    image:
        Float32 array of shape ``(H, W, 3)`` in [0, 1].
    attractor_objects:
        List of :class:`~imgcolorshine.engine.Attractor` instances.
    luminance:
        Transform the L (lightness) channel.
    saturation:
        Transform the C (chroma/saturation) channel.
    chroma:
        Alias for saturation — included for API compatibility.
    fast_hierar:
        Use hierarchical multi-resolution processing (reserved).
    fast_spatial:
        Use spatial acceleration structures (reserved).
    transformer:
        Configured :class:`~imgcolorshine.engine.ColorTransformer`.
    engine:
        Configured :class:`~imgcolorshine.engine.OKLCHEngine`.

    Returns
    -------
    np.ndarray
        Transformed float32 image of the same shape.
    """
    flags = {
        "luminance": luminance,
        "saturation": saturation or chroma,
        "hue": chroma,
    }
    return transformer.transform_image(image, attractor_objects, flags)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def process_image(
    input_image: str,
    attractors: tuple[str, ...],
    output_image: str | None = None,
    *,
    luminance: bool = True,
    saturation: bool = True,
    hue: bool = True,
    chroma: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """Transform *input_image* with one or more colour attractors.

    Parameters
    ----------
    input_image:
        Path to the source image (any format supported by OpenCV / PIL).
    attractors:
        Tuple of attractor strings, each in ``"color;tolerance;strength"`` form.
    output_image:
        Destination path.  Auto-generated from *input_image* when *None*.
    luminance:
        Enable L-channel (lightness) transformation.
    saturation:
        Enable C-channel (chroma/saturation) transformation.
    hue:
        Enable H-channel (hue) transformation.
    chroma:
        Alias for saturation; merged with *saturation* flag internally.
    verbose:
        Emit DEBUG-level log messages.
    **kwargs:
        Extra options forwarded to sub-functions:
        ``tile_size``, ``gpu``, ``lut_size``, ``fast_hierar``, ``fast_spatial``.
    """
    setup_logging(verbose=verbose)

    input_path = Path(input_image)

    if not attractors:
        msg = "At least one attractor must be provided"
        raise ValueError(msg)

    # Parse attractors
    logger.debug(f"Parsing {len(attractors)} attractor(s)")
    parsed: list[tuple[str, float, float]] = []
    for attr_str in attractors:
        color, tolerance, strength = parse_attractor(attr_str)
        parsed.append((color, tolerance, strength))
        logger.debug(f"  {color!r}  tol={tolerance}  str={strength}")

    # Resolve output path
    output_path = Path(output_image) if output_image is not None else generate_output_path(input_path)

    # Initialise components
    engine = OKLCHEngine()
    processor = ImageProcessor(tile_size=kwargs.get("tile_size", 1024))
    transformer = ColorTransformer(engine)

    # Build Attractor objects
    attractor_objects = [engine.create_attractor(c, t, s) for c, t, s in parsed]
    for _att, (c, t, s) in zip(attractor_objects, parsed, strict=False):
        logger.info(f"Attractor: {c!r}  tolerance={t}  strength={s}")

    # Load image
    logger.info(f"Loading: {input_path}")
    image: np.ndarray = processor.load_image(input_path)

    # Transform
    fast_hierar: bool = kwargs.get("fast_hierar", False)
    fast_spatial: bool = kwargs.get("fast_spatial", True)

    transformed = process_with_optimizations(
        image,
        attractor_objects,
        luminance,
        saturation,
        chroma,
        fast_hierar,
        fast_spatial,
        transformer,
        engine,
    )

    # Save
    logger.info(f"Saving: {output_path}")
    processor.save_image(transformed, output_path)
    logger.info(f"Done → {output_path}")
