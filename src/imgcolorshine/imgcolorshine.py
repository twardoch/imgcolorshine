#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["loguru", "numpy"]
# ///
# this_file: src/imgcolorshine/imgcolorshine.py

"""
Core processing logic for imgcolorshine.

Contains the main image transformation pipeline.
"""

import sys
from pathlib import Path

from loguru import logger

from imgcolorshine import ColorTransformer, ImageProcessor, OKLCHEngine


def setup_logging(verbose: bool = False):
    """Configure loguru logging based on verbosity."""
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")
    else:
        logger.add(sys.stderr, level="INFO", format="{message}")


def parse_attractor(attractor_str: str) -> tuple[str, float, float]:
    """Parse attractor string format: 'color;tolerance;strength'."""
    try:
        parts = attractor_str.split(";")
        if len(parts) != 3:
            msg = f"Invalid attractor format: {attractor_str}"
            raise ValueError(msg)

        color = parts[0].strip()
        tolerance = float(parts[1])
        strength = float(parts[2])

        if not 0 <= tolerance <= 100:
            msg = f"Tolerance must be 0-100, got {tolerance}"
            raise ValueError(msg)
        if not 0 <= strength <= 100:
            msg = f"Strength must be 0-100, got {strength}"
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


def process_image(
    input_image: str,
    attractors: tuple[str, ...],
    output_image: str | None = None,
    luminance: bool = True,
    saturation: bool = True,
    hue: bool = True,
    verbose: bool = False,
    tile_size: int = 1024,
    gpu: bool = True,
    lut_size: int = 0,
) -> None:
    """
    Process an image with color attractors.

    Main processing pipeline that handles logging setup, attractor parsing,
    image loading, transformation, and saving.

    Used in:
    - src/imgcolorshine/cli.py
    """
    setup_logging(verbose)

    # Convert to Path
    input_path = Path(input_image)

    # Validate inputs
    if not attractors:
        msg = "At least one attractor must be provided"
        raise ValueError(msg)

    if not any([luminance, saturation, hue]):
        msg = "At least one channel (luminance, saturation, hue) must be enabled"
        raise ValueError(msg)

    # Parse attractors
    logger.debug(f"Parsing {len(attractors)} attractors")
    parsed_attractors = []
    for attr_str in attractors:
        color, tolerance, strength = parse_attractor(attr_str)
        parsed_attractors.append((color, tolerance, strength))
        logger.debug(f"Attractor: color={color}, tolerance={tolerance}, strength={strength}")

    # Set output path
    if output_image is None:
        output_path = generate_output_path(input_path)
        logger.info(f"Output path auto-generated: {output_path}")
    else:
        output_path = Path(output_image)

    # Initialize components
    engine = OKLCHEngine()
    processor = ImageProcessor(tile_size=tile_size)
    transformer = ColorTransformer(engine)

    # Create attractor objects
    attractor_objects = []
    for color_str, tolerance, strength in parsed_attractors:
        attractor = engine.create_attractor(color_str, tolerance, strength)
        attractor_objects.append(attractor)
        logger.info(f"Created attractor: {color_str} (tolerance={tolerance}, strength={strength})")

    # Load image
    logger.info(f"Loading image: {input_path}")
    image = processor.load_image(input_path)

    # Try LUT transformation first if enabled
    transformed = None
    if lut_size > 0:
        try:
            logger.info(f"Building {lut_size}³ color LUT...")
            import numpy as np

            from imgcolorshine.fused_kernels import transform_pixel_fused
            from imgcolorshine.lut import ColorLUT

            # Create LUT
            lut = ColorLUT(size=lut_size if lut_size > 0 else 65)

            # Prepare attractor data
            attractors_lab = np.array([a.oklab_values for a in attractor_objects])
            tolerances = np.array([a.tolerance for a in attractor_objects])
            strengths = np.array([a.strength for a in attractor_objects])
            channels = [luminance, saturation, hue]

            # Build LUT using fused kernel
            def transform_func(rgb, attr_lab, tol, str_vals, l, s, h):
                return np.array(transform_pixel_fused(rgb[0], rgb[1], rgb[2], attr_lab, tol, str_vals, l, s, h))

            lut.build_lut(transform_func, attractors_lab, tolerances, strengths, channels)

            # Apply LUT
            logger.info("Applying LUT transformation...")
            transformed = lut.apply_lut(image)
            logger.info("LUT processing successful")

        except Exception as e:
            logger.warning(f"LUT processing failed: {e}, falling back to standard processing")
            transformed = None

    # Try GPU transformation if LUT failed or disabled
    if transformed is None and gpu:
        try:
            from imgcolorshine.gpu_backend import GPU_AVAILABLE

            if GPU_AVAILABLE:
                logger.info("Attempting GPU acceleration...")
                import numpy as np

                from imgcolorshine.gpu_transforms import process_image_gpu

                # Prepare attractor data
                attractors_lab = np.array([a.oklab_values for a in attractor_objects])
                tolerances = np.array([a.tolerance for a in attractor_objects])
                strengths = np.array([a.strength for a in attractor_objects])

                # Process on GPU
                transformed = process_image_gpu(
                    image, attractors_lab, tolerances, strengths, luminance, saturation, hue
                )

                if transformed is not None:
                    logger.info("GPU processing successful")
                else:
                    logger.warning("GPU processing failed, falling back to CPU")
                    gpu = False
            else:
                logger.debug("GPU not available, using CPU")
                gpu = False
        except ImportError:
            logger.debug("GPU libraries not installed, using CPU")
            gpu = False

    # CPU processing (fallback or if GPU disabled)
    if not gpu or transformed is None:
        logger.info("Transforming colors on CPU...")
        flags = {"luminance": luminance, "saturation": saturation, "hue": hue}
        transformed = transformer.transform_image(image, attractor_objects, flags)

    # Save image
    logger.info(f"Saving image: {output_path}")
    processor.save_image(transformed, output_path)

    logger.info(f"Processing complete: {input_path} → {output_path}")
