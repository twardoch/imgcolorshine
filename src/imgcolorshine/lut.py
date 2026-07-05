# this_file: src/imgcolorshine/lut.py
"""3D lookup-table (LUT) acceleration for colour transformations.

``LUTManager`` pre-computes a 3-D LUT that maps every sRGB colour to the
colour that would result from a set of attractors and channel flags, then
uses tri-linear interpolation to apply that LUT to an image in a single
vectorised pass — much faster than running the full per-pixel pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from imgcolorshine.fast_mypyc.engine import ColorTransformer


class LUTManager:
    """Build, cache and apply a 3-D colour look-up table.

    Parameters
    ----------
    lut_size:
        Number of grid points per channel (e.g. 17, 33, 65).  Larger values
        give more accurate results at the cost of build time.
    """

    def __init__(self, lut_size: int = 33) -> None:
        self.lut_size = lut_size
        self._cache: dict[str, tuple[np.ndarray, RegularGridInterpolator]] = {}
        logger.debug(f"LUTManager initialised with lut_size={lut_size}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_lut(
        self,
        transformer: ColorTransformer,
        attractor_objects: list[Any],
        flags: dict[str, bool],
    ) -> tuple[np.ndarray, RegularGridInterpolator]:
        """Compute the 3-D LUT and return (lut_data, interpolator)."""
        n = self.lut_size
        axis = np.linspace(0.0, 1.0, n, dtype=np.float32)

        # Build a grid of sRGB triplets.
        r_grid, g_grid, b_grid = np.meshgrid(axis, axis, axis, indexing="ij")
        input_grid = np.stack([r_grid, g_grid, b_grid], axis=-1)  # (n, n, n, 3)
        flat_input = input_grid.reshape(-1, 3)

        # Process through the transformation pipeline in chunks.
        chunk_size = 4096
        flat_output = np.empty_like(flat_input)
        for start in range(0, len(flat_input), chunk_size):
            chunk = flat_input[start : start + chunk_size]
            tile = chunk.reshape(1, -1, 3)
            transformed_tile = transformer.transform_image(tile, attractor_objects, flags)
            flat_output[start : start + len(chunk)] = transformed_tile.reshape(-1, 3)

        lut_data = flat_output.reshape(n, n, n, 3)

        interpolator = RegularGridInterpolator(
            (axis, axis, axis),
            lut_data,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        logger.debug(f"Built {n}³ LUT ({lut_data.nbytes // 1024} KB)")
        return lut_data, interpolator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lut(
        self,
        transformer: ColorTransformer,
        attractor_objects: list[Any],
        flags: dict[str, bool],
    ) -> tuple[np.ndarray, RegularGridInterpolator]:
        """Return (lut_data, interpolator), building if necessary."""
        cache_key = repr((id(transformer), [(id(a),) for a in attractor_objects], sorted(flags.items())))
        if cache_key not in self._cache:
            self._cache[cache_key] = self._build_lut(transformer, attractor_objects, flags)
        return self._cache[cache_key]

    def apply_lut(self, image: np.ndarray, interpolator: RegularGridInterpolator) -> np.ndarray:
        """Apply a pre-built interpolator to *image* (float32, shape H×W×3, range [0,1])."""
        original_shape = image.shape
        flat = image.reshape(-1, 3)
        result: np.ndarray = interpolator(flat).astype(np.float32)
        return result.reshape(original_shape)
