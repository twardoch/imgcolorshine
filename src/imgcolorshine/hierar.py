#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["numpy", "opencv-python", "loguru", "numba"]
# ///
# this_file: src/imgcolorshine/fast_hierar.py

"""
Hierarchical multi-resolution image processing for imgcolorshine.

Implements a coarse-to-fine processing strategy that reduces computation
by processing low-resolution versions first and only refining pixels that
differ significantly from the coarse approximation.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numba
import numpy as np
from loguru import logger


@dataclass
class PyramidLevel:
    """Represents one level in the image pyramid."""

    image: np.ndarray
    scale: float
    shape: tuple[int, int]
    level: int


class HierarchicalProcessor:
    """Multi-resolution image processing with adaptive refinement."""

    def __init__(
        self,
        min_size: int = 64,
        difference_threshold: float = 0.1,
        pyramid_factor: float = 0.5,
        use_adaptive_subdivision: bool = True,
        gradient_threshold: float = 0.05,
    ):
        """
        Initialize fast_hierar processor.

        Args:
            min_size: Minimum dimension for coarsest pyramid level
            difference_threshold: Threshold for refinement in perceptual units
            pyramid_factor: Downsampling factor between pyramid levels
            use_adaptive_subdivision: Enable gradient-based subdivision
            gradient_threshold: Threshold for detecting high-gradient regions
        """
        self.min_size = min_size
        self.difference_threshold = difference_threshold
        self.pyramid_factor = pyramid_factor
        self.use_adaptive_subdivision = use_adaptive_subdivision
        self.gradient_threshold = gradient_threshold
        self.pyramid_levels: list[PyramidLevel] = []

    def build_pyramid(self, image: np.ndarray) -> list[PyramidLevel]:
        """
        Build Gaussian pyramid from input image.

        Uses cv2.pyrDown for proper Gaussian filtering and downsampling.

        Args:
            image: Input image in RGB format

        Returns:
            List of pyramid levels from fine to coarse
        """
        levels = []
        current = image.copy()
        level = 0

        # Build pyramid until we reach minimum size
        while min(current.shape[:2]) > self.min_size:
            levels.append(
                PyramidLevel(
                    image=current.copy(),
                    scale=self.pyramid_factor**level,
                    shape=current.shape[:2],
                    level=level,
                )
            )

            # Downsample for next level
            current = cv2.pyrDown(current)
            level += 1

        # Add final coarsest level
        levels.append(
            PyramidLevel(
                image=current,
                scale=self.pyramid_factor**level,
                shape=current.shape[:2],
                level=level,
            )
        )

        logger.debug(f"Built pyramid with {len(levels)} levels: {[l.shape for l in levels]}")
        return levels

    @staticmethod
    @numba.njit
    def _compute_perceptual_distance(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """
        Compute perceptual distance between two Lab images.

        Args:
            lab1: First image in Lab space (H, W, 3)
            lab2: Second image in Lab space (H, W, 3)

        Returns:
            Distance map (H, W)
        """
        # Simple Euclidean distance in Lab space
        # For better accuracy, could use CIEDE2000 but it's more complex
        diff = lab1 - lab2
        return np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)

    def compute_difference_mask(
        self, fine_level: np.ndarray, coarse_upsampled: np.ndarray, threshold: float
    ) -> np.ndarray:
        """
        Create mask of pixels that need refinement.

        Compares fine level with upsampled coarse result to identify
        pixels that differ significantly and need reprocessing.

        Args:
            fine_level: Fine resolution image (RGB)
            coarse_upsampled: Upsampled coarse result (RGB)
            threshold: Perceptual distance threshold

        Returns:
            Boolean mask where True indicates pixels needing refinement
        """
        # For now, use simple RGB distance
        # In production, should convert to Lab/OKLCH for perceptual accuracy
        diff = np.abs(fine_level.astype(np.float32) - coarse_upsampled.astype(np.float32))
        distance = np.sqrt(np.sum(diff**2, axis=2))

        # Normalize by max possible distance (sqrt(3) for RGB)
        normalized_distance = distance / (255.0 * np.sqrt(3))

        # Create binary mask
        return normalized_distance > threshold

    def detect_gradient_regions(self, image: np.ndarray, gradient_threshold: float) -> np.ndarray:
        """
        Detect regions with high color gradients.

        Uses Sobel edge detection to find areas with rapid color changes
        that benefit from fine-resolution processing.

        Args:
            image: Input image (RGB)
            gradient_threshold: Threshold for gradient magnitude

        Returns:
            Boolean mask of high-gradient regions
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()

        # Create mask
        gradient_mask = gradient_magnitude > gradient_threshold

        # Dilate to ensure coverage of edge regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient_mask = cv2.dilate(gradient_mask.astype(np.uint8), kernel)

        return gradient_mask.astype(bool)

    def process_hierarchical(
        self,
        image: np.ndarray,
        transform_func: Callable,
        attractors: np.ndarray,
        tolerances: np.ndarray,
        strengths: np.ndarray,
        channels: list[bool],
    ) -> np.ndarray:
        """
        Process image hierarchically from coarse to fine resolution.

        Main algorithm that implements the multi-resolution processing strategy.
        Starts with coarsest level and progressively refines the result.

        Args:
            image: Input image (RGB format, 0-255 range)
            transform_func: Function to transform pixels
            attractors: Attractor colors in Lab space
            tolerances: Tolerance values for each attractor
            strengths: Strength values for each attractor
            channels: Boolean flags for L, C, H channels

        Returns:
            Transformed image in RGB format
        """
        # Build image pyramid
        pyramid = self.build_pyramid(image)

        if len(pyramid) == 1:
            # Image too small for pyramid, process directly
            logger.debug("Image too small for pyramid, processing directly")
            return transform_func(image, attractors, tolerances, strengths, channels)

        # Process coarsest level completely
        coarsest = pyramid[-1]
        logger.info(f"Processing coarsest level: {coarsest.shape}")

        # Transform the coarsest level
        result = transform_func(coarsest.image, attractors, tolerances, strengths, channels)

        # Statistics tracking
        total_pixels_refined = 0
        total_pixels = 0

        # Process from coarse to fine
        for i in range(len(pyramid) - 2, -1, -1):
            level = pyramid[i]
            h, w = level.shape[:2]
            total_pixels += h * w

            logger.debug(f"Processing pyramid level {i}: {level.shape}")

            # Upsample previous result to current resolution
            upsampled = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

            # Compute refinement mask
            diff_mask = self.compute_difference_mask(level.image, upsampled, self.difference_threshold)

            # Add gradient regions if enabled
            if self.use_adaptive_subdivision:
                gradient_mask = self.detect_gradient_regions(level.image, self.gradient_threshold)
                refinement_mask = diff_mask | gradient_mask
            else:
                refinement_mask = diff_mask

            # Count refined pixels
            num_refined = np.sum(refinement_mask)
            total_pixels_refined += num_refined

            # Process only masked pixels if any need refinement
            if num_refined > 0:
                logger.debug(f"Refining {num_refined} pixels ({num_refined / refinement_mask.size * 100:.1f}%)")

                # For efficient processing, we need to handle masked transformation
                # This is a simplified approach - in production, we'd optimize this
                refined_result = upsampled.copy()

                # Transform the entire level (optimization opportunity here)
                transformed_level = transform_func(level.image, attractors, tolerances, strengths, channels)

                # Apply only to masked pixels
                refined_result[refinement_mask] = transformed_level[refinement_mask]

                result = refined_result
            else:
                # No refinement needed, use upsampled result
                logger.debug("No refinement needed at this level")
                result = upsampled

        # Log statistics
        if total_pixels > 0:
            refinement_ratio = total_pixels_refined / total_pixels
            logger.info(f"Hierarchical processing complete: refined {refinement_ratio * 100:.1f}% of pixels")

        return result

    def process_hierarchical_tiled(
        self,
        image: np.ndarray,
        transform_func: Callable,
        attractors: np.ndarray,
        tolerances: np.ndarray,
        strengths: np.ndarray,
        channels: list[bool],
        tile_size: int = 512,
    ) -> np.ndarray:
        """
        Hierarchical processing with tiling for very large images.

        Combines fast_hierar processing with tile-based memory management
        for processing images that don't fit in memory.

        Args:
            image: Input image
            transform_func: Transformation function
            attractors: Attractor colors
            tolerances: Tolerance values
            strengths: Strength values
            channels: Channel flags
            tile_size: Size of tiles for processing

        Returns:
            Transformed image
        """
        h, w = image.shape[:2]

        # If image is small enough, process without tiling
        if h <= tile_size * 2 and w <= tile_size * 2:
            return self.process_hierarchical(image, transform_func, attractors, tolerances, strengths, channels)

        logger.info(f"Processing large image ({h}x{w}) with tiled fast_hierar approach")

        # Create output array
        result = np.zeros_like(image)

        # Process in tiles with overlap
        overlap = tile_size // 4  # 25% overlap

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Calculate tile boundaries with padding
                y1 = max(0, y - overlap // 2)
                y2 = min(h, y + tile_size + overlap // 2)
                x1 = max(0, x - overlap // 2)
                x2 = min(w, x + tile_size + overlap // 2)

                logger.debug(f"Processing tile [{y1}:{y2}, {x1}:{x2}]")

                # Extract tile
                tile = image[y1:y2, x1:x2]

                # Process tile hierarchically
                processed_tile = self.process_hierarchical(
                    tile, transform_func, attractors, tolerances, strengths, channels
                )

                # Blend into result (simple approach - could use feathering)
                # Take center region without overlap
                ty1 = overlap // 2 if y > 0 else 0
                ty2 = processed_tile.shape[0] - (overlap // 2 if y2 < h else 0)
                tx1 = overlap // 2 if x > 0 else 0
                tx2 = processed_tile.shape[1] - (overlap // 2 if x2 < w else 0)

                ry1 = y1 + ty1
                ry2 = y1 + ty2
                rx1 = x1 + tx1
                rx2 = x1 + tx2

                result[ry1:ry2, rx1:rx2] = processed_tile[ty1:ty2, tx1:tx2]

        return result
