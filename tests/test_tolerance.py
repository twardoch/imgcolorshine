#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["pytest", "numpy", "coloraide", "loguru"]
# ///
# this_file: tests/test_tolerance.py

"""
Unit tests for tolerance calculation and color transformation logic.

These tests verify that the linear tolerance mapping works correctly
and produces expected results for known color distances.
"""

import numpy as np
import pytest

from imgcolorshine.color_engine import OKLCHEngine
from imgcolorshine.transforms import MAX_DELTA_E, calculate_weights


class TestToleranceCalculation:
    """Test suite for tolerance-based weight calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = OKLCHEngine()

    def test_max_delta_e_value(self):
        """Verify MAX_DELTA_E is set to expected value."""
        assert MAX_DELTA_E == 2.5

    def test_linear_tolerance_mapping(self):
        """Test that tolerance maps linearly to delta_e_max."""
        # Create a dummy pixel and attractor
        pixel_lab = np.array([0.5, 0.0, 0.0])
        attractors_lab = np.array([[0.5, 0.0, 0.0]])  # Same color

        # Test various tolerance values (skip 0 to avoid division by zero in the algorithm)
        for tolerance in [1, 25, 50, 75, 100]:
            tolerances = np.array([tolerance])
            strengths = np.array([100])

            weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)

            # For identical colors, weight should equal strength/100
            expected_weight = 1.0  # strength=100, distance=0
            assert np.isclose(weights[0], expected_weight, rtol=1e-5)

    def test_tolerance_radius_effect(self):
        """Test that tolerance correctly controls the radius of influence."""
        # Create colors with known distances
        pixel_lab = np.array([0.5, 0.0, 0.0])

        # Test different distances
        test_cases = [
            # (distance, tolerance, should_affect)
            (0.49, 20, True),  # 0.49 < 20 * 2.5 / 100 = 0.5
            (0.5, 21, True),  # 0.5 < 21 * 2.5 / 100 = 0.525
            (0.51, 20, False),  # 0.51 > 20 * 2.5 / 100 = 0.5
            (0.99, 40, True),  # 0.99 < 40 * 2.5 / 100 = 1.0
            (1.0, 41, True),  # 1.0 < 41 * 2.5 / 100 = 1.025
            (1.01, 40, False),  # 1.01 > 40 * 2.5 / 100 = 1.0
            (1.99, 80, True),  # 1.99 < 80 * 2.5 / 100 = 2.0
            (2.0, 81, True),  # 2.0 < 81 * 2.5 / 100 = 2.025
            (2.01, 80, False),  # 2.01 > 80 * 2.5 / 100 = 2.0
        ]

        for distance, tolerance, should_affect in test_cases:
            # Create attractor at specified distance
            attractors_lab = np.array([[0.5, distance, 0.0]])
            tolerances = np.array([tolerance])
            strengths = np.array([100])

            weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)

            if should_affect:
                assert weights[0] > 0, f"Expected non-zero weight for distance={distance}, tolerance={tolerance}"
            else:
                assert weights[0] == 0, f"Expected zero weight for distance={distance}, tolerance={tolerance}"

    def test_strength_scaling(self):
        """Test that strength correctly scales the weight."""
        pixel_lab = np.array([0.5, 0.0, 0.0])
        attractors_lab = np.array([[0.5, 0.0, 0.0]])  # Same color
        tolerances = np.array([100])

        # Test different strength values
        for strength in [0, 25, 50, 75, 100]:
            strengths = np.array([strength])
            weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)

            # For identical colors, weight should equal strength/100
            expected_weight = strength / 100.0
            assert np.isclose(weights[0], expected_weight, rtol=1e-5)

    def test_falloff_function(self):
        """Test the raised cosine falloff function behavior."""
        pixel_lab = np.array([0.5, 0.0, 0.0])
        tolerances = np.array([100])
        strengths = np.array([100])

        # Test falloff at different normalized distances
        test_distances = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected_falloffs = [
            1.0,
            0.8536,
            0.5,
            0.1464,
            0.0,
        ]  # Raised cosine values

        for d_norm, expected_falloff in zip(test_distances, expected_falloffs, strict=True):
            # Create attractor at distance that gives desired d_norm
            distance = d_norm * MAX_DELTA_E
            attractors_lab = np.array([[0.5, distance, 0.0]])

            weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)

            # Weight should be strength * falloff
            expected_weight = expected_falloff
            assert np.isclose(weights[0], expected_weight, rtol=1e-3)

    def test_multiple_attractors(self):
        """Test weight calculation with multiple attractors."""
        pixel_lab = np.array([0.5, 0.0, 0.0])

        # Create three attractors at different distances
        attractors_lab = np.array(
            [
                [0.5, 0.0, 0.0],  # Same color
                [0.5, 0.5, 0.0],  # Medium distance
                [0.5, 2.0, 0.0],  # Far distance
            ]
        )

        tolerances = np.array([100, 50, 30])
        strengths = np.array([100, 80, 60])

        weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)

        # First attractor: same color, should have full weight
        assert np.isclose(weights[0], 1.0, rtol=1e-5)

        # Second attractor: should have partial weight
        assert 0 < weights[1] < 0.8

        # Third attractor: outside tolerance, should have zero weight
        assert weights[2] == 0.0

    def test_known_color_pairs(self):
        """Test with real color pairs and known perceptual distances."""
        # Test cases with approximate known distances
        test_cases = [
            # (color1, color2, approx_distance, tolerance_needed)
            ("red", "darkred", 0.3, 15),
            ("red", "orange", 0.4, 20),
            ("red", "yellow", 0.8, 35),
            ("red", "green", 1.2, 50),
            ("red", "blue", 1.5, 65),
            ("white", "black", 1.0, 45),
            ("gray", "darkgray", 0.25, 12),
        ]

        for color1_str, color2_str, _, min_tolerance in test_cases:
            # Convert colors to Oklab
            color1 = self.engine.parse_color(color1_str).convert("oklab")
            color2 = self.engine.parse_color(color2_str).convert("oklab")

            pixel_lab = np.array([color1["lightness"], color1["a"], color1["b"]])
            attractors_lab = np.array([[color2["lightness"], color2["a"], color2["b"]]])

            # Test that min_tolerance allows influence
            tolerances = np.array([min_tolerance])
            strengths = np.array([100])

            weights = calculate_weights(pixel_lab, attractors_lab, tolerances, strengths)

            # Should have some influence at min_tolerance
            assert weights[0] > 0, (
                f"Expected {color1_str} to be influenced by {color2_str} at tolerance={min_tolerance}"
            )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        pixel_lab = np.array([0.5, 0.0, 0.0])
        attractors_lab = np.array([[0.5, 1.0, 0.0]])

        # Test tolerance = 0
        weights = calculate_weights(pixel_lab, attractors_lab, np.array([0]), np.array([100]))
        assert weights[0] == 0.0

        # Test strength = 0
        weights = calculate_weights(pixel_lab, attractors_lab, np.array([100]), np.array([0]))
        assert weights[0] == 0.0

        # Test very small but non-zero values
        weights = calculate_weights(pixel_lab, attractors_lab, np.array([0.1]), np.array([0.1]))
        assert weights[0] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
