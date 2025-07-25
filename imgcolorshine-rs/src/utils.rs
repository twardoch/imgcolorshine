//! Utility functions for imgcolorshine-rs
//!
//! This module provides various utility functions used throughout the library.

use crate::error::{Result, ColorShineError};
use std::f32::consts::PI;

/// Fast approximation of cosine using a polynomial
#[inline]
pub fn fast_cos(x: f32) -> f32 {
    // Use standard library for now, can optimize later with polynomial approximation
    x.cos()
}

/// Calculate raised cosine falloff
/// 
/// This function implements the falloff curve used in the Python version:
/// falloff = 0.5 * (cos(d_norm * π) + 1.0)
#[inline]
pub fn raised_cosine_falloff(normalized_distance: f32) -> f32 {
    debug_assert!(normalized_distance >= 0.0 && normalized_distance <= 1.0);
    0.5 * (fast_cos(normalized_distance * PI) + 1.0)
}

/// Clamp a value to a range
#[inline]
pub fn clamp<T: PartialOrd>(value: T, min_val: T, max_val: T) -> T {
    if value < min_val {
        min_val
    } else if value > max_val {
        max_val
    } else {
        value
    }
}

/// Validate tolerance parameter
pub fn validate_tolerance(tolerance: f32) -> Result<()> {
    if tolerance < crate::TOLERANCE_MIN || tolerance > crate::TOLERANCE_MAX {
        return Err(ColorShineError::invalid_parameter(
            "tolerance",
            tolerance,
            &format!("{} to {}", crate::TOLERANCE_MIN, crate::TOLERANCE_MAX),
        ));
    }
    Ok(())
}

/// Validate strength parameter
pub fn validate_strength(strength: f32) -> Result<()> {
    if strength < crate::STRENGTH_MIN || strength > crate::STRENGTH_MAX {
        return Err(ColorShineError::invalid_parameter(
            "strength",
            strength,
            &format!("{} to {}", crate::STRENGTH_MIN, crate::STRENGTH_MAX),
        ));
    }
    Ok(())
}

/// Calculate percentile of a sorted array
/// 
/// This function calculates the value at the given percentile in a sorted array.
/// Uses linear interpolation between adjacent values when needed.
pub fn percentile(sorted_data: &[f32], percentile: f32) -> f32 {
    debug_assert!(!sorted_data.is_empty());
    debug_assert!(percentile >= 0.0 && percentile <= 100.0);

    let n = sorted_data.len();
    if n == 1 {
        return sorted_data[0];
    }

    let index = (percentile / 100.0) * (n - 1) as f32;
    let lower_index = index.floor() as usize;
    let upper_index = (lower_index + 1).min(n - 1);

    if lower_index == upper_index {
        sorted_data[lower_index]
    } else {
        let weight = index - lower_index as f32;
        sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight
    }
}

/// Circular mean for hue angles (in degrees)
/// 
/// This function calculates the weighted circular mean of hue angles,
/// which is necessary for proper hue blending in OKLCH space.
pub fn circular_mean_degrees(angles: &[f32], weights: &[f32]) -> f32 {
    debug_assert_eq!(angles.len(), weights.len());
    debug_assert!(!angles.is_empty());

    if angles.len() == 1 {
        return angles[0];
    }

    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    let mut total_weight = 0.0;

    for (&angle, &weight) in angles.iter().zip(weights.iter()) {
        let radians = angle.to_radians();
        sum_sin += weight * radians.sin();
        sum_cos += weight * radians.cos();
        total_weight += weight;
    }

    if total_weight == 0.0 {
        return angles[0]; // fallback
    }

    sum_sin /= total_weight;
    sum_cos /= total_weight;

    let result_radians = sum_sin.atan2(sum_cos);
    let result_degrees = result_radians.to_degrees();

    // Normalize to 0-360 range
    if result_degrees < 0.0 {
        result_degrees + 360.0
    } else {
        result_degrees
    }
}

/// Convert RGB to linear RGB (gamma decoding)
#[inline]
pub fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear RGB to sRGB (gamma encoding)
#[inline]
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raised_cosine_falloff() {
        // Test boundary conditions
        assert!((raised_cosine_falloff(0.0) - 1.0).abs() < f32::EPSILON);
        assert!((raised_cosine_falloff(1.0) - 0.0).abs() < f32::EPSILON);
        
        // Test middle value
        let mid = raised_cosine_falloff(0.5);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert!((percentile(&data, 0.0) - 1.0).abs() < f32::EPSILON);
        assert!((percentile(&data, 50.0) - 3.0).abs() < f32::EPSILON);
        assert!((percentile(&data, 100.0) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_circular_mean_degrees() {
        // Test simple case
        let angles = vec![10.0, 20.0];
        let weights = vec![1.0, 1.0];
        let mean = circular_mean_degrees(&angles, &weights);
        assert!((mean - 15.0).abs() < 1.0);

        // Test wraparound case
        let angles = vec![350.0, 10.0];
        let weights = vec![1.0, 1.0];
        let mean = circular_mean_degrees(&angles, &weights);
        assert!(mean < 10.0 || mean > 350.0); // Should be near 0 degrees
    }

    #[test]
    fn test_gamma_conversion() {
        // Test round-trip conversion
        let original = 0.5;
        let linear = srgb_to_linear(original);
        let back = linear_to_srgb(linear);
        assert!((original - back).abs() < 1e-6);
    }
}