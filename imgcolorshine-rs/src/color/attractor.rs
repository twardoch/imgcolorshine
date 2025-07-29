//! Color attractor implementation
//!
//! This module defines the `Attractor` type that represents a target color
//! that can influence surrounding colors in the image.

use crate::error::Result;
use crate::utils::{validate_tolerance, validate_strength};
use crate::color::conversions::{parse_css_color_to_oklch, oklch_to_oklab};

/// A color attractor that pulls nearby colors towards it
/// 
/// Attractors are defined by a target color, tolerance (influence radius),
/// and strength (intensity of the pull). They operate in OKLCH color space
/// for perceptually uniform transformations.
#[derive(Debug, Clone)]
pub struct Attractor {
    /// Target color in OKLCH space
    pub oklch: (f32, f32, f32), // (L, C, H)
    
    /// Target color in Oklab space (cached for performance)
    pub oklab: (f32, f32, f32), // (L, a, b)
    
    /// Tolerance: percentage of pixels to influence (0-100)
    pub tolerance: f32,
    
    /// Strength: intensity of transformation (0-200)
    pub strength: f32,
    
    /// Computed tolerance radius in Oklab space (calculated during processing)
    pub tolerance_radius: Option<f32>,
}

impl Attractor {
    /// Create a new attractor from OKLCH values
    /// 
    /// # Arguments
    /// * `oklch` - Target color in OKLCH space (lightness, chroma, hue)
    /// * `tolerance` - Percentage of pixels to influence (0-100)
    /// * `strength` - Intensity of transformation (0-200)
    pub fn new(oklch: (f32, f32, f32), tolerance: f32, strength: f32) -> Result<Self> {
        validate_tolerance(tolerance)?;
        validate_strength(strength)?;
        
        let oklab = oklch_to_oklab(oklch.0, oklch.1, oklch.2);
        
        Ok(Self {
            oklch,
            oklab,
            tolerance,
            strength,
            tolerance_radius: None,
        })
    }
    
    /// Create a new attractor from a CSS color string
    /// 
    /// # Arguments
    /// * `color_str` - CSS color string (hex, named color, oklch(), etc.)
    /// * `tolerance` - Percentage of pixels to influence (0-100)
    /// * `strength` - Intensity of transformation (0-200)
    pub fn from_css(color_str: &str, tolerance: f32, strength: f32) -> Result<Self> {
        let oklch = parse_css_color_to_oklch(color_str)?;
        Self::new(oklch, tolerance, strength)
    }
    
    /// Create a new attractor with default parameters
    /// 
    /// Uses default tolerance of 50% and strength of 75%
    pub fn from_css_default(color_str: &str) -> Result<Self> {
        Self::from_css(color_str, crate::DEFAULT_TOLERANCE, crate::DEFAULT_STRENGTH)
    }
    
    /// Set the computed tolerance radius
    /// 
    /// This is calculated during the first pass of the transformation algorithm
    /// based on the actual distribution of colors in the image.
    pub fn set_tolerance_radius(&mut self, radius: f32) {
        self.tolerance_radius = Some(radius);
    }
    
    /// Get the tolerance radius, panicking if not set
    pub fn tolerance_radius(&self) -> f32 {
        self.tolerance_radius.expect("Tolerance radius not computed yet")
    }
    
    /// Check if this attractor influences a pixel at the given Oklab distance
    pub fn influences_at_distance(&self, distance: f32) -> bool {
        match self.tolerance_radius {
            Some(radius) => distance <= radius,
            None => false, // Not computed yet
        }
    }
    
    /// Calculate the falloff weight for a pixel at the given distance
    /// 
    /// Returns a weight between 0.0 and 1.0 based on the attractor's
    /// strength and falloff curve.
    pub fn calculate_weight(&self, distance: f32) -> f32 {
        if !self.influences_at_distance(distance) {
            return 0.0;
        }
        
        let radius = self.tolerance_radius();
        let normalized_distance = distance / radius;
        
        // Calculate falloff using raised cosine
        let falloff = crate::utils::raised_cosine_falloff(normalized_distance);
        
        // Apply strength scaling
        if self.strength <= crate::STRENGTH_TRADITIONAL_MAX {
            // Traditional falloff mode (0-100)
            let strength_scaled = self.strength / 100.0;
            strength_scaled * falloff
        } else {
            // Extended intensity mode (101-200)
            let base_strength_factor = 1.0;
            let extra_strength_factor = (self.strength - crate::STRENGTH_TRADITIONAL_MAX) / 100.0;
            
            // Progressively flatten the falloff curve
            (base_strength_factor * falloff) + (extra_strength_factor * (1.0 - falloff))
        }
    }
}

impl std::fmt::Display for Attractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Attractor(OKLCH({:.2}, {:.2}, {:.1}°), tolerance={:.1}%, strength={:.1}%)",
            self.oklch.0, self.oklch.1, self.oklch.2,
            self.tolerance, self.strength
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attractor_creation() {
        let attractor = Attractor::new((0.7, 0.15, 30.0), 50.0, 75.0).unwrap();
        assert_eq!(attractor.oklch, (0.7, 0.15, 30.0));
        assert_eq!(attractor.tolerance, 50.0);
        assert_eq!(attractor.strength, 75.0);
    }

    #[test]
    fn test_attractor_from_css() {
        let attractor = Attractor::from_css("red", 60.0, 80.0).unwrap();
        assert_eq!(attractor.tolerance, 60.0);
        assert_eq!(attractor.strength, 80.0);
        // Red should have high chroma and hue around 0-30 degrees
        assert!(attractor.oklch.1 > 0.1); // High chroma
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid tolerance
        assert!(Attractor::new((0.7, 0.15, 30.0), -10.0, 75.0).is_err());
        assert!(Attractor::new((0.7, 0.15, 30.0), 150.0, 75.0).is_err());
        
        // Invalid strength
        assert!(Attractor::new((0.7, 0.15, 30.0), 50.0, -10.0).is_err());
        assert!(Attractor::new((0.7, 0.15, 30.0), 50.0, 250.0).is_err());
    }

    #[test]
    fn test_weight_calculation() {
        let mut attractor = Attractor::new((0.7, 0.15, 30.0), 50.0, 100.0).unwrap();
        attractor.set_tolerance_radius(0.1);
        
        // At distance 0, weight should be full strength
        let weight_zero = attractor.calculate_weight(0.0);
        assert!((weight_zero - 1.0).abs() < 0.01);
        
        // At tolerance radius, weight should be near zero
        let weight_edge = attractor.calculate_weight(0.1);
        assert!(weight_edge < 0.01);
        
        // Beyond tolerance radius, weight should be zero
        let weight_beyond = attractor.calculate_weight(0.2);
        assert_eq!(weight_beyond, 0.0);
    }

    #[test]
    fn test_extended_strength() {
        let mut attractor = Attractor::new((0.7, 0.15, 30.0), 50.0, 150.0).unwrap();
        attractor.set_tolerance_radius(0.1);
        
        // With extended strength, falloff should be flattened
        let weight_mid = attractor.calculate_weight(0.05);
        assert!(weight_mid > 0.5); // Should be higher than traditional falloff
    }
}