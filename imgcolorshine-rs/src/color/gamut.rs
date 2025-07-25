//! Gamut mapping implementation
//!
//! This module provides gamut mapping functionality to ensure that transformed
//! colors remain within the displayable sRGB color space.

use palette::{Srgb, Oklch, IntoColor};
use crate::error::Result;

/// Gamut mapper for ensuring colors stay within displayable ranges
/// 
/// The `GamutMapper` implements CSS Color Module 4 compliant gamut mapping
/// by reducing chroma while preserving lightness and hue when possible.
#[derive(Debug, Clone)]
pub struct GamutMapper {
    /// Tolerance for gamut boundary detection
    epsilon: f32,
    /// Maximum iterations for gamut mapping
    max_iterations: usize,
}

impl GamutMapper {
    /// Create a new gamut mapper with default settings
    pub fn new() -> Self {
        Self {
            epsilon: 0.001,
            max_iterations: 50,
        }
    }
    
    /// Create a gamut mapper with custom settings
    pub fn with_settings(epsilon: f32, max_iterations: usize) -> Self {
        Self {
            epsilon,
            max_iterations,
        }
    }
    
    /// Check if an OKLCH color is within the sRGB gamut
    /// 
    /// Returns true if the color can be accurately represented in sRGB
    pub fn is_in_gamut(&self, l: f32, c: f32, h: f32) -> bool {
        let oklch = Oklch::new(l, c, h);
        let srgb: Srgb = oklch.into_color();
        
        // Check if all RGB components are within [0, 1] range
        srgb.red >= 0.0 && srgb.red <= 1.0 &&
        srgb.green >= 0.0 && srgb.green <= 1.0 &&
        srgb.blue >= 0.0 && srgb.blue <= 1.0
    }
    
    /// Map an OKLCH color to the sRGB gamut
    /// 
    /// This function reduces chroma while preserving lightness and hue
    /// until the color fits within the sRGB gamut.
    pub fn map_to_gamut(&self, l: f32, c: f32, h: f32) -> (f32, f32, f32) {
        // If already in gamut, return unchanged
        if self.is_in_gamut(l, c, h) {
            return (l, c, h);
        }
        
        // Binary search to find maximum chroma that stays in gamut
        let mut low_chroma = 0.0;
        let mut high_chroma = c;
        let mut result_chroma = 0.0;
        
        for _ in 0..self.max_iterations {
            let mid_chroma = (low_chroma + high_chroma) / 2.0;
            
            if self.is_in_gamut(l, mid_chroma, h) {
                result_chroma = mid_chroma;
                low_chroma = mid_chroma;
            } else {
                high_chroma = mid_chroma;
            }
            
            if (high_chroma - low_chroma) < self.epsilon {
                break;
            }
        }
        
        (l, result_chroma, h)
    }
    
    /// Batch map OKLCH colors to sRGB gamut
    /// 
    /// Efficiently processes multiple colors at once
    pub fn batch_map_to_gamut(&self, oklch_data: &mut [f32]) {
        for oklch_chunk in oklch_data.chunks_exact_mut(3) {
            let (l, c, h) = self.map_to_gamut(oklch_chunk[0], oklch_chunk[1], oklch_chunk[2]);
            oklch_chunk[0] = l;
            oklch_chunk[1] = c;
            oklch_chunk[2] = h;
        }
    }
    
    /// Clamp RGB values to valid range as a fallback
    /// 
    /// This is a simpler but less accurate method that directly clamps
    /// RGB components to [0, 1] range without preserving perceptual properties.
    pub fn clamp_rgb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        (
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
        )
    }
    
    /// Get the maximum chroma for a given lightness and hue in sRGB gamut
    /// 
    /// This function finds the maximum chroma value that keeps the color
    /// within sRGB gamut for the specified lightness and hue.
    pub fn max_chroma_for_lh(&self, l: f32, h: f32) -> f32 {
        // Binary search for maximum chroma
        let mut low = 0.0;
        let mut high = 0.5; // Start with a reasonable upper bound
        
        // First, find an upper bound that's definitely out of gamut
        while self.is_in_gamut(l, high, h) && high < 2.0 {
            high *= 2.0;
        }
        
        // Now binary search for the exact boundary
        for _ in 0..self.max_iterations {
            let mid = (low + high) / 2.0;
            
            if self.is_in_gamut(l, mid, h) {
                low = mid;
            } else {
                high = mid;
            }
            
            if (high - low) < self.epsilon {
                break;
            }
        }
        
        low
    }
}

impl Default for GamutMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to apply gamut mapping to OKLCH color data
pub fn apply_gamut_mapping(oklch_data: &mut [f32]) {
    let mapper = GamutMapper::new();
    mapper.batch_map_to_gamut(oklch_data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamut_mapper_creation() {
        let mapper = GamutMapper::new();
        assert_eq!(mapper.epsilon, 0.001);
        assert_eq!(mapper.max_iterations, 50);
    }

    #[test]
    fn test_in_gamut_detection() {
        let mapper = GamutMapper::new();
        
        // Standard colors should be in gamut
        assert!(mapper.is_in_gamut(0.5, 0.1, 0.0)); // Moderate red-ish
        assert!(mapper.is_in_gamut(0.0, 0.0, 0.0)); // Black
        assert!(mapper.is_in_gamut(1.0, 0.0, 0.0)); // White
        
        // Extremely saturated colors might be out of gamut
        // (exact values depend on color space conversion precision)
    }

    #[test]
    fn test_gamut_mapping() {
        let mapper = GamutMapper::new();
        
        // Test mapping a potentially out-of-gamut color
        let (l, c, h) = (0.7, 0.8, 120.0); // Very saturated green
        let (mapped_l, mapped_c, mapped_h) = mapper.map_to_gamut(l, c, h);
        
        // Lightness and hue should be preserved
        assert!((mapped_l - l).abs() < 0.01);
        assert!((mapped_h - h).abs() < 0.01);
        
        // Chroma should be reduced or same
        assert!(mapped_c <= c);
        
        // Result should be in gamut
        assert!(mapper.is_in_gamut(mapped_l, mapped_c, mapped_h));
    }

    #[test]
    fn test_batch_gamut_mapping() {
        let mapper = GamutMapper::new();
        let mut oklch_data = vec![
            0.7, 0.8, 120.0, // Potentially out of gamut
            0.5, 0.1, 0.0,   // Likely in gamut
        ];
        
        mapper.batch_map_to_gamut(&mut oklch_data);
        
        // All colors should now be in gamut
        for oklch_chunk in oklch_data.chunks_exact(3) {
            assert!(mapper.is_in_gamut(oklch_chunk[0], oklch_chunk[1], oklch_chunk[2]));
        }
    }

    #[test]
    fn test_rgb_clamping() {
        let (r, g, b) = GamutMapper::clamp_rgb(-0.1, 1.5, 0.5);
        assert_eq!(r, 0.0);
        assert_eq!(g, 1.0);
        assert_eq!(b, 0.5);
    }

    #[test]
    fn test_max_chroma_calculation() {
        let mapper = GamutMapper::new();
        
        // Test maximum chroma for a mid-gray point at red hue
        let max_chroma = mapper.max_chroma_for_lh(0.5, 0.0);
        
        // Should be positive and reasonable
        assert!(max_chroma > 0.0);
        assert!(max_chroma < 1.0);
        
        // The color at max chroma should be in gamut
        assert!(mapper.is_in_gamut(0.5, max_chroma, 0.0));
        
        // Slightly higher chroma should be out of gamut
        assert!(!mapper.is_in_gamut(0.5, max_chroma + 0.01, 0.0));
    }
}