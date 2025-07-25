//! Color engine for managing attractors and orchestrating transformations
//!
//! This module provides the main `ColorEngine` that manages multiple attractors
//! and coordinates the color transformation process.

use std::collections::HashMap;
use crate::error::{Result, ColorShineError};
use crate::color::attractor::Attractor;
use crate::color::conversions::{
    batch_srgb_to_oklch, batch_oklch_to_srgb, batch_distance_to_attractor,
    srgb_to_oklch, oklch_to_srgb, oklch_to_oklab
};
use crate::utils::{percentile, circular_mean_degrees};

/// Represents different color spaces used in the transformation pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Standard RGB color space
    Srgb,
    /// OKLCH cylindrical color space (perceptually uniform)
    Oklch,
    /// Oklab rectangular color space (perceptually uniform)
    Oklab,
}

/// Transformation settings for controlling which color channels are affected
#[derive(Debug, Clone)]
pub struct TransformSettings {
    /// Whether to transform the lightness (L) channel
    pub transform_lightness: bool,
    /// Whether to transform the chroma/saturation (C) channel  
    pub transform_chroma: bool,
    /// Whether to transform the hue (H) channel
    pub transform_hue: bool,
}

impl Default for TransformSettings {
    fn default() -> Self {
        Self {
            transform_lightness: true,
            transform_chroma: true,
            transform_hue: true,
        }
    }
}

/// Main color engine for orchestrating color transformations
/// 
/// The `ColorEngine` manages a collection of attractors and provides
/// methods for applying them to image data using the two-pass algorithm.
#[derive(Debug)]
pub struct ColorEngine {
    /// Collection of attractors to apply
    attractors: Vec<Attractor>,
    /// Cached color conversion data
    conversion_cache: HashMap<String, (f32, f32, f32)>,
}

impl ColorEngine {
    /// Create a new empty color engine
    pub fn new() -> Self {
        Self {
            attractors: Vec::new(),
            conversion_cache: HashMap::new(),
        }
    }
    
    /// Add an attractor to the engine
    pub fn add_attractor(&mut self, attractor: Attractor) {
        self.attractors.push(attractor);
    }
    
    /// Add an attractor from CSS color string
    pub fn add_attractor_css(&mut self, color_str: &str, tolerance: f32, strength: f32) -> Result<()> {
        let attractor = Attractor::from_css(color_str, tolerance, strength)?;
        self.add_attractor(attractor);
        Ok(())
    }
    
    /// Get the number of attractors
    pub fn attractor_count(&self) -> usize {
        self.attractors.len()
    }
    
    /// Clear all attractors
    pub fn clear_attractors(&mut self) {
        self.attractors.clear();
    }
    
    /// Transform RGB pixel data using the two-pass algorithm
    /// 
    /// This implements the core algorithm from the Python version:
    /// 1. Pass 1: Calculate tolerance radii based on pixel distribution
    /// 2. Pass 2: Apply transformations with computed weights
    pub fn transform_pixels(
        &mut self,
        rgb_data: &[f32],
        settings: &TransformSettings,
    ) -> Result<Vec<f32>> {
        if rgb_data.len() % 3 != 0 {
            return Err(ColorShineError::computation_error("RGB data length must be divisible by 3"));
        }
        
        if self.attractors.is_empty() {
            return Ok(rgb_data.to_vec()); // No attractors, return unchanged
        }
        
        let pixel_count = rgb_data.len() / 3;
        
        // Convert RGB to OKLCH and Oklab for processing
        let mut oklch_data = vec![0.0; rgb_data.len()];
        let mut oklab_data = vec![0.0; rgb_data.len()];
        
        batch_srgb_to_oklch(rgb_data, &mut oklch_data);
        
        // Convert OKLCH to Oklab for distance calculations
        for (oklch_chunk, oklab_chunk) in oklch_data.chunks_exact(3).zip(oklab_data.chunks_exact_mut(3)) {
            let (l, a, b) = oklch_to_oklab(oklch_chunk[0], oklch_chunk[1], oklch_chunk[2]);
            oklab_chunk[0] = l;
            oklab_chunk[1] = a;
            oklab_chunk[2] = b;
        }
        
        // Pass 1: Calculate tolerance radii for each attractor
        self.calculate_tolerance_radii(&oklab_data)?;
        
        // Pass 2: Transform pixels
        self.apply_transformations(&mut oklch_data, settings)?;
        
        // Convert back to RGB
        let mut result_rgb = vec![0.0; rgb_data.len()];
        batch_oklch_to_srgb(&oklch_data, &mut result_rgb);
        
        Ok(result_rgb)
    }
    
    /// Pass 1: Calculate tolerance radii based on pixel distribution
    fn calculate_tolerance_radii(&mut self, oklab_data: &[f32]) -> Result<()> {
        let pixel_count = oklab_data.len() / 3;
        
        for attractor in &mut self.attractors {
            // Calculate distances from all pixels to this attractor
            let mut distances = vec![0.0; pixel_count];
            batch_distance_to_attractor(oklab_data, attractor.oklab, &mut distances);
            
            // Sort distances for percentile calculation
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Calculate tolerance radius at the specified percentile
            let radius = percentile(&distances, attractor.tolerance);
            attractor.set_tolerance_radius(radius);
        }
        
        Ok(())
    }
    
    /// Pass 2: Apply transformations with computed weights
    fn apply_transformations(
        &self,
        oklch_data: &mut [f32],
        settings: &TransformSettings,
    ) -> Result<()> {
        let pixel_count = oklch_data.len() / 3;
        
        // Pre-calculate distances for all attractors to all pixels
        let mut all_distances = Vec::new();
        for attractor in &self.attractors {
            let mut distances = vec![0.0; pixel_count];
            
            // Calculate distances in Oklab space
            for (i, oklch_chunk) in oklch_data.chunks_exact(3).enumerate() {
                let (l, a, b) = oklch_to_oklab(oklch_chunk[0], oklch_chunk[1], oklch_chunk[2]);
                let (attr_l, attr_a, attr_b) = attractor.oklab;
                
                let dl = l - attr_l;
                let da = a - attr_a;
                let db = b - attr_b;
                distances[i] = (dl * dl + da * da + db * db).sqrt();
            }
            
            all_distances.push(distances);
        }
        
        // Transform each pixel
        for (pixel_idx, oklch_chunk) in oklch_data.chunks_exact_mut(3).enumerate() {
            let original_l = oklch_chunk[0];
            let original_c = oklch_chunk[1];
            let original_h = oklch_chunk[2];
            
            // Collect weights and attractor values for blending
            let mut total_weight = 0.0;
            let mut attractor_contributions = Vec::new();
            
            for (attractor_idx, attractor) in self.attractors.iter().enumerate() {
                let distance = all_distances[attractor_idx][pixel_idx];
                let weight = attractor.calculate_weight(distance);
                
                if weight > 0.0 {
                    total_weight += weight;
                    attractor_contributions.push((weight, attractor.oklch));
                }
            }
            
            // If no attractors influence this pixel, leave it unchanged
            if total_weight == 0.0 {
                continue;
            }
            
            // Calculate source weight (remaining influence of original color)
            let source_weight = (1.0 - total_weight).max(0.0);
            
            // Blend colors channel by channel
            let mut new_l = original_l;
            let mut new_c = original_c;
            let mut new_h = original_h;
            
            if settings.transform_lightness {
                new_l = source_weight * original_l;
                for (weight, (attr_l, _, _)) in &attractor_contributions {
                    new_l += weight * attr_l;
                }
            }
            
            if settings.transform_chroma {
                new_c = source_weight * original_c;
                for (weight, (_, attr_c, _)) in &attractor_contributions {
                    new_c += weight * attr_c;
                }
            }
            
            if settings.transform_hue {
                // Hue requires circular mean
                let mut hues = vec![original_h];
                let mut weights = vec![source_weight];
                
                for (weight, (_, _, attr_h)) in &attractor_contributions {
                    hues.push(*attr_h);
                    weights.push(*weight);
                }
                
                new_h = circular_mean_degrees(&hues, &weights);
            }
            
            // Update the pixel
            oklch_chunk[0] = new_l;
            oklch_chunk[1] = new_c;
            oklch_chunk[2] = new_h;
        }
        
        Ok(())
    }
    
    /// Transform a single RGB pixel (utility function for testing)
    pub fn transform_pixel(
        &mut self,
        r: f32,
        g: f32,
        b: f32,
        settings: &TransformSettings,
    ) -> Result<(f32, f32, f32)> {
        let rgb_data = vec![r, g, b];
        let result = self.transform_pixels(&rgb_data, settings)?;
        Ok((result[0], result[1], result[2]))
    }
}

impl Default for ColorEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_engine_creation() {
        let engine = ColorEngine::new();
        assert_eq!(engine.attractor_count(), 0);
    }

    #[test]
    fn test_add_attractor() {
        let mut engine = ColorEngine::new();
        let attractor = Attractor::from_css("red", 50.0, 75.0).unwrap();
        engine.add_attractor(attractor);
        assert_eq!(engine.attractor_count(), 1);
    }

    #[test]
    fn test_add_attractor_css() {
        let mut engine = ColorEngine::new();
        engine.add_attractor_css("blue", 60.0, 80.0).unwrap();
        assert_eq!(engine.attractor_count(), 1);
    }

    #[test]
    fn test_transform_no_attractors() {
        let mut engine = ColorEngine::new();
        let rgb_data = vec![0.5, 0.6, 0.7];
        let settings = TransformSettings::default();
        
        let result = engine.transform_pixels(&rgb_data, &settings).unwrap();
        assert_eq!(result, rgb_data); // Should be unchanged
    }

    #[test]
    fn test_transform_with_attractor() {
        let mut engine = ColorEngine::new();
        engine.add_attractor_css("red", 100.0, 100.0).unwrap(); // Influence all pixels fully
        
        let rgb_data = vec![0.0, 0.0, 1.0]; // Blue pixel
        let settings = TransformSettings::default();
        
        let result = engine.transform_pixels(&rgb_data, &settings).unwrap();
        
        // The blue pixel should be pulled towards red
        assert!(result[0] > rgb_data[0]); // More red
        assert!(result[2] < rgb_data[2]); // Less blue
    }

    #[test]
    fn test_channel_selective_transformation() {
        let mut engine = ColorEngine::new();
        engine.add_attractor_css("red", 100.0, 100.0).unwrap();
        
        let rgb_data = vec![0.0, 0.0, 1.0]; // Blue pixel
        let settings = TransformSettings {
            transform_lightness: false,
            transform_chroma: false,
            transform_hue: true,
        };
        
        let result = engine.transform_pixels(&rgb_data, &settings).unwrap();
        
        // Only hue should change, lightness and chroma preserved
        let original_oklch = srgb_to_oklch(rgb_data[0], rgb_data[1], rgb_data[2]);
        let result_oklch = srgb_to_oklch(result[0], result[1], result[2]);
        
        assert!((original_oklch.0 - result_oklch.0).abs() < 0.1); // Lightness similar
        assert!((original_oklch.1 - result_oklch.1).abs() < 0.1); // Chroma similar
        // Hue should be different (hard to test exactly due to color space conversions)
    }
}