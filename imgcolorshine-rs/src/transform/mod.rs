//! Image transformation module
//!
//! This module provides high-level interfaces for transforming entire images
//! using the color attractor system.

use crate::error::Result;
use crate::color::{ColorEngine, Attractor};
use crate::color::engine::TransformSettings;

/// High-level color transformer for processing images
/// 
/// The `ColorTransformer` provides a convenient interface for applying
/// color transformations to image data using multiple attractors.
#[derive(Debug)]
pub struct ColorTransformer {
    /// The underlying color engine
    engine: ColorEngine,
    /// Default transformation settings
    default_settings: TransformSettings,
}

impl ColorTransformer {
    /// Create a new color transformer
    pub fn new() -> Self {
        Self {
            engine: ColorEngine::new(),
            default_settings: TransformSettings::default(),
        }
    }
    
    /// Add an attractor to the transformer
    pub fn add_attractor(&mut self, attractor: Attractor) {
        self.engine.add_attractor(attractor);
    }
    
    /// Add an attractor from CSS color string with default parameters
    pub fn add_attractor_default(&mut self, color_str: &str) -> Result<()> {
        let attractor = Attractor::from_css_default(color_str)?;
        self.add_attractor(attractor);
        Ok(())
    }
    
    /// Add an attractor from CSS color string with custom parameters
    pub fn add_attractor_css(&mut self, color_str: &str, tolerance: f32, strength: f32) -> Result<()> {
        self.engine.add_attractor_css(color_str, tolerance, strength)
    }
    
    /// Set the default transformation settings
    pub fn set_default_settings(&mut self, settings: TransformSettings) {
        self.default_settings = settings;
    }
    
    /// Transform RGB pixel data using default settings
    pub fn transform_pixels(&mut self, rgb_data: &[f32]) -> Result<Vec<f32>> {
        self.engine.transform_pixels(rgb_data, &self.default_settings)
    }
    
    /// Transform RGB pixel data with custom settings
    pub fn transform_pixels_with_settings(
        &mut self,
        rgb_data: &[f32],
        settings: &TransformSettings,
    ) -> Result<Vec<f32>> {
        self.engine.transform_pixels(rgb_data, settings)
    }
    
    /// Get the number of attractors
    pub fn attractor_count(&self) -> usize {
        self.engine.attractor_count()
    }
    
    /// Clear all attractors
    pub fn clear_attractors(&mut self) {
        self.engine.clear_attractors();
    }
}

impl Default for ColorTransformer {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export important types (already imported above)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_transformer_creation() {
        let transformer = ColorTransformer::new();
        assert_eq!(transformer.attractor_count(), 0);
    }

    #[test]
    fn test_add_attractor_default() {
        let mut transformer = ColorTransformer::new();
        transformer.add_attractor_default("red").unwrap();
        assert_eq!(transformer.attractor_count(), 1);
    }
}