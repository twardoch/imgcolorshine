//! Image I/O operations
//!
//! This module provides utilities for loading and saving images in various formats.

use image::{DynamicImage, ImageBuffer, Rgb, GenericImageView};
use crate::error::{Result, ColorShineError};
use std::path::Path;

/// Load an image from a file path
/// 
/// Supports common image formats: JPEG, PNG, WebP, TIFF, etc.
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
    let img = image::open(path)
        .map_err(|e| ColorShineError::ImageIo(e))?;
    Ok(img)
}

/// Save an image to a file path
/// 
/// The output format is determined by the file extension.
pub fn save_image<P: AsRef<Path>>(img: &DynamicImage, path: P) -> Result<()> {
    img.save(path)
        .map_err(|e| ColorShineError::ImageIo(e))?;
    Ok(())
}

/// Convert DynamicImage to RGB f32 pixel data
/// 
/// Returns pixel data in row-major order as [r, g, b, r, g, b, ...]
/// with values normalized to [0.0, 1.0] range.
pub fn image_to_rgb_f32(img: &DynamicImage) -> Vec<f32> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
    
    for pixel in rgb_img.pixels() {
        rgb_data.push(pixel[0] as f32 / 255.0);
        rgb_data.push(pixel[1] as f32 / 255.0);
        rgb_data.push(pixel[2] as f32 / 255.0);
    }
    
    rgb_data
}

/// Convert RGB f32 pixel data back to DynamicImage
/// 
/// Takes pixel data in row-major order as [r, g, b, r, g, b, ...]
/// with values in [0.0, 1.0] range and creates an image.
pub fn rgb_f32_to_image(rgb_data: &[f32], width: u32, height: u32) -> Result<DynamicImage> {
    if rgb_data.len() != (width * height * 3) as usize {
        return Err(ColorShineError::computation_error(
            "RGB data length doesn't match image dimensions"
        ));
    }
    
    let mut img_buffer = ImageBuffer::new(width, height);
    
    for (i, pixel) in img_buffer.pixels_mut().enumerate() {
        let base_idx = i * 3;
        let r = (rgb_data[base_idx] * 255.0).clamp(0.0, 255.0) as u8;
        let g = (rgb_data[base_idx + 1] * 255.0).clamp(0.0, 255.0) as u8;
        let b = (rgb_data[base_idx + 2] * 255.0).clamp(0.0, 255.0) as u8;
        *pixel = Rgb([r, g, b]);
    }
    
    Ok(DynamicImage::ImageRgb8(img_buffer))
}

/// Process an image file through a transformation function
/// 
/// This is a convenience function that loads an image, applies a transformation,
/// and saves the result to a new file.
pub fn process_image_file<P1, P2, F>(
    input_path: P1,
    output_path: P2,
    transform_fn: F,
) -> Result<()>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    F: FnOnce(&[f32]) -> Result<Vec<f32>>,
{
    // Load the image
    let img = load_image(input_path)?;
    let (width, height) = img.dimensions();
    
    // Convert to RGB f32 data
    let rgb_data = image_to_rgb_f32(&img);
    
    // Apply transformation
    let transformed_data = transform_fn(&rgb_data)?;
    
    // Convert back to image and save
    let result_img = rgb_f32_to_image(&transformed_data, width, height)?;
    save_image(&result_img, output_path)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_rgb_conversion_round_trip() {
        // Create a simple test image
        let width = 2;
        let height = 2;
        let rgb_data = vec![
            1.0, 0.0, 0.0, // Red
            0.0, 1.0, 0.0, // Green
            0.0, 0.0, 1.0, // Blue
            0.5, 0.5, 0.5, // Gray
        ];
        
        // Convert to image and back
        let img = rgb_f32_to_image(&rgb_data, width, height).unwrap();
        let converted_data = image_to_rgb_f32(&img);
        
        // Should be very close to original (allowing for quantization)
        for (original, converted) in rgb_data.iter().zip(converted_data.iter()) {
            assert!((original - converted).abs() < 0.01);
        }
    }

    #[test]
    fn test_invalid_dimensions() {
        let rgb_data = vec![1.0, 0.0, 0.0]; // Only one pixel
        let result = rgb_f32_to_image(&rgb_data, 2, 2); // But claim 2x2
        assert!(result.is_err());
    }
}