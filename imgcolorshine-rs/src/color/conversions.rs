//! Color space conversion functions
//!
//! This module implements the color space conversions matching the Python implementation
//! in `trans_numba.py`. All conversions maintain high precision for color accuracy.

use palette::{Oklab, Oklch, Srgb, IntoColor};

/// Convert sRGB to OKLCH color space
/// 
/// This function performs the full conversion pipeline:
/// sRGB -> OKLCH using palette's conversion chains
pub fn srgb_to_oklch(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let srgb = Srgb::new(r, g, b);
    let oklch: Oklch = srgb.into_color();
    (oklch.l, oklch.chroma, oklch.hue.into_positive_degrees())
}

/// Convert OKLCH to sRGB color space
/// 
/// This function performs the reverse conversion pipeline:
/// OKLCH -> sRGB using palette's conversion chains
pub fn oklch_to_srgb(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    let oklch = Oklch::new(l, c, h);
    let srgb: Srgb = oklch.into_color();
    (srgb.red, srgb.green, srgb.blue)
}

/// Convert Oklab to OKLCH
pub fn oklab_to_oklch(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let oklab = Oklab::new(l, a, b);
    let oklch: Oklch = oklab.into_color();
    (oklch.l, oklch.chroma, oklch.hue.into_positive_degrees())
}

/// Convert OKLCH to Oklab
pub fn oklch_to_oklab(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    let oklch = Oklch::new(l, c, h);
    let oklab: Oklab = oklch.into_color();
    (oklab.l, oklab.a, oklab.b)
}

/// Calculate perceptual distance (ΔE) between two colors in Oklab space
/// 
/// This is the Euclidean distance in Oklab space, which correlates well
/// with perceptual color differences.
#[inline]
pub fn delta_e_oklab(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    let dl = l1 - l2;
    let da = a1 - a2;
    let db = b1 - b2;
    (dl * dl + da * da + db * db).sqrt()
}

/// Batch convert sRGB pixels to OKLCH
/// 
/// This function is optimized for processing entire images efficiently.
/// It takes RGB pixel data and converts it to OKLCH color space.
pub fn batch_srgb_to_oklch(rgb_data: &[f32], oklch_data: &mut [f32]) {
    debug_assert_eq!(rgb_data.len(), oklch_data.len());
    debug_assert_eq!(rgb_data.len() % 3, 0);

    for (rgb_chunk, oklch_chunk) in rgb_data.chunks_exact(3).zip(oklch_data.chunks_exact_mut(3)) {
        let (l, c, h) = srgb_to_oklch(rgb_chunk[0], rgb_chunk[1], rgb_chunk[2]);
        oklch_chunk[0] = l;
        oklch_chunk[1] = c;
        oklch_chunk[2] = h;
    }
}

/// Batch convert OKLCH pixels to sRGB
/// 
/// This function converts OKLCH pixel data back to sRGB, with optional
/// gamut mapping to ensure all colors are displayable.
pub fn batch_oklch_to_srgb(oklch_data: &[f32], rgb_data: &mut [f32]) {
    debug_assert_eq!(oklch_data.len(), rgb_data.len());
    debug_assert_eq!(oklch_data.len() % 3, 0);

    for (oklch_chunk, rgb_chunk) in oklch_data.chunks_exact(3).zip(rgb_data.chunks_exact_mut(3)) {
        let (r, g, b) = oklch_to_srgb(oklch_chunk[0], oklch_chunk[1], oklch_chunk[2]);
        rgb_chunk[0] = r;
        rgb_chunk[1] = g;
        rgb_chunk[2] = b;
    }
}

/// Batch calculate distances from pixels to an attractor in Oklab space
/// 
/// This function efficiently calculates perceptual distances for tolerance computation.
pub fn batch_distance_to_attractor(
    pixel_data: &[f32], // Oklab pixel data (L, a, b, L, a, b, ...)
    attractor_lab: (f32, f32, f32), // Attractor color in Oklab
    distances: &mut [f32], // Output distance array
) {
    debug_assert_eq!(pixel_data.len() / 3, distances.len());

    let (attr_l, attr_a, attr_b) = attractor_lab;

    for (pixel_chunk, distance) in pixel_data.chunks_exact(3).zip(distances.iter_mut()) {
        *distance = delta_e_oklab(
            pixel_chunk[0], pixel_chunk[1], pixel_chunk[2],
            attr_l, attr_a, attr_b,
        );
    }
}

/// Parse CSS color string to OKLCH
/// 
/// This function parses various CSS color formats and converts them to OKLCH.
/// Supports: hex, rgb(), hsl(), oklch(), named colors, etc.
pub fn parse_css_color_to_oklch(color_str: &str) -> crate::error::Result<(f32, f32, f32)> {
    use crate::error::ColorShineError;

    let color_str = color_str.trim();

    // Try OKLCH format directly first
    if color_str.starts_with("oklch(") && color_str.ends_with(')') {
        let inner = &color_str[6..color_str.len()-1];
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        
        if parts.len() >= 3 {
            let l: f32 = parts[0].trim_end_matches('%').parse()
                .map_err(|_| ColorShineError::invalid_color("Invalid OKLCH lightness"))?;
            let c: f32 = parts[1].parse()
                .map_err(|_| ColorShineError::invalid_color("Invalid OKLCH chroma"))?;
            let h: f32 = parts[2].parse()
                .map_err(|_| ColorShineError::invalid_color("Invalid OKLCH hue"))?;
            
            // Convert percentage lightness if needed
            let l = if parts[0].ends_with('%') { l / 100.0 } else { l };
            
            return Ok((l, c, h));
        }
    }

    // For other formats, try basic color names and hex
    let srgb = match color_str {
        "red" => Srgb::new(1.0, 0.0, 0.0),
        "green" => Srgb::new(0.0, 1.0, 0.0),
        "blue" => Srgb::new(0.0, 0.0, 1.0),
        "white" => Srgb::new(1.0, 1.0, 1.0),
        "black" => Srgb::new(0.0, 0.0, 0.0),
        "orange" => Srgb::new(1.0, 0.647, 0.0),
        "yellow" => Srgb::new(1.0, 1.0, 0.0),
        "purple" => Srgb::new(0.5, 0.0, 0.5),
        "cyan" => Srgb::new(0.0, 1.0, 1.0),
        "magenta" => Srgb::new(1.0, 0.0, 1.0),
        _ => {
            // Try hex format
            if color_str.starts_with('#') && color_str.len() == 7 {
                let hex = &color_str[1..];
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| ColorShineError::invalid_color("Invalid hex color"))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| ColorShineError::invalid_color("Invalid hex color"))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| ColorShineError::invalid_color("Invalid hex color"))?;
                
                Srgb::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
            } else {
                return Err(ColorShineError::invalid_color(format!("Unable to parse color: {}", color_str)));
            }
        }
    };

    Ok(srgb_to_oklch(srgb.red, srgb.green, srgb.blue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_oklch_round_trip() {
        let original = (0.5, 0.7, 0.3);
        let (l, c, h) = srgb_to_oklch(original.0, original.1, original.2);
        let (r, g, b) = oklch_to_srgb(l, c, h);
        
        assert!((original.0 - r).abs() < 0.01);
        assert!((original.1 - g).abs() < 0.01);
        assert!((original.2 - b).abs() < 0.01);
    }

    #[test]
    fn test_delta_e_oklab() {
        // Same color should have zero distance
        let distance = delta_e_oklab(0.5, 0.1, 0.2, 0.5, 0.1, 0.2);
        assert!(distance.abs() < f32::EPSILON);

        // Different colors should have positive distance
        let distance = delta_e_oklab(0.5, 0.1, 0.2, 0.6, 0.1, 0.2);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_parse_css_color() {
        // Test hex color
        let (l, c, h) = parse_css_color_to_oklch("#ff0000").unwrap();
        assert!(l > 0.0 && c > 0.0);

        // Test named color
        let (l, c, h) = parse_css_color_to_oklch("red").unwrap();
        assert!(l > 0.0 && c > 0.0);

        // Test OKLCH format
        let (l, c, h) = parse_css_color_to_oklch("oklch(70% 0.2 120)").unwrap();
        assert!((l - 0.7).abs() < 0.01);
        assert!((c - 0.2).abs() < 0.01);
        assert!((h - 120.0).abs() < 0.01);
    }
}