//! Command-line interface for imgcolorshine-rs
//!
//! This module provides the CLI interface that matches the Python version's
//! command-line arguments and behavior.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use crate::error::{Result, ColorShineError};

/// Command-line arguments for imgcolorshine
#[derive(Parser, Debug)]
#[command(
    name = "imgcolorshine",
    version,
    about = "Transform image colors using OKLCH color attractors",
    long_about = "Transform image colors using physics-inspired attractors in perceptually uniform OKLCH color space"
)]
pub struct Args {
    /// Input image file
    #[arg(value_name = "INPUT_IMAGE", help = "Path to the input image file")]
    pub input_image: PathBuf,
    
    /// Color attractors in format "color;tolerance;strength"
    #[arg(
        value_name = "ATTRACTOR",
        help = "Color attractors in format \"color;tolerance;strength\""
    )]
    pub attractors: Vec<String>,
    
    /// Output image path
    #[arg(
        long,
        value_name = "PATH",
        help = "Output image path (auto-generated if not specified)"
    )]
    pub output_image: Option<PathBuf>,
    
    /// Transform lightness channel
    #[arg(long, default_value = "true", help = "Enable/disable lightness transformation")]
    pub luminance: bool,
    
    /// Transform chroma/saturation channel
    #[arg(long, default_value = "true", help = "Enable/disable chroma transformation")]
    pub saturation: bool,
    
    /// Transform hue channel
    #[arg(long, default_value = "true", help = "Enable/disable hue transformation")]
    pub hue: bool,
    
    /// Enable verbose logging
    #[arg(short, long, help = "Enable verbose logging")]
    pub verbose: bool,
    
    /// Number of threads to use (0 = auto)
    #[arg(long, default_value = "0", help = "Number of threads to use (0 = auto)")]
    pub threads: usize,
}

/// Parse a single attractor string in format "color;tolerance;strength"
pub fn parse_attractor_string(attractor_str: &str) -> Result<(String, f32, f32)> {
    let parts: Vec<&str> = attractor_str.split(';').collect();
    
    if parts.len() != 3 {
        return Err(ColorShineError::invalid_parameter(
            "attractor format",
            attractor_str,
            "color;tolerance;strength"
        ));
    }
    
    let color = parts[0].trim().to_string();
    
    let tolerance: f32 = parts[1].trim().parse()
        .map_err(|_| ColorShineError::invalid_parameter(
            "tolerance", parts[1], "number between 0 and 100"
        ))?;
    
    let strength: f32 = parts[2].trim().parse()
        .map_err(|_| ColorShineError::invalid_parameter(
            "strength", parts[2], "number between 0 and 200"
        ))?;
    
    Ok((color, tolerance, strength))
}

/// Generate output filename if not specified
pub fn generate_output_filename(input_path: &PathBuf) -> PathBuf {
    let mut output_path = input_path.clone();
    
    if let Some(stem) = input_path.file_stem() {
        if let Some(extension) = input_path.extension() {
            let new_filename = format!("{}_colorshine.{}", 
                stem.to_string_lossy(), 
                extension.to_string_lossy()
            );
            output_path.set_file_name(new_filename);
        }
    }
    
    output_path
}

/// Setup logging based on verbosity level
pub fn setup_logging(verbose: bool) {
    let log_level = if verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    
    env_logger::Builder::from_default_env()
        .filter_level(log_level)
        .init();
}

/// Setup thread pool for parallel processing
#[cfg(feature = "parallel")]
pub fn setup_thread_pool(num_threads: usize) -> Result<()> {
    use rayon::ThreadPoolBuilder;
    
    let threads = if num_threads == 0 {
        num_cpus::get()
    } else {
        num_threads
    };
    
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| ColorShineError::config_error(format!("Failed to setup thread pool: {}", e)))?;
    
    log::info!("Using {} threads for parallel processing", threads);
    Ok(())
}

#[cfg(not(feature = "parallel"))]
pub fn setup_thread_pool(_num_threads: usize) -> Result<()> {
    log::info!("Parallel processing not enabled");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_attractor_string() {
        let (color, tolerance, strength) = parse_attractor_string("red;50;75").unwrap();
        assert_eq!(color, "red");
        assert_eq!(tolerance, 50.0);
        assert_eq!(strength, 75.0);
    }

    #[test]
    fn test_parse_attractor_string_with_spaces() {
        let (color, tolerance, strength) = parse_attractor_string(" blue ; 60 ; 80 ").unwrap();
        assert_eq!(color, "blue");
        assert_eq!(tolerance, 60.0);
        assert_eq!(strength, 80.0);
    }

    #[test]
    fn test_parse_attractor_string_invalid_format() {
        assert!(parse_attractor_string("red;50").is_err());
        assert!(parse_attractor_string("red;50;75;extra").is_err());
    }

    #[test]
    fn test_parse_attractor_string_invalid_numbers() {
        assert!(parse_attractor_string("red;invalid;75").is_err());
        assert!(parse_attractor_string("red;50;invalid").is_err());
    }

    #[test]
    fn test_generate_output_filename() {
        let input_path = PathBuf::from("test.jpg");
        let output_path = generate_output_filename(&input_path);
        assert_eq!(output_path, PathBuf::from("test_colorshine.jpg"));
    }
}