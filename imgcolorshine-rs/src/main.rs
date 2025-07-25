//! imgcolorshine-rs command-line interface
//!
//! This is the main entry point for the imgcolorshine-rs CLI application.

use clap::Parser;
use log::{info, error};
use std::process;

use imgcolorshine_rs::{
    cli::{Args, parse_attractor_string, generate_output_filename, setup_logging, setup_thread_pool},
    transform::{ColorTransformer, TransformSettings},
    io::process_image_file,
    error::Result,
};

fn main() {
    // Parse command-line arguments
    let args = Args::parse();
    
    // Setup logging
    setup_logging(args.verbose);
    
    // Run the main application
    if let Err(e) = run(args) {
        error!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    info!("imgcolorshine-rs v{}", env!("CARGO_PKG_VERSION"));
    
    // Setup thread pool
    setup_thread_pool(args.threads)?;
    
    // Validate input file
    if !args.input_image.exists() {
        return Err(imgcolorshine_rs::error::ColorShineError::FileNotFound(
            args.input_image.to_string_lossy().to_string()
        ));
    }
    
    // Parse attractors
    if args.attractors.is_empty() {
        return Err(imgcolorshine_rs::error::ColorShineError::InvalidParameter {
            field: "attractors".to_string(),
            value: "empty".to_string(),
            expected: "at least one attractor".to_string(),
        });
    }
    
    let mut transformer = ColorTransformer::new();
    
    for attractor_str in &args.attractors {
        let (color, tolerance, strength) = parse_attractor_string(attractor_str)?;
        transformer.add_attractor_css(&color, tolerance, strength)?;
        info!("Added attractor: {} (tolerance: {:.1}%, strength: {:.1}%)", 
              color, tolerance, strength);
    }
    
    // Setup transformation settings
    let settings = TransformSettings {
        transform_lightness: args.luminance,
        transform_chroma: args.saturation,
        transform_hue: args.hue,
    };
    transformer.set_default_settings(settings);
    
    // Determine output path
    let output_path = args.output_image
        .unwrap_or_else(|| generate_output_filename(&args.input_image));
    
    info!("Processing: {} -> {}", 
          args.input_image.display(), 
          output_path.display());
    
    // Process the image
    process_image_file(
        &args.input_image,
        &output_path,
        |rgb_data| transformer.transform_pixels(rgb_data),
    )?;
    
    info!("Transformation complete! Output saved to: {}", output_path.display());
    
    Ok(())
}
