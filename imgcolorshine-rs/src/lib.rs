//! # imgcolorshine-rs
//!
//! Ultrafast Rust port of imgcolorshine - Transform image colors using OKLCH color attractors.
//!
//! This library provides a physics-inspired approach to color transformation, where colors 
//! are "attracted" towards target colors in perceptually uniform OKLCH color space.
//!
//! ## Features
//!
//! - **Perceptually Uniform**: Uses OKLCH color space for natural-looking transformations
//! - **High Performance**: Multi-threaded processing with SIMD optimization
//! - **Flexible**: Fine-grained control over color channels and transformation parameters
//! - **Memory Efficient**: Tile-based processing for large images
//! - **GPU Acceleration**: Optional GPU compute shader support
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use imgcolorshine_rs::{ColorTransformer, Attractor};
//! use palette::Oklch;
//!
//! // Create a color transformer
//! let mut transformer = ColorTransformer::new();
//!
//! // Add an attractor to pull colors towards orange
//! let orange_attractor = Attractor::new(
//!     Oklch::new(0.7, 0.15, 30.0), // OKLCH color
//!     50.0, // tolerance (50% of pixels)
//!     75.0, // strength
//! );
//! transformer.add_attractor(orange_attractor);
//!
//! // Transform an image
//! let result = transformer.transform_image("input.jpg", "output.jpg")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod color;
pub mod transform;
pub mod io;
pub mod cli;

#[cfg(feature = "gpu")]
pub mod gpu;

pub mod error;
pub mod utils;

// Re-export main types for convenience
pub use color::{Attractor, ColorEngine, ColorSpace};
pub use transform::ColorTransformer;
pub use color::engine::TransformSettings;
pub use error::{Result, ColorShineError};

#[cfg(feature = "gpu")]
pub use gpu::GpuAccelerator;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default tolerance value (50% of pixels influenced)
pub const DEFAULT_TOLERANCE: f32 = 50.0;

/// Default strength value (75% maximum influence)
pub const DEFAULT_STRENGTH: f32 = 75.0;

/// Tolerance range bounds
pub const TOLERANCE_MIN: f32 = 0.0;
pub const TOLERANCE_MAX: f32 = 100.0;

/// Strength range bounds  
pub const STRENGTH_MIN: f32 = 0.0;
pub const STRENGTH_MAX: f32 = 200.0;

/// Traditional strength regime boundary
pub const STRENGTH_TRADITIONAL_MAX: f32 = 100.0;