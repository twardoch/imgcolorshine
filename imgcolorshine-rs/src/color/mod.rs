//! Color space operations and attractor management
//!
//! This module provides the core color manipulation functionality, including
//! color space conversions, attractor definitions, and the color engine.

pub mod conversions;
pub mod attractor;
pub mod engine;
pub mod gamut;

pub use attractor::Attractor;
pub use engine::{ColorEngine, ColorSpace};
pub use conversions::{srgb_to_oklch, oklch_to_srgb, oklab_to_oklch, oklch_to_oklab};
pub use gamut::GamutMapper;