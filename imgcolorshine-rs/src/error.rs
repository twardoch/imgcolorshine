//! Error types for imgcolorshine-rs
//!
//! This module defines all error types used throughout the library.

use std::fmt;
use thiserror::Error;

/// Result type alias for imgcolorshine operations
pub type Result<T> = std::result::Result<T, ColorShineError>;

/// Main error type for imgcolorshine operations
#[derive(Error, Debug)]
pub enum ColorShineError {
    /// Image I/O errors
    #[error("Image I/O error: {0}")]
    ImageIo(#[from] image::ImageError),

    /// Invalid color format
    #[error("Invalid color format: {0}")]
    InvalidColor(String),

    /// Invalid parameter values
    #[error("Invalid parameter: {field} = {value} (expected {expected})")]
    InvalidParameter {
        field: String,
        value: String,
        expected: String,
    },

    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Unsupported image format
    #[error("Unsupported image format: {0}")]
    UnsupportedFormat(String),

    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    /// GPU acceleration error
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Color space conversion error
    #[error("Color space conversion error: {0}")]
    ColorSpaceError(String),

    /// Generic computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl ColorShineError {
    /// Create a new invalid parameter error
    pub fn invalid_parameter<T: fmt::Display>(field: &str, value: T, expected: &str) -> Self {
        Self::InvalidParameter {
            field: field.to_string(),
            value: value.to_string(),
            expected: expected.to_string(),
        }
    }

    /// Create a new invalid color error
    pub fn invalid_color<T: fmt::Display>(msg: T) -> Self {
        Self::InvalidColor(msg.to_string())
    }

    /// Create a new memory error
    pub fn memory_error<T: fmt::Display>(msg: T) -> Self {
        Self::MemoryError(msg.to_string())
    }

    /// Create a new color space error
    pub fn color_space_error<T: fmt::Display>(msg: T) -> Self {
        Self::ColorSpaceError(msg.to_string())
    }

    /// Create a new computation error
    pub fn computation_error<T: fmt::Display>(msg: T) -> Self {
        Self::ComputationError(msg.to_string())
    }

    /// Create a new configuration error
    pub fn config_error<T: fmt::Display>(msg: T) -> Self {
        Self::ConfigError(msg.to_string())
    }

    #[cfg(feature = "gpu")]
    /// Create a new GPU error
    pub fn gpu_error<T: fmt::Display>(msg: T) -> Self {
        Self::GpuError(msg.to_string())
    }
}