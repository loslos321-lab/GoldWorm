//! GoldWorm Architecture
//!
//! A production-grade native Rust cognitive engine (Rust 2024 edition):
//! - Non-linear 128-dimensional manifold geometry with Grassmannian fusion
//! - 302-neuron *C. elegans* connectome routing layer
//! - Smooth gradient optimization with logarithmic barrier regularization
//! - Vocabulary projection and token interface alignment
//!
//! # Module Architecture
//! - [`geometry`]: 128D token coordinates, Modified Gram-Schmidt orthogonalization,
//!   Grassmannian fusion, and `atan2` geodesics.
//! - [`bridge`]: Native token/logit projection via RPITIT batch traits mapping
//!   128D coordinates into arbitrary vocabulary spaces.
//! - [`worm_brain`]: 302-neuron *C. elegans* connectome routing layer for
//!   coordinate key propagation through biological synaptic topologies.
//! - [`hippocampus`]: Dual-Stream EchoReservoir with Hebbian association learning.
//! - [`observation`]: ANSI-based real-time observation dashboard.
//! - [`storage`]: Weight persistence via safetensors format.
//! - [`criticality`]: Quilez smooth-k annealing for creativity/determinism trade-off.
//! - [`training`]: Hebbian plasticity engine with Maxwell damping.
//! - [`tda`]: Topological Data Analysis for activation landscape monitoring.
//! - [`memory`]: Synaptic echo buffer and trajectory storage.
//!
//! # Critical Invariants (Rust 2024)
//! - All geometric state maintains true multi-dimensional variance; no scalar
//!   cloning across dimensions is permitted.
//! - All boundary regularization uses continuous, differentiable soft barriers.
//! - `#[inline]` on every hot-path function for monomorphized static dispatch.

pub mod bridge;
pub mod criticality;
pub mod geometry;
pub mod hippocampus;
pub mod memory;
pub mod neuron;
pub mod observation;
pub mod storage;
pub mod tda;
pub mod training;
pub mod worm_brain;

use thiserror::Error;

/// Unified error type for all core operations.
///
/// Every public fallible function returns [`Result<T>`] carrying a
/// `CoreError` so that multi-agent pipelines can intercept and classify
/// failures without panicking.  `#[repr(C)]` guarantees a deterministic
/// stack layout with zero auxiliary heap allocation in the discriminant.
#[derive(Debug, Error)]
#[repr(C)]
pub enum CoreError {
    /// Geometry computation failed (e.g., degenerate subspace, zero norm).
    #[error("geometry operation failed: {0}")]
    Geometry(String),

    /// Bridge projection failed (e.g., dimension mismatch, uninitialized weights).
    #[error("bridge projection failed: {0}")]
    Bridge(String),

    /// Dimension mismatch between expected and actual tensor shapes.
    #[error("invalid dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    /// A numerical operation produced an invalid value (inf, NaN, or out of domain).
    #[error("numerical error: {0}")]
    Numerical(String),
}

/// Shorthand result type used throughout the crate.
///
/// Monomorphized at every call site for zero-cost static dispatch.
pub type Result<T> = std::result::Result<T, CoreError>;
