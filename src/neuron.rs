//! Minimal dendritic tree stub for the 302-neuron connectome.
//!
//! In the full architecture this implements triple-quad packet folding.
//! For the public release, quad_routing defaults to false and this
//! serves as a no-op placeholder.

use ndarray::Array1;

/// A single dendritic packet placeholder.
#[derive(Clone, Debug)]
pub struct Packet {
    pub threshold: f32,
    pub basal_w: Vec<f32>,
}

impl Packet {
    pub fn new() -> Self {
        Self {
            threshold: 0.0,
            basal_w: Vec::new(),
        }
    }
}

impl Default for Packet {
    fn default() -> Self {
        Self::new()
    }
}

/// Dendritic tree placeholder.
#[derive(Clone, Debug)]
pub struct DendriticTree {
    pub packets: Vec<Packet>,
}

impl DendriticTree {
    /// Create a new placeholder dendritic tree.
    pub fn new() -> Self {
        Self {
            packets: vec![Packet::new(); 38],
        }
    }

    /// Forward pass through the dendritic tree.
    /// Returns the input unchanged (identity) when quad_routing is disabled.
    pub fn forward(&self, state: &Array1<f32>, _creative_k: f32) -> Array1<f32> {
        state.clone()
    }

    /// Zero all gradient accumulators.
    pub fn zero_gradients(&mut self) {}

    /// Reverse valve backward pass.
    /// No-op stub; returns the delta unchanged.
    pub fn reverse_valve_backward(
        &mut self,
        _scaled_state: &Array1<f32>,
        d_output: &Array1<f32>,
        _pruning_rate: f32,
        _creative_k: f32,
    ) -> Array1<f32> {
        d_output.clone()
    }

    /// Apply accumulated gradients.
    pub fn apply_gradients(&mut self, _lr: f32) {}

    /// Clamp all weights to valid range.
    pub fn clamp_weights(&mut self) {}
}

impl Default for DendriticTree {
    fn default() -> Self {
        Self::new()
    }
}
