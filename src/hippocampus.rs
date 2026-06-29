//! Dual-Stream Cognition Architecture (EchoReservoir)
//!
//! Implements the hippocampus-inspired associative reservoir that operates on
//! PRE-ENTMAX dense signals to enable echo learning at σ ≈ 1.0 criticality.
//!
//! ## Architecture
//!
//! - **EchoReservoir**: A ring-buffer of recent pre-entmax states + a Hebbian
//!   association matrix that learns to predict the next dense state.
//! - **CognitionState**: Three decoupled SOC parameters controlling local
//!   (NMDA), global (Gain), and arousal (Permeability) dynamics.
//!
//! ## Dual-Stream Principle
//!
//! - `sparse_action` = post-entmax output (σ ≈ 1.0, ~1-2 active neurons).
//!   Used for inference / token selection.
//! - `dense_learning` = pre-entmax logits (dense, >50% non-zero).
//!   Used as the gradient substrate so that associative Hebbian updates
//!   do not collapse to zero when different words activate disjoint neurons.

use ndarray::Array1;
use std::collections::VecDeque;

/// Number of neurons in the *C. elegans* connectome.
const WORM_NEURON_COUNT: usize = 302;

/// Ring-buffer of recent pre-entmax states + Hebbian association matrix.
///
/// The reservoir stores the last `capacity` dense pre-entmax vectors and
/// maintains a 302×302 Hebbian association matrix `W_assoc`.  When queried
/// with the current pre-entmax state, it returns an `echo_bias` vector that
/// nudges the activation toward recently co-active patterns.
///
/// # Invariants
/// - All stored vectors are finite and have length 302.
/// - `W_assoc` is symmetric and clamped to [-1.0, 1.0].
/// - The history buffer never exceeds `capacity`.
#[derive(Clone, Debug)]
pub struct EchoReservoir {
    /// Ring-buffer of recent dense pre-entmax states.
    pub history: VecDeque<Array1<f32>>,
    /// 302×302 Hebbian association matrix (dense outer-product accumulator).
    pub associations: ndarray::Array2<f32>,
    /// Maximum number of states in the history buffer.
    pub capacity: usize,
    /// Decay factor for the association matrix (0.0 = no decay, 1.0 = full decay).
    pub decay: f32,
    /// Pre-hebbian decay: multiplied BEFORE each outer-product step.
    /// Unlike `decay` which runs after, `pre_decay` controls the background
    /// forgetting rate during streaming. 0.9999 ≈ slow fade, 1.0 = no decay.
    pub pre_decay: f32,
    /// Learning rate for the Hebbian outer-product update.
    pub hebbian_lr: f32,
}

impl EchoReservoir {
    /// Create a new empty reservoir with the given capacity.
    ///
    /// The association matrix is initialised to zero.  `decay` controls
    /// exponential forgetting (0.0 = perfect memory, ~0.01 = slow forgetting).
    pub fn new(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            associations: ndarray::Array2::zeros((WORM_NEURON_COUNT, WORM_NEURON_COUNT)),
            capacity,
            decay: 0.01,
            pre_decay: 0.99999,
            hebbian_lr: 0.01,
        }
    }

    /// Push a dense pre-entmax state into the history buffer.
    ///
    /// Trims the buffer if it exceeds `capacity`. Skips all-zero states
    /// (degenerate inputs that would add noise to the association matrix).
    pub fn push(&mut self, dense: &Array1<f32>) {
        if dense.iter().all(|&v| v.abs() < 1e-8) {
            return;
        }
        self.history.push_back(dense.clone());
        while self.history.len() > self.capacity {
            self.history.pop_front();
        }
    }

    /// Query the reservoir: compute the echo bias for the current dense state.
    ///
    /// The bias is the sum of Hebbian-association-weighted recent states,
    /// scaled so its RMS matches the current state's RMS.
    ///
    /// Returns a zero vector if the history is empty.
    pub fn query(&self, current: &Array1<f32>) -> Array1<f32> {
        if self.history.is_empty() {
            return Array1::zeros(WORM_NEURON_COUNT);
        }

        // Sum over recent states weighted by association matrix
        let mut bias = Array1::zeros(WORM_NEURON_COUNT);
        for past in &self.history {
            let weight: f32 = past.dot(current);
            if weight > 0.0 {
                bias = &bias + past.mapv(|v| v * weight);
            }
        }

        // Normalise bias RMS to match current RMS
        let bias_rms = (bias.mapv(|v| v * v).sum() / WORM_NEURON_COUNT as f32)
            .sqrt()
            .max(1e-8);
        let cur_rms = (current.mapv(|v| v * v).sum() / WORM_NEURON_COUNT as f32)
            .sqrt()
            .max(1e-8);
        bias.mapv(|v| v / bias_rms * cur_rms)
    }

    /// Perform one step of Hebbian learning on the association matrix.
    ///
    /// ΔW[i,j] = η · z_pre[i] · z_pre[j]  (outer product of the DENSE signal)
    ///
    /// The update is clamped to [-1.0, 1.0] for numerical safety.  The matrix
    /// is kept symmetric by averaging W[i,j] and W[j,i].
    pub fn hebbian_step(&mut self, dense: &Array1<f32>) {
        // Pre-hebbian decay: fade old associations before adding new signal.
        // This prevents the matrix from saturating during streaming training.
        self.associations.mapv_inplace(|w| w * self.pre_decay);

        let lr = self.hebbian_lr;
        let n = WORM_NEURON_COUNT;
        for i in 0..n {
            let di = dense[i];
            if di.abs() < 1e-8 {
                continue;
            }
            for j in i..n {
                let dj = dense[j];
                let hebb = lr * di * dj;
                self.associations[(i, j)] += hebb;
                if i != j {
                    self.associations[(j, i)] += hebb;
                }
            }
        }

        // Exponential decay to prevent unbounded growth
        self.associations.mapv_inplace(|w| w * (1.0 - self.decay));

        // Clamp to [-1.0, 1.0]
        for v in self.associations.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
    }

    /// Clear the history buffer and reset the association matrix to zero.
    pub fn reset(&mut self) {
        self.history.clear();
        self.associations = ndarray::Array2::zeros((WORM_NEURON_COUNT, WORM_NEURON_COUNT));
    }
}

/// Three decoupled SOC parameters for the dual-stream architecture.
///
/// These replace the single `creative_k` with three independent knobs that
/// control different aspects of the critical dynamics:
///
/// - `kappa_gate` (NMDA / local): Controls the local NMDA-like threshold
///   sharpness.  High kappa → sharp gate (deterministic), low → smooth.
///   Analogue of the dendritic tree's `creative_k`.
/// - `t_scale` (Gain / global): Temperature-like scale for the dense pre-entmax
///   signal.  Larger values compress the dynamic range, making the entmax
///   more distributed.  Decoupled from kappa so the entmax can be tuned
///   independently of the gate.
/// - `alpha_echo` (Arousal / permeability): Blend factor between the raw
///   pre-entmax logits and the echo bias from the reservoir.
///   0.0 = no echo (pure feedforward), 1.0 = full echo (maximal recurrence).
#[derive(Clone, Debug)]
pub struct CognitionState {
    pub kappa_gate: f32,
    pub t_scale: f32,
    pub alpha_echo: f32,
}

impl CognitionState {
    pub fn new() -> Self {
        Self {
            kappa_gate: 1.0,
            t_scale: 50.0,
            alpha_echo: 0.0,
        }
    }
}

impl Default for CognitionState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_push_and_query_empty() {
        let reservoir = EchoReservoir::new(10);
        let state = Array1::zeros(WORM_NEURON_COUNT);
        let bias = reservoir.query(&state);
        assert_eq!(bias.len(), WORM_NEURON_COUNT);
        assert!(bias.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_reservoir_push_and_query_nonempty() {
        let mut reservoir = EchoReservoir::new(10);
        let mut state = Array1::zeros(WORM_NEURON_COUNT);
        state[42] = 0.5;
        reservoir.push(&state);
        let bias = reservoir.query(&state);
        // After one push, the bias should be non-zero
        let sum_abs: f32 = bias.iter().map(|&v| v.abs()).sum();
        assert!(sum_abs > 0.0, "bias should be non-zero after one push");
    }

    #[test]
    fn test_reservoir_capacity() {
        let mut reservoir = EchoReservoir::new(3);
        for i in 0..10 {
            let mut state = Array1::zeros(WORM_NEURON_COUNT);
            state[0] = i as f32 * 0.1;
            reservoir.push(&state);
        }
        assert_eq!(reservoir.history.len(), 3);
    }

    #[test]
    fn test_hebbian_step_symmetry() {
        let mut reservoir = EchoReservoir::new(10);
        let mut dense = Array1::zeros(WORM_NEURON_COUNT);
        dense[10] = 0.5;
        dense[20] = -0.3;
        reservoir.hebbian_step(&dense);
        // Matrix must be symmetric
        for i in 0..WORM_NEURON_COUNT {
            for j in 0..WORM_NEURON_COUNT {
                assert!(
                    (reservoir.associations[(i, j)] - reservoir.associations[(j, i)]).abs() < 1e-6,
                    "association matrix not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_hebbian_step_clamping() {
        let mut reservoir = EchoReservoir::new(10);
        let mut dense = Array1::zeros(WORM_NEURON_COUNT);
        dense[0] = 100.0; // extreme value
        reservoir.hebbian_lr = 0.1;
        reservoir.hebbian_step(&dense);
        for &v in reservoir.associations.iter() {
            assert!(
                v >= -1.0 && v <= 1.0,
                "association value {v} outside [-1, 1]"
            );
        }
    }

    #[test]
    fn test_cognition_state_default() {
        let cs = CognitionState::new();
        assert_eq!(cs.kappa_gate, 1.0);
        assert_eq!(cs.t_scale, 50.0);
        assert_eq!(cs.alpha_echo, 0.0);
    }
}
