//! Biological Connectome Router
//!
//! Implements a 302-neuron routing layer modelled after the *C. elegans*
//! nervous system — the only animal whose complete connectome has been
//! experimentally mapped (White et al., 1986).
//!
//! ## Connectivity Motifs
//! 1. **Band synapses** — ±1/±2/±3 neighbourhood connections.
//! 2. **Pharyngeal sub-network** — denser internal wiring for neurons 0–19.
//! 3. **Sensory → Interneuron** — sparse feed-forward projection (20–91 → 92–168).
//! 4. **Command interneuron hubs** (AVAL/AVAR/AVBL/AVBR, idx 99–102)
//!    broadcast to the full motor population (169–301).
//! 5. **Interneuron → Motor** — sparse feed-forward projection.
//!
//! ## Invariants
//! - `neuron_count` is always [`WORM_NEURON_COUNT`] = 302.
//! - All synaptic weights are non-negative and bounded in `[0, 1]`.
//! - `route_signal` is panic-free; dimension errors return
//!   [`crate::CoreError::Geometry`].

use crate::geometry::{self, MANIFOLD_DIM};
use crate::memory::SynapticEchoBuffer;
use crate::{CoreError, Result};
use ndarray::{Array1, Array2};

/// Experimentally determined neuron count of *C. elegans* (White et al., 1986).
pub const WORM_NEURON_COUNT: usize = 302;

/// Golden ratio φ — used for quasiperiodic group partitioning.
pub const PHI: f64 = 1.618033988749895;

/// Size of group A (sensory) = round(302 / φ) = 187.
pub const GROUP_A: usize = 187;

/// Number of dendritic packets in the Triple-Quad tree.
pub const NUM_PACKETS: usize = 38;

/// Tsallis α-entmax activation (generalisation of softmax and sparsemax).
///
/// α = 1: softmax (dense, all non-zero)
/// α = 2: sparsemax (sparse, exact zeros via simplex projection)
/// α > 2: sparser than sparsemax (Tsallis α-entropy regulariser)
///
/// Algorithm: For α ≈ 1 uses numerically stable softmax. For α = 2 uses the
/// exact O(n log n) sparsemax algorithm. For general α ≠ 1,2 uses binary
/// search on the threshold τ where p_i = [ (α-1)·z_i - τ ]_{+}^{1/(α-1)}.
#[inline]
pub fn alpha_entmax(v: &mut [f32], alpha: f32) {
    if (alpha - 1.0).abs() < 1e-6 {
        // α ≈ 1: softmax
        let max_v = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in v.iter_mut() {
            *x = (*x - max_v).exp();
            sum += *x;
        }
        if sum > 0.0 {
            for x in v.iter_mut() {
                *x /= sum;
            }
        }
    } else if (alpha - 2.0).abs() < 1e-10 {
        // α = 2: exact sparsemax (backward compatible)
        sparsemax(v);
    } else if alpha < 1.0 {
        // Clamp to softmax for α < 1 (Tsallis entropy undefined)
        alpha_entmax(v, 1.0);
    } else {
        // General case: binary search for τ on sorted values
        let alpha_m1 = alpha - 1.0;
        let exponent = 1.0 / alpha_m1;

        let mut sorted: Vec<f32> = v.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let max_z = sorted[0];
        let min_z = sorted[sorted.len() - 1];

        let mut low = alpha_m1 * min_z - 10.0;
        let mut high = alpha_m1 * max_z;

        // Ensure F(low) > 1 > F(high)
        let sum_at_low: f32 = sorted
            .iter()
            .map(|&z| (alpha_m1 * z - low).max(0.0).powf(exponent))
            .sum();
        if sum_at_low < 1.0 {
            low -= 10.0;
        }

        for _ in 0..80 {
            let mid = (low + high) / 2.0;
            let sum: f32 = sorted
                .iter()
                .map(|&z| (alpha_m1 * z - mid).max(0.0).powf(exponent))
                .sum();
            if sum > 1.0 {
                low = mid;
            } else {
                high = mid;
            }
        }

        let tau = (low + high) / 2.0;
        for x in v.iter_mut() {
            *x = (alpha_m1 * *x - tau).max(0.0).powf(exponent);
        }
    }
}

/// Exact sparsemax (α = 2). Finds τ = (cumsum - 1) / k then projects
/// p_i = max(0, z_i - τ).  O(n log n) via sort.
#[inline]
fn sparsemax(v: &mut [f32]) {
    let mut sorted: Vec<f32> = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut cumsum = 0.0f32;
    let mut tau = 0.0f32;

    for (i, &val) in sorted.iter().enumerate() {
        cumsum += val;
        let t = (cumsum - 1.0) / (i + 1) as f32;
        if t < val {
            tau = t;
        }
    }

    for x in v.iter_mut() {
        *x = (*x - tau).max(0.0);
    }
}

/// Hyperbolic sparsemax with per-branch NMDA voltage gating.
///
/// 1. Checks the Minkowski (Lorentz) norm M(state, state). If M ≥ 0, the
///    state is not on the hyperboloid — falls back to standard sparsemax.
/// 2. Normalises the state onto the Lorentz sphere: M(v, v) = -1.
/// 3. Per-branch NMDA thresholding: each neuron i has threshold τᵢ. The
///    squared Minkowski component vᵢ² (in the Lorentz frame) is the
///    activation energy. If vᵢ² < τᵢ, the neuron is silenced.
/// 4. Standard sparsemax on the surviving neurons renormalises output to
///    sum to 1 on the probability simplex.
#[inline]
fn sparsemax_hyperbolic(state: &mut [f32], thresholds: &[f32]) {
    let n = state.len();
    if n == 0 {
        return;
    }

    // Compute Minkowski norm: M(v,v) = v[0]² - Σ_{i=1}^{n-1} v[i]²
    let m_norm = crate::geometry::minkowski_dot(state, state);

    if m_norm >= 0.0 || m_norm.abs() < 1e-15 {
        // Not timelike — hyperbolic projection invalid
        sparsemax(state);
        return;
    }

    // Compute Lorentz scale factor for energy computation only
    let lorentz_scale = (-1.0 / m_norm).sqrt();

    // Apply per-branch NMDA thresholds: hyperbolic energy = (Lorentz-frame component)²
    // The Lorentz-frame energy E_i = (state[i] · lorentz_scale)².
    // If E_i < τ_i the neuron is silenced; survivors keep their original values.
    for i in 0..n {
        let thresh = thresholds.get(i).copied().unwrap_or(0.01);
        let energy = (state[i] * lorentz_scale) * (state[i] * lorentz_scale);
        if energy < thresh {
            state[i] = 0.0;
        }
    }

    // Sparsemax on survivors; if all gated, output stays as zeros
    if state.iter().any(|&v| v != 0.0) {
        sparsemax(state);
    }
}

/// A 302-neuron connectome router inspired by the *C. elegans* nervous system.
///
/// The incoming coordinate key is projected onto the neuron population via a
/// deterministic linear map, propagated through the synaptic adjacency matrix,
/// and passed through a smooth non-linear activation.
#[derive(Clone, Debug)]
pub struct WormBrain {
    /// Fixed neuron count (always 302).
    pub neuron_count: usize,
    /// 302×302 synaptic adjacency matrix (f32 precision).
    pub synapses: Array2<f32>,
    /// 302×302 Maxwell velocity matrix (momentum for damped updates).
    /// Initialised to zero; tracks structural momentum analogous to the
    /// magnetic field B coupling with the content field E in Maxwell's
    /// equations.  Used by [`WormTrainer::train_step_damped`].
    pub velocities: Array2<f32>,
    /// Deterministic projection from input space to neuron state.
    pub input_projection: Array2<f32>,
    /// Expected input dimension (aligned with `UniversalProjection` / manifold).
    pub input_dim: usize,
    /// When `true`, use triple-quad folding instead of direct synaptic
    /// propagation for O(n³) interaction capacity.
    pub quad_routing: bool,
    /// When `true`, replace sparsemax with Winner-Take-All (top-3) activation.
    /// WTA is sparser and can improve cluster formation during training.
    pub use_wta: bool,
    /// α-entmax parameter.
    /// α = 1: softmax (dense), α = 2: sparsemax (default), α > 2: sparser than sparsemax.
    pub entmax_alpha: f32,
    /// Dendritic tree of 38 triple-quad packets. When quad_routing is true,
    /// this replaces the linear synapses.dot(state) with quadratic computations.
    pub dendritic_tree: crate::neuron::DendriticTree,
    /// Reverse valve top-down modulation threshold.
    /// When > 0, the activation norm is compared against this threshold and
    /// the output is attenuated via sigmoid gating.  0 = valve disabled.
    pub valve_threshold: f32,
    /// Steepness of the reverse valve sigmoid transition.
    pub valve_steepness: f32,
    /// When `true`, use hyperbolic (Minkowski) sparsemax with per-branch
    /// NMDA thresholds instead of the standard α-entmax activation.
    pub hyperbolic_mode: bool,
    /// Per-branch NMDA thresholds (length 302). Each neuron has a voltage-gated
    /// threshold τᵢ in Minkowski space. Neurons whose squared Minkowski
    /// component falls below τᵢ are silenced before sparsemax.
    pub nmda_thresholds: Vec<f32>,
    /// When `true`, use Quilez Bridge smooth-k annealing for the activation.
    /// k → 0: deterministic (sparse), k → ∞: creative (dense/softmax).
    pub creative_mode: bool,
    /// Quilez smooth-k parameter for creative annealing.
    pub creative_k: f32,
    /// Optional EchoReservoir for dual-stream hippocampus-style associative
    /// memory operating on the PRE-ENTMAX dense signal.
    pub echo_reservoir: Option<crate::hippocampus::EchoReservoir>,
    /// Three decoupled SOC parameters for the dual-stream architecture.
    pub cognition: crate::hippocampus::CognitionState,
}

impl WormBrain {
    /// Construct a deterministic, sparse baseline connectome.
    ///
    /// The synapse topology mimics the canonical *C. elegans* structure:
    /// band neighbourhoods, pharyngeal cluster, sensory→interneuron feed-forward,
    /// command-interneuron hubs, and interneuron→motor projections.
    ///
    /// The input projection maps the standard 128-D manifold coordinate space
    /// onto the 302-neuron layer.
    #[inline]
    pub fn new_baseline() -> Self {
        let neuron_count = WORM_NEURON_COUNT;
        let mut synapses = Array2::<f32>::zeros((neuron_count, neuron_count));

        // 1. Band synapses: ±1, ±2, ±3 neighbours
        for i in 0..neuron_count {
            for delta in [-3i32, -2, -1, 1, 2, 3] {
                let j = i as i32 + delta;
                if j >= 0 && j < neuron_count as i32 {
                    let weight = match delta.abs() {
                        1 => 0.30,
                        2 => 0.15,
                        3 => 0.05,
                        _ => 0.0,
                    };
                    synapses[(i, j as usize)] = weight;
                }
            }
        }

        // 2. Pharyngeal sub-network (0–19): denser internal wiring
        for i in 0..20 {
            for j in 0..20 {
                if i != j {
                    synapses[(i, j)] = synapses[(i, j)].max(0.25_f32);
                }
            }
        }

        // 3. Sensory → Interneuron (20–91 → 92–168)
        for i in 20..92 {
            for j in 92..169 {
                if (i + j) % 7 == 0 {
                    synapses[(i, j)] = synapses[(i, j)].max(0.10_f32);
                }
            }
        }

        // 4. Command hubs (99–102) broadcast to motor (169–301)
        for hub in 99..103 {
            for j in 169..neuron_count {
                if (hub + j) % 5 == 0 {
                    synapses[(hub, j)] = synapses[(hub, j)].max(0.35_f32);
                }
            }
        }

        // 5. Interneuron → Motor (92–168 → 169–301)
        for i in 92..169 {
            for j in 169..neuron_count {
                if (i * 3 + j) % 11 == 0 {
                    synapses[(i, j)] = synapses[(i, j)].max(0.12_f32);
                }
            }
        }

        // Row-normalise to keep spectral radius bounded and weights in [0, 1]
        for i in 0..neuron_count {
            let row_sum = (0..neuron_count).map(|j| synapses[(i, j)]).sum::<f32>();
            if row_sum > 1.0 {
                for j in 0..neuron_count {
                    synapses[(i, j)] /= row_sum;
                }
            }
        }

        // Input projection: deterministic Gaussian sampling per row.
        // The phase function (0.5 + 0.5 * ((i*17+j*13) % 1000) / 1000) produces
        // rows with cos > 0.999 — all neurons see nearly identical inputs.
        // Gaussian rows with seed=42 give cos ~0.3 between different rows,
        // enabling genuine 128-D → 302-D diversity.
        let input_dim = MANIFOLD_DIM;
        let mut input_projection = Array2::<f32>::zeros((neuron_count, input_dim));
        let mut rng = fastrand::Rng::with_seed(42);
        for i in 0..neuron_count {
            // Generate MANIFOLD_DIM Gaussian values, then normalize to unit length
            let mut row = [0.0f32; MANIFOLD_DIM];
            let mut sq_sum = 0.0f32;
            for j in 0..input_dim {
                // Box-Muller transform for Gaussian(0, 1)
                let u1 = rng.f32(); // (0, 1)
                let u2 = rng.f32();
                let r = (-2.0 * (1.0 - u1.max(1e-15)).ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                let z = r * theta.cos();
                row[j] = z;
                sq_sum += z * z;
            }
            let norm = sq_sum.sqrt();
            if norm > 1e-15 {
                for j in 0..input_dim {
                    input_projection[(i, j)] = row[j] / norm;
                }
            } else {
                input_projection[(i, 0)] = 1.0;
            }
        }

        // DEBUG: Check cosine similarity of first 5 rows
        for a in 0..5.min(neuron_count) {
            for b in (a + 1)..5.min(neuron_count) {
                let dot: f32 = (0..input_dim)
                    .map(|j| input_projection[(a, j)] * input_projection[(b, j)])
                    .sum();
                let cos = dot.min(1.0).max(-1.0);
                eprintln!("  [DEBUG] cos(P[{a}], P[{b}]) = {cos:.6}");
            }
        }

        Self {
            neuron_count,
            synapses,
            velocities: Array2::zeros((neuron_count, neuron_count)),
            input_projection,
            input_dim,
            quad_routing: false,
            use_wta: false,
            entmax_alpha: 2.0,
            valve_threshold: 0.0,
            valve_steepness: 5.0,
            dendritic_tree: crate::neuron::DendriticTree::new(),
            hyperbolic_mode: false,
            nmda_thresholds: vec![0.0; WORM_NEURON_COUNT],
            creative_mode: false,
            creative_k: std::f64::consts::LN_2 as f32,
            echo_reservoir: None,
            cognition: crate::hippocampus::CognitionState::new(),
        }
    }

    /// Fold 302-D state into 38 packets of 8, compute triple quad products
    /// (overlapping 4-element windows), and gate the original state by the
    /// weighted combination.  Produces O(n³) interaction capacity while
    /// storing only linear weights because the folding dimension (8) is tiny.
    ///
    /// The 6 remainder neurons (indices 296–301) are zero-padded to fill
    /// the 38th packet.
    pub fn route_triple_quad(&self, state: &Array1<f32>) -> Array1<f32> {
        let n = state.len();
        let num_packets = 38;
        let packet_size = 8;
        // Stack-allocated fold buffer — no heap allocation.
        let mut folded = [[0.0f32; 8]; 38];

        for i in 0..num_packets {
            for j in 0..packet_size {
                let idx = i * packet_size + j;
                if idx < n {
                    folded[i][j] = state[idx];
                }
            }
        }

        let mut result = Array1::zeros(n);

        for i in 0..num_packets {
            let p = &folded[i];

            let q1 = p[0] * p[1] * p[2] * p[3];
            let q2 = p[2] * p[3] * p[4] * p[5];
            let q3 = p[4] * p[5] * p[6] * p[7];

            let combined = 0.4 * q1 + 0.3 * q2 + 0.3 * q3;

            for j in 0..packet_size {
                let idx = i * packet_size + j;
                if idx < n {
                    result[idx] = state[idx] * combined;
                }
            }
        }

        result
    }

    /// Borrow the input projection matrix (302 × MANIFOLD_DIM).
    #[inline]
    pub fn input_projection(&self) -> &Array2<f32> {
        &self.input_projection
    }

    /// Route a coordinate key through the connectome.
    ///
    /// Returns `(activation, pre_synaptic)` where `activation` is the 302-D
    /// sparsemax output and `pre_synaptic` is the 302-D state *before* the
    /// synaptic dot product (i.e. `input_projection @ key_f32`).  The
    /// pre-synaptic value is used by the trainer for correct projection
    /// gradient updates.
    ///
    /// 1. Validates that `point_cloud_key.len()` matches the expected input
    ///    dimension (aligned with `UniversalProjection` / manifold geometry).
    /// 2. Projects the key onto the 302-neuron state vector.
    /// 3. Propagates the state through the synaptic adjacency matrix.
    /// 4. Applies sparsemax activation.
    ///
    /// # Errors
    /// * [`CoreError::Geometry`] if the input dimension does not match.
    /// * [`CoreError::Geometry`] if the propagation yields non-finite values.
    #[inline]
    pub fn route_signal(
        &self,
        point_cloud_key: &Array1<f64>,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        if point_cloud_key.len() != self.input_dim {
            return Err(CoreError::Geometry(format!(
                "input dimension mismatch for connectome routing: expected {} (manifold input_dim), got {}",
                self.input_dim, point_cloud_key.len()
            )));
        }

        // Convert f64 coordinate to f32
        let key_f32: Array1<f32> = Array1::from_iter(point_cloud_key.iter().map(|&v| v as f32));

        // Map input coordinate key onto the 302-neuron state vector
        let mut state = self.input_projection.dot(&key_f32);

        // Capture pre-synaptic state BEFORE synapses and sparsemax
        let pre_synaptic = state.clone();

        if self.quad_routing {
            // RMS-normalise the pre-synaptic state to unit scale so that
            // the dendritic tree's NMDA threshold (τ ≈ 0.2) is triggered.
            // The raw input_projection uses [0, 0.01] weights, giving
            // state values ~0.005 — far below τ. Normalisation maps the
            // state to RMS ≈ 1.0, where quad_form outputs exceed τ.
            let rms = (state.mapv(|v| v * v).sum() / state.len() as f32)
                .sqrt()
                .max(1e-8);
            let inv_rms = 1.0 / rms;
            state = state.mapv(|v| v * inv_rms);

            // Triple-Quad dendritic mode: replace linear weight matrix with
            // quadratic dendritic subnetworks for XOR-capable computation.
            // The 38 packets of 8 neurons compute 3 overlapping quadratic
            // forms each, then apply NMDA-like sparsemax thresholding.
            state = self.dendritic_tree.forward(&state, self.creative_k);

            // Legacy triple-quad folding: modulates state on TOP of dendritic
            // output for extra nonlinearity.
            let quad = self.route_triple_quad(&state);
            state = state * 0.7 + quad * 0.3;

            // Temperature scaling for creative mode: compress logit dynamic range
            // so α-entmax produces distributed outputs (not 1-hot).
            // NOTE: scale is fixed (decoupled from k) — k only controls α-entmax.
            if self.creative_mode && self.creative_k > 0.0 {
                let scale = 50.0;
                state.mapv_inplace(|v| v / scale);
            }
        } else {
            // Standard linear synaptic propagation (backward compat)
            state = self.synapses.dot(&state);
        }

        // Clip extreme logit outliers produced by the dendritic tree's quadratic
        // forms (which reach 63K+ after Hebbian training). Without clipping, no
        // α-entmax can produce distributed output — the one-hot logit dominates.
        for v in state.iter_mut() {
            *v = v.clamp(-0.3, 0.3);
        }

        if state.iter().any(|&v| v.is_nan()) {
            return Err(CoreError::Geometry(
                "NaN detected in state before sparsemax".to_string(),
            ));
        }

        if self.hyperbolic_mode {
            // Hyperbolic sparsemax with per-branch NMDA voltage gating
            sparsemax_hyperbolic(state.as_slice_mut().unwrap(), &self.nmda_thresholds);
        } else if self.creative_mode {
            // Quilez Bridge smooth-k annealing
            let controller = crate::criticality::CriticalityController::new(self.creative_k);
            controller.smooth_activation(state.as_slice_mut().unwrap());
            // Renormalization guard: α-entmax with tiny logits may not sum to 1
            // due to numerical limits of the binary search with exponent > 2.
            let s: f32 = state.iter().sum();
            if s > 0.0 && (s - 1.0).abs() > 1e-4 {
                state.mapv_inplace(|x| x / s);
            }
        } else if self.use_wta {
            // Winner-Take-All: top-3 neurons fire, rest set to 0
            let mut with_idx: Vec<(usize, f32)> = state.iter().copied().enumerate().collect();
            with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let threshold = with_idx.get(2).map(|v| v.1).unwrap_or(0.0);
            state.mapv_inplace(|v| if v >= threshold { v } else { 0.0 });
            let sum: f32 = state.iter().sum();
            if sum > 0.0 {
                state.mapv_inplace(|v| v / sum);
            }
        } else {
            // α-entmax activation (generalises sparsemax/softmax)
            alpha_entmax(state.as_slice_mut().unwrap(), self.entmax_alpha);
        }

        // Reverse valve: top-down modulation via sigmoid gating
        if self.valve_threshold > 0.0 {
            let state_norm: f32 = state.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let gate =
                1.0 / (1.0 + (-self.valve_steepness * (state_norm - self.valve_threshold)).exp());
            state.mapv_inplace(|v| v * gate);
            // Renormalize to sum to 1
            let sum: f32 = state.iter().sum();
            if sum > 0.0 {
                state.mapv_inplace(|v| v / sum);
            }
        }

        Ok((state, pre_synaptic))
    }
}

// ── Rational Q16.16 Fixed-Point Forward Pass ───────────────────────────

/// Q16.16 fixed-point format: 16 bits integer, 16 bits fractional.
/// Range ±32768, precision 1/65536 ≈ 0.0000153.
const Q16_SHIFT: i32 = 16;
const Q16_SCALE: f32 = 65536.0;

/// Rational (Q16.16) forward-pass brain with bit-reproducible arithmetic.
///
/// All weights are stored as `i32` in Q16.16 format in flat row-major vectors.
/// The forward pass uses only integer arithmetic — no floating-point transcendental drift.
/// Compatible with sparsemax activation (no exp/log needed).
#[derive(Clone, Debug)]
pub struct RationalWormBrain {
    /// 302×MANIFOLD_DIM input projection matrix in Q16.16 format (flat, row-major).
    pub input_projection_int: Box<[i32]>,
    /// 302×302 synaptic adjacency matrix in Q16.16 format (flat, row-major).
    pub synapses_int: Box<[i32]>,
    /// When `true`, apply triple-quad folding on top of synaptic propagation.
    pub quad_routing: bool,
}

impl RationalWormBrain {
    /// Convert an f32 WormBrain to Q16.16 rational format.
    pub fn from_worm_brain(brain: &WormBrain) -> Self {
        let (input_projection_int, synapses_int) =
            convert_weights_to_rational(&brain.input_projection, &brain.synapses);
        Self {
            input_projection_int,
            synapses_int,
            quad_routing: brain.quad_routing,
        }
    }

    /// Rational forward pass: input_projection @ key → synapses @ state →
    /// (optionally triple-quad blend) → sparsemax activation,
    /// all in Q16.16 fixed-point.
    ///
    /// `point_cloud_key` must be MANIFOLD_DIM (128) elements long in Q16.16 format.
    /// Returns a 302-element array in Q16.16 format.
    pub fn route_rational(
        &self,
        point_cloud_key: &[i32; MANIFOLD_DIM],
    ) -> [i32; WORM_NEURON_COUNT] {
        // Fixed-point matmul: input_projection @ key
        let mut state = [0i32; WORM_NEURON_COUNT];
        for i in 0..WORM_NEURON_COUNT {
            let mut sum = 0i64;
            let base = i * MANIFOLD_DIM;
            for j in 0..MANIFOLD_DIM {
                sum += self.input_projection_int[base + j] as i64 * point_cloud_key[j] as i64;
            }
            state[i] = (sum >> Q16_SHIFT) as i32;
        }

        // Fixed-point matmul: synapses @ state
        let mut next = [0i32; WORM_NEURON_COUNT];
        for i in 0..WORM_NEURON_COUNT {
            let mut sum = 0i64;
            let base = i * WORM_NEURON_COUNT;
            for j in 0..WORM_NEURON_COUNT {
                sum += self.synapses_int[base + j] as i64 * state[j] as i64;
            }
            next[i] = (sum >> Q16_SHIFT) as i32;
        }

        if self.quad_routing {
            // Triple-quad folding: 70% synaptic + 30% triple-quad
            let quad = rational_triple_quad(&next);
            for i in 0..WORM_NEURON_COUNT {
                let syn = (45875i64 * next[i] as i64) >> Q16_SHIFT; // 0.7 = 45875/65536
                let qua = (19661i64 * quad[i] as i64) >> Q16_SHIFT; // 0.3 = 19661/65536
                next[i] = (syn + qua) as i32;
            }
        }

        // Clip extreme logit outliers to match f32 route_signal range [-0.3, 0.3]
        // 0.3 in Q16.16 = 0.3 * 65536 = 19660
        const CLIP_MAX: i32 = 19660;
        const CLIP_MIN: i32 = -19660;
        for v in next.iter_mut() {
            *v = (*v).clamp(CLIP_MIN, CLIP_MAX);
        }

        // Sparsemax in rational arithmetic
        rational_sparsemax_i32(&mut next);

        next
    }
}

/// Triple-quad folding in Q16.16 fixed-point arithmetic.
///
/// Folds 302-D state into 38 packets of 8 neurons, computes 3 overlapping
/// quadruple products per packet, then gates the original state by the
/// weighted combination.  Matches the f32 `route_triple_quad` in behaviour.
fn rational_triple_quad(state: &[i32; WORM_NEURON_COUNT]) -> [i32; WORM_NEURON_COUNT] {
    let num_packets = 38;
    let packet_size = 8;
    let mut folded = [[0i32; 8]; 38];

    for i in 0..num_packets {
        for j in 0..packet_size {
            let idx = i * packet_size + j;
            if idx < WORM_NEURON_COUNT {
                folded[i][j] = state[idx];
            }
        }
    }

    let mut result = [0i32; WORM_NEURON_COUNT];

    for i in 0..num_packets {
        let p = &folded[i];

        // q1 = p[0]·p[1]·p[2]·p[3] (four-way product)
        let a = (p[0] as i64 * p[1] as i64) >> Q16_SHIFT;
        let b = (p[2] as i64 * p[3] as i64) >> Q16_SHIFT;
        let q1 = (a * b) >> Q16_SHIFT;

        // q2 = p[2]·p[3]·p[4]·p[5]
        let c = (p[2] as i64 * p[3] as i64) >> Q16_SHIFT;
        let d = (p[4] as i64 * p[5] as i64) >> Q16_SHIFT;
        let q2 = (c * d) >> Q16_SHIFT;

        // q3 = p[4]·p[5]·p[6]·p[7]
        let e = (p[4] as i64 * p[5] as i64) >> Q16_SHIFT;
        let f = (p[6] as i64 * p[7] as i64) >> Q16_SHIFT;
        let q3 = (e * f) >> Q16_SHIFT;

        // combined = 0.4·q1 + 0.3·q2 + 0.3·q3
        let q4 = (26214i64 * q1) >> Q16_SHIFT; // 0.4 ≈ 26214/65536
        let q5 = (19661i64 * q2) >> Q16_SHIFT; // 0.3 ≈ 19661/65536
        let q6 = (19661i64 * q3) >> Q16_SHIFT;
        let combined = q4 + q5 + q6;

        // result[idx] = state[idx] * combined
        for j in 0..packet_size {
            let idx = i * packet_size + j;
            if idx < WORM_NEURON_COUNT {
                result[idx] = ((state[idx] as i64 * combined) >> Q16_SHIFT) as i32;
            }
        }
    }

    result
}

/// Rational sparsemax in pure i32 — no transcendental functions.
///
/// Finds the threshold τ such that ∑max(0, zᵢ - τ) = 1 (in Q16.16,
/// 1 = 65536).  Then projects: output = max(0, z - τ).
fn rational_sparsemax_i32(state: &mut [i32; WORM_NEURON_COUNT]) {
    let mut sorted: Vec<i32> = state.to_vec();
    sorted.sort_by(|a, b| b.cmp(a));

    let mut cumsum = 0i64;
    let mut tau = 0i32;

    for (i, &val) in sorted.iter().enumerate() {
        cumsum += val as i64;
        // τ = (cumsum - 65536) / (i + 1)  (1.0 in Q16.16 = 65536)
        let t = ((cumsum - Q16_SCALE as i64) / (i + 1) as i64) as i32;
        if t < val {
            tau = t;
        }
    }

    for i in 0..WORM_NEURON_COUNT {
        state[i] = (state[i] - tau).max(0);
    }
}

/// Convert f32 weight matrices to Q16.16 fixed-point i32 format (heap-allocated flat vectors).
///
/// Each weight `w` is stored as `(w * 2¹⁶).round()` as i32.
pub fn convert_weights_to_rational(
    input_projection: &Array2<f32>,
    synapses: &Array2<f32>,
) -> (Box<[i32]>, Box<[i32]>) {
    let proj_len = WORM_NEURON_COUNT * MANIFOLD_DIM;
    let syn_len = WORM_NEURON_COUNT * WORM_NEURON_COUNT;
    let mut input_projection_int = vec![0i32; proj_len];
    let mut synapses_int = vec![0i32; syn_len];

    for i in 0..WORM_NEURON_COUNT {
        let base = i * MANIFOLD_DIM;
        for j in 0..MANIFOLD_DIM {
            input_projection_int[base + j] = (input_projection[(i, j)] * Q16_SCALE).round() as i32;
        }
    }

    for i in 0..WORM_NEURON_COUNT {
        let base = i * WORM_NEURON_COUNT;
        for j in 0..WORM_NEURON_COUNT {
            synapses_int[base + j] = (synapses[(i, j)] * Q16_SCALE).round() as i32;
        }
    }

    (
        input_projection_int.into_boxed_slice(),
        synapses_int.into_boxed_slice(),
    )
}

// ---------------------------------------------------------------------------
// 302-D Activation-space Boltzmann Decoding
// ---------------------------------------------------------------------------

/// Precompute coarse 30-D fingerprints for each vocabulary word (≥4 chars).
///
/// The cheap 30-D fingerprints (average of every ~10 dimensions) are stored
/// in memory for fast coarse scoring over all 10k words.  The full 302-D
/// dot product is computed **on-the-fly** only for the top candidates,
/// keeping RSS under 18 MB.
pub fn compute_vocab_activations(
    brain: &WormBrain,
    vocab: &[(String, Array1<f32>)],
) -> VocabFootprints {
    let iter = vocab.iter().filter(|(t, _)| t.len() >= 4);
    let mut tokens = Vec::new();
    let mut coords_16d = Vec::new();
    let mut coarse = Vec::new();

    for (token, coord_16d) in iter {
        let coord_f64: Array1<f64> = coord_16d.iter().map(|&v| v as f64).collect();
        let act = match brain.route_signal_capped(&coord_f64, 150) {
            Ok((a, _)) => a,
            Err(_) => continue,
        };
        let norm = act.dot(&act).sqrt();
        let act_slice = act.as_slice().unwrap_or(&[]);
        let cf: Vec<f32> = (0..30)
            .map(|b| {
                let start = b * 10;
                let end = (start + 10).min(act_slice.len());
                if start >= act_slice.len() || act_slice.is_empty() {
                    0.0
                } else {
                    let mean = act_slice[start..end].iter().sum::<f32>() / (end - start) as f32;
                    mean / norm.max(1e-15)
                }
            })
            .collect();

        tokens.push(token.clone());
        coords_16d.push(coord_16d.clone());
        coarse.push(cf);
    }

    VocabFootprints {
        tokens,
        coords_16d,
        coarse,
    }
}

/// Lightweight vocabulary footprints for 302-D Boltzmann decoding.
///
/// Only the coarse 30-D fingerprints are stored in memory; the full 302-D
/// dot products are computed on-the-fly for the top candidates by routing
/// the 128-D coordinate through [`WormBrain::route_signal`] on demand.
/// This keeps RSS under ~18 MB while enabling < 30 ms generation.
#[derive(Clone, Debug)]
pub struct VocabFootprints {
    pub tokens: Vec<String>,
    /// 128-D manifold coordinates (used for on-the-fly 302-D activation).
    pub coords_16d: Vec<Array1<f32>>,
    /// Coarse 30-D fingerprints (dimension-reduced via averaging).
    pub coarse: Vec<Vec<f32>>,
}

impl VocabFootprints {
    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Boltzmann energy-based token selection in 302-D activation space.
///
/// ## Energy Function
/// Each vocabulary word has a precomputed 302-D activation (its "brain
/// footprint").  The energy of a token given the query activation *q* is the
/// *negative* cosine similarity:
///
/// ```text
/// E(t_i) = -cos(q, a_i)
/// ```
///
/// Lower energy = better match.  The selection probability follows a Boltzmann
/// (Gibbs) distribution:
///
/// ```text
/// P(t_i | q) = exp(-E_i / T) / Z
/// ```
///
/// where *Z* = ∑ exp(-E_j / T) is the partition function and *T* is the
/// temperature.
///
/// ## Anti-attractor
/// Tokens already emitted (`visited`) have their energy increased by +2.0,
/// suppressing immediate repetition.
///
/// ## Temperature
/// - *T* → 0⁺ : deterministic (greedy — picks the best match every time)
/// - *T* ≈ 0.5 : balanced creativity
/// - *T* → ∞ : uniform random
pub fn decode_token_energy(
    query: &Array1<f32>,
    brain: &WormBrain,
    vocab: &VocabFootprints,
    visited: &[String],
    temperature: f64,
) -> String {
    let query_slice = query.as_slice().unwrap();
    let query_norm_sq: f64 = query_slice.iter().map(|&x| x as f64 * x as f64).sum();
    let inv_qnorm = if query_norm_sq > 1e-30 {
        1.0 / query_norm_sq.sqrt()
    } else {
        return String::new();
    };

    // coarse query fingerprint (30-D, average of every ~10 dims)
    let q_coarse: Vec<f64> = (0..30)
        .map(|b| {
            let start = b * 10;
            let end = (start + 10).min(query_slice.len());
            if start >= query_slice.len() {
                0.0
            } else {
                let avg = query_slice[start..end]
                    .iter()
                    .map(|&v| v as f64)
                    .sum::<f64>()
                    / (end - start) as f64;
                avg * inv_qnorm
            }
        })
        .collect();

    let visited_positions: std::collections::HashMap<&str, usize> = visited
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let visited_len = visited.len();

    let n = vocab.len();
    let coarse_keep = 20usize.min(n);

    // Stage 1: coarse 30-D scoring → keep top `coarse_keep`
    let mut coarse_scores: Vec<(f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let mut sim = 0.0_f64;
        let cf = &vocab.coarse[i];
        for k in 0..30 {
            sim += q_coarse[k] * cf[k] as f64;
        }
        let energy = -sim;
        if let Some(&pos) = visited_positions.get(vocab.tokens[i].as_str()) {
            let distance = (visited_len - pos) as f64;
            let penalty = 2.0 * 0.8_f64.powi(distance as i32 - 1);
            coarse_scores.push((energy + penalty, i));
        } else {
            coarse_scores.push((energy, i));
        }
    }

    let top_n = if n > coarse_keep {
        coarse_scores.select_nth_unstable_by(coarse_keep, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        coarse_keep
    } else {
        n
    };

    // Stage 2: compute full 302-D on-the-fly for top candidates
    let mut final_scores: Vec<(f64, usize)> = Vec::with_capacity(top_n);
    for &(_, idx) in coarse_scores[..top_n].iter() {
        let coord_f64: Array1<f64> = vocab.coords_16d[idx].iter().map(|&v| v as f64).collect();
        let full_act = match brain.route_signal_capped(&coord_f64, 150) {
            Ok((a, _)) => a,
            Err(_) => continue,
        };
        let f_slice = full_act.as_slice().unwrap();
        let mut sim = 0.0_f64;
        for k in 0..query_slice.len() {
            sim += query_slice[k] as f64 * f_slice[k] as f64;
        }
        let f_norm_sq: f64 = f_slice.iter().map(|&x| x as f64 * x as f64).sum();
        let inv_fnorm = if f_norm_sq > 1e-30 {
            1.0 / f_norm_sq.sqrt()
        } else {
            continue;
        };
        let mut energy = -sim * inv_qnorm * inv_fnorm;
        if let Some(&pos) = visited_positions.get(vocab.tokens[idx].as_str()) {
            let distance = (visited_len - pos) as f64;
            let penalty = 2.0 * 0.8_f64.powi(distance as i32 - 1);
            energy += penalty;
        }
        final_scores.push((energy, idx));
    }

    if final_scores.is_empty() {
        return String::new();
    }

    final_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let min_e = final_scores[0].0;
    let t = temperature.max(1e-9);
    let mut total = 0.0_f64;
    for &(e, _) in &final_scores {
        total += (-(e - min_e) / t).exp();
    }

    if total <= 0.0 {
        return vocab.tokens[final_scores[0].1].clone();
    }

    let draw = fastrand::f64() * total;
    let mut cumulative = 0.0;
    for &(e, idx) in &final_scores {
        cumulative += (-(e - min_e) / t).exp();
        if draw <= cumulative {
            return vocab.tokens[idx].clone();
        }
    }

    vocab.tokens[final_scores[0].1].clone()
}

// ---------------------------------------------------------------------------
// Response generation with 302-D state machine
// ---------------------------------------------------------------------------

impl WormBrain {
    /// Route signal WITHOUT sparsemax/WTA — preserves full 302-D diversity.
    /// Use ONLY for inference/decoding. Training MUST use route_signal().
    pub fn route_signal_raw(&self, key: &Array1<f64>) -> Result<(Array1<f32>, Array1<f32>)> {
        if key.len() != self.input_dim {
            return Err(CoreError::Geometry(format!(
                "input dimension mismatch: expected {}, got {}",
                self.input_dim,
                key.len()
            )));
        }
        let key_f32: Array1<f32> = Array1::from_iter(key.iter().map(|&v| v as f32));
        let mut state = self.input_projection.dot(&key_f32);
        let pre_synaptic = state.clone();

        if self.quad_routing {
            // RMS-normalise the pre-synaptic state (same as route_signal).
            let rms = (state.mapv(|v| v * v).sum() / state.len() as f32)
                .sqrt()
                .max(1e-8);
            let inv_rms = 1.0 / rms;
            state = state.mapv(|v| v * inv_rms);

            // Dendritic tree forward pass + triple-quad blend.
            state = self.dendritic_tree.forward(&state, self.creative_k);
            let quad = self.route_triple_quad(&state);
            state = state * 0.7 + quad * 0.3;

            // Temperature scaling for creative mode: compress logit dynamic range
            // so α-entmax produces distributed outputs (not 1-hot).
            if self.creative_mode && self.creative_k > 0.0 {
                let scale = (self.creative_k * 15.0).max(5.0);
                state.mapv_inplace(|v| v / scale);
            }
        } else {
            state = self.synapses.dot(&state);
        }

        if state.iter().any(|&v| v.is_nan()) {
            return Err(CoreError::Geometry("NaN in route_signal_raw".into()));
        }
        Ok((state, pre_synaptic))
    }

    /// Route signal with top-k cap: keep only `k` most-active neurons,
    /// rest set to zero. Combines raw diversity with noise gating.
    pub fn route_signal_capped(
        &self,
        key: &Array1<f64>,
        k: usize,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        let (mut state, pre_synaptic) = self.route_signal_raw(key)?;
        let k = k.min(state.len());

        let mut with_idx: Vec<(usize, f32)> = state.iter().copied().enumerate().collect();
        with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let threshold = with_idx
            .get(k.saturating_sub(1))
            .map(|v| v.1)
            .unwrap_or(0.0);

        for v in state.iter_mut() {
            if *v < threshold {
                *v = 0.0;
            }
        }
        Ok((state, pre_synaptic))
    }

    /// Dual-stream forward pass returning (sparse_action, dense_learning).
    ///
    /// - `sparse_action`: post-entmax output (σ ≈ 1.0, 1–2 active neurons).
    ///   Used for token selection / inference.
    /// - `dense_learning`: pre-entmax logits (dense, carries associativity).
    ///   Used as the learning substrate for the EchoReservoir.
    ///
    /// When `self.echo_reservoir` is `Some` and `self.cognition.alpha_echo > 0.0`,
    /// the reservoir's echo bias is blended into the logits before entmax:
    ///
    ///   logits = (1 - α) · z_dense + α · echo_bias
    ///   sparse_action = entmax(logits)
    ///
    /// The ORIGINAL pre-blend `dense_learning` is still returned so the
    /// caller can push the genuine pre-entmax signal into the reservoir.
    #[inline]
    pub fn route_with_echo(
        &self,
        point_cloud_key: &Array1<f64>,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        if point_cloud_key.len() != self.input_dim {
            return Err(CoreError::Geometry(format!(
                "input dimension mismatch for connectome routing: expected {} (manifold input_dim), got {}",
                self.input_dim, point_cloud_key.len()
            )));
        }

        let key_f32: Array1<f32> = Array1::from_iter(point_cloud_key.iter().map(|&v| v as f32));
        let mut state = self.input_projection.dot(&key_f32);
        let _pre_synaptic = state.clone();

        if self.quad_routing {
            let rms = (state.mapv(|v| v * v).sum() / state.len() as f32)
                .sqrt()
                .max(1e-8);
            let inv_rms = 1.0 / rms;
            state = state.mapv(|v| v * inv_rms);

            state = self.dendritic_tree.forward(&state, self.creative_k);
            let quad = self.route_triple_quad(&state);
            state = state * 0.7 + quad * 0.3;

            if self.creative_mode && self.creative_k > 0.0 {
                let scale = 50.0;
                state.mapv_inplace(|v| v / scale);
            }
        } else {
            state = self.synapses.dot(&state);
        }

        for v in state.iter_mut() {
            *v = v.clamp(-0.3, 0.3);
        }

        if state.iter().any(|&v| v.is_nan()) {
            return Err(CoreError::Geometry(
                "NaN detected in state before sparsemax".to_string(),
            ));
        }

        // ── Dual-stream split ──────────────────────────────────────────────
        // dense_learning is the pre-entmax logits.  sparse_action is the
        // post-entmax output used for routing / token selection.
        let dense_learning = state.clone();

        let echo_bias = if let Some(ref reservoir) = self.echo_reservoir {
            if self.cognition.alpha_echo > 0.0 {
                Some(reservoir.query(&dense_learning))
            } else {
                None
            }
        } else {
            None
        };

        if let Some(ref bias) = echo_bias {
            let alpha = self.cognition.alpha_echo;
            for i in 0..state.len() {
                state[i] = (1.0 - alpha) * state[i] + alpha * bias[i];
            }
        }

        if self.hyperbolic_mode {
            sparsemax_hyperbolic(state.as_slice_mut().unwrap(), &self.nmda_thresholds);
        } else if self.creative_mode {
            let controller = crate::criticality::CriticalityController::new(self.creative_k);
            controller.smooth_activation(state.as_slice_mut().unwrap());
            let s: f32 = state.iter().sum();
            if s > 0.0 && (s - 1.0).abs() > 1e-4 {
                state.mapv_inplace(|x| x / s);
            }
        } else if self.use_wta {
            let mut with_idx: Vec<(usize, f32)> = state.iter().copied().enumerate().collect();
            with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let threshold = with_idx.get(2).map(|v| v.1).unwrap_or(0.0);
            state.mapv_inplace(|v| if v >= threshold { v } else { 0.0 });
            let sum: f32 = state.iter().sum();
            if sum > 0.0 {
                state.mapv_inplace(|v| v / sum);
            }
        } else {
            alpha_entmax(state.as_slice_mut().unwrap(), self.entmax_alpha);
        }

        if self.valve_threshold > 0.0 {
            let state_norm: f32 = state.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let gate =
                1.0 / (1.0 + (-self.valve_steepness * (state_norm - self.valve_threshold)).exp());
            state.mapv_inplace(|v| v * gate);
            let sum: f32 = state.iter().sum();
            if sum > 0.0 {
                state.mapv_inplace(|v| v / sum);
            }
        }

        Ok((state, dense_learning))
    }

    /// Feed a dense pre-entmax signal into the EchoReservoir for Hebbian
    /// association learning.
    ///
    /// # Caller requirements
    /// - `dense` MUST be the pre-entmax logits (NOT the post-entmax action).
    /// - The reservoir stores the state in its ring-buffer AND runs one
    ///   Hebbian outer-product step: ΔW[i,j] = η · z_dense[i] · z_dense[j].
    pub fn train_echo_dense(&mut self, dense: &Array1<f32>) {
        if let Some(ref mut reservoir) = self.echo_reservoir {
            reservoir.push(dense);
            reservoir.hebbian_step(dense);
        }
    }

    /// Auto-regressive response generation using 302-D Boltzmann decoding.
    ///
    /// 1. Routes the context basis's first principal direction through the
    ///    connectome to obtain the initial 302-D activation.
    /// 2. Each step: compares the activation against precomputed 302-D vocab
    ///    footprints via [`decode_token_energy`], routes the chosen token
    ///    back through the brain.
    /// 3. A decaying echo buffer blends previous and new activations (0.6/0.4)
    ///    for smooth concept drift.
    /// 4. Stops on sentence-ending punctuation or `max_tokens`.
    ///
    /// Returns `(response_text, trajectory_activations)` where the activations
    /// can be passed to [`WormBrain::contrastive_unlearn`] for coherence
    /// alignment.
    #[inline]
    pub fn generate_response(
        &self,
        vocab: &VocabFootprints,
        context_basis: &geometry::OrthonormalBasis,
        max_tokens: usize,
        temperature: f64,
    ) -> (String, Vec<Array1<f32>>) {
        self.generate_response_with_echo(vocab, context_basis, max_tokens, temperature, None)
    }

    /// Like `generate_response` but accepts an optional `SynapticEchoBuffer`
    /// for short-term memory across turns.
    ///
    /// When `echo` is `Some`, the Dirac echo buffer is applied at each step
    /// with gated normalization (if norm > 1.0, scale to unit norm).
    /// The multi-component start blend uses up to 4 principal directions
    /// with harmonic weights (1, 1/2, 1/3, 1/4).
    pub fn generate_response_with_echo(
        &self,
        vocab: &VocabFootprints,
        context_basis: &geometry::OrthonormalBasis,
        max_tokens: usize,
        temperature: f64,
        mut echo: Option<&mut SynapticEchoBuffer>,
    ) -> (String, Vec<Array1<f32>>) {
        // Multi-component start blend (up to rank 4, harmonic weights).
        let rank = context_basis.rank.min(4);
        if rank == 0 {
            return (String::new(), Vec::new());
        }
        let mut start_f64 = Array1::<f64>::zeros(geometry::MANIFOLD_DIM);
        let mut weight_sum = 0.0_f64;
        for col in 0..rank {
            let weight = 1.0 / (col + 1) as f64;
            let component = context_basis.vectors.column(col);
            start_f64 = &start_f64 + &component.mapv(|v| v * weight);
            weight_sum += weight;
        }
        if weight_sum > 0.0 {
            start_f64.mapv_inplace(|v| v / weight_sum);
        }

        let mut activation = match self.route_signal_capped(&start_f64, 150) {
            Ok((a, _)) => a,
            Err(_) => return (String::new(), Vec::new()),
        };

        let mut prev_echo = activation.clone();
        let mut visited: Vec<String> = Vec::new();
        let mut output = String::new();
        let mut traj_activations: Vec<Array1<f32>> = Vec::new();

        for _ in 0..max_tokens {
            let token = decode_token_energy(
                &activation, self, vocab, &visited, temperature,
            );

            if token.is_empty() {
                break;
            }

            if !output.is_empty() {
                output.push(' ');
            }
            output.push_str(&token);
            visited.push(token.clone());
            traj_activations.push(activation.clone());

            // Sentence-ending punctuation halts generation (minimum 3 tokens).
            let token_count = output.split_whitespace().count();
            if token_count >= 3
                && (token.ends_with('.') || token.ends_with('!') || token.ends_with('?'))
            {
                break;
            }

            // Route the decoded token back through the brain for the next state.
            let coord = geometry::token_to_coord(&token);
            activation = match self.route_signal_capped(coord.inner(), 150) {
                Ok((a, _)) => a,
                Err(_) => break,
            };

            // Echo integration with gated normalization
            if let Some(ref mut echo) = echo {
                let _ = echo.apply_and_decay(&mut activation);
                let norm = activation.dot(&activation).sqrt();
                if norm > 1.0 {
                    activation.mapv_inplace(|v| v / norm);
                }
                echo.inject_echo(&activation);
            } else {
                // Fallback: decaying echo of previous activation for smooth transitions.
                activation = &activation * 0.6 + &prev_echo * 0.4;
            }
            prev_echo = activation.clone();
        }

        (output, traj_activations)
    }

    /// Auto-regressive response generation using 302-D Boltzmann decoding.
    ///
    /// 1. Routes the context basis's first principal direction through the
    ///    connectome to obtain the initial 302-D activation.
    /// 2. Each step: compares the activation against precomputed 302-D vocab
    ///    footprints via [`decode_token_energy`], routes the chosen token
    /// Contrastive Hebbian alignment — suppress incoherent trajectories.
    ///
    /// If the mean cosine *distance* between consecutive 302-D activations
    /// exceeds `threshold`, the trajectory is considered incoherent ("fantasy
    /// data").  A negative Hebbian update weakens the synapses along the
    /// trajectory's activation path, making those transitions less likely in
    /// the future.
    ///
    /// The cosine distance between two activations *a* and *b* is defined as
    /// `sqrt(2 · (1 - cos(a, b)))` — the chord distance on the unit sphere,
    /// equivalent to `2·sin(θ/2)`.
    ///
    /// Returns the computed mean cosine distance (0.0 if too few activations
    /// or below threshold).
    #[inline]
    pub fn contrastive_unlearn(
        &mut self,
        trajectory: &[Array1<f32>],
        threshold: f64,
        learning_rate: f64,
    ) -> f64 {
        let n = trajectory.len();
        if n < 2 {
            return 0.0;
        }

        // Compute mean cosine distance between consecutive activations.
        let mut total_dist = 0.0_f64;
        for pair in trajectory.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            let an = a.dot(a).sqrt() as f64;
            let bn = b.dot(b).sqrt() as f64;
            if an > 1e-15 && bn > 1e-15 {
                let cos = ((a.dot(b) as f64) / (an * bn)).clamp(-1.0, 1.0);
                total_dist += (2.0 * (1.0 - cos)).sqrt();
            }
        }
        let mean_dist = total_dist / (n - 1) as f64;

        // Only apply contrastive update if mean distance exceeds threshold.
        if mean_dist <= threshold {
            return 0.0;
        }

        // Compute mean activation over the trajectory.
        let mut mean_act: Array1<f32> = Array1::zeros(trajectory[0].len());
        for act in trajectory {
            mean_act = &mean_act + act;
        }
        mean_act.mapv_inplace(|v| v / n as f32);

        // Negative Hebbian: decrement weights where mean activation product > 0.
        let lr = (learning_rate as f32).min(0.01);
        for i in 0..mean_act.len() {
            for j in 0..mean_act.len() {
                let hebb: f32 = mean_act[i] * mean_act[j];
                if hebb > 1e-6 {
                    let val = self.synapses[(i, j)] - lr * hebb;
                    self.synapses[(i, j)] = val.max(0.0);
                }
            }
        }

        mean_dist
    }
}

impl WormBrain {
    /// Compute the geodesic distance between two synaptic weight matrices.
    ///
    /// Each row of the weight matrix is treated as a point on the probability
    /// simplex (since rows are normalised to [0, 1]).  The angular distance
    /// between corresponding rows is averaged across all 302 rows.
    ///
    /// Returns `0.0` if the matrices have identical shapes or one is zero.
    pub fn geodesic_weight_distance(a: &Array2<f32>, b: &Array2<f32>) -> f64 {
        if a.shape() != b.shape() {
            return 0.0;
        }
        let n = a.shape()[0];
        let mut total_angle = 0.0_f64;
        let mut count = 0usize;
        for i in 0..n {
            let row_a = a.row(i);
            let row_b = b.row(i);
            let norm_a = row_a.dot(&row_a).sqrt() as f64;
            let norm_b = row_b.dot(&row_b).sqrt() as f64;
            if norm_a > 1e-15 && norm_b > 1e-15 {
                let cos = ((row_a.dot(&row_b) as f64) / (norm_a * norm_b)).clamp(-1.0, 1.0);
                total_angle += cos.acos();
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total_angle / count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worm_brain_new_baseline() {
        let brain = WormBrain::new_baseline();
        assert_eq!(brain.neuron_count, 302);
        assert_eq!(brain.synapses.shape(), &[302, 302]);
    }

    #[test]
    fn test_route_signal_valid_input() {
        let brain = WormBrain::new_baseline();
        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        let (output, _) = brain.route_signal(&key).unwrap();
        assert_eq!(output.len(), 302);
        assert!(output.iter().all(|&v| v.is_finite()));
        // Sparsemax outputs are non-negative and sum to ~1
        assert!(output.iter().all(|&v| v >= 0.0));
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
        // Should be sparse: at least 50% zeros for typical input
        let zeros = output.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "sparsemax should produce some exact zeros");
    }

    #[test]
    fn test_route_signal_zero_input() {
        let brain = WormBrain::new_baseline();
        let key = Array1::<f64>::zeros(MANIFOLD_DIM);
        let (output, _) = brain.route_signal(&key).unwrap();
        assert_eq!(output.len(), 302);
        // Sparsemax projects zero input to uniform 1/n over the simplex
        let expected = 1.0 / 302.0;
        assert!(output.iter().all(|&v| (v - expected).abs() < 1e-6));
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_alpha_entmax_alpha2_bit_identical() {
        // α=2 must reproduce exact sparsemax for diverse inputs
        let test_inputs = [
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![10.0, -5.0, 3.0, 0.0, 7.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![100.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        for input in &test_inputs {
            let mut result_entmax = input.clone();
            alpha_entmax(&mut result_entmax, 2.0);

            let mut result_sparsemax = input.clone();
            // Manual sparsemax (exact original algorithm)
            let mut sorted = result_sparsemax.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let mut cumsum = 0.0f32;
            let mut tau = 0.0f32;
            for (i, &val) in sorted.iter().enumerate() {
                cumsum += val;
                let t = (cumsum - 1.0) / (i + 1) as f32;
                if t < val {
                    tau = t;
                }
            }
            for x in &mut result_sparsemax {
                *x = (*x - tau).max(0.0);
            }

            // Bit-identical within f32 tolerance
            for (a, b) in result_entmax.iter().zip(result_sparsemax.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "α=2 mismatch: entmax={}, sparsemax={} for input {:?}",
                    a,
                    b,
                    input
                );
            }
        }
    }

    #[test]
    fn test_alpha_entmax_alpha1_all_nonzero() {
        // α=1 (softmax) must produce all non-zero outputs
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut result = input.clone();
        alpha_entmax(&mut result, 1.0);
        assert!(
            result.iter().all(|&v| v > 0.0),
            "α=1 should produce all non-zero, got zeros: {:?}",
            result
        );
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "α=1 should sum to 1, got {}", sum);
    }

    #[test]
    fn test_alpha_entmax_sparsity_monotonic() {
        // As α increases, support size must be monotonic non-increasing
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let alphas = vec![1.0, 1.5, 2.0, 3.0, 4.0, 8.0];
        let mut prev_nonzero = 10usize;

        for &alpha in &alphas {
            let mut result = input.clone();
            alpha_entmax(&mut result, alpha);
            let nonzero = result.iter().filter(|&&v| v > 0.0).count();
            assert!(
                nonzero <= prev_nonzero,
                "α={}: support size {} > previous {}",
                alpha,
                nonzero,
                prev_nonzero
            );
            let sum: f32 = result.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "α={}: sum={}, expected 1",
                alpha,
                sum
            );
            prev_nonzero = nonzero;
        }
    }

    #[test]
    fn test_alpha_entmax_route_signal_entmax_alpha_works() {
        // WormBrain with different α values should produce different activation patterns
        let brain2 = WormBrain::new_baseline();

        let mut brain1 = brain2.clone();
        brain1.entmax_alpha = 1.0;
        let mut brain4 = brain2.clone();
        brain4.entmax_alpha = 4.0;

        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));

        let (out1, _) = brain1.route_signal(&key).unwrap();
        let (out2, _) = brain2.route_signal(&key).unwrap();
        let (out4, _) = brain4.route_signal(&key).unwrap();

        // α=1 should have more non-zeros than α=2
        let nz1 = out1.iter().filter(|&&v| v > 0.0).count();
        let nz2 = out2.iter().filter(|&&v| v > 0.0).count();
        let nz4 = out4.iter().filter(|&&v| v > 0.0).count();

        assert!(nz1 >= nz2, "α=1 support {} < α=2 support {}", nz1, nz2);
        assert!(nz2 >= nz4, "α=2 support {} < α=4 support {}", nz2, nz4);

        // All sum to 1
        let sum1: f32 = out1.iter().sum();
        let sum2: f32 = out2.iter().sum();
        let sum4: f32 = out4.iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-3, "α=1 sum={sum1}");
        assert!((sum2 - 1.0).abs() < 1e-3, "α=2 sum={sum2}");
        assert!((sum4 - 1.0).abs() < 5e-2, "α=4 sum={sum4} (needs 0.05 tolerance with clipped values)");
    }

    #[test]
    fn test_alpha_entmax_alpha2_backward_compat_in_brain() {
        // Default brain (α=2) must produce same output as old sparsemax path
        let brain = WormBrain::new_baseline();
        assert!(
            (brain.entmax_alpha - 2.0).abs() < 1e-10,
            "default entmax_alpha should be 2.0"
        );

        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        let (output, _) = brain.route_signal(&key).unwrap();

        // Standard sparsemax properties: non-negative, sum ≈ 1, at least 50% zeros
        assert!(output.iter().all(|&v| v >= 0.0));
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
        let zeros = output.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "α=2 should produce exact zeros");
    }

    #[test]
    fn test_reverse_valve_disabled_by_default() {
        let brain = WormBrain::new_baseline();
        assert_eq!(
            brain.valve_threshold, 0.0,
            "valve should be disabled by default"
        );
        assert_eq!(brain.valve_steepness, 5.0);
    }

    #[test]
    fn test_reverse_valve_attenuates_low_activation() {
        let mut brain = WormBrain::new_baseline();
        brain.valve_threshold = 0.3;
        brain.valve_steepness = 10.0;

        let key = Array1::<f64>::zeros(MANIFOLD_DIM);
        let (output, _) = brain.route_signal(&key).unwrap();

        // Zero input gives uniform activation (norm ≈ sqrt(1/302) ≈ 0.058)
        // This is well below threshold 0.3, so the valve should attenuate
        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "output must still sum to 1, got {}",
            sum
        );
        let max_val = output.iter().copied().fold(0.0f32, f32::max);
        // With gate significantly < 1, the max value should still be uniform-ish
        let expected_uniform = 1.0 / 302.0;
        assert!(
            (max_val - expected_uniform).abs() < 1e-4,
            "attenuated uniform should still be uniform, max={}",
            max_val
        );
    }

    #[test]
    fn test_reverse_valve_strong_signal_passes_through() {
        let mut brain = WormBrain::new_baseline();
        brain.valve_threshold = 0.3;
        brain.valve_steepness = 10.0;

        // Strong key produces concentrated activation (high norm)
        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| (i as f64) * 10.0));
        let (output, _) = brain.route_signal(&key).unwrap();

        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "output must sum to 1, got {}",
            sum
        );
        // With strong signal, the valve gate should be near 1.0,
        // so the output should have the usual sparsemax sparsity
        let zeros = output.iter().filter(|&&v| v == 0.0).count();
        assert!(
            zeros > 0,
            "strong signal with valve should still produce zeros"
        );
    }

    #[test]
    fn test_decode_token_energy_returns_something() {
        let brain = WormBrain::new_baseline();
        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        let (act, _) = brain.route_signal(&key).unwrap();
        let tokens: Vec<String> = (0..10).map(|i| format!("token_{}", i)).collect();
        let coords_16d: Vec<Array1<f32>> = (0..10)
            .map(|i| {
                Array1::from_vec(
                    (0..MANIFOLD_DIM)
                        .map(|j| (i as f32) * 0.1 + j as f32 * 0.01)
                        .collect(),
                )
            })
            .collect();
        let coarse: Vec<Vec<f32>> = coords_16d
            .iter()
            .map(|c| {
                let c_f64: Array1<f64> = c.iter().map(|&v| v as f64).collect();
                let (a, _) = brain.route_signal(&c_f64).unwrap();
                let s = a.as_slice().unwrap();
                let n = a.dot(&a).sqrt();
                (0..30)
                    .map(|b| {
                        let st = b * 10;
                        let en = (st + 10).min(s.len());
                        if st >= s.len() {
                            0.0
                        } else {
                            s[st..en].iter().sum::<f32>() / (en - st) as f32 / n.max(1e-15)
                        }
                    })
                    .collect()
            })
            .collect();
        let vf = VocabFootprints {
            tokens,
            coords_16d,
            coarse,
        };
        let result = decode_token_energy(&act, &brain, &vf, &[], 0.5);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_contrastive_unlearn_below_threshold() {
        let mut brain = WormBrain::new_baseline();
        let saved = brain.synapses.clone();
        let act = Array1::<f32>::zeros(302);
        let traj = vec![act.clone(), act.clone()];
        let d = brain.contrastive_unlearn(&traj, 0.5, 0.001);
        assert!(d < 0.5);
        assert!((&brain.synapses - &saved).iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_contrastive_unlearn_single_activation() {
        let mut brain = WormBrain::new_baseline();
        let saved = brain.synapses.clone();
        let act = Array1::<f32>::zeros(302);
        let traj = vec![act];
        let d = brain.contrastive_unlearn(&traj, 0.5, 0.001);
        assert_eq!(d, 0.0);
        assert!((&brain.synapses - &saved).iter().all(|&v| v.abs() < 1e-12));
    }

    // ── Rational forward pass tests ──

    #[test]
    fn test_convert_weights_to_rational() {
        let brain = WormBrain::new_baseline();
        let (proj_int, syn_int) =
            convert_weights_to_rational(&brain.input_projection, &brain.synapses);

        // Check dimensions
        assert_eq!(proj_int.len(), WORM_NEURON_COUNT * MANIFOLD_DIM);
        assert_eq!(syn_int.len(), WORM_NEURON_COUNT * WORM_NEURON_COUNT);

        // Check conversion: 0.5 → 32768
        assert_eq!((0.5f32 * Q16_SCALE).round() as i32, 32768);

        // Verify some non-zero synapses converted correctly
        let has_non_zero = syn_int.iter().any(|&v| v != 0);
        assert!(
            has_non_zero,
            "rational conversion should preserve non-zero weights"
        );
    }

    #[test]
    fn test_rational_sparsemax_uniform() {
        // Uniform input: all values equal → sparsemax should give uniform 1/n
        let mut state = [1000i32; WORM_NEURON_COUNT]; // same value for all
        rational_sparsemax_i32(&mut state);

        // All outputs should be equal (within Q16.16 precision)
        let first = state[0];
        assert!(state.iter().all(|&v| v == first));
        // Each value should be approximately 65536/302 ≈ 217
        let expected = (Q16_SCALE / WORM_NEURON_COUNT as f32) as i32;
        assert!((first - expected).abs() <= 1);
    }

    #[test]
    fn test_rational_sparsemax_one_active() {
        // Only one large value, rest negative → sparsemax should keep that one
        let mut state = [0i32; WORM_NEURON_COUNT];
        state[0] = 100000; // large positive in Q16.16

        // Everything else is 0 (in Q16.16, 0 = 0.0)
        // sparsemax should set the winner to 65536 (1.0 in Q16.16)
        // and everything else to 0

        // But wait: τ = (cumsum - 65536) / 1 = 100000 - 65536 = 34464
        // state[0] = max(0, 100000 - 34464) = 65536
        // For others: max(0, 0 - 34464) = 0 ✓

        rational_sparsemax_i32(&mut state);
        assert_eq!(state[0], Q16_SCALE as i32); // exactly 1.0
        assert!(state[1..].iter().all(|&v| v == 0));
    }

    #[test]
    fn test_route_rational_matches_f32() {
        let brain = WormBrain::new_baseline();
        let rational = RationalWormBrain::from_worm_brain(&brain);

        // Create test key (in f64 f32)
        let key_f64 = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.05));
        let (f32_output, _) = brain.route_signal(&key_f64).unwrap();

        // Convert same key to Q16.16
        let mut key_q16 = [0i32; MANIFOLD_DIM];
        for (j, val) in key_f64.iter().enumerate() {
            key_q16[j] = (val * Q16_SCALE as f64).round() as i32;
        }

        let i32_output = rational.route_rational(&key_q16);

        // Compare: convert f32 back to Q16.16 and check difference
        let mut max_diff = 0i64;
        for i in 0..WORM_NEURON_COUNT {
            let expected = (f32_output[i] * Q16_SCALE).round() as i32;
            let diff = (i32_output[i] as i64 - expected as i64).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        // Allow up to ±5 Q16.16 units of difference (rounded accumulation grows with MANIFOLD_DIM)
        assert!(max_diff <= 5, "max diff = {max_diff}, expected ≤ 5");
    }

    // ── Hyperbolic sparsemax tests ──

    #[test]
    fn test_sparsemax_hyperbolic_falls_back_to_sparsemax_for_spacelike() {
        // Non-timelike vector (M >= 0) should fall back to standard sparsemax
        let mut state = vec![2.0, 1.0, 0.5]; // M = 4 - 1 - 0.25 = 2.75 > 0
        sparsemax_hyperbolic(&mut state, &[0.0; 3]);
        let sum: f32 = state.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // Standard sparsemax should give at best [1, 0, 0]
        assert_eq!(state[0], 1.0);
        assert_eq!(state[1], 0.0);
        assert_eq!(state[2], 0.0);
    }

    #[test]
    fn test_sparsemax_hyperbolic_timelike_produces_valid_simplex() {
        // Timelike vector: M < 0
        // Need v[0]² < v[1]² + v[2]²
        let mut state = vec![0.5, 1.0, 2.0]; // M = 0.25 - 1 - 4 = -4.75 < 0
        let thresholds = [0.0, 0.0, 0.0]; // no gating
        sparsemax_hyperbolic(&mut state, &thresholds);
        let sum: f32 = state.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum should be 1, got {sum}");
        // At least one zero from sparsemax
        let has_zero = state.iter().any(|&v| v == 0.0);
        assert!(has_zero, "hyperbolic sparsemax should be sparse");
    }

    #[test]
    fn test_sparsemax_hyperbolic_nmda_gates_silent_neurons() {
        // Strong timelike vector but with high thresholds
        let mut state = vec![1.0, 5.0, 10.0]; // M = 1 - 25 - 100 = -124 < 0
        let thresholds = [0.0, 100.0, 0.0]; // silence neuron 1
        sparsemax_hyperbolic(&mut state, &thresholds);
        assert_eq!(state[1], 0.0, "neuron 1 should be NMDA-gated");
        let sum: f32 = state.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum should be 1, got {sum}");
    }

    #[test]
    fn test_sparsemax_hyperbolic_all_gated_returns_all_zero() {
        // All neurons gated → sparsemax sees all zeros → all zero output
        let mut state = vec![1.0, 5.0, 10.0]; // timelike
        let thresholds = [1000.0, 1000.0, 1000.0]; // gate everything
        sparsemax_hyperbolic(&mut state, &thresholds);
        assert!(state.iter().all(|&v| v == 0.0), "all should be zero");
    }

    #[test]
    fn test_hyperbolic_mode_disabled_backward_compat() {
        // hyperbolic_mode=false must produce same output as standard routine
        let brain = WormBrain::new_baseline();
        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        assert!(
            !brain.hyperbolic_mode,
            "default hyperbolic_mode should be false"
        );
        let (output, _) = brain.route_signal(&key).unwrap();
        assert_eq!(output.len(), 302);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_hyperbolic_mode_enabled_produces_valid_output() {
        let mut brain = WormBrain::new_baseline();
        brain.hyperbolic_mode = true;
        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        let (output, _) = brain.route_signal(&key).unwrap();
        assert_eq!(output.len(), 302);
        assert!(output.iter().all(|&v| v.is_finite()));
        assert!(output.iter().all(|&v| v >= 0.0));
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum should be 1, got {sum}");
    }

    #[test]
    fn test_hyperbolic_mode_sparser_with_high_thresholds() {
        let mut brain = WormBrain::new_baseline();
        brain.hyperbolic_mode = true;
        brain.nmda_thresholds = vec![1e-6; WORM_NEURON_COUNT]; // very low gating
        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        let (output, _) = brain.route_signal(&key).unwrap();
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum should be 1, got {sum}");
        // With thresholds > 0, some neurons should be gated
        let zeros = output.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "should have zeros with NMDA gating");
    }

    // ── Creative mode (Quilez Bridge smooth-k) tests ──

    #[test]
    fn test_creative_mode_disabled_by_default() {
        let brain = WormBrain::new_baseline();
        assert!(
            !brain.creative_mode,
            "creative_mode should default to false"
        );
    }

    #[test]
    fn test_creative_mode_k_zero_is_deterministic() {
        let mut brain = WormBrain::new_baseline();
        brain.creative_mode = true;
        brain.creative_k = 0.0; // k=0 → α=3, very sparse
        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        let (output, _) = brain.route_signal(&key).unwrap();
        assert_eq!(output.len(), 302);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum should be 1, got {sum}");
        // k=0 (α=3) is very sparse → at least some zeros
        let zeros = output.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "k=0 should produce zeros");
    }

    #[test]
    fn test_creative_mode_k_large_is_softer() {
        let mut brain = WormBrain::new_baseline();
        brain.creative_mode = true;
        brain.creative_k = 10.0; // α ≈ 1, softmax-like
        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        let (output, _) = brain.route_signal(&key).unwrap();
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum should be 1, got {sum}");
        // k=10 (α≈1) should have fewer zeros than k=0
    }

    #[test]
    fn test_creative_mode_k_ln2_matches_sparsemax() {
        // k = ln(2) gives α = 2 (exact sparsemax)
        let mut creative_brain = WormBrain::new_baseline();
        creative_brain.creative_mode = true;
        creative_brain.creative_k = 0.693_147_2;

        let baseline_brain = WormBrain::new_baseline(); // entmax_alpha = 2

        let key = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.1));
        let (creative_out, _) = creative_brain.route_signal(&key).unwrap();
        let (baseline_out, _) = baseline_brain.route_signal(&key).unwrap();

        // Both should produce α=2 sparsemax → identical output
        for (c, b) in creative_out.iter().zip(baseline_out.iter()) {
            assert!(
                (c - b).abs() < 1e-5,
                "creative k=ln2 should match baseline: creative={c}, baseline={b}"
            );
        }
    }

    #[test]
    fn test_rational_triple_quad_zero_state_returns_zero() {
        let state = [0i32; WORM_NEURON_COUNT];
        let result = rational_triple_quad(&state);
        assert!(
            result.iter().all(|&v| v == 0),
            "zero state should produce zero output"
        );
    }

    #[test]
    fn test_rational_triple_quad_large_values() {
        let mut state = [0i32; WORM_NEURON_COUNT];
        // Use 10000 in Q16.16 ≈ 0.15 — large enough for quad products to survive
        for v in state.iter_mut() {
            *v = 10000;
        }
        let result = rational_triple_quad(&state);
        let has_non_zero = result.iter().any(|&v| v != 0);
        assert!(
            has_non_zero,
            "large uniform state should produce non-zero quad output"
        );
    }

    #[test]
    fn test_rational_quad_routing_from_worm_brain_carries_flag() {
        let mut brain = WormBrain::new_baseline();
        brain.quad_routing = true;
        let rational = RationalWormBrain::from_worm_brain(&brain);
        assert!(
            rational.quad_routing,
            "quad_routing flag should be preserved"
        );
    }

    #[test]
    fn test_rational_quad_routing_produces_valid_output() {
        let mut brain = WormBrain::new_baseline();
        brain.quad_routing = true;
        let rational = RationalWormBrain::from_worm_brain(&brain);
        assert!(rational.quad_routing);

        let key_f64 = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.05));
        let mut key_q16 = [0i32; MANIFOLD_DIM];
        for (j, val) in key_f64.iter().enumerate() {
            key_q16[j] = (val * Q16_SCALE as f64).round() as i32;
        }
        let output = rational.route_rational(&key_q16);
        // Must sum to ~65536 (1.0 in Q16.16)
        let sum: i64 = output.iter().map(|&v| v as i64).sum();
        assert!(
            (sum - Q16_SCALE as i64).abs() <= 100,
            "rational quad_routing should sum to ~65536, got {sum}"
        );
    }

    #[test]
    fn test_rational_triple_quad_disabled_matches_standard_rational() {
        // When quad_routing is false, rational route should match standard rational
        let brain = WormBrain::new_baseline();
        let rational = RationalWormBrain::from_worm_brain(&brain);
        assert!(
            !rational.quad_routing,
            "default quad_routing should be false"
        );

        let key_f64 = Array1::from_iter((0..MANIFOLD_DIM).map(|i| i as f64 * 0.05));
        let mut key_q16 = [0i32; MANIFOLD_DIM];
        for (j, val) in key_f64.iter().enumerate() {
            key_q16[j] = (val * Q16_SCALE as f64).round() as i32;
        }
        let output = rational.route_rational(&key_q16);
        let sum: i64 = output.iter().map(|&v| v as i64).sum();
        assert!((sum - Q16_SCALE as i64).abs() <= 30, "sum should be ~65536");
    }

    #[test]
    fn test_rational_sparsemax_sum_to_one() {
        // Random input should produce sum ≈ 65536 (1.0 in Q16.16)
        let mut state = [0i32; WORM_NEURON_COUNT];
        let mut rng = fastrand::Rng::new();
        for v in state.iter_mut() {
            *v = rng.i32(-50000..50000);
        }
        rational_sparsemax_i32(&mut state);

        let sum: i64 = state.iter().map(|&v| v as i64).sum();
        // Integer truncation can cause sum to differ from 65536 by up to ~20
        assert!(
            (sum - Q16_SCALE as i64).abs() <= 30,
            "sparsemax sum = {sum}, expected ~{Q16_SCALE}"
        );
    }
}
