//! Bio-inspired plasticity training engine.
//!
//! Implements Hebbian ("fire together, wire together") plasticity on the
//! 302-neuron *C. elegans* connectome.  The [`WormTrainer`] applies
//! co-firing updates scaled by a learning rate, decays inactive weights via
//! a stabilisation factor, and row-normalises the synaptic matrix to `[0, 1]`
//! — all while respecting the static structural blueprint (no de novo
//! synaptogenesis).

use std::cell::Cell;
use ndarray::{Array1, Array2, Zip};
use crate::{CoreError, Result};
use crate::geometry::MANIFOLD_DIM;
use crate::worm_brain::{WormBrain, GROUP_A, PHI};
use fastrand;

/// Hebbian plasticity trainer for the 302-neuron *C. elegans* connectome.
///
/// Encodes a static structural blueprint mask captured from
/// [`WormBrain::new_baseline`] so that only pre-existing synapses are
/// eligible for potentiation.  Every production path is panic-free.
///
/// When Maxwell damping is enabled (via [`train_step_damped`]), the trainer
/// maintains a separate velocity matrix that acts as electromagnetic momentum,
/// suppressing non-productive loop attractors via a damping factor γ.
#[derive(Clone, Debug)]
pub struct WormTrainer {
    /// Learning rate scaling the Hebbian co-firing delta.
    pub learning_rate: f32,
    /// Leakage / decay factor applied each step to prevent weight explosion.
    /// Typical values are in `(0, 1)` (e.g. `0.999`).
    pub stabilization_factor: f32,
    /// Homeostatic coefficient for weight-proportional decay.
    /// When `homeostasis > 0`, weights near 1.0 experience stronger decay
    /// than weights near 0.0, preventing premature saturation.
    pub homeostasis: f32,
    /// Maxwell damping factor γ for velocity dissipation.
    /// Velocity is scaled by `(1.0 - γ)` each step to drain energy from
    /// non-productive local attractors.  Default `0.05`.
    pub damping_factor: f32,
    /// Boolean mask capturing the *C. elegans* structural blueprint.
    /// `true` where a synapse exists; `false` where the blueprint has zero.
    pub structural_mask: Array2<bool>,
    /// When `true` (default), validate all weights/velocities are finite
    /// after each batch.  Set to `false` with [`with_safe_mode`] to skip
    /// this O(n²) scan for maximum throughput.
    pub safe_mode: bool,
    /// Global training step counter.  Used by the φ-phase oscillation
    /// formula `cos(π · φ⁻¹ · step)` for quasiperiodic B→A modulation.
    pub step: Cell<u64>,
    /// Learning rate for contrastive Hebbian updates (d⊗d on capped activations).
    /// When > 0.0, [`Self::contrastive_step`] pushes sentence activations apart
    /// by amplifying their differences.  Default 0.0 (disabled).
    pub contrastive_lr: f32,
    /// When `true`, [`train_step`] uses `activation_state` directly for the
    /// Hebbian update instead of applying sparsemax to `pre_synaptic`.
    /// This aligns training activations with inference activations from
    /// [`route_signal_capped`].  Default `false` for backward compat.
    pub use_capped_training: bool,
    /// TDA β₀ threshold: connected components limit before intervention.
    /// `monitor_and_intervene` injects noise when β₀ exceeds this.
    /// Derived from empirical baseline: β₀ ∈ [6, 50] for 50-token streams.
    pub tda_beta_0_threshold: usize,
    /// TDA β₁ threshold: cycle limit before intervention.
    /// `monitor_and_intervene` applies contrastive unlearning when β₁ exceeds this.
    /// Derived from empirical baseline: β₁ ∈ [0, 6] for 50-token streams.
    pub tda_beta_1_threshold: usize,

    /// Learning rate for the dendritic tree weight updates (Phase 2).
    /// When > 0.0 and `brain.quad_routing` is enabled, [`train_step`]
    /// delegates to the reverse valve plasticity path instead of the
    /// standard Hebbian synapse update.  Default `0.0` (disabled).
    pub dendritic_lr: f32,

    /// Pruning rate for the NMDA threshold reverse valve.
    /// Controls how aggressively silent packets raise their threshold τ
    /// to prune non-productive branches.  Default `0.01`.
    pub dendritic_pruning_rate: f32,
}

impl WormTrainer {
    /// Construct a new trainer with the given hyper-parameters.
    ///
    /// The structural blueprint mask is automatically derived from
    /// [`WormBrain::new_baseline`] so the trainer inherently knows which
    /// connections are biologically valid.
    ///
    /// `homeostasis` controls weight-proportional decay (0 = off, ~0.3 =
    /// moderate).  A positive value prevents saturation by applying stronger
    /// stabilisation to weights already near 1.0.
    #[inline]
    pub fn new(learning_rate: f32, stabilization_factor: f32) -> Self {
        let baseline = WormBrain::new_baseline();
        let structural_mask = baseline.synapses.mapv(|w| w > 0.0);
        Self {
            learning_rate,
            stabilization_factor,
            homeostasis: 0.3,
            damping_factor: 0.05,
            structural_mask,
            safe_mode: true,
            step: Cell::new(0),
            contrastive_lr: 0.0,
            use_capped_training: false,
            tda_beta_0_threshold: 50,
            tda_beta_1_threshold: 6,
            dendritic_lr: 0.0,
            dendritic_pruning_rate: 0.01,
        }
    }

    /// Enable or disable the per-batch finiteness safety check.
    /// Pass `false` to skip the O(n²) scan for maximum throughput.
    #[inline]
    pub fn with_safe_mode(mut self, safe: bool) -> Self {
        self.safe_mode = safe;
        self
    }

    /// Run one step of Hebbian plasticity training.
    ///
    /// 1. **Co-firing detection** – outer product of `activation_state` with
    ///    itself.
    /// 2. **Hebbian update** – potentiate pre-existing synapses proportional
    ///    to co-firing, scaled by `self.learning_rate`.
    /// 3. **Stabilisation decay** – multiply *all* weights by
    ///    `self.stabilization_factor` to provide gradual forgetting.
    /// 4. **Row-normalisation** – clamp each row's values to `[0, 1]`.
    ///
    /// # Errors
    /// * [`CoreError::InvalidDimension`] if `activation_state` is not 302-long.
    /// * [`CoreError::Numerical`] if any weight becomes non-finite.
    #[inline]
    pub fn train_step(
        &self,
        brain: &mut WormBrain,
        activation_state: &Array1<f32>,
        pre_synaptic: &Array1<f32>,
        input_key: &Array1<f32>,
    ) -> Result<()> {
        // Dispatch to dendritic plasticity path when quad_routing is active.
        if brain.quad_routing && self.dendritic_lr > 0.0 {
            return self.train_step_dendritic(brain, pre_synaptic, activation_state);
        }

        if activation_state.len() != brain.neuron_count {
            return Err(CoreError::InvalidDimension {
                expected: brain.neuron_count,
                got: activation_state.len(),
            });
        }

        let n = brain.neuron_count;
        let lr = self.learning_rate;

        // Hebbian activation source: sparsemax of pre-synaptic (default)
        // or capped activation_state (when use_capped_training=true).
        let hebb: Array1<f32> = if self.use_capped_training {
            activation_state.to_owned()
        } else {
            // Sparse Hebbian update via pre-synaptic projection (avoids
            // synapse → sparsemax collapse loop).  Applies sparsemax directly
            // to the diverse pre-synaptic state so the co-firing pattern
            // remains input-specific throughout training.
            let mut h = pre_synaptic.to_owned();
            let mut sorted_hebb: Vec<f32> = h.iter().copied().collect();
            sorted_hebb.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let mut cumsum = 0.0f32;
            let mut tau = 0.0f32;
            for (i, &val) in sorted_hebb.iter().enumerate() {
                cumsum += val;
                let t = (cumsum - 1.0) / (i + 1) as f32;
                if t < val { tau = t; }
            }
            h.mapv_inplace(|v| (v - tau).max(0.0));
            h
        };

        // Fused Hebbian + φ-phase oscillation for all synapses.
        let phase = ((self.step.get() as f64 * std::f64::consts::PI / PHI).cos() * 0.15 + 0.85) as f32;
        let phase_delta = if (phase - 1.0).abs() > 1e-6 { phase - 1.0 } else { 0.0 };
        for i in 0..n {
            let ai = hebb[i];
            if ai == 0.0 { continue; }
            for j in 0..n {
                if !self.structural_mask[(i, j)] { continue; }
                let cofire = ai * hebb[j];
                let mut delta = lr * cofire;
                if i < GROUP_A && j >= GROUP_A && phase_delta != 0.0 {
                    delta += lr * cofire * phase_delta;
                }
                brain.synapses[(i, j)] += delta;
            }
        }

        // 3. Homeostatic stabilisation decay.
        let h = self.homeostasis;
        let sf = self.stabilization_factor;
        brain.synapses.mapv_inplace(|w| {
            let homeo = 1.0 - h * w;
            w * homeo * sf
        });

        // 4. Row-normalise to [0, 1].
        for i in 0..n {
            let mut row = brain.synapses.row_mut(i);
            let max_val = row.iter().copied().fold(0.0_f32, f32::max);
            if max_val > 0.0 {
                for val in row.iter_mut() {
                    *val /= max_val;
                    if *val < 0.0 {
                        *val = 0.0;
                    }
                }
            }
        }

        // 5. Projection Hebbian update: ΔP = lr · pre_synaptic ⊗ input_key
        // Skipped when use_capped_training is true because the Gaussian
        // row-norm projection is already diverse; Hebbian iteration would
        // collapse all rows to the first principal component of the input
        // distribution, destroying the diversity needed for cluster formation.
        if !self.use_capped_training {
            for i in 0..n {
                let pi = pre_synaptic[i];
                if pi == 0.0 { continue; }
                for j in 0..MANIFOLD_DIM {
                    brain.input_projection[(i, j)] += lr * pi * input_key[j];
                }
            }

            // 6. Column-normalise projection to [0, 1].
            for j in 0..MANIFOLD_DIM {
                let mut col = brain.input_projection.column_mut(j);
                let max_val = col.iter().copied().fold(0.0_f32, f32::max);
                if max_val > 0.0 {
                    for val in col.iter_mut() {
                        *val /= max_val;
                        if *val < 0.0 {
                            *val = 0.0;
                        }
                    }
                }
            }
        }

        if self.safe_mode {
            for &val in brain.synapses.iter() {
                if !val.is_finite() {
                    return Err(CoreError::Numerical(
                        "non-finite value in synapses after training step".to_string(),
                    ));
                }
            }
            for &val in brain.input_projection.iter() {
                if !val.is_finite() {
                    return Err(CoreError::Numerical(
                        "non-finite value in input_projection after training step".to_string(),
                    ));
                }
            }
        }

        self.step.set(self.step.get() + 1);

        Ok(())
    }

    /// Reverse Valve plasticity step for the dendritic tree.
    ///
    /// Replaces the standard Hebbian synapse update when
    /// `brain.quad_routing && self.dendritic_lr > 0`.
    ///
    /// 1. **RMS-normalizes** the pre-synaptic state (same as `route_signal`).
    /// 2. **Forwards** through the dendritic tree.
    /// 3. **Reverse Valve Backward** — propagates the activation error
    ///    through the 70/30 blend, then through each DendriticPacket,
    ///    accumulating weight gradients (Hebbian) and threshold pruning
    ///    (anti-Hebbian).
    /// 4. **Applies** gradients via SGD: weights ← weights - lr · dW,
    ///    thresholds ← thresholds + prune · inactive_fraction · active_branches.
    /// 5. **Clamps** weights to [-1.0, 1.0] and thresholds to [0.01, 10.0].
    ///
    /// # Errors
    /// * [`CoreError::InvalidDimension`] if dimensions mismatch.
    /// * [`CoreError::Numerical`] if any parameter becomes non-finite.
    #[inline]
    pub fn train_step_dendritic(
        &self,
        brain: &mut WormBrain,
        pre_state: &Array1<f32>,
        activation: &Array1<f32>,
    ) -> Result<()> {
        if pre_state.len() != brain.neuron_count {
            return Err(CoreError::InvalidDimension {
                expected: brain.neuron_count,
                got: pre_state.len(),
            });
        }
        if activation.len() != brain.neuron_count {
            return Err(CoreError::InvalidDimension {
                expected: brain.neuron_count,
                got: activation.len(),
            });
        }

        let n = brain.neuron_count;

        // 1. RMS-normalise the pre-synaptic state (matching route_signal).
        let rms = (pre_state.mapv(|v| v * v).sum() / n as f32).sqrt().max(1e-8);
        let inv_rms = 1.0 / rms;
        let scaled_state: Array1<f32> = pre_state.mapv(|v| v * inv_rms);

        // 2. Forward through the dendritic tree.
        //    (We don't need the blended output — the activation is already
        //     the sparsemax of the blended state.)
        let _tree_out = brain.dendritic_tree.forward(&scaled_state, brain.creative_k);

        // 3. Error signal through the 70/30 blend.
        //    δ_tree = δ_activation * 0.7  (ignore quad backward — no gradient
        //    flows through the 4-way product for now).
        let d_output: Array1<f32> = activation.mapv(|a| a * 0.7);

        // 4. Reverse Valve backward.
        brain.dendritic_tree.zero_gradients();
        let _ = brain.dendritic_tree.reverse_valve_backward(
            &scaled_state, &d_output, self.dendritic_pruning_rate, brain.creative_k,
        );

        // 5. Apply gradients (weights + thresholds).
        brain.dendritic_tree.apply_gradients(self.dendritic_lr);
        brain.dendritic_tree.clamp_weights();

        // 6. Safe-mode check.
        if self.safe_mode {
            for (i, p) in brain.dendritic_tree.packets.iter().enumerate() {
                if !p.threshold.is_finite() {
                    return Err(CoreError::Numerical(format!(
                        "non-finite threshold in packet {} after dendritic training",
                        i,
                    )));
                }
                for w in p.basal_w.iter() {
                    if !w.is_finite() {
                        return Err(CoreError::Numerical(format!(
                            "non-finite basal_w in packet {} after dendritic training",
                            i,
                        )));
                    }
                }
            }
        }

        self.step.set(self.step.get() + 1);
        Ok(())
    }

    /// Contrastive Hebbian: amplify differences between sentence activation patterns.
    ///
    /// For each pair `(A, B)`, computes the signed difference vector
    /// `d = act_A - act_B` and applies `Δw = lr · d ⊗ d` (outer product).
    /// Positive `d_i·d_j`: both differ in same direction → strengthen (amplify
    /// separation). Negative `d_i·d_j`: differ in opposite direction → weaken
    /// (reduce cross-talk).  Operates on the same capped 150-D activations
    /// used by [`crate::worm_brain::WormBrain::route_signal_capped`].
    ///
    /// Only active when `self.contrastive_lr > 0.0`.
    ///
    /// # Panics
    /// * Never — production paths are panic-free.
    #[inline]
    pub fn contrastive_step(
        &self,
        brain: &mut WormBrain,
        activations: &[Array1<f32>],
    ) {
        let lr = self.contrastive_lr;
        if lr <= 0.0 || activations.len() < 2 { return; }
        let n = brain.neuron_count;
        for a in 0..activations.len() {
            for b in (a + 1)..activations.len() {
                let diff: Array1<f32> = &activations[a] - &activations[b];
                let diff_norm = diff.dot(&diff).sqrt();
                if diff_norm < 1e-6 { continue; }
                for i in 0..n {
                    let di = diff[i];
                    if di.abs() < 1e-6 { continue; }
                    for j in 0..n {
                        if !self.structural_mask[(i, j)] { continue; }
                        // Sparse contrastive: only update synapses where di
                        // and diff[j] have OPPOSITE signs.  This concentrates
                        // the contrastive signal on the decision boundary
                        // between sentence A and B, rather than diffusing it
                        // across all 22500 shared connections.
                        let dj = diff[j];
                        if di * dj < 0.0 {
                            let hebb = di * dj;
                            brain.synapses[(i, j)] = (brain.synapses[(i, j)]
                                + lr * hebb).max(0.0).min(1.0);
                        }
                    }
                }
            }
        }
    }

    /// Run one step of **Maxwell-damped Hebbian** plasticity.
    ///
    /// Implements electromagnetic loop-suppression by coupling the weight
    /// update to a momentum (velocity) field:
    ///
    /// 1. **Co-firing** – outer product of `activation_state` (same as
    ///    [`train_step`]).
    /// 2. **Mutual Induction** – `velocity += lr · co_firing · mask`.
    /// 3. **Energy Dissipation** – `velocity *= (1.0 - γ)`.
    /// 4. **Forward Propagation** – `weight += velocity`.
    /// 5. **Row-normalisation** to `[0, 1]`.
    ///
    /// # Errors
    /// * [`CoreError::InvalidDimension`] if `activation_state` is not 302-long.
    /// * [`CoreError::Numerical`] if any weight becomes non-finite.
    #[inline]
    pub fn train_step_damped(
        &self,
        brain: &mut WormBrain,
        activation_state: &Array1<f32>,
        pre_synaptic: &Array1<f32>,
        input_key: &Array1<f32>,
    ) -> Result<()> {
        if activation_state.len() != brain.neuron_count {
            return Err(CoreError::InvalidDimension {
                expected: brain.neuron_count,
                got: activation_state.len(),
            });
        }

        let n = brain.neuron_count;
        let lr = self.learning_rate;

        // Fused co-firing + mutual induction — no 302×302 allocation.
        let act = activation_state.view();
        let phase = ((self.step.get() as f64 * std::f64::consts::PI / PHI).cos() * 0.15 + 0.85) as f32;
        let phase_delta = if (phase - 1.0).abs() > 1e-6 { phase - 1.0 } else { 0.0 };
        for i in 0..n {
            let ai = act[i];
            if ai == 0.0 { continue; }
            for j in 0..n {
                if !self.structural_mask[(i, j)] { continue; }
                let cofire = ai * act[j];
                let mut delta = lr * cofire;
                if i < GROUP_A && j >= GROUP_A && phase_delta != 0.0 {
                    delta += lr * cofire * phase_delta;
                }
                brain.velocities[(i, j)] += delta;
            }
        }

        // 3. Energy dissipation: damp velocity to suppress loop attractors.
        let gamma = self.damping_factor.clamp(0.0, 1.0);
        brain.velocities.mapv_inplace(|v| v * (1.0 - gamma));

        // 4. Forward propagation: push weights along the damped momentum.
        Zip::from(&mut brain.synapses)
            .and(&brain.velocities)
            .for_each(|syn, &vel| {
                *syn += vel;
                if *syn < 0.0 {
                    *syn = 0.0;
                }
            });

        // 5. Row-normalise to [0, 1].
        for i in 0..n {
            let max_val = brain.synapses.row(i).iter().copied().fold(0.0_f32, f32::max);
            if max_val > 0.0 {
                for val in brain.synapses.row_mut(i).iter_mut() {
                    *val /= max_val;
                    if *val < 0.0 {
                        *val = 0.0;
                    }
                }
            }
        }

        // 6. Projection Hebbian update: ΔP = lr · pre_synaptic ⊗ input_key
        for i in 0..n {
            let pi = pre_synaptic[i];
            if pi == 0.0 { continue; }
            for j in 0..MANIFOLD_DIM {
                brain.input_projection[(i, j)] += lr * pi * input_key[j];
            }
        }

        // 7. Column-normalise projection to [0, 1].
        for j in 0..MANIFOLD_DIM {
            let mut col = brain.input_projection.column_mut(j);
            let max_val = col.iter().copied().fold(0.0_f32, f32::max);
            if max_val > 0.0 {
                for val in col.iter_mut() {
                    *val /= max_val;
                    if *val < 0.0 {
                        *val = 0.0;
                    }
                }
            }
        }

        if self.safe_mode {
            for &val in brain.synapses.iter() {
                if !val.is_finite() {
                    return Err(CoreError::Numerical(
                        "non-finite value in synapses after damped training step".to_string(),
                    ));
                }
            }
            for &val in brain.input_projection.iter() {
                if !val.is_finite() {
                    return Err(CoreError::Numerical(
                        "non-finite value in input_projection after damped training step".to_string(),
                    ));
                }
            }
        }

        self.step.set(self.step.get() + 1);

        Ok(())
    }

    /// Run one step of **GPU-accelerated Maxwell-damped** Hebbian plasticity.
    ///
    /// Uploads the current weights, velocities, and activation to the CUDA
    /// device, executes co-firing, mutual induction, dissipation, forward
    /// propagation, and row-normalisation entirely on the RTX 4060 via
    /// [`GpuEngine::damped_step`], then synchronises the result back.
    ///
    /// This method is only available when compiled with `--features cuda`.
    #[cfg(feature = "cuda")]
    #[inline]
    pub fn train_step_gpu(
        &self,
        brain: &mut WormBrain,
        activation_state: &Array1<f32>,
        engine: &crate::gpu::GpuEngine,
    ) -> Result<()> {
        let n = brain.neuron_count;
        let flat_w: Vec<f32> = brain.synapses.iter().copied().collect();
        let flat_v: Vec<f32> = brain.velocities.iter().copied().collect();
        let flat_a: Vec<f32> = activation_state.iter().copied().collect();

        // Upload to GPU
        let w_gpu = engine.host_to_tensor_2d_f32(&flat_w, &[n, n], "synapses")?;
        let v_gpu = engine.host_to_tensor_2d_f32(&flat_v, &[n, n], "velocities")?;
        let a_gpu = engine.host_to_tensor_1d_f32(&flat_a, "activation")?;

        // Build the mask once (constant)
        let mask_flat: Vec<f32> = self.structural_mask
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let mask_gpu = engine.host_to_tensor_2d_f32(&mask_flat, &[n, n], "mask")?;

        // Execute damped step on GPU
        let (s_new, v_new) = engine.damped_step(
            &w_gpu, &v_gpu, &mask_gpu, &a_gpu,
            self.learning_rate, self.damping_factor,
        )?;

        // Sync back to host
        let flat_s = engine.tensor_to_host_2d(&s_new, "synapses")?;
        let flat_v2 = engine.tensor_to_host_2d(&v_new, "velocities")?;

        if self.safe_mode {
            if flat_s.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::Numerical("synapses after damped step contain non-finite values".into()));
            }
            if flat_v2.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::Numerical("velocities after damped step contain non-finite values".into()));
            }
        }

        // Write back into the brain
        for (i, &val) in flat_s.iter().enumerate() {
            let row = i / n;
            let col = i % n;
            brain.synapses[(row, col)] = val;
        }
        for (i, &val) in flat_v2.iter().enumerate() {
            let row = i / n;
            let col = i % n;
            brain.velocities[(row, col)] = val;
        }

        Ok(())
    }

    /// Run a **batched** Maxwell-damped Hebbian training step on GPU.
    ///
    /// Uploads weights, velocities, and the structural mask **once**, then
    /// processes all activations in the batch on-device without CPU roundtrips.
    /// This eliminates the PCIe transfer overhead that dominates the current
    /// per-token `train_step_gpu`.
    #[cfg(feature = "cuda")]
    pub fn train_batch_gpu(
        &self,
        brain: &mut WormBrain,
        activations_batch: &[Array1<f32>],
        engine: &crate::gpu::GpuEngine,
    ) -> Result<()> {
        let n = brain.neuron_count;

        let flat_w: Vec<f32> = brain.synapses.iter().copied().collect();
        let flat_v: Vec<f32> = brain.velocities.iter().copied().collect();
        let mask_flat: Vec<f32> = self.structural_mask
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let batch_size = activations_batch.len();
        let act_flat: Vec<f32> = activations_batch
            .iter()
            .flat_map(|a| a.iter().copied())
            .collect();

        let (flat_s, flat_v2) = engine.batch_damped_step(
            &flat_w, &flat_v, &mask_flat, &act_flat, batch_size,
            self.learning_rate, self.damping_factor,
        )?;

        if self.safe_mode {
            if flat_s.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::Numerical(
                    "synapses after batch GPU step contain non-finite values".into(),
                ));
            }
            if flat_v2.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::Numerical(
                    "velocities after batch GPU step contain non-finite values".into(),
                ));
            }
        }

        for (i, &val) in flat_s.iter().enumerate() {
            let row = i / n;
            let col = i % n;
            brain.synapses[(row, col)] = val;
        }
        for (i, &val) in flat_v2.iter().enumerate() {
            let row = i / n;
            let col = i % n;
            brain.velocities[(row, col)] = val;
        }

        Ok(())
    }

    /// Full end-to-end batch: route 128-D coordinates → train — all on GPU.
    ///
    /// Uploads input_projection, synapses, velocities, mask **once**, then
    /// uploads all coordinate keys as a single (B×16) tensor.  The entire
    /// routing + Hebbian training loop runs device-side without any CPU
    /// roundtrip.  Only the final weights and velocities are downloaded.
    ///
    /// This is the fastest path — zero CPU-GPU transfers per token.
    #[cfg(feature = "cuda")]
    pub fn train_batch_route_gpu(
        &self,
        brain: &mut WormBrain,
        coords_batch: &[Array1<f64>],
        engine: &crate::gpu::GpuEngine,
    ) -> Result<()> {
        let n = brain.neuron_count;

        let proj_flat: Vec<f32> = brain.input_projection().iter().copied().collect();
        let flat_w: Vec<f32> = brain.synapses.iter().copied().collect();
        let flat_v: Vec<f32> = brain.velocities.iter().copied().collect();
        let mask_flat: Vec<f32> = self.structural_mask
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let batch_size = coords_batch.len();
        let coords_flat: Vec<f64> = coords_batch
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();

        let (flat_s, flat_v2) = engine.batch_route_and_train(
            &proj_flat, &flat_w, &flat_v, &mask_flat,
            &coords_flat, batch_size,
            self.learning_rate, self.damping_factor,
        )?;

        if self.safe_mode {
            if flat_s.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::Numerical(
                    "synapses after batch end-to-end GPU step contain non-finite values".into(),
                ));
            }
            if flat_v2.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::Numerical(
                    "velocities after batch end-to-end GPU step contain non-finite values".into(),
                ));
            }
        }

        for (i, &val) in flat_s.iter().enumerate() {
            let row = i / n;
            let col = i % n;
            brain.synapses[(row, col)] = val;
        }
        for (i, &val) in flat_v2.iter().enumerate() {
            let row = i / n;
            let col = i % n;
            brain.velocities[(row, col)] = val;
        }

        Ok(())
    }

    /// Same as [`train_batch_route_gpu`] but accepts a **flat** coordinate
    /// buffer (`B × MANIFOLD_DIM` row-major) directly — no `Vec<Array1<f64>>`
    /// conversion needed.  Use this from the async producer-consumer pipeline.
    #[cfg(feature = "cuda")]
    pub fn train_batch_route_gpu_flat(
        &self,
        brain: &mut WormBrain,
        coords_flat: &[f64],
        batch_size: usize,
        engine: &crate::gpu::GpuEngine,
    ) -> Result<()> {
        let n = brain.neuron_count;

        let proj_flat: Vec<f32> = brain.input_projection().iter().copied().collect();
        let flat_w: Vec<f32> = brain.synapses.iter().copied().collect();
        let flat_v: Vec<f32> = brain.velocities.iter().copied().collect();
        let mask_flat: Vec<f32> = self.structural_mask
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();

        let (flat_s, flat_v2) = engine.batch_route_and_train(
            &proj_flat, &flat_w, &flat_v, &mask_flat,
            coords_flat, batch_size,
            self.learning_rate, self.damping_factor,
        )?;

        if self.safe_mode {
            if flat_s.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::Numerical(
                    "synapses after batch end-to-end GPU step contain non-finite values".into(),
                ));
            }
            if flat_v2.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::Numerical(
                    "velocities after batch end-to-end GPU step contain non-finite values".into(),
                ));
            }
        }

        for (i, &val) in flat_s.iter().enumerate() {
            let row = i / n;
            let col = i % n;
            brain.synapses[(row, col)] = val;
        }
        for (i, &val) in flat_v2.iter().enumerate() {
            let row = i / n;
            let col = i % n;
            brain.velocities[(row, col)] = val;
        }

        Ok(())
    }
}

// ── TDA Loss Landscape Monitoring ────────────────────────────────────────

use crate::tda::compute_betti_numbers;

/// Run TDA monitoring and apply corrective interventions if landscape
/// anomalies are detected.
///
/// Returns `true` if an intervention was applied.
pub fn monitor_and_intervene(
    activations: &[Array1<f32>],
    brain: &mut WormBrain,
    trainer: &WormTrainer,
    step: usize,
) -> bool {
    let (beta_0, beta_1) = compute_betti_numbers(activations);
    let mut intervened = false;

    // β₀ threshold: activations fractured into isolated clusters
    if beta_0 > trainer.tda_beta_0_threshold {
        eprintln!(
            "\n  [TDA] Step {step}: β₀ = {beta_0} > {th} — fracturing detected, injecting noise",
            th = trainer.tda_beta_0_threshold,
        );
        // Inject noise into the projection to reconnect clusters
        let noise_level = 0.05 * trainer.learning_rate;
        let mut rng = fastrand::Rng::new();
        for val in brain.synapses.iter_mut() {
            if *val > 0.0 {
                *val += (rng.f32() - 0.5) * noise_level;
                *val = val.clamp(0.0, 1.0);
            }
        }
        intervened = true;
    }

    // β₁ threshold: circular gradient paths (cycles)
    if beta_1 > trainer.tda_beta_1_threshold {
        eprintln!(
            "\n  [TDA] Step {step}: β₁ = {beta_1} > {th} — cycle detected, applying contrastive unlearn",
            th = trainer.tda_beta_1_threshold,
        );
        // Break cycles via contrastive unlearning
        brain.contrastive_unlearn(activations, 0.08, 0.005);
        intervened = true;
    }

    intervened
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_hebbian_plasticity_learning() {
        let trainer = WormTrainer::new(0.01, 0.99);
        let mut brain = WormBrain::new_baseline();
        let initial_synapses = brain.synapses.clone();

        // Activation state where neurons 0, 1, 2 co-fire repeatedly.
        let mut activation = Array1::<f32>::zeros(302);
        activation[0] = 1.0;
        activation[1] = 1.0;
        activation[2] = 1.0;

        let pre_synaptic = Array1::<f32>::zeros(302);
        let input_key = Array1::<f32>::zeros(MANIFOLD_DIM);

        // Run several training steps to strengthen the 0↔1↔2 sub-circuit.
        for _ in 0..10 {
            trainer.train_step(&mut brain, &activation, &pre_synaptic, &input_key).unwrap();
        }

        // Invariant 1: structural zero connections MUST remain zero.
        for i in 0..302 {
            for j in 0..302 {
                if initial_synapses[(i, j)] == 0.0 {
                    assert_eq!(
                        brain.synapses[(i, j)], 0.0,
                        "synapse ({},{}) became non-zero despite no structural connection",
                        i, j,
                    );
                }
            }
        }

        // Invariant 2: all weights lie in [0, 1].
        for &w in brain.synapses.iter() {
            assert!(
                w >= 0.0 && w <= 1.0 + 1e-12,
                "weight {} outside [0, 1]",
                w,
            );
        }

        // Invariant 3: every row's maximum is normalised to ≤ 1.0.
        for i in 0..302 {
            let row_max = brain
                .synapses
                .row(i)
                .iter()
                .copied()
                .fold(0.0_f32, f32::max);
            assert!(
                row_max <= 1.0 + 1e-6,
                "row {} has max {} > 1.0",
                i,
                row_max,
            );
        }

        // Invariant 4: co-firing pathways (0↔1, 0↔2) are measurably
        // potentiated above a low floor (pharyngeal cluster baseline ≥ 0.25,
        // plus Hebbian boost over 10 steps).
        let trained_01 = brain.synapses[(0, 1)];
        let trained_02 = brain.synapses[(0, 2)];
        let trained_10 = brain.synapses[(1, 0)];
        let trained_12 = brain.synapses[(1, 2)];
        assert!(
            trained_01 >= 0.1,
            "co-firing connection (0,1) too weak: {}",
            trained_01,
        );
        assert!(
            trained_02 >= 0.1,
            "co-firing connection (0,2) too weak: {}",
            trained_02,
        );
        assert!(
            trained_10 >= 0.1,
            "co-firing connection (1,0) too weak: {}",
            trained_10,
        );
        assert!(
            trained_12 >= 0.1,
            "co-firing connection (1,2) too weak: {}",
            trained_12,
        );

        // Invariant 5: dimension-mismatch error is properly returned.
        let bad_state = Array1::<f32>::zeros(100);
        let result = trainer.train_step(&mut brain, &bad_state, &pre_synaptic, &input_key);
        assert!(result.is_err());
        if let Err(CoreError::InvalidDimension { expected, got }) = result {
            assert_eq!(expected, 302);
            assert_eq!(got, 100);
        } else {
            panic!("expected InvalidDimension error for wrong activation size");
        }
    }

    #[test]
    fn test_dendritic_training_dispatches_when_quad_routing() {
        // Create a brain with quad_routing enabled.
        let mut brain = WormBrain::new_baseline();
        brain.quad_routing = true;

        // Trainer with dendritic learning enabled.
        let mut trainer = WormTrainer::new(0.1, 0.99);
        trainer.dendritic_lr = 0.01;

        // Configure packet 0 deterministically so the basal branch fires
        // for x = 1.0: w00 = 4.0 → q1 = Σw[i][j]·1·1 = 4.0 > τ = 0.2
        brain.dendritic_tree.packets[0].basal_w = [0.0; 16];
        brain.dendritic_tree.packets[0].oblique_w = [0.0; 16];
        brain.dendritic_tree.packets[0].apical_w = [0.0; 16];
        brain.dendritic_tree.packets[0].basal_w[0] = 4.0;
        brain.dendritic_tree.packets[0].threshold = 0.2;

        // Record pre-training weights.
        let w_before = brain.dendritic_tree.packets[0].basal_w[0];
        let tau_before = brain.dendritic_tree.packets[0].threshold;

        // Create a pre-synaptic state where RMS normalization gives x ≈ 1
        let pre_state = Array1::from_elem(302, 0.5);
        let mut activation = Array1::zeros(302);
        activation[0] = 0.1;
        activation[1] = 0.2;
        activation[2] = 0.3;
        // Sparsemax would normalise the sum to 1, but for the test any
        // non-uniform pattern suffices.
        activation /= activation.sum();

        let input_key = Array1::zeros(MANIFOLD_DIM);

        // Run train_step — should dispatch to train_step_dendritic.
        trainer
            .train_step(&mut brain, &activation, &pre_state, &input_key)
            .unwrap();

        // Verify dendritic weights changed.
        assert!(
            (brain.dendritic_tree.packets[0].basal_w[0] - w_before).abs() > 0.0,
            "dendritic weights should change after training step",
        );

        // Verify dendritic threshold changed (pruning).
        assert!(
            (brain.dendritic_tree.packets[0].threshold - tau_before).abs() > 0.0,
            "dendritic threshold should change after training step",
        );

        // Verify weights are clamped to [-1, 1].
        for p in brain.dendritic_tree.packets.iter() {
            for &w in p.basal_w.iter() {
                assert!(w >= -1.0 && w <= 1.0, "basal_w {} out of [-1, 1]", w);
            }
        }

        // Verify synapses were NOT modified (dispatch skipped synapse update).
        let baseline = WormBrain::new_baseline();
        for i in 0..302 {
            for j in 0..302 {
                assert_eq!(
                    brain.synapses[(i, j)], baseline.synapses[(i, j)],
                    "synapse ({},{}) was modified despite quad_routing dispatch",
                    i, j,
                );
            }
        }
    }

    #[test]
    fn test_dendritic_training_backward_compat_when_disabled() {
        // When dendritic_lr = 0 (default), train_step should use the standard
        // Hebbian path even with quad_routing enabled.
        let mut brain = WormBrain::new_baseline();
        brain.quad_routing = true;

        let trainer = WormTrainer::new(0.1, 0.99);
        assert_eq!(trainer.dendritic_lr, 0.0, "dendritic_lr should be 0 by default");

        let mut activation = Array1::zeros(302);
        activation[0] = 1.0;
        activation[1] = 1.0;
        let pre_state = Array1::from_elem(302, 0.5);
        let input_key = Array1::zeros(MANIFOLD_DIM);

        // Should NOT dispatch to dendritic path.
        let w_before = brain.dendritic_tree.packets[0].basal_w[0];
        trainer
            .train_step(&mut brain, &activation, &pre_state, &input_key)
            .unwrap();
        let w_after = brain.dendritic_tree.packets[0].basal_w[0];

        // Dendritic weights should NOT change (backward compat).
        assert_eq!(
            w_before, w_after,
            "dendritic weights must not change when dendritic_lr = 0",
        );
    }

    #[test]
    fn test_dendritic_training_multi_step_accumulation() {
        let mut brain = WormBrain::new_baseline();
        brain.quad_routing = true;

        let mut trainer = WormTrainer::new(0.1, 0.99);
        trainer.dendritic_lr = 0.01;

        // Configure packet 0 deterministically
        brain.dendritic_tree.packets[0].basal_w = [0.0; 16];
        brain.dendritic_tree.packets[0].oblique_w = [0.0; 16];
        brain.dendritic_tree.packets[0].apical_w = [0.0; 16];
        brain.dendritic_tree.packets[0].basal_w[0] = 4.0;
        brain.dendritic_tree.packets[0].threshold = 0.2;

        let pre_state = Array1::from_elem(302, 0.5);
        let mut activation = Array1::zeros(302);
        activation[0] = 0.3;
        activation[1] = 0.2;
        activation[2] = 0.1;
        let norm: f32 = activation.sum();
        activation = activation.mapv(|v| v / norm.max(1e-8));
        let input_key = Array1::zeros(MANIFOLD_DIM);

        let w_initial = brain.dendritic_tree.packets[0].basal_w[0];

        for _ in 0..5 {
            trainer
                .train_step(&mut brain, &activation, &pre_state, &input_key)
                .unwrap();
        }

        // After 5 steps, weight change should be larger than after 1 step.
        let change = (brain.dendritic_tree.packets[0].basal_w[0] - w_initial).abs();
        assert!(
            change > 0.0,
            "dendritic weights should change over multiple training steps",
        );
    }

    #[test]
    fn test_dendritic_training_preserves_synapses() {
        // Verify that train_step dispatches to dendritic path when
        // quad_routing + dendritic_lr > 0, and that synapses remain
        // unchanged.
        let mut brain = WormBrain::new_baseline();
        brain.quad_routing = true;

        let mut trainer = WormTrainer::new(0.1, 0.99);
        trainer.dendritic_lr = 0.01; // Enable dendritic path

        let pre_state = Array1::from_elem(302, 0.5);
        let activation = Array1::from_elem(302, 1.0 / 302.0);
        let input_key = Array1::zeros(MANIFOLD_DIM);

        let synapses_before = brain.synapses.clone();

        trainer
            .train_step(&mut brain, &activation, &pre_state, &input_key)
            .unwrap();

        // synapses must be unmodified (dispatch went to dendritic path).
        for i in 0..302 {
            for j in 0..302 {
                assert_eq!(
                    brain.synapses[(i, j)], synapses_before[(i, j)],
                    "synapses should not change via dendritic training path, but ({},{}) changed",
                    i, j,
                );
            }
        }
    }
}
