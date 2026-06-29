//! CUDA-accelerated Maxwell-damped Hebbian training backend.
//!
//! Uses `candle-core` with the `cuda` feature to execute all large
//! 302×302 tensor operations on the NVIDIA RTX 4060.  The host-side
//! `GpuEngine` manages device placement and provides the key primitives
//! needed by [`WormTrainer::train_step_gpu`].
//!
//! # Device Selection
//! [`GpuEngine::try_new`] probes `Device::Cuda(0)`.  If CUDA is unavailable
//! it returns `None` — the caller should fall back to the CPU path.

use candle_core::{Device, Tensor, D};
use crate::{CoreError, Result};

/// GPU compute context for Maxwell-damped Hebbian training.
///
/// Holds a reference to the CUDA device and provides batched tensor
/// operations for co-firing, mutual induction, dissipation, forward
/// propagation, and row-normalisation — all on the RTX 4060.
pub struct GpuEngine {
    device: Device,
}

impl GpuEngine {
    /// Initialise the GPU engine.  Returns `Ok(None)` if no CUDA device
    /// is available (caller should fall back to CPU).
    pub fn try_new() -> Result<Option<Self>> {
        match Device::cuda_if_available(0) {
            Ok(device) => {
                Ok(Some(Self { device }))
            }
            Err(_) => Ok(None),
        }
    }

    /// Return a reference to the underlying CUDA device.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    // ── Core tensor conversions ──────────────────────────────────────

    /// Convert a host `&[f64]` slice to a 1-D CUDA `f32` tensor.
    pub fn host_to_tensor_1d(&self, data: &[f64], label: &str) -> Result<Tensor> {
        let f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        Tensor::new(f32.as_slice(), &self.device)
            .map_err(|e| CoreError::Bridge(format!("{label} → GPU tensor: {e}")))
    }

    /// Convert a host `&[f32]` slice to a 1-D CUDA `f32` tensor (no conversion).
    pub fn host_to_tensor_1d_f32(&self, data: &[f32], label: &str) -> Result<Tensor> {
        Tensor::new(data, &self.device)
            .map_err(|e| CoreError::Bridge(format!("{label} → GPU tensor: {e}")))
    }

    /// Convert a host `&[f64]` slice to a 2-D CUDA `f32` tensor with the
    /// given shape (row-major layout).
    pub fn host_to_tensor_2d(
        &self,
        data: &[f64],
        shape: &[usize],
        label: &str,
    ) -> Result<Tensor> {
        let f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        Tensor::from_vec(f32, shape, &self.device)
            .map_err(|e| CoreError::Bridge(format!("{label} → GPU tensor: {e}")))
    }

    /// Convert a host `&[f32]` slice to a 2-D CUDA `f32` tensor (no conversion).
    pub fn host_to_tensor_2d_f32(
        &self,
        data: &[f32],
        shape: &[usize],
        label: &str,
    ) -> Result<Tensor> {
        Tensor::from_vec(data.to_vec(), shape, &self.device)
            .map_err(|e| CoreError::Bridge(format!("{label} → GPU tensor: {e}")))
    }

    /// Copy a 2-D CUDA `f32` tensor back to the host as a flat `Vec<f32>`.
    pub fn tensor_to_host_2d(&self, t: &Tensor, label: &str) -> Result<Vec<f32>> {
        let cpu = t.to_device(&Device::Cpu)
            .map_err(|e| CoreError::Bridge(format!("{label} → CPU: {e}")))?;
        let flat_f32: Vec<f32> = cpu.to_vec2::<f32>()
            .map_err(|e| CoreError::Bridge(format!("{label} vec2: {e}")))?
            .into_iter()
            .flatten()
            .collect();
        Ok(flat_f32)
    }

    /// Copy a 2-D CUDA `f32` tensor back as `Vec<Vec<f64>>` (row-major, with conversion).
    pub fn tensor_to_host_rows(&self, t: &Tensor, label: &str) -> Result<Vec<Vec<f64>>> {
        let cpu = t.to_device(&Device::Cpu)
            .map_err(|e| CoreError::Bridge(format!("{label} → CPU: {e}")))?;
        let rows: Vec<Vec<f32>> = cpu.to_vec2::<f32>()
            .map_err(|e| CoreError::Bridge(format!("{label} vec2: {e}")))?;
        Ok(rows.into_iter()
            .map(|r| r.into_iter().map(|v| v as f64).collect())
            .collect())
    }

    // ── Maxwell-damped training step (all on GPU) ─────────────────────

    /// Execute one Maxwell-damped Hebbian step entirely on the GPU.
    ///
    /// # Arguments
    /// * `synapses`  — 302×302 weight tensor (f32 on CUDA).
    /// * `velocities` — 302×302 momentum tensor (f32 on CUDA).
    /// * `mask`      — 302×302 structural blueprint (1.0 where synapse exists, 0 elsewhere).
    /// * `activation` — 302-length firing vector (f32 on CUDA, 1-D).
    /// * `lr`        — Hebbian learning rate.
    /// * `gamma`     — Maxwell damping factor (velocity dissipation).
    ///
    /// # Returns
    /// `(updated_synapses, updated_velocities)` — both remain on the CUDA device.
    pub fn damped_step(
        &self,
        synapses: &Tensor,
        velocities: &Tensor,
        mask: &Tensor,
        activation: &Tensor,
        lr: f32,
        gamma: f32,
    ) -> Result<(Tensor, Tensor)> {
        let n: usize = 302;
        let gamma_f32 = gamma;

        // 1. Co-firing: outer product = col · row  ⟹  n×1 × 1×n = n×n
        let col = activation.reshape((n, 1))
            .map_err(|e| CoreError::Bridge(format!("co-firing col reshape: {e}")))?;
        let row = activation.reshape((1, n))
            .map_err(|e| CoreError::Bridge(format!("co-firing row reshape: {e}")))?;
        let co_firing = col.matmul(&row)
            .map_err(|e| CoreError::Bridge(format!("co-firing matmul: {e}")))?;

        // 2. Mutual Induction: v += lr * co_firing * mask
        let induction = (co_firing * mask)
            .map_err(|e| CoreError::Bridge(format!("induction mask: {e}")))?;
        let lr_tensor = Tensor::new(&[lr], &self.device)
            .map_err(|e| CoreError::Bridge(format!("lr tensor: {e}")))?;
        let induction_scaled = induction.broadcast_mul(&lr_tensor)
            .map_err(|e| CoreError::Bridge(format!("induction lr: {e}")))?;
        let v_new = (velocities + induction_scaled)
            .map_err(|e| CoreError::Bridge(format!("velocity add: {e}")))?;

        // 3. Energy Dissipation: v *= (1 - gamma)
        let damp = 1.0_f32 - gamma_f32;
        let damp_tensor = Tensor::new(&[damp], &self.device)
            .map_err(|e| CoreError::Bridge(format!("damp tensor: {e}")))?;
        let v_new = v_new.broadcast_mul(&damp_tensor)
            .map_err(|e| CoreError::Bridge(format!("velocity damp: {e}")))?;

        // 4. Forward Propagation: w += v
        let s_new = (synapses + &v_new)
            .map_err(|e| CoreError::Bridge(format!("synapse add: {e}")))?;

        // Clamp negatives to 0
        let zero = Tensor::zeros(&[n, n], candle_core::DType::F32, &self.device)
            .map_err(|e| CoreError::Bridge(format!("zero tensor: {e}")))?;
        let s_new = s_new.maximum(&zero)
            .map_err(|e| CoreError::Bridge(format!("clamp negatives: {e}")))?;

        // 5. Row-normalise: each row divided by its max
        let s_new = self.normalize_rows(&s_new, n)?;

        Ok((s_new, v_new))
    }

    /// Row-normalise a 2-D tensor so each row's values are in [0, 1].
    pub(crate) fn normalize_rows(&self, t: &Tensor, n: usize) -> Result<Tensor> {
        let max_vals = t.max(D::Minus1)
            .map_err(|e| CoreError::Bridge(format!("row max: {e}")))?;
        let max_vals = max_vals.reshape((n, 1))
            .map_err(|e| CoreError::Bridge(format!("max reshape: {e}")))?;
        let t = t.broadcast_div(&max_vals)
            .map_err(|e| CoreError::Bridge(format!("row normalise: {e}")))?;

        // Clamp any negatives that may have survived
        let zero = Tensor::zeros(&[n, n], candle_core::DType::F32, &self.device)
            .map_err(|e| CoreError::Bridge(format!("zero tensor: {e}")))?;
        t.maximum(&zero)
            .map_err(|e| CoreError::Bridge(format!("clamp post-norm: {e}")))
    }

    /// Process a batch of activations entirely on GPU — no per-step roundtrips.
    ///
    /// Uploads synapses/velocities/mask AND the entire batch as a single
    /// (B×302) tensor, then iterates on-device.  Downloads once at the end.
    ///
    /// # Arguments
    /// * `synapses`         — flat 302×302 weight matrix (f32, row-major)
    /// * `velocities`       — flat 302×302 momentum   (f32, row-major)
    /// * `mask_flat`        — flat 302×302 structural blueprint (1.0/0.0 f32)
    /// * `activations_flat` — flat B×302 activations (f32, row-major)
    /// * `batch_size`       — number of activations in the batch
    /// * `lr`               — Hebbian learning rate
    /// * `gamma`            — Maxwell damping factor
    pub fn batch_damped_step(
        &self,
        synapses: &[f32],
        velocities: &[f32],
        mask_flat: &[f32],
        activations_flat: &[f32],
        batch_size: usize,
        lr: f32,
        gamma: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n: usize = 302;
        let damp_f32 = 1.0_f32 - gamma;

        let w_gpu = self.host_to_tensor_2d_f32(synapses, &[n, n], "syn")?;
        let mut v_gpu = self.host_to_tensor_2d_f32(velocities, &[n, n], "vel")?;
        let mask_gpu = self.host_to_tensor_2d_f32(mask_flat, &[n, n], "mask")?;
        let lr_t = Tensor::new(&[lr], &self.device)
            .map_err(|e| CoreError::Bridge(format!("lr tensor: {e}")))?;
        let damp_t = Tensor::new(&[damp_f32], &self.device)
            .map_err(|e| CoreError::Bridge(format!("damp tensor: {e}")))?;
        let zero = Tensor::zeros(&[n, n], candle_core::DType::F32, &self.device)
            .map_err(|e| CoreError::Bridge(format!("zero tensor: {e}")))?;

        // Upload all activations in one shot as (B, n)
        let acts_gpu = self.host_to_tensor_2d_f32(activations_flat, &[batch_size, n], "acts_batch")?;
        let chunks = acts_gpu.chunk(batch_size, 0)
            .map_err(|e| CoreError::Bridge(format!("batch chunk: {e}")))?;

        for chunk in chunks {
            let a_gpu = chunk.squeeze(0)
                .map_err(|e| CoreError::Bridge(format!("batch squeeze: {e}")))?;

            let col = a_gpu.reshape((n, 1))
                .map_err(|e| CoreError::Bridge(format!("batch col reshape: {e}")))?;
            let row = a_gpu.reshape((1, n))
                .map_err(|e| CoreError::Bridge(format!("batch row reshape: {e}")))?;
            let co_firing = col.matmul(&row)
                .map_err(|e| CoreError::Bridge(format!("batch matmul: {e}")))?;

            let induction = (co_firing * &mask_gpu)
                .map_err(|e| CoreError::Bridge(format!("batch induction mask: {e}")))?;
            let induction = induction.broadcast_mul(&lr_t)
                .map_err(|e| CoreError::Bridge(format!("batch induction lr: {e}")))?;
            v_gpu = (v_gpu + induction)
                .map_err(|e| CoreError::Bridge(format!("batch vel add: {e}")))?;

            v_gpu = v_gpu.broadcast_mul(&damp_t)
                .map_err(|e| CoreError::Bridge(format!("batch vel damp: {e}")))?;
        }

        let mut s_new = (w_gpu + &v_gpu)
            .map_err(|e| CoreError::Bridge(format!("batch syn add: {e}")))?;
        s_new = s_new.maximum(&zero)
            .map_err(|e| CoreError::Bridge(format!("batch clamp: {e}")))?;
        s_new = self.normalize_rows(&s_new, n)?;

        let flat_w = self.tensor_to_host_2d(&s_new, "syn")?;
        let flat_v = self.tensor_to_host_2d(&v_gpu, "vel")?;
        Ok((flat_w, flat_v))
    }

    /// Full end-to-end batch: route coordinate → train — all on GPU.
    ///
    /// Uploads input_projection, synapses, velocities, mask (all f32), and
    /// the full batch of 128-D coordinates (f64, converted to f32 on upload)
    /// as a single (B×16) tensor.  For each coordinate in the batch:
    ///   1. activation = proj @ coord
    ///   2. activation = synapses @ activation
    ///   3. activation = tanh(activation)
    ///   4. Hebbian outer-product + damped velocity update
    ///
    /// Downloads updated W and V once.  Zero CPU-GPU transfers during the
    /// batch iteration.
    ///
    /// # Arguments
    /// * `proj_flat`   — flat 302×16 input projection (f32, row-major)
    /// * `synapses`    — flat 302×302 weights         (f32, row-major)
    /// * `velocities`  — flat 302×302 momentum        (f32, row-major)
    /// * `mask_flat`   — flat 302×302 blueprint       (1.0/0.0 f32)
    /// * `coords_flat` — flat B×16 coordinate vectors (f64, row-major)
    /// * `batch_size`  — number of tokens in the batch
    /// * `lr`          — Hebbian learning rate
    /// * `gamma`       — Maxwell damping factor
    pub fn batch_route_and_train(
        &self,
        proj_flat: &[f32],
        synapses: &[f32],
        velocities: &[f32],
        mask_flat: &[f32],
        coords_flat: &[f64],
        batch_size: usize,
        lr: f32,
        gamma: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = 302usize;
        let d = crate::geometry::MANIFOLD_DIM;
        let damp_f32 = 1.0_f32 - gamma;

        // Upload once — all stay on GPU (use f32 methods, no conversion)
        let proj_gpu = self.host_to_tensor_2d_f32(proj_flat, &[n, d], "proj")?;
        let w_gpu = self.host_to_tensor_2d_f32(synapses, &[n, n], "syn")?;
        let mut v_gpu = self.host_to_tensor_2d_f32(velocities, &[n, n], "vel")?;
        let mask_gpu = self.host_to_tensor_2d_f32(mask_flat, &[n, n], "mask")?;
        let lr_t = Tensor::new(&[lr], &self.device)
            .map_err(|e| CoreError::Bridge(format!("lr tensor: {e}")))?;
        let damp_t = Tensor::new(&[damp_f32], &self.device)
            .map_err(|e| CoreError::Bridge(format!("damp tensor: {e}")))?;
        let zero = Tensor::zeros(&[n, n], candle_core::DType::F32, &self.device)
            .map_err(|e| CoreError::Bridge(format!("zero tensor: {e}")))?;

        // Upload all coords as (B, MANIFOLD_DIM)
        let coords_gpu = self.host_to_tensor_2d(coords_flat, &[batch_size, d], "coords")?;
        let coord_chunks = coords_gpu.chunk(batch_size, 0)
            .map_err(|e| CoreError::Bridge(format!("coord chunk: {e}")))?;

        for chunk in coord_chunks {
            // chunk is (1, MANIFOLD_DIM) → squeeze to (MANIFOLD_DIM,)
            let coord = chunk.squeeze(0)
                .map_err(|e| CoreError::Bridge(format!("coord squeeze: {e}")))?;

            // 1. Project: state = proj @ coord  → (302,)
            let coord_2d = coord.reshape((d, 1))
                .map_err(|e| CoreError::Bridge(format!("coord reshape: {e}")))?;
            let state = proj_gpu.matmul(&coord_2d)
                .map_err(|e| CoreError::Bridge(format!("proj matmul: {e}")))?;

            // 2. Synaptic propagation: state = syn @ state  → (302,)
            let state = w_gpu.matmul(&state)
                .map_err(|e| CoreError::Bridge(format!("route matmul: {e}")))?;

            // 3. Tanh activation
            let a_gpu = state.tanh()
                .map_err(|e| CoreError::Bridge(format!("tanh: {e}")))?;

            // 4. Co-firing outer product
            let col = a_gpu.reshape((n, 1))
                .map_err(|e| CoreError::Bridge(format!("rf col: {e}")))?;
            let row = a_gpu.reshape((1, n))
                .map_err(|e| CoreError::Bridge(format!("rf row: {e}")))?;
            let co_firing = col.matmul(&row)
                .map_err(|e| CoreError::Bridge(format!("rf matmul: {e}")))?;

            // 5. Mutual induction
            let induction = (co_firing * &mask_gpu)
                .map_err(|e| CoreError::Bridge(format!("rf induction mask: {e}")))?;
            let induction = induction.broadcast_mul(&lr_t)
                .map_err(|e| CoreError::Bridge(format!("rf induction lr: {e}")))?;
            v_gpu = (v_gpu + induction)
                .map_err(|e| CoreError::Bridge(format!("rf vel add: {e}")))?;

            // 6. Damping
            v_gpu = v_gpu.broadcast_mul(&damp_t)
                .map_err(|e| CoreError::Bridge(format!("rf vel damp: {e}")))?;
        }

        // 7. Forward propagation
        let mut s_new = (w_gpu + &v_gpu)
            .map_err(|e| CoreError::Bridge(format!("rf syn add: {e}")))?;
        s_new = s_new.maximum(&zero)
            .map_err(|e| CoreError::Bridge(format!("rf clamp: {e}")))?;
        s_new = self.normalize_rows(&s_new, n)?;

        let flat_w = self.tensor_to_host_2d(&s_new, "syn")?;
        let flat_v = self.tensor_to_host_2d(&v_gpu, "vel")?;
        Ok((flat_w, flat_v))
    }

    /// Check that all elements of a tensor are finite.
    fn check_finite(&self, t: &Tensor, label: &str) -> Result<()> {
        let data = t.to_vec1::<f32>()
            .map_err(|e| CoreError::Bridge(format!("{label} to_vec1: {e}")))?;
        let has_nan = data.iter().any(|&x| x.is_nan());
        let has_inf = data.iter().any(|&x| x.is_infinite());
        if has_nan {
            return Err(CoreError::Bridge(format!("{label} contains NaN")));
        }
        if has_inf {
            return Err(CoreError::Bridge(format!("{label} contains Inf")));
        }
        Ok(())
    }
}

/// Persistent GPU training state that eliminates per-batch weight transfers.
///
/// Keeps synapses, velocities, mask, and input_projection resident on the
/// CUDA device.  Each batch only uploads the coordinate tensor (65 KB for
/// B=1024) — no weight upload/download until checkpoint.
pub struct PersistentGpuState {
    pub(crate) engine: GpuEngine,
    pub(crate) proj: Tensor,
    pub(crate) synapses: Tensor,
    pub(crate) velocities: Tensor,
    pub(crate) mask: Tensor,
}

impl PersistentGpuState {
    /// Upload all weight tensors once and return a persistent handle.
    pub fn new(
        eng: &GpuEngine,
        proj_flat: &[f32],
        synapses: &[f32],
        velocities: &[f32],
        mask_flat: &[f32],
    ) -> Result<Self> {
        let n = crate::worm_brain::WORM_NEURON_COUNT;
        let d = crate::geometry::MANIFOLD_DIM;
        Ok(Self {
            engine: GpuEngine { device: eng.device().clone() },
            proj: eng.host_to_tensor_2d_f32(proj_flat, &[n, d], "proj")?,
            synapses: eng.host_to_tensor_2d_f32(synapses, &[n, n], "syn")?,
            velocities: eng.host_to_tensor_2d_f32(velocities, &[n, n], "vel")?,
            mask: eng.host_to_tensor_2d_f32(mask_flat, &[n, n], "mask")?,
        })
    }

    /// Run one batch: route coordinates, train Hebbian updates, all on GPU.
    /// Only coordinates are uploaded — weights stay resident.
    pub fn train_batch(
        &mut self,
        coords_flat: &[f64],
        batch_size: usize,
        lr: f32,
        gamma: f32,
    ) -> Result<f32> {
        let n = crate::worm_brain::WORM_NEURON_COUNT;
        let d = crate::geometry::MANIFOLD_DIM;
        let damp_f32 = 1.0_f32 - gamma;
        let dev = self.engine.device();

        let lr_t = Tensor::new(&[lr], dev)
            .map_err(|e| CoreError::Bridge(format!("lr: {e}")))?;
        let damp_t = Tensor::new(&[damp_f32], dev)
            .map_err(|e| CoreError::Bridge(format!("damp: {e}")))?;
        let zero = Tensor::zeros(&[n, n], candle_core::DType::F32, dev)
            .map_err(|e| CoreError::Bridge(format!("zero: {e}")))?;

        // Upload coords as [B, d] → transpose to [d, B]
        let coords_gpu = self.engine.host_to_tensor_2d(coords_flat, &[batch_size, d], "coords")?;
        let coords_t = coords_gpu.transpose(0, 1)
            .map_err(|e| CoreError::Bridge(format!("coords T: {e}")))?;
        drop(coords_gpu);

        // Batched projection: all_states = proj @ coords^T → [302, B]
        let all_states = self.proj.matmul(&coords_t)
            .map_err(|e| CoreError::Bridge(format!("batch proj: {e}")))?;
        drop(coords_t);

        // Process each token sequentially (W changes after each token)
        let chunks = all_states.chunk(batch_size, 1)
            .map_err(|e| CoreError::Bridge(format!("chunk: {e}")))?;
        drop(all_states);

        let mut batch_norm_sum = 0.0_f64;
        let mut token_cnt = 0u32;

        for chunk in chunks {
            let state = chunk.squeeze(1)
                .map_err(|e| CoreError::Bridge(format!("squeeze: {e}")))?;
            let state_2d = state.reshape((n, 1))
                .map_err(|e| CoreError::Bridge(format!("reshape: {e}")))?;

            let state = self.synapses.matmul(&state_2d)
                .map_err(|e| CoreError::Bridge(format!("prop: {e}")))?;
            let a_gpu = state.tanh()
                .map_err(|e| CoreError::Bridge(format!("tanh: {e}")))?;

            // Activation L2 norm for display metrics
            let l2_sq: f32 = a_gpu.sqr()
                .map_err(|e| CoreError::Bridge(format!("sqr: {e}")))?
                .sum_all()
                .map_err(|e| CoreError::Bridge(format!("sum: {e}")))?
                .to_vec0::<f32>()
                .map_err(|e| CoreError::Bridge(format!("scalar: {e}")))?;
            batch_norm_sum += (l2_sq as f64).sqrt();
            token_cnt += 1;

            let col = a_gpu.reshape((n, 1))
                .map_err(|e| CoreError::Bridge(format!("col: {e}")))?;
            let row = a_gpu.reshape((1, n))
                .map_err(|e| CoreError::Bridge(format!("row: {e}")))?;
            let co_firing = col.matmul(&row)
                .map_err(|e| CoreError::Bridge(format!("outer: {e}")))?;

            let induction = (co_firing * &self.mask)
                .map_err(|e| CoreError::Bridge(format!("mask: {e}")))?;
            let induction = induction.broadcast_mul(&lr_t)
                .map_err(|e| CoreError::Bridge(format!("lr mul: {e}")))?;
            self.velocities = (self.velocities.clone() + induction)
                .map_err(|e| CoreError::Bridge(format!("v add: {e}")))?;
            self.velocities = self.velocities.broadcast_mul(&damp_t)
                .map_err(|e| CoreError::Bridge(format!("v damp: {e}")))?;
        }

        // Apply velocity → weight update
        let mut s_new = (self.synapses.clone() + &self.velocities)
            .map_err(|e| CoreError::Bridge(format!("w add: {e}")))?;
        s_new = s_new.maximum(&zero)
            .map_err(|e| CoreError::Bridge(format!("clamp: {e}")))?;
        s_new = self.engine.normalize_rows(&s_new, n)?;
        self.synapses = s_new;

        let mean_norm = if token_cnt > 0 {
            (batch_norm_sum / token_cnt as f64) as f32
        } else {
            0.0
        };

        Ok(mean_norm)
    }

    /// Copy the current weight matrix back to host (for checkpointing).
    pub fn download_synapses(&self) -> Result<Vec<f32>> {
        self.engine.tensor_to_host_2d(&self.synapses, "syn")
    }

    /// Copy the current velocity matrix back to host.
    pub fn download_velocities(&self) -> Result<Vec<f32>> {
        self.engine.tensor_to_host_2d(&self.velocities, "vel")
    }
}
