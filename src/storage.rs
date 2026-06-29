//! Weight persistence via safetensors.
//!
//! Exports the trained 302×302 synaptic weight matrix to the standard
//! safetensors format for checkpoint storage and downstream reload.

use std::path::Path;
use ndarray::Array2;
use crate::{CoreError, Result, bridge};
use crate::geometry::MANIFOLD_DIM;
use crate::worm_brain::WormBrain;

/// Serialise the synaptic weight matrix and input projection of a trained
/// [`WormBrain`] to a safetensors file on disk.
///
/// Tensors stored under the keys `"synaptic_weights"` (302×302 F32) and
/// `"input_projection"` (302×16 F32).
///
/// # Errors
/// * [`CoreError::Bridge`] if the tensor view cannot be constructed.
/// * [`CoreError::Bridge`] if safetensors serialisation fails.
/// * [`CoreError::Bridge`] if the file cannot be written.
#[inline]
pub fn save_brain_state(brain: &WormBrain, path: &Path) -> Result<()> {
    let mut tensors = std::collections::HashMap::new();

    // ── Synaptic weights (302×302) ──
    let syn_data: Vec<u8> = brain
        .synapses
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let syn_tensor = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        brain.synapses.shape().to_vec(),
        &syn_data,
    )
    .map_err(|e| CoreError::Bridge(format!("failed to create synapses TensorView: {}", e)))?;
    tensors.insert("synaptic_weights", syn_tensor);

    // ── Input projection (302×16) ──
    let proj_data: Vec<u8> = brain
        .input_projection()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let proj_tensor = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        vec![brain.neuron_count, brain.input_dim],
        &proj_data,
    )
    .map_err(|e| CoreError::Bridge(format!("failed to create input_projection TensorView: {}", e)))?;
    tensors.insert("input_projection", proj_tensor);

    let serialized = safetensors::serialize(&tensors, &None)
        .map_err(|e| CoreError::Bridge(format!("safetensors serialization failed: {}", e)))?;

    std::fs::write(path, &serialized).map_err(|e| {
        CoreError::Bridge(format!("failed to write safetensors to {}: {}", path.display(), e))
    })?;

    Ok(())
}

/// Deserialise a safetensors checkpoint and return the synaptic weight matrix
/// and input projection matrix.
///
/// The file must contain a tensor under the key `"synaptic_weights"` with
/// shape `(302, 302)` and a tensor under `"input_projection"` with shape
/// `(302, 16)`, both dtype `F32` or `F64`.
///
/// For backward compatibility, if the `"input_projection"` tensor is missing,
/// a baseline projection is returned.
///
/// # Errors
/// * [`CoreError::Bridge`] if the file cannot be opened, mapped, or parsed.
/// * [`CoreError::Bridge`] if the required tensor key is missing.
/// * [`CoreError::Bridge`] if the tensor shape is not `(302, 302)`.
#[inline]
pub fn load_brain_state(path: &Path) -> Result<(Array2<f32>, Array2<f32>)> {
    let file = std::fs::File::open(path)
        .map_err(|e| CoreError::Bridge(format!("failed to open {}: {}", path.display(), e)))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| CoreError::Bridge(format!("failed to mmap {}: {}", path.display(), e)))?;
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| CoreError::Bridge(format!("failed to deserialize safetensors: {}", e)))?;

    // ── Synaptic weights ──
    let syn_tensor = st.tensor("synaptic_weights").map_err(|_| {
        CoreError::Bridge("safetensors missing required tensor \"synaptic_weights\"".to_string())
    })?;
    let syn_shape = syn_tensor.shape().to_vec();
    if syn_shape.len() != 2 || syn_shape[0] != 302 || syn_shape[1] != 302 {
        return Err(CoreError::Bridge(format!(
            "expected (302, 302) tensor, got {:?}",
            syn_shape
        )));
    }
    let syn_data = bridge::tensor_bytes_to_f32_vec(syn_tensor.data(), syn_tensor.dtype())?;
    let synapses = Array2::from_shape_vec((302, 302), syn_data)
        .map_err(|e| CoreError::Bridge(format!("synaptic_weights reshape failed: {}", e)))?;

    // ── Input projection ──
    let proj = match st.tensor("input_projection") {
        Ok(proj_tensor) => {
            let proj_shape = proj_tensor.shape().to_vec();
            if proj_shape.len() != 2 || proj_shape[0] != 302 || proj_shape[1] != MANIFOLD_DIM {
                return Err(CoreError::Bridge(format!(
                    "expected (302, {MANIFOLD_DIM}) tensor for input_projection, got {:?}",
                    proj_shape
                )));
            }
            let proj_data =
                bridge::tensor_bytes_to_f32_vec(proj_tensor.data(), proj_tensor.dtype())?;
            Array2::from_shape_vec((302, MANIFOLD_DIM), proj_data)
                .map_err(|e| CoreError::Bridge(format!("input_projection reshape failed: {}", e)))?
        }
        // Backward compat: return baseline projection
        Err(_) => crate::worm_brain::WormBrain::new_baseline().input_projection().clone(),
    };

    Ok((synapses, proj))
}

/// Extended save: synapses + input_projection + EchoReservoir + CognitionState.
///
/// Additional tensors stored:
/// - `"associations"` — (302, 302) F32 Hebbian matrix
/// - `"reservoir_pre_decay"` — (1,) F32
/// - `"reservoir_hebbian_lr"` — (1,) F32
/// - `"reservoir_decay"` — (1,) F32
/// - `"cognition_kappa_gate"` — (1,) F32
/// - `"cognition_t_scale"` — (1,) F32
/// - `"cognition_alpha_echo"` — (1,) F32
#[inline]
pub fn save_brain_state_extended(
    brain: &WormBrain,
    path: &Path,
) -> Result<()> {
    // Keep backing byte buffers alive until after serialize().
    // TensorView borrows &[u8], so the data must outlive the HashMap.
    let mut tensors = std::collections::HashMap::new();

    // ── Synaptic weights ──
    let syn_data: Vec<u8> = brain
        .synapses
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let syn_tensor = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        brain.synapses.shape().to_vec(),
        &syn_data,
    )
    .map_err(|e| CoreError::Bridge(format!("failed to create synapses TensorView: {}", e)))?;
    tensors.insert("synaptic_weights", syn_tensor);

    // ── Input projection ──
    let proj_data: Vec<u8> = brain
        .input_projection
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let proj_tensor = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        vec![brain.neuron_count, brain.input_dim],
        &proj_data,
    )
    .map_err(|e| CoreError::Bridge(format!("failed to create input_projection TensorView: {}", e)))?;
    tensors.insert("input_projection", proj_tensor);

    // ── EchoReservoir (optional) ──
    let assoc_data: Vec<u8>;
    let pre_decay_data: Vec<u8>;
    let hebbian_lr_data: Vec<u8>;
    let decay_data: Vec<u8>;
    if let Some(ref reservoir) = brain.echo_reservoir {
        assoc_data = reservoir
            .associations
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let assoc_tensor = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            reservoir.associations.shape().to_vec(),
            &assoc_data,
        )
        .map_err(|e| CoreError::Bridge(format!("failed to create associations TensorView: {}", e)))?;
        tensors.insert("associations", assoc_tensor);

        pre_decay_data = reservoir.pre_decay.to_le_bytes().to_vec();
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![1],
            &pre_decay_data,
        )
        .map_err(|e| CoreError::Bridge(format!("failed to create tensor 'reservoir_pre_decay': {e}")))?;
        tensors.insert("reservoir_pre_decay", tv);

        hebbian_lr_data = reservoir.hebbian_lr.to_le_bytes().to_vec();
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![1],
            &hebbian_lr_data,
        )
        .map_err(|e| CoreError::Bridge(format!("failed to create tensor 'reservoir_hebbian_lr': {e}")))?;
        tensors.insert("reservoir_hebbian_lr", tv);

        decay_data = reservoir.decay.to_le_bytes().to_vec();
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![1],
            &decay_data,
        )
        .map_err(|e| CoreError::Bridge(format!("failed to create tensor 'reservoir_decay': {e}")))?;
        tensors.insert("reservoir_decay", tv);
    }

    // ── CognitionState ──
    let c = &brain.cognition;
    let kappa_data = c.kappa_gate.to_le_bytes().to_vec();
    let tv = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        vec![1],
        &kappa_data,
    )
    .map_err(|e| CoreError::Bridge(format!("failed to create tensor 'cognition_kappa_gate': {e}")))?;
    tensors.insert("cognition_kappa_gate", tv);

    let tscale_data = c.t_scale.to_le_bytes().to_vec();
    let tv = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        vec![1],
        &tscale_data,
    )
    .map_err(|e| CoreError::Bridge(format!("failed to create tensor 'cognition_t_scale': {e}")))?;
    tensors.insert("cognition_t_scale", tv);

    let alpha_data = c.alpha_echo.to_le_bytes().to_vec();
    let tv = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        vec![1],
        &alpha_data,
    )
    .map_err(|e| CoreError::Bridge(format!("failed to create tensor 'cognition_alpha_echo': {e}")))?;
    tensors.insert("cognition_alpha_echo", tv);

    let serialized = safetensors::serialize(&tensors, &None)
        .map_err(|e| CoreError::Bridge(format!("safetensors serialization failed: {}", e)))?;

    std::fs::write(path, &serialized).map_err(|e| {
        CoreError::Bridge(format!("failed to write safetensors to {}: {}", path.display(), e))
    })?;

    Ok(())
}

/// Extended load: returns (synapses, input_projection, optional EchoReservoir, CognitionState).
///
/// Backward-compatible: if `"associations"` is missing the reservoir is `None`.
pub fn load_brain_state_extended(
    path: &Path,
) -> Result<(Array2<f32>, Array2<f32>, Option<crate::hippocampus::EchoReservoir>, crate::hippocampus::CognitionState)> {
    let file = std::fs::File::open(path)
        .map_err(|e| CoreError::Bridge(format!("failed to open {}: {}", path.display(), e)))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| CoreError::Bridge(format!("failed to mmap {}: {}", path.display(), e)))?;
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| CoreError::Bridge(format!("failed to deserialize safetensors: {}", e)))?;

    // ── Synaptic weights ──
    let syn_tensor = st.tensor("synaptic_weights").map_err(|_| {
        CoreError::Bridge("safetensors missing required tensor \"synaptic_weights\"".to_string())
    })?;
    let syn_shape = syn_tensor.shape().to_vec();
    if syn_shape.len() != 2 || syn_shape[0] != 302 || syn_shape[1] != 302 {
        return Err(CoreError::Bridge(format!(
            "expected (302, 302) tensor, got {:?}",
            syn_shape
        )));
    }
    let syn_data = bridge::tensor_bytes_to_f32_vec(syn_tensor.data(), syn_tensor.dtype())?;
    let synapses = Array2::from_shape_vec((302, 302), syn_data)
        .map_err(|e| CoreError::Bridge(format!("synaptic_weights reshape failed: {}", e)))?;

    // ── Input projection ──
    let proj = match st.tensor("input_projection") {
        Ok(proj_tensor) => {
            let proj_shape = proj_tensor.shape().to_vec();
            if proj_shape.len() != 2 || proj_shape[0] != 302 || proj_shape[1] != MANIFOLD_DIM {
                return Err(CoreError::Bridge(format!(
                    "expected (302, {MANIFOLD_DIM}) tensor for input_projection, got {:?}",
                    proj_shape
                )));
            }
            let proj_data =
                bridge::tensor_bytes_to_f32_vec(proj_tensor.data(), proj_tensor.dtype())?;
            Array2::from_shape_vec((302, MANIFOLD_DIM), proj_data)
                .map_err(|e| CoreError::Bridge(format!("input_projection reshape failed: {}", e)))?
        }
        Err(_) => crate::worm_brain::WormBrain::new_baseline().input_projection().clone(),
    };

    // ── EchoReservoir (optional) ──
    let reservoir = if let Ok(assoc_tensor) = st.tensor("associations") {
        let assoc_shape = assoc_tensor.shape().to_vec();
        if assoc_shape.len() == 2 && assoc_shape[0] == 302 && assoc_shape[1] == 302 {
            let assoc_data =
                bridge::tensor_bytes_to_f32_vec(assoc_tensor.data(), assoc_tensor.dtype())?;
            let associations = Array2::from_shape_vec((302, 302), assoc_data)
                .map_err(|e| CoreError::Bridge(format!("associations reshape failed: {}", e)))?;

            let pre_decay = st.tensor("reservoir_pre_decay")
                .ok()
                .and_then(|t| bridge::tensor_bytes_to_f32_vec(t.data(), t.dtype()).ok())
                .and_then(|v| v.first().copied())
                .unwrap_or(0.99999);
            let hebbian_lr = st.tensor("reservoir_hebbian_lr")
                .ok()
                .and_then(|t| bridge::tensor_bytes_to_f32_vec(t.data(), t.dtype()).ok())
                .and_then(|v| v.first().copied())
                .unwrap_or(0.01);
            let decay = st.tensor("reservoir_decay")
                .ok()
                .and_then(|t| bridge::tensor_bytes_to_f32_vec(t.data(), t.dtype()).ok())
                .and_then(|v| v.first().copied())
                .unwrap_or(0.01);

            let mut res = crate::hippocampus::EchoReservoir::new(256);
            res.associations = associations;
            res.pre_decay = pre_decay;
            res.hebbian_lr = hebbian_lr;
            res.decay = decay;
            Some(res)
        } else {
            None
        }
    } else {
        None
    };

    // ── CognitionState ──
    let cognition = crate::hippocampus::CognitionState {
        kappa_gate: st.tensor("cognition_kappa_gate")
            .ok()
            .and_then(|t| bridge::tensor_bytes_to_f32_vec(t.data(), t.dtype()).ok())
            .and_then(|v| v.first().copied())
            .unwrap_or(1.0),
        t_scale: st.tensor("cognition_t_scale")
            .ok()
            .and_then(|t| bridge::tensor_bytes_to_f32_vec(t.data(), t.dtype()).ok())
            .and_then(|v| v.first().copied())
            .unwrap_or(50.0),
        alpha_echo: st.tensor("cognition_alpha_echo")
            .ok()
            .and_then(|t| bridge::tensor_bytes_to_f32_vec(t.data(), t.dtype()).ok())
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0),
    };

    Ok((synapses, proj, reservoir, cognition))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worm_brain::WormBrain;

    #[test]
    fn test_save_brain_state_roundtrip() {
        let brain = WormBrain::new_baseline();
        let temp = std::env::temp_dir().join("utophiecorn_test.safetensors");

        // Export
        save_brain_state(&brain, &temp).unwrap();
        assert!(temp.exists(), "safetensors file should exist");

        // Verify file is non-empty and has reasonable size
        let meta = std::fs::metadata(&temp).unwrap();
        assert!(meta.len() > 0, "safetensors file should not be empty");
        // 302×302×4 bytes ≈ 364_816 plus header overhead
        assert!(meta.len() > 364_000, "file too small for 302×302 F32 matrix");

        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn test_save_load_brain_state_roundtrip() {
        let brain = WormBrain::new_baseline();
        let temp = std::env::temp_dir().join("utophiecorn_roundtrip.safetensors");

        // Export → Import → verify identical matrices
        save_brain_state(&brain, &temp).unwrap();
        let (loaded_syn, loaded_proj) = load_brain_state(&temp).unwrap();
        assert_eq!(loaded_syn.shape(), &[302, 302]);
        assert_eq!(loaded_proj.shape(), &[302, MANIFOLD_DIM]);
        // f32 comparison: allow rounding
        for (a, b) in brain.synapses.iter().zip(loaded_syn.iter()) {
            assert!((a - b).abs() < 1e-6, "syn roundtrip mismatch: {} vs {}", a, b);
        }
        for (a, b) in brain.input_projection().iter().zip(loaded_proj.iter()) {
            assert!((a - b).abs() < 1e-6, "proj roundtrip mismatch: {} vs {}", a, b);
        }

        let _ = std::fs::remove_file(&temp);
    }
}