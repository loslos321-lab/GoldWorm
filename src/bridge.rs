//! UtoPiCorn_LM Token Interface Alignment
//!
//! Builds the native token/logit projection interface. Maps N-D manifold
//! coordinates into an arbitrary target space — vocabulary logits, biological
//! neural layers (e.g. C. elegans connectome), or any downstream geometry —
//! without hardcoded shape constants, avoiding any hidden uncalibrated
//! random fallbacks.
//!
//! ## Bridge Invariants
//! - The projection matrix is always initialized to deterministic zeros.
//! - Weights must be explicitly set via [`VocabularyProjection::set_weights`]
//!   before inference; calling [`project_to_logits`] on uninitialized weights
//!   returns a zero vector (deterministic, not random).
//! - All dimension changes are validated at the API boundary.
//! - No `unwrap()` or `panic!` calls in the production library path.
//!
//! ## Rust 2024 RPITIT Upgrades
//! - `BatchProjector` and `TokenEncoder` traits use native RPITIT to return
//!   `impl Iterator` directly, eliminating `Box<dyn Iterator>` heap allocations.
//! - Explicit `+ use<'a>` lifetime capture on batch iterators that borrow
//!   from the input coordinate slice.
//! - `#[inline]` on every hot-path method for zero-cost monomorphization.

use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use crate::{CoreError, Result, geometry::TokenCoord};

/// Trait for batch token projection.
///
/// Uses native **RPITIT** to return an iterator without `Box<dyn Iterator>`
/// allocations, guaranteeing static dispatch and zero auxiliary heap overhead.
///
/// The `+ use<'a>` captures the input slice lifetime explicitly, aligning with
/// Rust 2024's stricter `impl Trait` lifetime capture rules for non-method
/// (free/associated) functions.
pub trait BatchProjector {
    /// Project a batch of coordinates into logit vectors.
    fn project_batch<'a>(
        &self,
        coords: &'a [TokenCoord],
    ) -> impl Iterator<Item = Result<Array1<f64>>> + use<'a, '_, Self>;
}

/// Trait for encoding token coordinates into probability distributions.
///
/// Uses native RPITIT with explicit lifetime capture to stream probability
/// vectors without dynamic dispatch or heap-allocated trait objects.
pub trait TokenEncoder {
    /// Encode a batch of coordinates into probability vectors.
    fn encode_batch<'a>(
        &self,
        coords: &'a [TokenCoord],
    ) -> impl Iterator<Item = Result<Array1<f64>>> + use<'a, '_, Self>;
}

/// Maps N-D token coordinates into an arbitrary target logit space.
///
/// The projection is an affine map: `logits = W · coord + b`.
/// Both `target_dim` and `coordinate_dim` are set at construction time
/// and can represent vocabularies, neuron populations, or any manifold.
#[derive(Clone, Debug)]
pub struct VocabularyProjection {
    /// Weight matrix of shape `(target_dim, coordinate_dim)`.
    weights: Array2<f64>,
    /// Optional bias vector of shape `(target_dim,)`.
    bias: Option<Array1<f64>>,
    /// Target output dimension (e.g. vocabulary size, neuron count).
    /// Kept as `vocab_size` for backward compatibility with existing callers.
    pub vocab_size: usize,
    /// Authoritative target output dimension; equals `vocab_size`.
    pub target_dim: usize,
    /// Input coordinate dimension (formerly fixed to 16; now fully dynamic).
    pub coordinate_dim: usize,
    /// Temperature scaling applied to logits before probability conversion.
    /// `temperature = 1.0` means no scaling.
    pub temperature: f64,
    /// Whether weights have been explicitly set (not just default zeros).
    weights_initialized: bool,
}

impl VocabularyProjection {
    /// Create a new `VocabularyProjection`.
    ///
    /// The weight matrix is initialized to zeros. No random values are
    /// injected. Callers **must** supply calibrated weights via
    /// [`set_weights`] before meaningful inference.
    ///
    /// # Arguments
    /// * `vocab_size` – Size of the target output dimension (vocabulary,
    ///   neuron population, etc.).
    /// * `embedding_dim` – Dimension of the input coordinate space. Any
    ///   positive integer is accepted; no longer constrained to 16.
    ///
    /// # Errors
    /// * [`CoreError::InvalidDimension`] if `embedding_dim` is zero.
    #[inline]
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Result<Self> {
        if embedding_dim == 0 {
            return Err(CoreError::InvalidDimension {
                expected: 1,
                got: 0,
            });
        }

        Ok(Self {
            weights: Array2::zeros((vocab_size, embedding_dim)),
            bias: Some(Array1::zeros(vocab_size)),
            vocab_size,
            target_dim: vocab_size,
            coordinate_dim: embedding_dim,
            temperature: 1.0,
            weights_initialized: false,
        })
    }

    /// Set the projection weights from a pre-calibrated matrix.
    ///
    /// # Arguments
    /// * `weights` – Matrix of shape `(target_dim, coordinate_dim)`.
    ///
    /// # Errors
    /// * [`CoreError::InvalidDimension`] if the shape does not match.
    #[inline]
    pub fn set_weights(&mut self, weights: Array2<f64>) -> Result<()> {
        let shape = weights.shape();
        if shape != &[self.target_dim, self.coordinate_dim] {
            return Err(CoreError::InvalidDimension {
                expected: self.target_dim,
                got: shape[0],
            });
        }
        self.weights = weights;
        self.weights_initialized = true;
        Ok(())
    }

    /// Set the bias vector.
    ///
    /// # Arguments
    /// * `bias` – Vector of length `vocab_size`.
    ///
    /// # Errors
    /// * [`CoreError::InvalidDimension`] if the length does not match.
    #[inline]
    pub fn set_bias(&mut self, bias: Array1<f64>) -> Result<()> {
        if bias.len() != self.target_dim {
            return Err(CoreError::InvalidDimension {
                expected: self.target_dim,
                got: bias.len(),
            });
        }
        self.bias = Some(bias);
        Ok(())
    }

    /// Clear the bias (set to zero implicit bias).
    #[inline]
    pub fn clear_bias(&mut self) {
        self.bias = None;
    }

    /// Project a 128-D token coordinate into the vocabulary logit space.
    ///
    /// Performs the affine map `logits = W · coord + b` followed by
    /// temperature scaling `logits / temperature`.
    ///
    /// If weights have not been explicitly set, the result is a deterministic
    /// zero vector (or the bias, if any), never a random fallback.
    ///
    /// # Arguments
    /// * `coord` – A 128-D [`TokenCoord`].
    ///
    /// # Errors
    /// * [`CoreError::Bridge`] if the projection produces non-finite values.
    #[inline]
    pub fn project_to_logits(&self, coord: &TokenCoord) -> Result<Array1<f64>> {
        let coord_arr = coord.inner();

        // Guard: coordinate dimension must match the projection matrix columns.
        // Checked here rather than relying on ndarray to panic.
        if coord_arr.len() != self.coordinate_dim {
            return Err(CoreError::InvalidDimension {
                expected: self.coordinate_dim,
                got: coord_arr.len(),
            });
        }

        // Matrix-vector multiplication: logits = W · coord
        let mut logits = self.weights.dot(coord_arr);

        // Add bias if present
        if let Some(ref b) = self.bias {
            logits = &logits + b;
        }

        // Temperature scaling
        if self.temperature != 1.0 && self.temperature > 0.0 {
            logits = &logits / self.temperature;
        }

        // Validate finiteness
        for &v in logits.iter() {
            if !v.is_finite() {
                return Err(CoreError::Bridge(
                    "logit projection produced non-finite values".to_string(),
                ));
            }
        }

        Ok(logits)
    }

    /// Convert a logit vector to token probabilities via numerically stable softmax.
    ///
    /// # Arguments
    /// * `logits` – Logit vector of length `vocab_size`.
    ///
    /// # Returns
    /// Probability vector of the same length, summing to 1.0.
    ///
    /// # Errors
    /// * [`CoreError::Bridge`] if the softmax produces non-finite values.
    #[inline]
    pub fn logits_to_probs(logits: &Array1<f64>) -> Result<Array1<f64>> {
        if logits.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let max_logit = logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let shifted = logits.map(|x| x - max_logit);
        let exp_shifted = shifted.map(|x| x.exp());
        let sum_exp = exp_shifted.sum();

        if sum_exp == 0.0 || !sum_exp.is_finite() {
            return Err(CoreError::Bridge(
                "softmax denominator is zero or non-finite".to_string(),
            ));
        }

        Ok(&exp_shifted / sum_exp)
    }

    /// Project a coordinate directly to token probabilities.
    ///
    /// Convenience method combining [`project_to_logits`] and [`logits_to_probs`].
    #[inline]
    pub fn project_to_probs(&self, coord: &TokenCoord) -> Result<Array1<f64>> {
        let logits = self.project_to_logits(coord)?;
        Self::logits_to_probs(&logits)
    }

    /// Check whether the projection weights have been explicitly calibrated.
    #[inline]
    pub fn is_calibrated(&self) -> bool {
        self.weights_initialized
    }

    /// Load a `VocabularyProjection` from a safetensors file via zero-copy mmap.
    ///
    /// The file is memory-mapped and the header is parsed in-place. The weight
    /// tensor is searched under the keys `"lm_head.weight"`, `"output.weight"`,
    /// and `"model.embed_tokens.weight"` (in that order). The bias tensor is
    /// optional and is searched under `"lm_head.bias"`, `"output.bias"`, and
    /// `"bias"`.
    ///
    /// # Arguments
    /// * `path` – Path to the `.safetensors` file.
    ///
    /// # Errors
    /// * [`CoreError::Bridge`] if the file cannot be opened, mapped, or parsed.
    /// * [`CoreError::Bridge`] if the required weight key is missing.
    /// * [`CoreError::Bridge`] if the weight tensor is not 2-D.
    /// * [`CoreError::Bridge`] if a bias tensor is present but its length does
    ///   not match the inferred `target_dim`.
    #[inline]
    pub fn load_from_safetensors<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(&path).map_err(|e| {
            CoreError::Bridge(format!("failed to open safetensors file: {}", e))
        })?;

        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            CoreError::Bridge(format!("failed to mmap safetensors file: {}", e))
        })?;

        let st = SafeTensors::deserialize(&mmap).map_err(|e| {
            CoreError::Bridge(format!("failed to deserialize safetensors: {}", e))
        })?;

        // --- Weight tensor ---
        const WEIGHT_KEYS: &[&str] = &[
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight",
        ];
        let weight_view = WEIGHT_KEYS
            .iter()
            .find_map(|&k| st.tensor(k).ok())
            .ok_or_else(|| {
                CoreError::Bridge(format!(
                    "safetensors missing weight key (tried: {})",
                    WEIGHT_KEYS.join(", ")
                ))
            })?;

        let weight_shape = weight_view.shape();
        if weight_shape.len() != 2 {
            return Err(CoreError::Bridge(format!(
                "weight tensor must be 2-D, got shape {:?}",
                weight_shape
            )));
        }

        // Dimensions are inferred dynamically from the tensor header.
        // No hardcoded shape assertions — the loader is geometry-agnostic.
        let target_dim = weight_shape[0];
        let coordinate_dim = weight_shape[1];

        let weight_vec = tensor_bytes_to_f64_vec(weight_view.data(), weight_view.dtype())?;
        let weights = Array2::from_shape_vec((target_dim, coordinate_dim), weight_vec)
            .map_err(|e| CoreError::Bridge(format!("weight tensor reshape failed: {}", e)))?;

        // --- Bias tensor (optional) ---
        const BIAS_KEYS: &[&str] = &["lm_head.bias", "output.bias", "bias"];
        let bias_opt = BIAS_KEYS.iter().find_map(|&k| st.tensor(k).ok());

        let mut proj = Self::new(target_dim, coordinate_dim)?;
        proj.set_weights(weights)?;

        if let Some(bias_view) = bias_opt {
            let bias_shape = bias_view.shape();
            if bias_shape.len() != 1 || bias_shape[0] != target_dim {
                return Err(CoreError::Bridge(format!(
                    "bias tensor shape {:?} does not match target_dim {}",
                    bias_shape, target_dim
                )));
            }
            let bias_vec = tensor_bytes_to_f64_vec(bias_view.data(), bias_view.dtype())?;
            let bias = Array1::from_vec(bias_vec);
            proj.set_bias(bias)?;
        }

        Ok(proj)
    }
}

/// Backward-project a 302-neuron activation state onto the token manifold
/// and return the token whose coordinate is nearest (cosine similarity).
///
/// The input projection matrix `P` has shape `(302, 16)`.  The backward map
/// uses `Pᵀ` (16 × 302) to reduce the activation vector to a 128-D "thought
/// coordinate", then searches a token pool for the closest match.
///
/// If the token pool is empty, returns an empty string.
#[inline]
pub fn decode_state_to_token(
    neuron_activations: &Array1<f64>,
    input_projection: &Array2<f64>,
    token_coords: &[(String, Array1<f32>)],
) -> String {
    if token_coords.is_empty() {
        return String::new();
    }

    // Backward projection: 128-D ← 302-D via transpose of the input map.
    let backward = input_projection.t();
    let thought = backward.dot(neuron_activations);

    let thought_norm = thought.dot(&thought).sqrt();

    // Nearest-neighbour search via cosine similarity.
    let mut best_idx = 0;
    let mut best_sim = f64::NEG_INFINITY;

    for (i, (_, tc)) in token_coords.iter().enumerate() {
        let tc_norm = tc.dot(tc).sqrt() as f64;
        if thought_norm > 1e-15 && tc_norm > 1e-15 {
            let sim = thought.dot(&tc.mapv(|v| v as f64)) / (thought_norm * tc_norm);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
    }

    token_coords[best_idx].0.clone()
}

/// Convert raw safetensors byte data into a `Vec<f64>` respecting the dtype.
///
/// Supports `F64` (direct little-endian conversion) and `F32` (cast to f64).
/// All other dtypes are rejected with a [`CoreError::Bridge`] error.
#[inline]
pub(crate) fn tensor_bytes_to_f64_vec(data: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f64>> {
    match dtype {
        safetensors::Dtype::F64 => {
            data.chunks_exact(8)
                .map(|chunk| {
                    let bytes: [u8; 8] = chunk.try_into().map_err(|_| {
                        CoreError::Bridge("misaligned F64 tensor bytes".to_string())
                    })?;
                    Ok(f64::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<_>>>()
        }
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
                        CoreError::Bridge("misaligned F32 tensor bytes".to_string())
                    })?;
                    Ok(f32::from_le_bytes(bytes) as f64)
                })
                .collect::<Result<Vec<_>>>()
        }
        other => Err(CoreError::Bridge(format!(
            "unsupported safetensors dtype: {:?}; expected F64 or F32",
            other
        ))),
    }
}

/// Convert raw safetensors byte data into a `Vec<f32>` respecting the dtype.
///
/// Supports `F32` (direct little-endian conversion) and `F64` (cast to f32).
/// All other dtypes are rejected with a [`CoreError::Bridge`] error.
#[inline]
pub(crate) fn tensor_bytes_to_f32_vec(data: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>> {
    match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
                        CoreError::Bridge("misaligned F32 tensor bytes".to_string())
                    })?;
                    Ok(f32::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<_>>>()
        }
        safetensors::Dtype::F64 => {
            data.chunks_exact(8)
                .map(|chunk| {
                    let bytes: [u8; 8] = chunk.try_into().map_err(|_| {
                        CoreError::Bridge("misaligned F64 tensor bytes".to_string())
                    })?;
                    Ok(f64::from_le_bytes(bytes) as f32)
                })
                .collect::<Result<Vec<_>>>()
        }
        other => Err(CoreError::Bridge(format!(
            "unsupported safetensors dtype: {:?}; expected F32 or F64",
            other
        ))),
    }
}

impl BatchProjector for VocabularyProjection {
    #[inline]
    fn project_batch<'a>(
        &self,
        coords: &'a [TokenCoord],
    ) -> impl Iterator<Item = Result<Array1<f64>>> + use<'a, '_> {
        coords.iter().map(|c| self.project_to_logits(c))
    }
}

impl TokenEncoder for VocabularyProjection {
    #[inline]
    fn encode_batch<'a>(
        &self,
        coords: &'a [TokenCoord],
    ) -> impl Iterator<Item = Result<Array1<f64>>> + use<'a, '_> {
        coords.iter().map(|c| self.project_to_probs(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::MANIFOLD_DIM;

    #[test]
    fn test_bridge_dimensions() {
        let bridge = VocabularyProjection::new(100, MANIFOLD_DIM).unwrap();
        assert_eq!(bridge.vocab_size, 100);
        assert!(!bridge.is_calibrated());
    }

    #[test]
    fn test_uninitialized_projection_is_zero() {
        let bridge = VocabularyProjection::new(10, MANIFOLD_DIM).unwrap();
        let coord = TokenCoord::zeros();
        let logits = bridge.project_to_logits(&coord).unwrap();
        assert_eq!(logits.len(), 10);
        // Without bias, all logits should be zero
        assert!(logits.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let probs = VocabularyProjection::logits_to_probs(&logits).unwrap();
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        // Previously rejected embedding_dim != 16; now any positive dim is valid.
        // The residual guard is zero-dimension construction.
        let result = VocabularyProjection::new(100, 0);
        assert!(result.is_err(), "expected error for zero coordinate_dim");
        if let Err(CoreError::InvalidDimension { expected, got }) = result {
            assert_eq!(expected, 1);
            assert_eq!(got, 0);
        } else {
            panic!("expected InvalidDimension error for zero coordinate_dim");
        }

        // set_weights still enforces shape consistency (target_dim × coordinate_dim).
        let mut proj = VocabularyProjection::new(100, 8).unwrap();
        let bad_weights = ndarray::Array2::<f64>::zeros((50, 8)); // wrong target_dim
        assert!(proj.set_weights(bad_weights).is_err());
    }

    #[test]
    fn test_batch_projector_static_dispatch() {
        let bridge = VocabularyProjection::new(5, MANIFOLD_DIM).unwrap();
        let coords = vec![TokenCoord::zeros(); 3];
        let batch: Vec<_> = bridge.project_batch(&coords).collect();
        assert_eq!(batch.len(), 3);
        for logits in batch {
            let logits = logits.unwrap();
            assert_eq!(logits.len(), 5);
            assert!(logits.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_encoder_static_dispatch() {
        let bridge = VocabularyProjection::new(4, MANIFOLD_DIM).unwrap();
        let coords = vec![TokenCoord::zeros(); 2];
        let batch: Vec<_> = bridge.encode_batch(&coords).collect();
        assert_eq!(batch.len(), 2);
        for probs in batch {
            let probs = probs.unwrap();
            assert_eq!(probs.len(), 4);
            let sum: f64 = probs.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_safetensors_mock_loading() -> crate::Result<()> {
        // Use a biologically-grounded target_dim (C. elegans: 302 neurons)
        // to verify that the loader is no longer locked to 152_064.
        // coordinate_dim stays MANIFOLD_DIM to match TokenCoord::zeros() dimensionality.
        let target_dim: usize = 302;
        let coordinate_dim: usize = crate::geometry::MANIFOLD_DIM;

        // --- Build deterministic F32 weight bytes ---
        let weight_f32: Vec<f32> = (0..target_dim * coordinate_dim)
            .map(|i| (i as f32) * 0.05)
            .collect();
        let weight_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        // --- Build deterministic F32 bias bytes ---
        let bias_f32: Vec<f32> = (0..target_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let bias_bytes: Vec<u8> = bias_f32
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        // --- Serialize via safetensors ---
        let mut tensors = std::collections::HashMap::new();
        let weight_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![target_dim, coordinate_dim],
            &weight_bytes,
        )
        .unwrap();
        let bias_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![target_dim],
            &bias_bytes,
        )
        .unwrap();
        tensors.insert("lm_head.weight", weight_view);
        tensors.insert("lm_head.bias", bias_view);

        let serialized = safetensors::serialize(&tensors, &None).unwrap();

        // --- Write temp file ---
        let temp_path =
            std::env::temp_dir().join("utophiecorn_mock.safetensors");
        std::fs::write(&temp_path, &serialized).unwrap();

        // --- Load via the engine ---
        let proj = VocabularyProjection::load_from_safetensors(&temp_path)?;

        // --- Dynamic field assertions ---
        assert!(proj.is_calibrated());
        // target_dim and coordinate_dim must be populated from the tensor header.
        assert_eq!(proj.target_dim, target_dim);
        assert_eq!(proj.coordinate_dim, coordinate_dim);
        // vocab_size is kept for backward compatibility and equals target_dim.
        assert_eq!(proj.vocab_size, target_dim);

        // --- End-to-end projection: W·0 + b = b invariant ---
        let coord = TokenCoord::zeros();
        let logits = proj.project_to_logits(&coord)?;
        assert_eq!(logits.len(), target_dim);

        // For zero input: logit[i] == bias[i] == i * 0.01 (cast F32 → F64).
        // bias[0] = 0 * 0.01 = 0.0
        assert!(logits[0].abs() < 1e-5, "logits[0] should equal bias[0] = 0.0");
        // bias[1] = 1 * 0.01 = 0.01 (within F32 round-trip tolerance)
        assert!((logits[1] - 0.01_f32 as f64).abs() < 1e-5,
            "logits[1] should equal bias[1] ≈ 0.01");

        // --- Cleanup ---
        let _ = std::fs::remove_file(&temp_path);

        Ok(())
    }
}
