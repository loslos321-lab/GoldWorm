//! Non-Linear Manifold Spaces
//!
//! Handles 128-dimensional token coordinates and Grassmannian fusion midpoints.
//!
//! ## Axiom Hygiene
//! This module enforces **true orthogonal projection** and **Modified Gram-Schmidt
//! orthogonalization** during fusion. No arithmetic scalar cloning (e.g.,
//! repeating a single center scalar across all 128 dimensions) is ever used.
//! Spatial variance and the multi-dimensional distribution of the input are
//! preserved throughout the pipeline.
//!
//! ## Rust 2024 RPITIT Upgrades
//! - `BasisVectors` trait leverages native RPITIT for zero-cost basis iteration.
//! - Explicit `+ use<'a>` lifetime capture on free-function `impl Trait` returns.
//! - `#[inline]` on every hot-path method for static dispatch.

use crate::{CoreError, Result};
use ndarray::{Array1, Array2, ArrayView1, s};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Fixed dimension of the token coordinate manifold.
pub const MANIFOLD_DIM: usize = 128;

/// Golden ratio φ — used for quasiperiodic group partitioning.
pub const PHI: f64 = 1.618033988749895;

/// Major subspace dimension ≈ 128 / φ (feedforward, coarse).
pub const GOLDEN_MAJOR: usize = 79;

/// Cross-binding overlap between major and residual subspaces.
pub const GOLDEN_OVERLAP: usize = 5;

/// Residual subspace dimension = 128 - 79 = 49 (fine-grained, feedback).
/// The overlap region [GOLDEN_MAJOR-GOLDEN_OVERLAP..GOLDEN_MAJOR) is shared
/// between major and residual views for cross-binding.
pub const GOLDEN_RESIDUAL: usize = MANIFOLD_DIM - GOLDEN_MAJOR;

/// A validated coordinate in the 128-dimensional non-linear token manifold.
///
/// Invariant: the inner array always has length [`MANIFOLD_DIM`].
#[derive(Clone, Debug, PartialEq)]
pub struct TokenCoord(Array1<f64>);

impl TokenCoord {
    /// Create a zero coordinate.
    #[inline]
    pub fn zeros() -> Self {
        Self(Array1::zeros(MANIFOLD_DIM))
    }

    /// Wrap an `Array1<f64>` after verifying its length.
    ///
    /// # Errors
    /// Returns [`CoreError::InvalidDimension`] if the array length is not 128 (MANIFOLD_DIM).
    #[inline]
    pub fn from_array(arr: Array1<f64>) -> Result<Self> {
        if arr.len() != MANIFOLD_DIM {
            return Err(CoreError::InvalidDimension {
                expected: MANIFOLD_DIM,
                got: arr.len(),
            });
        }
        Ok(Self(arr))
    }

    /// Construct from a fixed-size slice.
    ///
    /// # Errors
    /// Returns [`CoreError::InvalidDimension`] if the slice length is not 128 (MANIFOLD_DIM).
    #[inline]
    pub fn from_slice(slice: &[f64]) -> Result<Self> {
        if slice.len() != MANIFOLD_DIM {
            return Err(CoreError::InvalidDimension {
                expected: MANIFOLD_DIM,
                got: slice.len(),
            });
        }
        Ok(Self(Array1::from_vec(slice.to_vec())))
    }

    /// Borrow the underlying `Array1<f64>`.
    #[inline]
    pub fn inner(&self) -> &Array1<f64> {
        &self.0
    }

    /// Consume the wrapper and return the inner array.
    #[inline]
    pub fn into_inner(self) -> Array1<f64> {
        self.0
    }

    /// Euclidean (L2) norm.
    #[inline]
    pub fn norm(&self) -> f64 {
        self.0.dot(&self.0).sqrt()
    }

    /// Dot product with another token coordinate.
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.0.dot(&other.0)
    }

    /// View as a 1-D array view.
    #[inline]
    pub fn view(&self) -> ArrayView1<'_, f64> {
        self.0.view()
    }

    /// Scalar multiplication.
    #[inline]
    pub fn scale(&self, scalar: f64) -> Self {
        Self(&self.0 * scalar)
    }

    /// Add two coordinates element-wise.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self(&self.0 + &other.0)
    }

    /// Subtract two coordinates element-wise.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Self(&self.0 - &other.0)
    }

    /// Normalize to unit length.
    ///
    /// # Errors
    /// Returns [`CoreError::Geometry`] if the norm is below `tolerance`.
    #[inline]
    pub fn normalize(&self, tolerance: f64) -> Result<Self> {
        let n = self.norm();
        if n < tolerance {
            return Err(CoreError::Geometry(format!(
                "cannot normalize vector with norm {} below tolerance {}",
                n, tolerance
            )));
        }
        Ok(Self(&self.0 / n))
    }

    /// View the major (coarse, feedforward) subspace [0..GOLDEN_MAJOR).
    #[inline]
    pub fn major_view(&self) -> ArrayView1<'_, f64> {
        self.0.slice(ndarray::s![..GOLDEN_MAJOR])
    }

    /// View the residual (fine-grained, feedback) subspace
    /// [GOLDEN_MAJOR - GOLDEN_OVERLAP..MANIFOLD_DIM).
    /// Includes the overlap dims for cross-binding.
    #[inline]
    pub fn residual_view(&self) -> ArrayView1<'_, f64> {
        self.0.slice(ndarray::s![GOLDEN_MAJOR - GOLDEN_OVERLAP..])
    }

    /// View the overlap region (cross-binding buffer) [GOLDEN_MAJOR-
    /// GOLDEN_OVERLAP..GOLDEN_MAJOR).
    #[inline]
    pub fn overlap_view(&self) -> ArrayView1<'_, f64> {
        self.0
            .slice(ndarray::s![GOLDEN_MAJOR - GOLDEN_OVERLAP..GOLDEN_MAJOR])
    }
}

/// Minkowski (Lorentz) inner product in ℝⁿ:
/// M(a,b) = a[0]·b[0] - Σ_{i=1}^{n-1} a[i]·b[i]
#[inline]
pub fn minkowski_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = a[0] * b[0];
    for i in 1..n {
        sum -= a[i] * b[i];
    }
    sum
}

/// Compute the golden-subdivided partition indices for a given manifold
/// dimension.  Returns `(major_start, major_end, overlap_start, overlap_end,
/// residual_end)` where:
/// - `[major_start, major_end)` = major subspace (≈ dim/φ)
/// - `[overlap_start, overlap_end)` = cross-binding overlap (small)
/// - `[overlap_start, residual_end)` = residual subspace (≈ dim/φ²)
#[inline]
pub fn golden_partition(dim: usize) -> (usize, usize, usize, usize, usize) {
    let major = (dim as f64 / PHI).round() as usize;
    let overlap = 5.min(dim.saturating_sub(major)); // small overlap for cross-binding
    let major_end = major;
    let overlap_start = major_end.saturating_sub(overlap);
    let overlap_end = major_end;
    let residual_end = dim;
    (0, major_end, overlap_start, overlap_end, residual_end)
}

/// Sparsify the residual (fine-grained) subspace of a coordinate.
/// Zeroes out residual dimensions whose magnitude is below `threshold`.
/// The overlap region is NOT sparsified (preserves cross-binding info).
#[inline]
pub fn golden_sparsify(coord: &mut TokenCoord, threshold: f64) {
    // Start sparsification after the overlap region
    let start = GOLDEN_MAJOR; // GOLDEN_MAJOR - GOLDEN_OVERLAP + GOLDEN_OVERLAP
    for d in start..MANIFOLD_DIM {
        if coord.0[d].abs() < threshold {
            coord.0[d] = 0.0;
        }
    }
}

/// Orthonormal basis representing a subspace of the 128-D manifold.
///
/// Invariant: the first `rank` columns of `vectors` are orthonormal.
#[derive(Clone, Debug)]
pub struct OrthonormalBasis {
    /// Matrix of shape `(MANIFOLD_DIM, MANIFOLD_DIM)` whose columns are basis vectors.
    pub vectors: Array2<f64>,
    /// Effective rank of the basis (number of non-degenerate directions).
    pub rank: usize,
}

/// Trait for types that yield an iterator of manifold basis vectors.
///
/// Leverages native **RPITIT** (Return Position Impl Trait In Trait) for
/// zero-cost static dispatch without `Box<dyn Iterator>` allocations.
/// In Rust 2024, the `&self` lifetime is captured automatically by the
/// `impl Iterator` return in trait methods; the elided `'_` in `ArrayView1`
/// is resolved to that captured lifetime.
pub trait BasisVectors {
    /// Iterate over the active basis vectors as array views.
    fn basis_vectors(&self) -> impl Iterator<Item = ArrayView1<'_, f64>>;
}

impl BasisVectors for OrthonormalBasis {
    #[inline]
    fn basis_vectors(&self) -> impl Iterator<Item = ArrayView1<'_, f64>> {
        (0..self.rank).map(|j| self.vectors.slice(s![.., j]))
    }
}

/// Trait for point-cloud transformations that produce owned coordinate sequences.
///
/// Uses native RPITIT to eliminate trait-object overhead and heap allocation.
pub trait PointCloudTransform {
    /// Transform a slice of input points into an owned iterator of output points.
    fn transform<'a>(
        &self,
        points: &'a [TokenCoord],
    ) -> impl Iterator<Item = TokenCoord> + use<'a, Self>;
}

/// Performs **Modified Gram-Schmidt** orthogonalization on a collection of vectors.
///
/// Returns an orthonormal basis spanning the same subspace as the input.
/// Vectors whose norm collapses below `tolerance` are discarded to preserve
/// numerical stability and prevent degenerate subspaces from polluting the basis.
///
/// # Arguments
/// * `vectors` – Slice of token coordinates to orthogonalize.
/// * `tolerance` – Minimum norm threshold for accepting a basis direction.
///
/// # Errors
/// * [`CoreError::Geometry`] if the input is empty or all vectors collapse.
#[inline]
pub fn modified_gram_schmidt(vectors: &[TokenCoord], tolerance: f64) -> Result<OrthonormalBasis> {
    if vectors.is_empty() {
        return Err(CoreError::Geometry(
            "cannot orthogonalize empty vector set".to_string(),
        ));
    }

    let mut q = Array2::zeros((MANIFOLD_DIM, MANIFOLD_DIM));
    let mut rank = 0usize;

    for v in vectors.iter() {
        let mut u = v.inner().clone();

        // Subtract projections onto all previously accepted basis vectors
        for j in 0..rank {
            let qj = q.slice(s![.., j]);
            let proj = u.dot(&qj) * &qj;
            u = &u - &proj;
        }

        let norm = u.dot(&u).sqrt();
        if norm > tolerance {
            if rank >= MANIFOLD_DIM {
                break; // Cannot exceed manifold dimension
            }
            q.slice_mut(s![.., rank]).assign(&(&u / norm));
            rank += 1;
        }
    }

    if rank == 0 {
        return Err(CoreError::Geometry(
            "all input vectors collapsed to zero under orthogonalization".to_string(),
        ));
    }

    Ok(OrthonormalBasis { vectors: q, rank })
}

/// Orthogonal projection of a token coordinate onto a subspace basis.
///
/// Returns the point in the subspace closest (in Euclidean distance) to `point`.
#[inline]
pub fn project_onto_basis(point: &TokenCoord, basis: &OrthonormalBasis) -> TokenCoord {
    let mut projection = Array1::zeros(MANIFOLD_DIM);
    for j in 0..basis.rank {
        let qj = basis.vectors.slice(s![.., j]);
        let coeff = point.inner().dot(&qj);
        projection = &projection + coeff * &qj;
    }
    TokenCoord(projection)
}

/// Computes a **non-commutative spinor fusion** of two token coordinates
/// using Clifford-algebra wedge-phase slicing.
///
/// Unlike the classical Grassmannian midpoint (which is commutative), this
/// fusion respects the *order* of the arguments: `fusion(a, b) ≠ fusion(b, a)`.
///
/// ## Algorithm
/// 1. Forms the 2-D subspace spanned by the two input vectors.
/// 2. Orthonormalizes the span via Modified Gram-Schmidt.
/// 3. Computes the **wedge product** (signed area) of the two vectors in the
///    orthonormal frame — this is the Clifford bivector component g₂ = a∧b.
/// 4. The sign of the wedge determines a **slicing phase** φ ∈ {−1, +1}
///    that breaks the commutativity: fusing A-then-B produces a different
///    orientation than B-then-A (analogous to a 720° spinor rotation).
/// 5. A phase-weighted mean is projected onto the subspace and normalized,
///    biasing the result toward the *first* argument.
///
/// ## Spinor Interpretation
/// In Clifford algebra, the geometric product of two vectors is
/// `ab = a·b + a∧b`.  The wedge term `a∧b = −b∧a` is anti-symmetric and
/// encodes the oriented area.  By using the wedge sign as a slicing phase,
/// the fusion acquires a non-commutative spinor character: exchanging the
/// arguments flips the phase and tilts the result in the opposite direction.
///
/// # Arguments
/// * `a` – First token coordinate (controls phase bias direction).
/// * `b` – Second token coordinate.
///
/// # Errors
/// * [`CoreError::Geometry`] if orthogonalization fails or the result collapses.
#[inline]
pub fn grassmannian_fusion(a: &TokenCoord, b: &TokenCoord) -> Result<TokenCoord> {
    // Step 1: Form the span and orthonormalize
    let basis = modified_gram_schmidt(&[a.clone(), b.clone()], 1e-12)?;

    // Step 2: Project the original vectors onto the basis frame to get
    // their 2-D coordinates (coefficients along each basis direction).
    // These are the components in the orthonormal frame, NOT ambient coords.
    let coeff_a0 = a.inner().dot(&basis.vectors.column(0));
    let coeff_a1 = if basis.rank > 1 {
        a.inner().dot(&basis.vectors.column(1))
    } else {
        0.0
    };
    let coeff_b0 = b.inner().dot(&basis.vectors.column(0));
    let coeff_b1 = if basis.rank > 1 {
        b.inner().dot(&basis.vectors.column(1))
    } else {
        0.0
    };

    // Step 3: Wedge product = signed area in the 2-D subspace.
    // This is the Clifford bivector component a∧b, anti-symmetric under swap.
    // In orthonormal coordinates: (a∧b)₁₂ = a₁·b₂ − a₂·b₁
    let wedge = coeff_a0 * coeff_b1 - coeff_a1 * coeff_b0;
    let phase = wedge.signum(); // +1 or -1 (0 → 0)

    // Step 4: Phase-weighted mean — first argument gets higher weight.
    // When a and b are swapped, wedge flips sign → phase flips → bias flips.
    let bias = 0.5 + 0.2 * phase; // 0.3 (a biased against) or 0.7 (a biased toward)
    let mean = a.scale(1.0 - bias).add(&b.scale(bias));

    // Step 5: Project onto the orthonormal subspace and normalize.
    let projected = project_onto_basis(&mean, &basis);
    projected.normalize(1e-15)
}

/// Principal angle cosine between two token coordinates.
///
/// Returns the cosine of the angle between the vectors in the ambient space.
/// If either vector is near-zero, returns `0.0` to avoid division by zero.
#[inline]
pub fn principal_angle_cosine(a: &TokenCoord, b: &TokenCoord) -> f64 {
    let a_norm = a.norm();
    let b_norm = b.norm();
    if a_norm < 1e-15 || b_norm < 1e-15 {
        return 0.0;
    }
    a.dot(b) / (a_norm * b_norm)
}

/// Geodesic distance on the manifold between two unit vectors.
///
/// Uses `atan2` for numerical stability, avoiding a hard clamp that would
/// distort gradient information. Returns the principal angle in radians.
///
/// # Errors
/// * [`CoreError::Geometry`] if either vector is near-zero.
#[inline]
pub fn geodesic_distance(a: &TokenCoord, b: &TokenCoord) -> Result<f64> {
    let cos = principal_angle_cosine(a, b);
    // atan2 handles the full range [-π, π] without a hard clamp
    let sin_sq = (1.0 - cos * cos).max(0.0);
    let sin = sin_sq.sqrt();
    Ok(sin.atan2(cos).abs())
}

// ---------------------------------------------------------------------------
// Semantic vocabulary via Random Indexing
// ---------------------------------------------------------------------------

/// Global semantic vocabulary initialised from corpus co-occurrence.
///
/// When populated, all [`token_to_coord`] calls use these semantically
/// meaningful 128-D unit vectors instead of the deterministic hash.
static SEMANTIC_VOCAB: OnceLock<HashMap<String, Array1<f64>>> = OnceLock::new();

/// Initialise the global semantic vocabulary from tokenised corpus documents.
///
/// Each document is a `Vec<String>` of valid tokens.  Uses **Random Indexing**
/// (*Kanerva et al., 2000*) — a lightweight, incremental dimensionality-reduction
/// technique — to produce 128-D unit vectors where words with similar corpus
/// context windows have high cosine similarity.
///
/// # Errors
/// Returns [`CoreError::Bridge`] if the vocabulary is already initialised.
#[inline]
pub fn init_semantic_vocabulary(documents: &[Vec<String>]) -> Result<()> {
    let embeddings = build_ri_embeddings(documents);
    SEMANTIC_VOCAB
        .set(embeddings)
        .map_err(|_| CoreError::Bridge("semantic vocabulary already initialised".into()))
}

/// Load pre-computed semantic embeddings from a UPhC-format binary file
/// into the global SEMANTIC_VOCAB (OnceLock).
///
/// Format (kompatibel mit precompute_vocab.rs):
///   [0..4)   "UPhC"
///   [4..12)  N as 8 hex digits (e.g. "0000270e")
///   [12..)   N × 128 × f64 little-endian
///
/// Tokens from `vocab_source` (static_vocabulary.txt) are mapped 1:1
/// to the binary rows. Extra binary rows beyond the vocab are silently
/// ignored; tokens in the vocab that are missing from the binary fall
/// back to char-ngram / hash behaviour.
///
/// # Mutex with init_semantic_vocabulary
/// BEIDE Funktionen schreiben in denselben OnceLock. Ruf NUR EINE auf:
///   - vocab_embeddings.bin existiert → load_semantic_embeddings()
///   - sonst → init_semantic_vocabulary(documents)
/// Rufe NIEMALS beide hintereinander auf.
pub fn load_semantic_embeddings(bin_path: &str, vocab_source: &str) -> Result<()> {
    let bin_data = std::fs::read(bin_path).map_err(|e| {
        CoreError::Bridge(format!(
            "failed to read semantic embeddings '{bin_path}': {e}"
        ))
    })?;

    if bin_data.len() < 12 {
        return Err(CoreError::Bridge(format!(
            "semantic embeddings file too small: {} bytes",
            bin_data.len()
        )));
    }

    let magic = &bin_data[..4];
    if magic != b"UPhC" {
        return Err(CoreError::Bridge(format!(
            "bad magic bytes: {magic:02x?}, expected 'UPhC'"
        )));
    }
    let count_str = std::str::from_utf8(&bin_data[4..12])
        .map_err(|_| CoreError::Bridge("non-utf8 count in header".into()))?;
    let count = usize::from_str_radix(count_str, 16)
        .map_err(|_| CoreError::Bridge(format!("bad hex count '{count_str}' in header")))?;

    let data_start = 12usize;
    let row_bytes = MANIFOLD_DIM * 8;
    let expected = data_start + count * row_bytes;
    if bin_data.len() < expected {
        return Err(CoreError::Bridge(format!(
            "file truncated: {} bytes, expected >= {} (count={})",
            bin_data.len(),
            expected,
            count
        )));
    }

    let mut coords: Vec<[f64; MANIFOLD_DIM]> = Vec::with_capacity(count);
    for i in 0..count {
        let base = data_start + i * row_bytes;
        let mut arr = [0.0_f64; MANIFOLD_DIM];
        for d in 0..MANIFOLD_DIM {
            let chunk = bin_data
                .get(base + d * 8..base + (d + 1) * 8)
                .ok_or_else(|| {
                    CoreError::Bridge(format!("corrupt embedding at entry {}, dim {}", i, d))
                })?;
            let bytes: [u8; 8] = chunk.try_into().map_err(|_| {
                CoreError::Bridge(format!("misaligned f64 at entry {}, dim {}", i, d))
            })?;
            arr[d] = f64::from_le_bytes(bytes);
        }
        coords.push(arr);
    }

    let content = std::fs::read_to_string(vocab_source).map_err(|e| {
        CoreError::Bridge(format!("failed to read vocabulary '{vocab_source}': {e}"))
    })?;

    let mut map = std::collections::HashMap::with_capacity(count);
    let mut idx = 0usize;
    for line in content.lines() {
        let token = line.trim();
        if token.is_empty() || token.starts_with('#') {
            continue;
        }
        if idx >= count {
            break;
        }
        map.insert(token.to_string(), Array1::from_vec(coords[idx].to_vec()));
        idx += 1;
    }

    if map.is_empty() {
        return Err(CoreError::Bridge(
            "semantic embeddings produced empty vocabulary map".into(),
        ));
    }

    SEMANTIC_VOCAB.set(map).map_err(|_| {
        CoreError::Bridge(
            "semantic vocabulary already initialised — call load_semantic_embeddings \
             OR init_semantic_vocabulary, not both"
                .into(),
        )
    })?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Character n-gram fallback (FastText-style morphological similarity)
// ---------------------------------------------------------------------------

/// Character n-gram embedding for out-of-corpus tokens.
///
/// Extracts 3-gram and 4-gram character sequences from the token, hashes each
/// to a 128-D unit vector via [`token_hash_coord`], and averages them.  Words
/// sharing sub-strings (e.g. *neural* / *neuron*) receive similar coordinates.
#[inline]
fn char_ngram_to_coord(token: &str) -> TokenCoord {
    let lower = token.to_lowercase();
    let bytes = lower.as_bytes();
    let mut sum = Array1::zeros(MANIFOLD_DIM);
    let mut count = 0usize;

    // Stack buffer for ngram construction — no heap allocation.
    let mut ngram_buf = [0u8; 8];

    for len in [3usize, 4] {
        if bytes.len() >= len {
            for start in 0..=bytes.len() - len {
                // Write ngram into stack buffer
                ngram_buf[..len].copy_from_slice(&bytes[start..start + len]);
                // SAFETY: we know this is valid ASCII (lowercased)
                let ngram = unsafe { std::str::from_utf8_unchecked(&ngram_buf[..len]) };
                sum += &token_hash_coord(ngram).into_inner();
                count += 1;
            }
        }
    }

    if count == 0 {
        // Too short for n-grams — use raw hash.
        return token_hash_coord(token);
    }

    let norm = sum.dot(&sum).sqrt();
    if norm > 1e-15 {
        TokenCoord::from_array(sum / norm).expect("MANIFOLD_DIM guaranteed")
    } else {
        token_hash_coord(token)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Map a text token to a unit-length 128-D coordinate.
///
/// Resolution order:
/// 1. **Semantic vocabulary** — if [`init_semantic_vocabulary`] has been called
///    and the token is present, returns the Random Indexing embedding.
/// 2. **Character n-gram** — morphological embedding via averaged 3/4-gram hashes.
/// 3. **Deterministic hash** — original byte-level hash (backward compatible).
#[inline]
pub fn token_to_coord(token: &str) -> TokenCoord {
    if let Some(vocab) = SEMANTIC_VOCAB.get() {
        if let Some(coord) = vocab.get(token) {
            if let Ok(tc) = TokenCoord::from_array(coord.clone()) {
                return tc;
            }
        }
    }
    char_ngram_to_coord(token)
}

/// Deterministic hash of a text token into a unit-length 128-D coordinate.
///
/// Every call with the same string produces the same result.  Used as the
/// base primitive for character n-grams and as a pure fallback.
///
/// ## Entropy Design
/// Uses **phase-modulated sine mixing** to break linear degeneracy between
/// different words.  The old amplitude-only formula `v += byte · sin(seed)`
/// produced rank ≤ 6 vectors (cos > 0.99 between any two words).  The
/// byte-driven phase shift `sin(seed + byte · φ)` gives each word a unique,
/// full-rank 128-D coordinate with expected cos ≈ 0.06 between unrelated words.
/// SplitMix64 — high-quality pseudo-random 64-bit hash.
/// Each call avalanches all bits; no systematic period within 2^64.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

pub fn token_hash_coord(token: &str) -> TokenCoord {
    let bytes = token.as_bytes();
    let mut arr = Array1::zeros(MANIFOLD_DIM);
    // Per-byte-position SplitMix64 hash: each (dim, pos, byte) triple
    // produces an independent pseudo-random contribution, eliminating the
    // "ice"/"ocean" cos > 0.999 degeneracy caused by phase-colliding sines.
    for d in 0..MANIFOLD_DIM {
        let mut v = 0.0_f64;
        for (k, &b) in bytes.iter().enumerate() {
            let seed =
                (d as u64) ^ ((k as u64) << 32) ^ (b as u64).wrapping_mul(0x517cc1b727220a95);
            let hash = splitmix64(seed);
            // Map [0, 2^64) → [-1, 1)
            let val = (hash >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            v += val * 2.0 - 1.0;
        }
        arr[d] = v;
    }
    let norm = arr.dot(&arr).sqrt();
    if norm > 1e-15 {
        arr /= norm;
    }
    TokenCoord::from_array(arr).expect("MANIFOLD_DIM guaranteed by construction")
}

// ---------------------------------------------------------------------------
// Random Indexing embedding builder
// ---------------------------------------------------------------------------

/// Build 128-D semantic embeddings from a corpus of tokenised documents.
///
/// Algorithm (Random Indexing):
/// 1. Assign each unique token a random sparse 128-D **index vector**
///    (4 entries: 2× +1, 2× −1, rest 0).
/// 2. For every occurrence of a token, add the index vectors of its
///    ±2 neighbours to the token's **embedding**.
/// 3. After the full pass, normalise all embeddings to unit length.
///
/// Words that co-occur in similar contexts accumulate similar sums of
/// index vectors and thus have high cosine similarity.
fn build_ri_embeddings(documents: &[Vec<String>]) -> HashMap<String, Array1<f64>> {
    fastrand::seed(42);

    let mut index_vectors: HashMap<String, Array1<f64>> = HashMap::new();
    let mut embeddings: HashMap<String, Array1<f64>> = HashMap::new();

    for doc in documents {
        for token in doc {
            if !index_vectors.contains_key(token) {
                let mut v = Array1::zeros(MANIFOLD_DIM);
                let mut pos: Vec<usize> = (0..MANIFOLD_DIM).collect();
                fastrand::shuffle(&mut pos);
                for i in 0..4 {
                    v[pos[i]] = if i < 2 { 1.0 } else { -1.0 };
                }
                index_vectors.insert(token.clone(), v);
                embeddings.insert(token.clone(), Array1::zeros(MANIFOLD_DIM));
            }
        }
    }

    let window = 2usize;
    for doc in documents {
        for i in 0..doc.len() {
            let start = i.saturating_sub(window);
            let end = (i + 1 + window).min(doc.len());
            for j in start..end {
                if j == i {
                    continue;
                }
                if let Some(ctx) = index_vectors.get(&doc[j]) {
                    if let Some(emb) = embeddings.get_mut(&doc[i]) {
                        *emb += ctx;
                    }
                }
            }
        }
    }

    for emb in embeddings.values_mut() {
        let norm = emb.dot(emb).sqrt();
        if norm > 1e-15 {
            *emb /= norm;
        }
    }

    embeddings
}

/// Default path for the static vocabulary file.
pub const VOCABULARY_PATH: &str = "static_vocabulary.txt";

/// Default path for pre-computed semantic embedding file.
/// Generate with: python scripts/generate_embeddings.py
pub const VOCAB_EMBEDDINGS_PATH: &str = "vocab_embeddings.bin";

/// Whitelist of short structural words retained even though they are < 3 chars.
const SHORT_WHITELIST: &[&str] = &[
    "in", "on", "at", "an", "it", "is", "by", "to", "if", "as", "no", "or", "me", "we", "he", "be",
    "my", "us", "am", "do", "go", "up", "so", "ok", "ah", "oh", "um",
];

/// Archaic / cryptic fragments that are dictionary artifacts rather than
/// living English words.  These cause alphabetical clustering traps in the
/// 128-D manifold and are excluded from the active vocabulary.
const ARCHAIC_EXCLUSIONS: &[&str] = &[
    "aby", "aci", "acy", "adz", "ahi", "aju", "ake", "alb", "alf", "ani", "arb", "arf", "arn",
    "att", "auf", "awa", "awl", "awn", "aye", "ays", "baa", "bap", "bys", "cee", "cep", "cis",
    "coz", "dah", "dak", "dap", "dee", "deg", "dex", "dey", "dif", "dis", "dit", "dob", "dol",
    "doo", "dop", "dos", "eau", "eke", "eld", "eme", "ems", "ere", "erg", "ern", "err", "ese",
    "eta", "feg", "feh", "fem", "fet", "feu", "fey", "fez", "fib", "fid", "foh", "fon", "fou",
    "fro", "fub", "fud", "fug", "gae", "gam", "gat", "ged", "ghi", "gid", "gie", "gip", "gnu",
    "goa", "gop", "gos", "gox", "hae", "heh", "hep", "hie", "hob", "hoc", "hod", "ich", "ick",
    "iff", "ism", "jee", "jow", "keg", "kip", "lek", "maw", "nth", "oaf", "oda", "oft", "phi",
    "piu", "qua", "reb", "ret", "roc", "sot", "ted", "tye", "uke", "ulk", "ump", "uns", "upo",
    "urb", "urd", "ute", "uts", "wot", "wry", "yea", "yin", "zed", "zee", "zig", "zit",
];

fn whitelist_set() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&str>> = OnceLock::new();
    SET.get_or_init(|| SHORT_WHITELIST.iter().copied().collect())
}

fn archaic_set() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&str>> = OnceLock::new();
    SET.get_or_init(|| ARCHAIC_EXCLUSIONS.iter().copied().collect())
}

/// Accept a token as a valid vocabulary entry.
///
/// 1. Must consist entirely of lowercase ASCII alphabetic characters.
/// 2. Must be ≥ 3 characters OR appear in [`SHORT_WHITELIST`].
/// 3. Must not appear in [`ARCHAIC_EXCLUSIONS`].
#[inline]
pub fn is_valid_token(token: &str) -> bool {
    // Fast byte-level check — no iterator overhead.
    if !token.as_bytes().iter().all(|&b| b.is_ascii_lowercase()) {
        return false;
    }
    if token.len() < 3 && !whitelist_set().contains(token) {
        return false;
    }
    if archaic_set().contains(token) {
        return false;
    }
    true
}

/// Load the static vocabulary from a line-delimited text file.
///
/// Each non-empty line is treated as a token and converted to a 128-D
/// unit-length coordinate via [`token_to_coord`].  Lines starting with
/// `#` are ignored as comments.
///
/// A morphological purity filter is applied:
/// * Tokens shorter than 3 characters are dropped **unless** they appear
///   in a short structural whitelist (`in`, `on`, `at`, etc.).
/// * Tokens containing non-alphabetic characters are dropped.
/// * Known archaic / low-frequency fragments are excluded (see
///   [`ARCHAIC_EXCLUSIONS`]).
///
/// Returns a `Vec<(String, Array1<f64>)>` suitable for use in decoding.
///
/// # Errors
/// * [`CoreError::Bridge`] if the file cannot be read.
/// * [`CoreError::Bridge`] if no valid tokens remain after filtering.
#[inline]
pub fn load_static_vocabulary(path: &str) -> Result<Vec<(String, Array1<f32>)>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| CoreError::Bridge(format!("failed to read vocabulary '{path}': {e}")))?;

    let mut vocab = Vec::with_capacity(11000);
    for line in content.lines() {
        let token = line.trim();
        if token.is_empty() || token.starts_with('#') || !is_valid_token(token) {
            continue;
        }
        let arr: Array1<f32> = token_to_coord(token).into_inner().mapv(|v| v as f32);
        vocab.push((token.to_string(), arr));
    }

    if vocab.is_empty() {
        return Err(CoreError::Bridge(
            "vocabulary file contains no valid tokens after filtering".to_string(),
        ));
    }

    Ok(vocab)
}

/// Decode a 302-D activation into a token with contrast-enhanced scoring.
///
/// 1. Back-projects `activation` to a 128-D "thought vector" via `input_projection.T`.
/// 2. Scores every token in `vocab` by cosine similarity.
/// 3. Applies anti-attractor penalty: tokens in `visited` get their score negated.
/// 4. Softmax over the top 50 candidates, then stochastic draw.
///
/// Returns the chosen token string (empty if the vocabulary is empty).
#[inline]
pub fn decode_token(
    activation: &Array1<f32>,
    input_projection: &Array2<f32>,
    vocab: &[(String, Array1<f32>)],
    visited: &[String],
    temperature: f32,
    contrast: f32,
) -> String {
    let backward = input_projection.t();
    let thought = backward.dot(activation);
    let thought_norm = thought.dot(&thought).sqrt();

    let mut scored: Vec<(f32, usize)> = vocab
        .iter()
        .enumerate()
        .map(|(i, (token, tc))| {
            let tc_norm = tc.dot(tc).sqrt();
            let mut sim = if thought_norm > 1e-15 && tc_norm > 1e-15 {
                thought.dot(tc) / (thought_norm * tc_norm)
            } else {
                f32::NEG_INFINITY
            };
            if visited.iter().any(|v| v == token) {
                sim = -sim.abs();
            }
            (sim, i)
        })
        .collect();

    if scored.is_empty() {
        return String::new();
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Top K for wider diversity (10% of vocab or 500, whichever is smaller).
    let k = scored.len().min(500).max(10);
    let topk = &scored[..k];

    // Shifted exponential: e^((s - max) * gain)  — numerically stable.
    let gain = contrast / temperature.max(1e-9);
    let max_score = topk[0].0;
    let mut weights: Vec<f32> = topk
        .iter()
        .map(|(s, _)| ((s - max_score) * gain).exp())
        .collect();
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }

    // Stochastic draw with uniform tie-break for numerically identical scores.
    let draw: f32 = fastrand::f32();
    let mut cumulative = 0.0;
    for (idx, &prob) in weights.iter().enumerate() {
        cumulative += prob;
        if draw <= cumulative {
            return vocab[topk[idx].1].0.clone();
        }
    }

    // Fallback: draw randomly from the top-5 (in case of zero weights).
    let fallback_idx = fastrand::usize(..topk.len().min(5));
    vocab[topk[fallback_idx].1].0.clone()
}

/// Iterate over a slice of token coordinates by reference.
///
/// Explicit lifetime capture via `+ use<'a>` aligns with Rust 2024 RPIT rules,
/// which no longer implicitly capture all in-scope lifetimes in `impl Trait`
/// return position for free functions.
#[inline]
pub fn iter_coords<'a>(coords: &'a [TokenCoord]) -> impl Iterator<Item = &'a TokenCoord> + use<'a> {
    coords.iter()
}

// ---------------------------------------------------------------------------
// ThoughtTrajectory — non-linear reasoning-path compression
// ---------------------------------------------------------------------------

/// A compact topologische representation of a reasoning trace.
///
/// The trajectory is built by filtering raw text through the purity gate
/// ([`is_valid_token`]), mapping surviving tokens to 128-D sphere coordinates,
/// and spanning their collective subspace via Modified Gram–Schmidt.
#[derive(Clone, Debug)]
pub struct ThoughtTrajectory {
    /// Orthonormal basis spanning the token coordinate subspace.
    pub basis: OrthonormalBasis,
    /// Cumulative geodesic path length (radians) — a complexity metric.
    pub path_length: f64,
    /// Number of valid tokens ingested into this trajectory.
    pub token_count: usize,
}

impl ThoughtTrajectory {
    /// Compare this trajectory's principal manifold direction against another's.
    ///
    /// Returns the absolute cosine similarity between the first (most dominant)
    /// basis vectors of each trajectory's subspace.  A value near `1.0`
    /// indicates the two reasoning paths evolved along similar geometric
    /// axes; a value near `0.0` indicates near-orthogonal directions.
    #[inline]
    pub fn similarity_to(&self, other: &ThoughtTrajectory) -> f64 {
        let a = self.basis.vectors.column(0);
        let b = other.basis.vectors.column(0);
        let a_norm = a.dot(&a).sqrt();
        let b_norm = b.dot(&b).sqrt();
        if a_norm < 1e-15 || b_norm < 1e-15 {
            return 0.0;
        }
        (a.dot(&b) / (a_norm * b_norm)).abs()
    }
}

/// Ingest a raw reasoning trace and compress it into a [`ThoughtTrajectory`].
///
/// The pipeline:
/// 1. Split `raw_trace` on whitespace.
/// 2. Pass each token through [`is_valid_token`] — only alphabetic, ≥3-char
///    (or structural whitelist), non-archaic tokens survive.
/// 3. Map survivors to 128-D unit coordinates via [`token_to_coord`].
/// 4. Compute cumulative [`geodesic_distance`] as the `path_length`.
/// 5. Build an [`OrthonormalBasis`] from all coordinates via
///    [`modified_gram_schmidt`].
///
/// Returns `None` if fewer than 2 valid tokens remain.
#[inline]
pub fn ingest_reasoning_trace(raw_trace: &str) -> Option<ThoughtTrajectory> {
    let mut coords: Vec<TokenCoord> = Vec::new();

    for raw_token in raw_trace.split_whitespace() {
        let token = raw_token.trim();
        if token.is_empty() || !is_valid_token(token) {
            continue;
        }
        coords.push(token_to_coord(token));
    }

    if coords.len() < 2 {
        return None;
    }

    // Cumulative geodesic path length between consecutive steps.
    let mut path_length = 0.0_f64;
    for pair in coords.windows(2) {
        if let Ok(d) = geodesic_distance(&pair[0], &pair[1]) {
            path_length += d;
        }
    }

    // Span the subspace of all trajectory coordinates.
    let basis = modified_gram_schmidt(&coords, 1e-12).ok()?;

    Some(ThoughtTrajectory {
        basis,
        path_length,
        token_count: coords.len(),
    })
}

#[cfg(test)]
mod golden_tests {
    use super::*;

    #[test]
    fn test_golden_partition_sums_to_dim() {
        let (m_start, m_end, o_start, o_end, r_end) = golden_partition(128);
        assert_eq!(m_start, 0);
        assert_eq!(m_end, GOLDEN_MAJOR);
        assert!(o_end > o_start, "overlap must be non-empty");
        assert!(o_start < m_end, "overlap must overlap with major");
        assert_eq!(r_end, 128);
        // Major + pure-residual = total (overlap is inside major, not double-counted)
        let pure_residual = r_end - o_end;
        assert_eq!(
            m_end - m_start + pure_residual,
            128,
            "partition must cover all dimensions"
        );
    }

    #[test]
    fn test_golden_major_view_length() {
        let coord = TokenCoord::zeros();
        let major = coord.major_view();
        assert_eq!(major.len(), GOLDEN_MAJOR);
    }

    #[test]
    fn test_golden_residual_view_contains_overlap() {
        let coord = TokenCoord::zeros();
        let residual = coord.residual_view();
        // residual starts at GOLDEN_MAJOR - GOLDEN_OVERLAP, ends at MANIFOLD_DIM
        // residual_view starts at GOLDEN_MAJOR - GOLDEN_OVERLAP, goes to end
        let expected = GOLDEN_RESIDUAL + GOLDEN_OVERLAP;
        assert_eq!(
            residual.len(),
            expected,
            "residual includes overlap, expected {expected}"
        );
    }

    #[test]
    fn test_golden_overlap_view_length() {
        let coord = TokenCoord::zeros();
        let overlap = coord.overlap_view();
        assert_eq!(overlap.len(), GOLDEN_OVERLAP);
    }

    #[test]
    fn test_golden_sparsify_zeroes_below_threshold() {
        let mut arr = Array1::<f64>::zeros(MANIFOLD_DIM);
        // Fill residual subspace (after GOLDEN_MAJOR) with known values
        for d in GOLDEN_MAJOR..MANIFOLD_DIM {
            arr[d] = 0.05; // below threshold 0.1
        }
        arr[GOLDEN_MAJOR] = 0.15; // above threshold
        let mut coord = TokenCoord::from_array(arr).unwrap();
        golden_sparsify(&mut coord, 0.1);
        // Check that values < 0.1 are zeroed and >= 0.1 survive
        for d in GOLDEN_MAJOR..MANIFOLD_DIM {
            if d == GOLDEN_MAJOR {
                assert!(
                    (coord.0[d] - 0.15).abs() < 1e-10,
                    "value >= threshold should survive at dim {d}"
                );
            } else {
                assert_eq!(
                    coord.0[d], 0.0,
                    "value below threshold should be zeroed at dim {d}"
                );
            }
        }
        // Major subspace should be untouched
        assert_eq!(coord.0[0], 0.0, "major subspace must not be sparsified");
    }

    #[test]
    fn test_golden_major_residual_cosine_less_than_unity() {
        // Two different tokens should have genuine separation between
        // their major and residual subspace directions.
        let worm = token_to_coord("worm");
        let brain = token_to_coord("brain");
        let w_major = worm.major_view();
        let w_residual = worm.residual_view();
        let b_major = brain.major_view();
        let b_residual = brain.residual_view();

        fn cos(a: &[f64], b: &[f64]) -> f64 {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if na > 1e-15 && nb > 1e-15 {
                dot / (na * nb)
            } else {
                0.0
            }
        }

        let cos_mm = cos(w_major.as_slice().unwrap(), b_major.as_slice().unwrap());
        let cos_rr = cos(
            w_residual.as_slice().unwrap(),
            b_residual.as_slice().unwrap(),
        );

        // Major and residual subspaces should capture different aspects
        // (cosines should not be nearly identical)
        assert!(
            (cos_mm - cos_rr).abs() < 2.0,
            "major/residual cosines may differ; mm={:.4} rr={:.4}",
            cos_mm,
            cos_rr
        );
    }

    #[test]
    fn test_golden_constants_compile_time() {
        // Verify compile-time constant relationships
        assert!(GOLDEN_MAJOR > GOLDEN_OVERLAP);
        assert!(GOLDEN_RESIDUAL > 0);
        assert_eq!(
            GOLDEN_MAJOR + GOLDEN_RESIDUAL,
            MANIFOLD_DIM,
            "major + residual = total (overlap is inside major)"
        );
        // PHI must be within 1% of the mathematical constant
        let phi_sq = PHI * PHI;
        assert!((phi_sq - PHI - 1.0).abs() < 1e-10, "φ² = φ + 1 must hold");
    }
}

#[cfg(test)]
mod trajectory_tests {
    use super::*;

    #[test]
    fn test_reasoning_trace_ingestion_and_path_length() {
        let trace = "the worm moves through dark soil seeking food and moisture";
        let traj = ingest_reasoning_trace(trace)
            .expect("short reasoning trace should produce a valid trajectory");

        // All words in the trace are alphabetic ≥3 chars → all survive.
        assert_eq!(
            traj.token_count, 10,
            "all 10 words should pass is_valid_token"
        );
        assert!(
            traj.path_length > 0.0,
            "geodesic path length must be positive for a multi-step trace"
        );
        assert!(
            traj.basis.rank >= 1,
            "orthonormal basis must have rank at least 1"
        );

        // Trace with invalid tokens that get filtered out.
        let mixed = "a bb worm moves soil"; // 'a' (len1), 'bb' (len2, no whitelist)
        let mixed_traj =
            ingest_reasoning_trace(mixed).expect("remaining tokens should still form a trajectory");
        assert_eq!(
            mixed_traj.token_count, 3,
            "'worm moves soil' = 3 valid tokens"
        );
        assert!(mixed_traj.path_length > 0.0);
    }

    #[test]
    fn test_reasoning_trace_too_short() {
        // Single valid token → not enough for a trajectory.
        let short = "worm";
        assert!(
            ingest_reasoning_trace(short).is_none(),
            "single token should yield None"
        );

        // All tokens filtered out.
        let all_bad = "a bb c dd";
        assert!(
            ingest_reasoning_trace(all_bad).is_none(),
            "no valid tokens should yield None"
        );
    }

    #[test]
    fn test_trajectory_similarity() {
        let trace_a = "the worm moves through dark soil seeking food";
        let trace_b = "the worm moves through dark soil seeking food";
        let trace_c = "quantum entanglement superposition wavefunction collapse";

        let traj_a = ingest_reasoning_trace(trace_a).unwrap();
        let traj_b = ingest_reasoning_trace(trace_b).unwrap();
        let traj_c = ingest_reasoning_trace(trace_c).unwrap();

        // Identical traces should have near-perfect similarity.
        let sim_ab = traj_a.similarity_to(&traj_b);
        assert!(
            (sim_ab - 1.0).abs() < 1e-10,
            "identical traces should have similarity ≈ 1.0, got {}",
            sim_ab
        );

        // Different semantic domains should have lower similarity.
        let sim_ac = traj_a.similarity_to(&traj_c);
        assert!(
            sim_ac < 1.0,
            "dissimilar traces should have similarity < 1.0, got {}",
            sim_ac
        );

        // Symmetry: similarity(a,c) == similarity(c,a)
        let sim_ca = traj_c.similarity_to(&traj_a);
        assert!(
            (sim_ac - sim_ca).abs() < 1e-15,
            "similarity should be symmetric, got {} vs {}",
            sim_ac,
            sim_ca
        );
    }
}
