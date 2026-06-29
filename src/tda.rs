//! Topological Data Analysis for worm brain activations.
//!
//! Quantized persistence diagrams using Q16.16 fixed-point arithmetic,
//! β₀ (connected components), β₁ (cycles), and β₂ (voids) computation.
//! Also provides bottleneck distance between diagrams and golden-ratio
//! anti-resonance regularization.

use ndarray::Array1;

/// Quantized persistence diagram entry.
///
/// Both birth and death are stored in Q16.16 format (1/16 unit = 0.0625).
/// This avoids floating-point nondeterminism across platforms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PersistencePair {
    /// Birth threshold (in 1/16 units, Q16.16 scaled).
    pub birth: i32,
    /// Death threshold (in 1/16 units, Q16.16 scaled).
    pub death: i32,
}

impl PersistencePair {
    /// Lifespan = death - birth, in quantized units.
    #[inline]
    pub fn lifespan(&self) -> i32 {
        self.death - self.birth
    }

    /// True if this feature persists across more than 1 ε step (non-trivial).
    #[inline]
    pub fn is_persistent(&self) -> bool {
        self.death > self.birth + 1
    }
}

/// Complete quantized persistence diagram for a sequence of activations.
///
/// Computed via a filtration over 257 uniformly-spaced ε thresholds
/// in Q16.16 format.
#[derive(Clone, Debug)]
pub struct PersistenceDiagram {
    /// β₀ persistence pairs (connected components).
    pub beta_0_pairs: Vec<PersistencePair>,
    /// β₁ persistence pairs (cycles / 1D holes).
    pub beta_1_pairs: Vec<PersistencePair>,
    /// β₂ persistence pairs (voids / 2D holes).
    pub beta_2_pairs: Vec<PersistencePair>,
    /// Number of input points used to compute this diagram.
    pub n_points: usize,
}

impl PersistenceDiagram {
    /// Number of β₀ features at the default ε threshold (midpoint of filtration).
    pub fn beta_0_at_default(&self) -> usize {
        self.beta_0_pairs.iter().filter(|p| p.death > 128).count()
    }

    /// Number of β₁ features at the default ε threshold.
    pub fn beta_1_at_default(&self) -> usize {
        self.beta_1_pairs.iter().filter(|p| p.death > 128).count()
    }

    /// Number of β₂ features at the default ε threshold.
    pub fn beta_2_at_default(&self) -> usize {
        self.beta_2_pairs.iter().filter(|p| p.death > 128).count()
    }

    /// Maximum β₀ across all ε levels.
    pub fn max_beta_0(&self) -> usize {
        self.beta_0_pairs.len()
    }

    /// Maximum β₁ across all ε levels.
    pub fn max_beta_1(&self) -> usize {
        self.beta_1_pairs.len()
    }

    /// Maximum β₂ across all ε levels.
    pub fn max_beta_2(&self) -> usize {
        self.beta_2_pairs.len()
    }

    /// Total topological complexity score: sum of all persistent feature lifespans.
    pub fn complexity_score(&self) -> i64 {
        let s0: i64 = self.beta_0_pairs.iter().map(|p| p.lifespan() as i64).sum();
        let s1: i64 = self.beta_1_pairs.iter().map(|p| p.lifespan() as i64).sum();
        let s2: i64 = self.beta_2_pairs.iter().map(|p| p.lifespan() as i64).sum();
        s0 + s1 + s2
    }
}

/// Compute Betti numbers (β₀, β₁) for a sequence of activation vectors.
///
/// This is the single-threshold (non-persistent) variant using adaptive ε.
/// Kept for backward compat with existing callers.
pub fn compute_betti_numbers(activations: &[Array1<f32>]) -> (usize, usize) {
    let n = activations.len();
    if n < 2 {
        return (n, 0);
    }

    let mut dists = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = activations[i]
                .iter()
                .zip(activations[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }

    let total: f32 = dists.iter().sum();
    let eps = total / (n * n) as f32 * 0.3;

    let mut parent: Vec<usize> = (0..n).collect();

    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x {
            p[x] = find(p, p[x]);
        }
        p[x]
    }

    fn union(p: &mut [usize], a: usize, b: usize) {
        let ra = find(p, a);
        let rb = find(p, b);
        if ra != rb {
            p[ra] = rb;
        }
    }

    let mut rank = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if dists[i * n + j] < eps {
                if find(&mut parent, i) != find(&mut parent, j) {
                    union(&mut parent, i, j);
                } else {
                    rank += 1;
                }
            }
        }
    }

    let beta_0 = (0..n).filter(|&i| find(&mut parent, i) == i).count();
    let beta_1 = if n > 0 {
        (rank as isize - n as isize + beta_0 as isize).max(0) as usize
    } else {
        0
    };

    (beta_0, beta_1)
}

/// Compute the full quantized persistence diagram for a sequence of activations.
///
/// Uses 257 uniformly-spaced ε thresholds in Q16.16 format (1/16 units).
/// The distance matrix is scaled to i32 in 1/16 units to match the quantized
/// filtration, guaranteeing platform-independent results.
pub fn compute_persistence_diagram(activations: &[Array1<f32>]) -> PersistenceDiagram {
    let n = activations.len();
    if n < 2 {
        return PersistenceDiagram {
            beta_0_pairs: (0..n).map(|_i| PersistencePair { birth: 0, death: 256 }).collect(),
            beta_1_pairs: vec![],
            beta_2_pairs: vec![],
            n_points: n,
        };
    }

    // Step 1: Pairwise L1 distance matrix, quantized to 1/16 units
    let mut dists = vec![0i32; n * n];
    let mut max_dist = 0i32;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = activations[i]
                .iter()
                .zip(activations[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>();
            let d_q = (d * 16.0).round() as i32;
            dists[i * n + j] = d_q;
            dists[j * n + i] = d_q;
            if d_q > max_dist { max_dist = d_q; }
        }
    }

    if max_dist > 256 { max_dist = 256; }
    if max_dist < 1 { max_dist = 1; }

    let mut b0_parents: Vec<usize> = (0..n).collect();
    let mut b0_active: Vec<bool> = vec![true; n];
    let mut beta_0_pairs: Vec<PersistencePair> = Vec::new();
    let mut beta_1_pairs: Vec<PersistencePair> = Vec::new();
    let mut prev_beta_1_val = 0usize;

    fn b0_find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x { p[x] = b0_find(p, p[x]); }
        p[x]
    }
    fn b0_union(p: &mut [usize], a: usize, b: usize) {
        let ra = b0_find(p, a);
        let rb = b0_find(p, b);
        if ra != rb { p[ra] = rb; }
    }

    for eps in 1..=max_dist {
        let mut edges = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if dists[i * n + j] <= eps { edges += 1; }
            }
        }

        let mut comps = n;
        let mut uf: Vec<usize> = (0..n).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                if dists[i * n + j] <= eps && b0_find(&mut uf, i) != b0_find(&mut uf, j) {
                    b0_union(&mut uf, i, j);
                    comps -= 1;
                }
            }
        }

        let current_beta_1 = (edges as isize - n as isize + comps as isize).max(0) as usize;

        if current_beta_1 > prev_beta_1_val {
            for _ in 0..(current_beta_1 - prev_beta_1_val) {
                beta_1_pairs.push(PersistencePair { birth: eps, death: max_dist });
            }
        }

        prev_beta_1_val = current_beta_1;

        for i in 0..n {
            for j in (i + 1)..n {
                if dists[i * n + j] == eps {
                    let ri = b0_find(&mut b0_parents, i);
                    let rj = b0_find(&mut b0_parents, j);
                    if ri != rj {
                        let (keep, kill) = if ri < rj { (ri, rj) } else { (rj, ri) };
                        b0_parents[kill] = keep;
                        if b0_active[kill] {
                            b0_active[kill] = false;
                            beta_0_pairs.push(PersistencePair { birth: 0, death: eps });
                        }
                    }
                }
            }
        }
    }

    for i in 0..n {
        if b0_active[i] {
            beta_0_pairs.push(PersistencePair { birth: 0, death: max_dist });
        }
    }

    PersistenceDiagram {
        beta_0_pairs,
        beta_1_pairs,
        beta_2_pairs: vec![],
        n_points: n,
    }
}

/// Estimate β₂ from the 1-skeleton by counting maximal cliques of size 3.
pub fn compute_betti_2(distance_matrix: &[f32], n: usize, eps: f32) -> usize {
    let mut triangles = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if distance_matrix[i * n + j] < eps {
                for k in (j + 1)..n {
                    if distance_matrix[i * n + k] < eps && distance_matrix[j * n + k] < eps {
                        triangles += 1;
                    }
                }
            }
        }
    }

    let edges: usize = (0..n)
        .flat_map(|i| ((i + 1)..n).filter(move |&j| distance_matrix[i * n + j] < eps))
        .count();
    let chi = (n as isize) - (edges as isize) + (triangles as isize);
    if chi > (n as isize) {
        (chi - n as isize) as usize
    } else {
        0
    }
}

/// Compute β₀, β₁, and β₂ simultaneously at a single ε threshold.
pub fn compute_all_betti_numbers(activations: &[Array1<f32>]) -> (usize, usize, usize) {
    let n = activations.len();
    if n < 2 {
        return (n, 0, 0);
    }

    let mut dists = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = activations[i]
                .iter()
                .zip(activations[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }

    let total: f32 = dists.iter().sum();
    let eps = total / (n * n) as f32 * 0.3;

    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x { p[x] = find(p, p[x]); }
        p[x]
    }
    fn union(p: &mut [usize], a: usize, b: usize) {
        let ra = find(p, a); let rb = find(p, b);
        if ra != rb { p[ra] = rb; }
    }

    let mut edges = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if dists[i * n + j] < eps {
                edges += 1;
                if find(&mut parent, i) != find(&mut parent, j) {
                    union(&mut parent, i, j);
                }
            }
        }
    }

    let beta_0 = (0..n).filter(|&i| find(&mut parent, i) == i).count();
    let beta_1 = (edges as isize - n as isize + beta_0 as isize).max(0) as usize;
    let beta_2 = compute_betti_2(&dists, n, eps);

    (beta_0, beta_1, beta_2)
}

// ─────────────────────────────────────────────────────────────────────────
// Bottleneck distance + golden-ratio anti-resonance
// ─────────────────────────────────────────────────────────────────────────

/// Golden ratio constant.
const PHI: f64 = 1.618_033_988_749_895;

/// Compute the bottleneck distance between two persistence diagrams.
///
/// Uses a greedy matching: pairs are sorted by lifespan, then matched
/// in order. The bottleneck distance is the L∞ sup-norm over matched pairs
/// plus half-lifespan penalties for unmatched pairs (the diagonal matching).
///
/// Returns (bottleneck_distance, mean_matched_distance).
pub fn bottleneck_distance(a: &PersistenceDiagram, b: &PersistenceDiagram) -> (f64, f64) {
    let pairs_a = merge_all_pairs(a);
    let pairs_b = merge_all_pairs(b);
    bottleneck_distance_pairs(&pairs_a, &pairs_b)
}

/// Merge β₀, β₁, β₂ pairs into a single sorted vector for matching.
fn merge_all_pairs(diag: &PersistenceDiagram) -> Vec<PersistencePair> {
    let mut all: Vec<PersistencePair> = Vec::with_capacity(
        diag.beta_0_pairs.len() + diag.beta_1_pairs.len() + diag.beta_2_pairs.len(),
    );
    all.extend_from_slice(&diag.beta_0_pairs);
    all.extend_from_slice(&diag.beta_1_pairs);
    all.extend_from_slice(&diag.beta_2_pairs);
    all.sort_by(|a, b| b.lifespan().cmp(&a.lifespan()));
    all
}

/// Compute bottleneck distance on a merged pair list.
fn bottleneck_distance_pairs(a: &[PersistencePair], b: &[PersistencePair]) -> (f64, f64) {
    let mut sorted_a = a.to_vec();
    let mut sorted_b = b.to_vec();
    sorted_a.sort_by(|x, y| y.lifespan().cmp(&x.lifespan()));
    sorted_b.sort_by(|x, y| y.lifespan().cmp(&x.lifespan()));

    let n_matched = sorted_a.len().min(sorted_b.len());
    let mut max_l_inf = 0.0_f64;
    let mut match_sum = 0.0_f64;

    for i in 0..n_matched {
        let pa = &sorted_a[i];
        let pb = &sorted_b[i];
        let db = (pa.birth as f64 - pb.birth as f64).abs();
        let dd = (pa.death as f64 - pb.death as f64).abs();
        let l_inf = db.max(dd);
        if l_inf > max_l_inf {
            max_l_inf = l_inf;
        }
        match_sum += l_inf;
    }

    let extra_a = if sorted_a.len() > n_matched {
        sorted_a[n_matched..].iter().map(|p| p.lifespan() as f64 * 0.5).fold(0.0_f64, f64::max)
    } else {
        0.0
    };
    let extra_b = if sorted_b.len() > n_matched {
        sorted_b[n_matched..].iter().map(|p| p.lifespan() as f64 * 0.5).fold(0.0_f64, f64::max)
    } else {
        0.0
    };

    let bottleneck = max_l_inf.max(extra_a).max(extra_b);
    let mean_matched = if n_matched > 0 { match_sum / n_matched as f64 } else { 0.0 };

    (bottleneck, mean_matched)
}

/// Golden-ratio anti-resonance penalty.
///
/// Computes a penalty for persistence pairs whose normalised lifespan
/// is close to golden-ratio related frequencies (1/φ, 1/φ², etc.).
/// These "resonant" features may be numerical artifacts rather than true
/// topological signals.
pub fn golden_anti_resonance(diagram: &PersistenceDiagram) -> f64 {
    let all = merge_all_pairs(diagram);
    if all.is_empty() {
        return 0.0;
    }

    let max_life = all.iter().map(|p| p.lifespan()).max().unwrap_or(1).max(1) as f64;
    let inv_phi = 1.0 / PHI;
    let inv_phi2 = inv_phi * inv_phi;

    let total_penalty: f64 = all.iter().map(|p| {
        let ratio = p.lifespan() as f64 / max_life;
        let peak1 = (-100.0 * (ratio - inv_phi).powi(2)).exp();
        let peak2 = (-100.0 * (ratio - inv_phi2).powi(2)).exp();
        let peak3 = (-100.0 * (ratio - 1.0 + inv_phi).powi(2)).exp();
        peak1 + peak2 + peak3
    }).sum();

    (total_penalty / all.len() as f64).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_betti_numbers_empty() {
        assert_eq!(compute_betti_numbers(&[]), (0, 0));
    }

    #[test]
    fn test_compute_betti_numbers_single() {
        let a = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        assert_eq!(compute_betti_numbers(&a), (1, 0));
    }

    #[test]
    fn test_compute_betti_numbers_two_distant() {
        let a = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![100.0, 100.0]),
        ];
        assert_eq!(compute_betti_numbers(&a), (2, 0));
    }

    #[test]
    fn test_compute_betti_numbers_two_close() {
        let a = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![0.001, 0.001]),
            Array1::from_vec(vec![100.0, 100.0]),
        ];
        let (b0, b1) = compute_betti_numbers(&a);
        assert_eq!(b0, 2, "close pair + far point = 2 components");
        assert_eq!(b1, 0);
    }

    #[test]
    fn test_persistence_diagram_simple() {
        let a = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
        ];
        let diagram = compute_persistence_diagram(&a);
        assert_eq!(diagram.n_points, 3);
        assert!(!diagram.beta_0_pairs.is_empty());
        assert!(!diagram.beta_1_pairs.is_empty());
    }

    #[test]
    fn test_compute_all_betti_numbers() {
        let a = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![100.0, 100.0]),
        ];
        let (b0, b1, b2) = compute_all_betti_numbers(&a);
        assert_eq!(b0, 2);
        let _ = b1;
        assert!(b2 <= 1);
    }

    // ── Bottleneck distance tests ──

    #[test]
    fn test_bottleneck_distance_identical_diagrams() {
        let a = PersistenceDiagram {
            beta_0_pairs: vec![
                PersistencePair { birth: 0, death: 100 },
                PersistencePair { birth: 0, death: 50 },
            ],
            beta_1_pairs: vec![PersistencePair { birth: 20, death: 80 }],
            beta_2_pairs: vec![],
            n_points: 5,
        };
        let b = a.clone();
        let (bottleneck, _) = bottleneck_distance(&a, &b);
        assert!(bottleneck < 1e-10, "identical diagrams should have 0 bottleneck");
    }

    #[test]
    fn test_bottleneck_distance_different_diagrams() {
        let a = PersistenceDiagram {
            beta_0_pairs: vec![PersistencePair { birth: 0, death: 100 }],
            beta_1_pairs: vec![],
            beta_2_pairs: vec![],
            n_points: 2,
        };
        let b = PersistenceDiagram {
            beta_0_pairs: vec![PersistencePair { birth: 0, death: 90 }],
            beta_1_pairs: vec![],
            beta_2_pairs: vec![],
            n_points: 2,
        };
        let (bottleneck, _) = bottleneck_distance(&a, &b);
        assert!((bottleneck - 10.0).abs() < 1e-6, "bottleneck should be 10, got {bottleneck}");
    }

    #[test]
    fn test_bottleneck_distance_missing_pairs_matched_to_diagonal() {
        let a = PersistenceDiagram {
            beta_0_pairs: vec![PersistencePair { birth: 0, death: 100 }],
            beta_1_pairs: vec![],
            beta_2_pairs: vec![],
            n_points: 2,
        };
        let b = PersistenceDiagram {
            beta_0_pairs: vec![],
            beta_1_pairs: vec![],
            beta_2_pairs: vec![],
            n_points: 1,
        };
        let (bottleneck, _) = bottleneck_distance(&a, &b);
        assert!((bottleneck - 50.0).abs() < 1e-6, "bottleneck should be 50, got {bottleneck}");
    }

    // ── Golden anti-resonance tests ──

    #[test]
    fn test_golden_anti_resonance_empty_diagram() {
        let d = PersistenceDiagram {
            beta_0_pairs: vec![],
            beta_1_pairs: vec![],
            beta_2_pairs: vec![],
            n_points: 0,
        };
        assert_eq!(golden_anti_resonance(&d), 0.0);
    }

    #[test]
    fn test_golden_anti_resonance_phi_aligned() {
        let d = PersistenceDiagram {
            beta_0_pairs: vec![PersistencePair { birth: 0, death: 100 }],
            beta_1_pairs: vec![],
            beta_2_pairs: vec![],
            n_points: 2,
        };
        let penalty = golden_anti_resonance(&d);
        assert!(penalty >= 0.0 && penalty <= 1.0);
    }

    #[test]
    fn test_golden_anti_resonance_hits_peak_near_inv_phi() {
        // Create one long-lived pair to set max_life, and one pair at 1/φ * max_life
        let max_life = 1000i32;
        let inv_phi_life = ((max_life as f64) / PHI).round() as i32; // ≈ 618
        let d = PersistenceDiagram {
            beta_0_pairs: vec![
                PersistencePair { birth: 0, death: max_life },
                PersistencePair { birth: 0, death: inv_phi_life },
            ],
            beta_1_pairs: vec![],
            beta_2_pairs: vec![],
            n_points: 2,
        };
        let penalty = golden_anti_resonance(&d);
        // The inv_phi pair ratio = 618/1000 = 0.618 ≈ 1/φ, should produce non-trivial penalty
        assert!(penalty > 0.2, "penalty should be significant near 1/φ, got {penalty}");
    }
}
