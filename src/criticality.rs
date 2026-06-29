//! Criticality Controller
//!
//! Implements the Quilez Bridge smooth-k annealing for controlling the
//! creativity/determinism trade-off in the sparsemax activation.
//!
//! The smooth-k parameter k controls the degree of blending:
//!   k → 0:  creative (smooth blend, softmax-like, all neurons active)
//!   k → ∞:  deterministic (sparsemax becomes WTA-like, one dominant neuron)
//!
//! # Quilez Bridge
//!
//! The smooth maximum smax(a,b,k) = (a·exp(k·a) + b·exp(k·b)) / (exp(k·a) + exp(k·b))
//! interpolates between max(a,b) (k→∞) and arithmetic mean (k→0).
//! This is mapped to the α-entmax parameter via α(k) = 1 + 2·exp(-k) so that:
//!   k = 0   → α = 3  (sparser than sparsemax)
//!   k = ∞   → α = 1  (softmax — all active)
//!   k = ln 2 → α = 2 (exact sparsemax)

/// Criticality controller with Quilez smooth-k annealing.
///
/// The controller tracks a single scalar k that is annealed over the
/// training schedule. The effect on activation sparsity is monotonic:
/// higher k → more creative (denser), lower k → more deterministic (sparser).
#[derive(Clone, Debug)]
pub struct CriticalityController {
    /// Quilez smooth-k parameter.
    /// - k → 0:   deterministic (α=3, very sparse, WTA-like)
    /// - k → ∞:   creative (α=1, softmax, all active)
    /// - k ≈ 0.69: α=2 (exact sparsemax)
    pub k: f32,
}

impl CriticalityController {
    /// Create a new controller with the given k value.
    #[inline]
    pub fn new(k: f32) -> Self {
        Self { k }
    }

    /// Default: k = 0.69 gives α ≈ 2 (exact sparsemax, standard behaviour).
    #[inline]
    pub fn default() -> Self {
        Self { k: std::f64::consts::LN_2 as f32 }
    }

    /// Map k to the α-entmax parameter.
    ///
    /// α(k) = 1 + 2·exp(-k)
    ///
    /// | k      | α    | behaviour       |
    /// |--------|------|-----------------|
    /// | 0      | 3    | very sparse     |
    /// | 0.69   | 2    | sparsemax       |
    /// | 2      | 1.27 | between         |
    /// | → ∞    | → 1  | softmax         |
    #[inline]
    pub fn alpha(&self) -> f32 {
        1.0 + 2.0 * (-self.k).exp()
    }

    /// Anneal k by a multiplicative factor (call each training step).
    ///
    /// `step` is a positive number that increases k towards ∞ (more creative).
    #[inline]
    pub fn anneal(&mut self, factor: f32) {
        self.k = self.k * factor;
    }

    /// Apply smooth sparsemax to a state slice using the current k.
    ///
    /// Delegates to `alpha_entmax` with α = alpha().
    #[inline]
    pub fn smooth_activation(&self, state: &mut [f32]) {
        let alpha = self.alpha();
        super::worm_brain::alpha_entmax(state, alpha);
    }
}

/// Criticality dashboard: real-time metrics for the branching process.
///
/// Computes the five core criticality observables from the 302-D activation
/// state and the training trajectory:
/// 1. **Branching ratio σ** — variance of the activation distribution.
/// 2. **Betti entropy** — Shannon entropy of the normalised activation.
/// 3. **Correlation length** — average pairwise cosine in activation space.
/// 4. **Avalanche exponent** — power-law exponent of cascade sizes.
/// 5. **Holonomy** — angular sum from gradient compression.
#[derive(Clone, Debug, Default)]
pub struct CriticalityDashboard {
    /// Branching ratio: variance of the activation distribution.
    /// σ ≈ 0: uniform (softmax), σ ≫ 0: sparse (nearly one-hot).
    pub branching_ratio: f64,
    /// Shannon entropy of the normalised activation (in nats).
    pub betti_entropy: f64,
    /// Mean pairwise cosine similarity between consecutive activations.
    pub correlation_length: f64,
    /// Power-law exponent of the avalanche size distribution.
    pub avalanche_exponent: f64,
    /// Holonomy: accumulated angular sum around the gradient trajectory.
    pub holonomy: f64,
}

impl CriticalityDashboard {
    /// Compute dashboard metrics from the current activation state and
    /// optional trajectory history.
    ///
    /// - `activation`: 302-D sparsemax output (probability simplex).
    /// - `trajectory`: optional recent activation history for correlation.
    /// - `holonomy`: holonomy from gradient compression (or 0.0).
    pub fn compute(
        activation: &[f32],
        trajectory: Option<&[Vec<f32>]>,
        holonomy: f64,
    ) -> Self {
        let branching_ratio = Self::compute_branching_ratio(activation);
        let betti_entropy = Self::compute_betti_entropy(activation);
        let correlation_length = trajectory
            .map(Self::compute_correlation_length)
            .unwrap_or(0.0);
        let avalanche_exponent = trajectory
            .map(Self::compute_avalanche_exponent)
            .unwrap_or(0.0);

        Self {
            branching_ratio,
            betti_entropy,
            correlation_length,
            avalanche_exponent,
            holonomy,
        }
    }

    /// Branching ratio σ: variance of the activation distribution.
    ///
    /// For a probability distribution p_i (sum = 1):
    ///   σ² = (1/n) Σ (p_i - 1/n)²  (population variance)
    /// σ ranges from 0 (uniform) to ~1/n (one-hot at index 0 gives
    /// σ² = (1-1/n)²/n + (n-1)*(0-1/n)²/n = ~1/n^2).
    #[inline]
    pub fn compute_branching_ratio(activation: &[f32]) -> f64 {
        let n = activation.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let inv_n = 1.0 / n;
        let variance: f64 = activation
            .iter()
            .map(|&p| {
                let diff = p as f64 - inv_n;
                diff * diff
            })
            .sum::<f64>()
            / n;
        variance.sqrt() * n // scale by n for interpretability
    }

    /// Betti entropy: Shannon entropy of the activation in nats.
    ///
    /// H = -Σ p_i · ln(p_i). Maximum = ln(n) for uniform, 0 for one-hot.
    #[inline]
    pub(crate) fn compute_betti_entropy(activation: &[f32]) -> f64 {
        let mut entropy = 0.0_f64;
        for &p in activation {
            let p = p as f64;
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Correlation length: mean pairwise cosine between consecutive
    /// activations in a trajectory.
    #[inline]
    fn compute_correlation_length(trajectory: &[Vec<f32>]) -> f64 {
        let n = trajectory.len();
        if n < 2 {
            return 0.0;
        }
        let mut total_cos = 0.0_f64;
        let mut count = 0usize;
        for pair in trajectory.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na > 1e-15 && nb > 1e-15 {
                total_cos += (dot / (na * nb)).clamp(-1.0, 1.0) as f64;
                count += 1;
            }
        }
        if count > 0 {
            total_cos / count as f64
        } else {
            0.0
        }
    }

    /// Avalanche exponent: rough estimate of the power-law exponent of
    /// cascade sizes. Cascade = number of consecutive activations whose
    /// total energy exceeds the mean + 1σ.
    #[inline]
    fn compute_avalanche_exponent(trajectory: &[Vec<f32>]) -> f64 {
        let n = trajectory.len();
        if n < 5 {
            return 0.0;
        }

        // Compute total energy per activation
        let energies: Vec<f64> = trajectory
            .iter()
            .map(|act| act.iter().map(|&v| v as f64).sum::<f64>())
            .collect();

        let mean_e: f64 = energies.iter().sum::<f64>() / n as f64;
        let var_e: f64 = energies
            .iter()
            .map(|&e| {
                let d = e - mean_e;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std_e = var_e.sqrt();
        let threshold = mean_e + std_e;

        // Find cascade sizes: consecutive steps above threshold
        let mut cascade_sizes: Vec<usize> = Vec::new();
        let mut current = 0usize;
        for &e in &energies {
            if e > threshold {
                current += 1;
            } else if current > 0 {
                cascade_sizes.push(current);
                current = 0;
            }
        }
        if current > 0 {
            cascade_sizes.push(current);
        }

        if cascade_sizes.len() < 3 {
            return 0.0;
        }

        // Rough power-law estimate: τ ≈ 1 + n / Σ ln(s_i)
        // Using the Hill-type estimator for power-law exponent
        let min_size = *cascade_sizes.iter().min().unwrap() as f64;
        let sum_log: f64 = cascade_sizes
            .iter()
            .map(|&s| (s as f64 / min_size).ln())
            .sum();
        let n_av = cascade_sizes.len() as f64;
        if sum_log > 1e-15 {
            1.0 + n_av / sum_log
        } else {
            2.0 // default: mean-field exponent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_criticality_controller_default() {
        let cc = CriticalityController::default();
        // default k = ln(2) ≈ 0.693
        assert!((cc.k - 0.693_147_2).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_at_k_zero() {
        let cc = CriticalityController::new(0.0);
        // α(0) = 1 + 2·exp(0) = 1 + 2 = 3
        assert!((cc.alpha() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_at_k_ln2() {
        let cc = CriticalityController::new(0.693_147_2);
        // α(ln2) = 1 + 2·exp(-ln2) = 1 + 2/2 = 2
        assert!((cc.alpha() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_alpha_at_large_k() {
        let cc = CriticalityController::new(10.0);
        // α(10) ≈ 1 + 2·exp(-10) ≈ 1.00009
        let alpha = cc.alpha();
        assert!(alpha > 1.0 && alpha < 1.001, "α({}) should approach 1", alpha);
    }

    #[test]
    fn test_anneal_increases_alpha_towards_one() {
        let mut cc = CriticalityController::new(0.0);
        cc.anneal(1.5); // k = 0 * 1.5 = 0
        assert!((cc.alpha() - 3.0).abs() < 1e-6);

        cc.k = 0.5;
        let alpha_before = cc.alpha();
        cc.anneal(2.0); // k = 1.0
        let alpha_after = cc.alpha();
        // k increased, so α decreased (closer to 1)
        assert!(alpha_after < alpha_before, "anneal should move α towards 1");
    }

    #[test]
    fn test_smooth_activation_sum_to_one() {
        let cc = CriticalityController::new(0.693_147_2); // sparsemax
        let mut state = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        cc.smooth_activation(&mut state);
        let sum: f32 = state.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum should be 1, got {sum}");
        assert!(state.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_smooth_activation_k_zero_is_sparser_than_k_large() {
        // k = 0 (α = 3) should be sparser than k = large (α ≈ 1)
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut det = input.clone();
        let deterministic = CriticalityController::new(0.0);
        deterministic.smooth_activation(&mut det);
        let det_zeros = det.iter().filter(|&&v| v == 0.0).count();

        let mut creat = input.clone();
        let creative = CriticalityController::new(10.0);
        creative.smooth_activation(&mut creat);
        let creat_zeros = creat.iter().filter(|&&v| v == 0.0).count();

        assert!(
            det_zeros >= creat_zeros,
            "k=0 (α=3) should be at least as sparse as k=10 (α≈1): det_zeros={det_zeros}, creat_zeros={creat_zeros}"
        );
    }

    // ── CriticalityDashboard tests ──

    #[test]
    fn test_dashboard_branching_ratio_uniform() {
        let n = 302usize;
        let uniform = vec![1.0 / n as f32; n];
        let ratio = CriticalityDashboard::compute_branching_ratio(&uniform);
        // f32→f64 conversion causes tiny error; accept < 0.001
        assert!(ratio < 0.001, "uniform should have near-zero branching ratio, got {ratio}");
    }

    #[test]
    fn test_dashboard_branching_ratio_one_hot() {
        let mut one_hot = vec![0.0f32; 302];
        one_hot[0] = 1.0;
        let ratio = CriticalityDashboard::compute_branching_ratio(&one_hot);
        assert!(ratio > 0.0, "one-hot should have positive branching ratio");
    }

    #[test]
    fn test_dashboard_betti_entropy_uniform() {
        let n = 302usize;
        let uniform = vec![1.0 / n as f32; n];
        let entropy = CriticalityDashboard::compute_betti_entropy(&uniform);
        let expected = (n as f64).ln();
        assert!((entropy - expected).abs() < 1e-6, "uniform entropy should be ln(n)");
    }

    #[test]
    fn test_dashboard_betti_entropy_one_hot() {
        let mut one_hot = vec![0.0f32; 302];
        one_hot[0] = 1.0;
        let entropy = CriticalityDashboard::compute_betti_entropy(&one_hot);
        assert!(entropy < 1e-10, "one-hot entropy should be near 0, got {entropy}");
    }

    #[test]
    fn test_dashboard_correlation_length_identical() {
        let act = vec![0.1f32; 302];
        let traj = vec![act.clone(), act];
        let corr = CriticalityDashboard::compute_correlation_length(&traj);
        assert!((corr - 1.0).abs() < 1e-6, "identical activations should have cos=1");
    }

    #[test]
    fn test_dashboard_correlation_length_single() {
        let traj = vec![vec![1.0f32; 302]];
        let corr = CriticalityDashboard::compute_correlation_length(&traj);
        assert_eq!(corr, 0.0, "single activation should give 0 correlation");
    }

    #[test]
    fn test_dashboard_avalanche_exponent_short_trajectory() {
        let short = vec![vec![0.1f32; 302]; 3];
        let exp = CriticalityDashboard::compute_avalanche_exponent(&short);
        assert_eq!(exp, 0.0, "short trajectory should give 0");
    }

    #[test]
    fn test_dashboard_compute_roundtrip() {
        let activation = vec![1.0 / 302.0; 302];
        let dashboard = CriticalityDashboard::compute(&activation, None, 0.42);
        assert!(dashboard.branching_ratio < 1.0, "branching ratio bounded, got {}", dashboard.branching_ratio);
        assert!((dashboard.betti_entropy - (302.0_f64).ln()).abs() < 1e-6);
        assert_eq!(dashboard.holonomy, 0.42);
    }

    #[test]
    fn test_dashboard_with_trajectory() {
        let activation = vec![0.0f32; 302];
        let traj = vec![
            vec![0.1f32; 302],
            vec![0.2f32; 302],
            vec![0.3f32; 302],
            vec![0.4f32; 302],
            vec![0.5f32; 302],
            vec![0.6f32; 302],
        ];
        let dashboard = CriticalityDashboard::compute(&activation, Some(&traj), 0.0);
        assert!(dashboard.correlation_length > 0.9, "similar activations should give high correlation");
    }
}
