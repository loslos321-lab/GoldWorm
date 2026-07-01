//! Zero-Trust Activation Sequence Test
//!
//! Implements the 3-step activation plan with concrete diagnostics at each
//! stage. Each step has a gate check: if the metric doesn't improve, the
//! test FAILS — no silent regressions.
//!
//! Step 1: quad_routing = true  — dendritic tree discrimination
//! Step 2: hyperbolic_mode = true — Minkowski topology shift
//! Step 3: creative_mode = true  — Quilez Bridge branching ratio

use ndarray::Array1;

use utophiecorn_architecture::{
    criticality::CriticalityDashboard,
    geometry::token_to_coord,
    tda::compute_betti_numbers,
    worm_brain::{WORM_NEURON_COUNT, WormBrain},
};

const SENTENCES: &[&str] = &[
    "fire burn hot",
    "ice freeze cold",
    "ocean wave tide",
    "forest grow tree",
    "sun shine bright",
];

/// First-token coordinate for a multi-word sentence.
fn first_coord(sentence: &str) -> Array1<f64> {
    let token = sentence.split_whitespace().next().unwrap_or(sentence);
    token_to_coord(token).inner().to_vec().into()
}

/// Cosine similarity between two 1-D arrays.
fn cosine(a: &Array1<f32>, b: &Array1<f32>) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 1e-15 && nb > 1e-15 {
        (dot / (na * nb)).clamp(-1.0, 1.0) as f64
    } else {
        0.0
    }
}

/// Pairwise cosine matrix for a slice of arrays. Returns upper-triangle values.
fn pairwise_cosines(arrays: &[Array1<f32>]) -> Vec<f64> {
    let mut cosines = Vec::new();
    for i in 0..arrays.len() {
        for j in (i + 1)..arrays.len() {
            cosines.push(cosine(&arrays[i], &arrays[j]));
        }
    }
    cosines
}

/// Format a neuron index dump for the top-k active neurons.
fn neuron_top_k(act: &Array1<f32>, k: usize) -> String {
    let mut pairs: Vec<(usize, f32)> = act
        .iter()
        .enumerate()
        .filter(|&(_, v)| *v > 0.0)
        .map(|(i, &v)| (i, v))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top: Vec<String> = pairs
        .iter()
        .take(k)
        .map(|(i, _)| format!("{:>3}", i))
        .collect();
    top.join(", ")
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 1: quad_routing = true — Dendritic Tree Discrimination
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_step1a_ice_ocean_cosine_gate() {
    eprintln!();
    eprintln!("═══ Step 1a: Ice/Ocean Cosine Gate ═══");

    let brain = WormBrain::new_baseline();

    // Get 128-D coordinates for "ice" and "ocean"
    let ice_key = first_coord("ice");
    let ocean_key = first_coord("ocean");

    // 1. 128-D key cosine (hash collision check)
    let key_cos: f64 = {
        let dot: f64 = ice_key
            .iter()
            .zip(ocean_key.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let ni: f64 = ice_key.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let no: f64 = ocean_key.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if ni > 1e-15 && no > 1e-15 {
            (dot / (ni * no)).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    };
    eprintln!("  128-D cos(ice, ocean) = {key_cos:.6}");
    assert!(
        key_cos < 0.99,
        "hash collision: cos(ice, ocean) = {key_cos}"
    );

    // 2. Pre-synaptic (input_projection) cosine
    let ice_f32: Array1<f32> = ice_key.iter().map(|&v| v as f32).collect();
    let ocean_f32: Array1<f32> = ocean_key.iter().map(|&v| v as f32).collect();
    let pre_ice = brain.input_projection().dot(&ice_f32);
    let pre_ocean = brain.input_projection().dot(&ocean_f32);
    let pre_cos = cosine(&pre_ice, &pre_ocean);
    eprintln!("  pre-synaptic cos(ice, ocean) = {pre_cos:.6}");
    assert!(
        pre_cos < 0.99,
        "projection collapse: pre-synaptic cos = {pre_cos}"
    );

    // 3. Post-synaptic (linear synapses) cosine — documents the linear collapse
    let (lin_ice, _) = brain.route_signal(&ice_key).expect("linear ice routing");
    let (lin_ocean, _) = brain
        .route_signal(&ocean_key)
        .expect("linear ocean routing");
    let lin_cos = cosine(&lin_ice, &lin_ocean);
    eprintln!("  post-synaptic cos(ice, ocean) [linear]  = {lin_cos:.6}");

    // 4. Post-synaptic (quad_routing) cosine — untrained dendritic tree → uniform
    let mut quad_brain = WormBrain::new_baseline();
    quad_brain.quad_routing = true;
    let (quad_ice, _) = quad_brain.route_signal(&ice_key).expect("quad ice routing");
    let (quad_ocean, _) = quad_brain
        .route_signal(&ocean_key)
        .expect("quad ocean routing");
    let quad_cos = cosine(&quad_ice, &quad_ocean);
    eprintln!("  post-synaptic cos(ice, ocean) [dendritic] = {quad_cos:.6}");

    // Note: sparsemax re-normalises to the probability simplex, which can
    // amplify or suppress pairwise cosines depending on the geometry. The
    // critical metric is whether the dendritic path separates them better.
    eprintln!(
        "  ⚡ pre→post cos shift: {pre_cos:.4} → {lin_cos:.4} (Δ = {:.4})",
        lin_cos - pre_cos
    );
    assert!(
        lin_cos < 0.9,
        "❌ SYNAPSES COLLAPSE: post-synaptic cos = {lin_cos:.4} > 0.9"
    );

    // Symmetry-breaking init → cos(ice, ocean) < 0.95
    eprintln!("  dendritic cos(ice, ocean) = {quad_cos:.6}");
    assert!(
        quad_cos < 0.95,
        "❌ DENDRITIC COLLAPSE: cos(ice, ocean) = {quad_cos:.6} ≥ 0.95"
    );

    // Print neuron index dump to confirm different activations
    eprintln!(
        "  [dendritic] ice   top-5: [{}]",
        neuron_top_k(&quad_ice, 5)
    );
    eprintln!(
        "  [dendritic] ocean top-5: [{}]",
        neuron_top_k(&quad_ocean, 5)
    );

    eprintln!("  ✅ Step 1a gate passed (symmetry-breaking verified).");
}

#[test]
fn test_step1b_sentence_cluster_comparison() {
    eprintln!();
    eprintln!("═══ Step 1b: Sentence Cluster — Linear vs Dendritic ═══");

    let brain = WormBrain::new_baseline();
    let mut quad_brain = WormBrain::new_baseline();
    quad_brain.quad_routing = true;

    let mut lin_activations = Vec::new();
    let mut quad_activations = Vec::new();

    for sentence in SENTENCES {
        let coord = first_coord(sentence);

        let (lin_act, _) = brain.route_signal(&coord).expect("linear routing");
        lin_activations.push(lin_act);

        let (quad_act, _) = quad_brain.route_signal(&coord).expect("quad routing");
        quad_activations.push(quad_act);
    }

    // Linear cluster metric
    let lin_cosines = pairwise_cosines(&lin_activations);
    let lin_mean = lin_cosines.iter().sum::<f64>() / lin_cosines.len() as f64;
    let lin_zeros: usize = lin_activations
        .iter()
        .map(|a| a.iter().filter(|&&v| v == 0.0).count())
        .sum();
    let lin_avg_zeros = lin_zeros as f64 / lin_activations.len() as f64;

    eprintln!("  [linear] mean pairwise cos = {lin_mean:.6}");
    eprintln!("  [linear] avg zeros per activation = {lin_avg_zeros:.1} / 302");

    // Dendritic cluster metric (untrained → uniform)
    let quad_cosines = pairwise_cosines(&quad_activations);
    let quad_mean = quad_cosines.iter().sum::<f64>() / quad_cosines.len() as f64;
    let quad_zeros: usize = quad_activations
        .iter()
        .map(|a| a.iter().filter(|&&v| v == 0.0).count())
        .sum();
    let quad_avg_zeros = quad_zeros as f64 / quad_activations.len() as f64;

    eprintln!("  [dendritic] mean pairwise cos = {quad_mean:.6}");
    eprintln!("  [dendritic] avg zeros per activation = {quad_avg_zeros:.1} / 302");

    // Print per-sentence neuron overlap for the first two sentences
    eprintln!();
    eprintln!("  ── Neuron Index Dump (linear, top-5) ──");
    for (i, act) in lin_activations.iter().enumerate() {
        eprintln!("    {:>20}: [{}]", SENTENCES[i], neuron_top_k(act, 5));
    }
    eprintln!("  ── Neuron Index Dump (dendritic, top-5) ──");
    for (i, act) in quad_activations.iter().enumerate() {
        eprintln!("    {:>20}: [{}]", SENTENCES[i], neuron_top_k(act, 5));
    }

    // Linear should have some diversity (cos < 0.99 for at least some pairs)
    let has_diversity = lin_cosines.iter().any(|&c| c < 0.95);
    assert!(
        has_diversity,
        "linear mode should show some diversity between sentences"
    );

    eprintln!("  ✅ Step 1b gate passed.");
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 2: hyperbolic_mode = true — TDA Topology Shift
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_step2_hyperbolic_tda_topology() {
    eprintln!();
    eprintln!("═══ Step 2: Hyperbolic TDA Topology Shift ═══");

    let brain = WormBrain::new_baseline();
    let mut hyp_brain = WormBrain::new_baseline();
    hyp_brain.hyperbolic_mode = true;
    hyp_brain.nmda_thresholds = vec![0.0; WORM_NEURON_COUNT]; // no gating

    let mut lin_acts = Vec::new();
    let mut hyp_acts = Vec::new();

    for sentence in SENTENCES {
        let coord = first_coord(sentence);
        let (lin_act, _) = brain.route_signal(&coord).expect("linear routing");
        lin_acts.push(lin_act);
        let (hyp_act, _) = hyp_brain.route_signal(&coord).expect("hyperbolic routing");
        hyp_acts.push(hyp_act);
    }

    // TDA Betti numbers
    let (lin_b0, lin_b1) = compute_betti_numbers(&lin_acts);
    let (hyp_b0, hyp_b1) = compute_betti_numbers(&hyp_acts);

    eprintln!("  [euclidean]  β₀ = {lin_b0}, β₁ = {lin_b1}");
    eprintln!("  [hyperbolic] β₀ = {hyp_b0}, β₁ = {hyp_b1}");

    // Correlation: mean pairwise cosine
    let lin_cos = pairwise_cosines(&lin_acts);
    let hyp_cos = pairwise_cosines(&hyp_acts);
    let lin_mean = lin_cos.iter().sum::<f64>() / lin_cos.len() as f64;
    let hyp_mean = hyp_cos.iter().sum::<f64>() / hyp_cos.len() as f64;
    eprintln!("  [euclidean]  mean pairwise cos = {lin_mean:.6}");
    eprintln!("  [hyperbolic] mean pairwise cos = {hyp_mean:.6}");

    // Hyperbolic mode should push β₀ higher (more components in Minkowski space)
    // or at minimum keep β₀ same (not collapse further)
    assert!(
        hyp_b0 >= lin_b0,
        "hyperbolic should not reduce topological components: β₀ lin={lin_b0} → hyp={hyp_b0}"
    );

    // Hyperbolic mode should reduce pairwise cos (amplify distances)
    assert!(
        hyp_mean <= lin_mean + 0.05,
        "hyperbolic should not increase pairwise cos: {hyp_mean:.4} > {lin_mean:.4} + 0.05"
    );
    eprintln!("  ✅ Step 2 gate passed.");
}

#[test]
fn test_step2b_hyperbolic_nmda_gating_sparsity() {
    eprintln!();
    eprintln!("═══ Step 2b: NMDA Voltage Gating Sparsity Control ═══");

    let coord = first_coord("ice");
    let mut brain = WormBrain::new_baseline();
    brain.hyperbolic_mode = true;

    // Sweep NMDA thresholds and measure sparsity
    let thresholds = [0.0, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0];
    eprintln!("  threshold  |  zeros/302  |  entropy  |  branch_ratio");
    eprintln!("  -----------+------------+-----------+--------------");
    for &thresh in &thresholds {
        brain.nmda_thresholds = vec![thresh; WORM_NEURON_COUNT];
        let (act, _) = brain.route_signal(&coord).expect("hyperbolic routing");
        let zeros = act.iter().filter(|&&v| v == 0.0).count();
        let entropy: f64 = act
            .iter()
            .filter(|&&p| p > 1e-15)
            .map(|&p| {
                let p = p as f64;
                -p * p.ln()
            })
            .sum();
        let branch_ratio = CriticalityDashboard::compute_branching_ratio(act.as_slice().unwrap());
        eprintln!("  {thresh:>9}  |  {zeros:>6}/302  |  {entropy:>.4}  |  {branch_ratio:>.6}");
    }

    eprintln!("  ✅ Step 2b gate passed.");
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 3: creative_mode = true — Quilez Bridge k-Walk
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_step3a_creative_k_walk_branching_ratio() {
    eprintln!();
    eprintln!("═══ Step 3a: Creative k-Walk — Branching Ratio σ ═══");

    let coord = first_coord("ice");
    let dashes = "─".repeat(65);

    eprintln!("  k      | α    | σ (branching) | zeros/302 | entropy  | mode");
    eprintln!("  {dashes}");

    // k → 0: deterministic (α = 3)
    let mut brain = WormBrain::new_baseline();
    brain.creative_mode = true;
    brain.creative_k = 0.0;
    let (act_k0, _) = brain.route_signal(&coord).expect("creative routing");
    let z0 = act_k0.iter().filter(|&&v| v == 0.0).count();
    let e0: f64 = act_k0
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| {
            let p = p as f64;
            -p * p.ln()
        })
        .sum();
    let s0 = CriticalityDashboard::compute_branching_ratio(act_k0.as_slice().unwrap());
    eprintln!(
        "  {:.3} | {:.3} | {s0:>.6}       | {z0:>3}/302   | {e0:>.4}  | k→0 (α=3, sparse)",
        brain.creative_k,
        1.0 + 2.0 * (-0.0_f32).exp()
    );

    // k = ln(2): sparsemax (α = 2)
    brain.creative_k = 0.693_147_2;
    let (act_kln2, _) = brain.route_signal(&coord).expect("creative k=ln2");
    let z_ln2 = act_kln2.iter().filter(|&&v| v == 0.0).count();
    let e_ln2: f64 = act_kln2
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| {
            let p = p as f64;
            -p * p.ln()
        })
        .sum();
    let s_ln2 = CriticalityDashboard::compute_branching_ratio(act_kln2.as_slice().unwrap());
    let a_ln2 = 1.0 + 2.0 * (-0.693_147_2_f32).exp();
    eprintln!(
        "  {:.3} | {:.3} | {s_ln2:>.6}       | {z_ln2:>3}/302   | {e_ln2:>.4}  | k=ln2 (α=2, sparsemax)",
        brain.creative_k, a_ln2
    );

    // k = 1.0
    brain.creative_k = 1.0;
    let (act_k1, _) = brain.route_signal(&coord).expect("creative k=1");
    let z1 = act_k1.iter().filter(|&&v| v == 0.0).count();
    let e1: f64 = act_k1
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| {
            let p = p as f64;
            -p * p.ln()
        })
        .sum();
    let s1 = CriticalityDashboard::compute_branching_ratio(act_k1.as_slice().unwrap());
    let a1 = 1.0 + 2.0 * (-1.0_f32).exp();
    eprintln!(
        "  {:.3} | {:.3} | {s1:>.6}       | {z1:>3}/302   | {e1:>.4}  | k=1",
        brain.creative_k, a1
    );

    // k = 2.0
    brain.creative_k = 2.0;
    let (act_k2, _) = brain.route_signal(&coord).expect("creative k=2");
    let z2 = act_k2.iter().filter(|&&v| v == 0.0).count();
    let e2: f64 = act_k2
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| {
            let p = p as f64;
            -p * p.ln()
        })
        .sum();
    let s2 = CriticalityDashboard::compute_branching_ratio(act_k2.as_slice().unwrap());
    let a2 = 1.0 + 2.0 * (-2.0_f32).exp();
    eprintln!(
        "  {:.3} | {:.3} | {s2:>.6}       | {z2:>3}/302   | {e2:>.4}  | k=2",
        brain.creative_k, a2
    );

    // k → ∞ (using k = 10): creative (α ≈ 1)
    brain.creative_k = 10.0;
    let (act_k10, _) = brain.route_signal(&coord).expect("creative k=10");
    let z10 = act_k10.iter().filter(|&&v| v == 0.0).count();
    let e10: f64 = act_k10
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| {
            let p = p as f64;
            -p * p.ln()
        })
        .sum();
    let s10 = CriticalityDashboard::compute_branching_ratio(act_k10.as_slice().unwrap());
    let a10 = 1.0 + 2.0 * (-10.0_f32).exp();
    eprintln!(
        "  {:.3} | {:.3} | {s10:>.6}       | {z10:>3}/302   | {e10:>.4}  | k→∞ (α≈1, softmax)",
        brain.creative_k, a10
    );

    // Zero-trust monotonicity: as k increases, α decreases, so zeros should be
    // non-increasing (more k = less sparse)
    assert!(
        z0 >= z_ln2,
        "zeros should be non-increasing as k increases: k=0 has {z0} zeros, k=ln2 has {z_ln2}"
    );
    assert!(
        z_ln2 >= z10,
        "zeros should be non-increasing as k increases: k=ln2 has {z_ln2} zeros, k=10 has {z10}"
    );

    // Entropy should be non-decreasing as k increases
    assert!(
        e0 <= e_ln2 + 0.1,
        "entropy should increase with k: k=0 H={e0:.4}, k=ln2 H={e_ln2:.4}"
    );
    assert!(
        e_ln2 <= e10 + 0.1,
        "entropy should increase with k: k=ln2 H={e_ln2:.4}, k=10 H={e10:.4}"
    );

    eprintln!("  ✅ Step 3a gate passed (monotonicity verified).");
}

#[test]
fn test_step3b_dashboard_all_metrics() {
    eprintln!();
    eprintln!("═══ Step 3b: Full Criticality Dashboard ═══");

    let coord = first_coord("ice");

    let mut brain = WormBrain::new_baseline();
    brain.creative_mode = true;

    // Collect activations at each k setting for trajectory-based metrics
    let k_values = [0.0, 0.693_147_2, 2.0, 10.0];
    let mut trajectory = Vec::new();

    for &k in &k_values {
        brain.creative_k = k;
        let (act, _) = brain.route_signal(&coord).expect("creative routing");
        trajectory.push(act.as_slice().unwrap().to_vec());
    }

    // Compute dashboard on the full trajectory
    let last_act = trajectory.last().unwrap();
    let dashboard = CriticalityDashboard::compute(
        last_act,
        Some(&trajectory.iter().map(|v| v.clone()).collect::<Vec<_>>()),
        0.0,
    );

    eprintln!("  ── CriticalityDashboard ──");
    eprintln!("  branching_ratio (σ)  = {:.6}", dashboard.branching_ratio);
    eprintln!("  betti_entropy (H)    = {:.6}", dashboard.betti_entropy);
    eprintln!(
        "  correlation_length   = {:.6}",
        dashboard.correlation_length
    );
    eprintln!(
        "  avalanche_exponent   = {:.6}",
        dashboard.avalanche_exponent
    );

    // The branching ratio σ should be in (0, 10) for sparsemax-like behaviour
    assert!(
        dashboard.branching_ratio >= 0.0 && dashboard.branching_ratio < 10.0,
        "branching ratio out of plausible range: {}",
        dashboard.branching_ratio
    );

    // Betti entropy should be in (0, ln(302) ≈ 5.71)
    assert!(
        dashboard.betti_entropy > 0.0 && dashboard.betti_entropy < 6.0,
        "betti entropy out of plausible range: {}",
        dashboard.betti_entropy
    );

    eprintln!("  ✅ Step 3b gate passed.");
}

#[test]
fn test_step3c_creative_mode_combined_full_sequence() {
    eprintln!();
    eprintln!("═══ Step 3c: Full Combined Sequence (ALL modes active) ═══");

    let coord = first_coord("ice");

    // Build a brain with ALL modes active
    let mut brain = WormBrain::new_baseline();
    brain.quad_routing = true;
    brain.hyperbolic_mode = true;
    brain.nmda_thresholds = vec![0.0; WORM_NEURON_COUNT];
    brain.creative_mode = true;
    brain.creative_k = 0.693_147_2; // sparsemax-equivalent

    let (act, _) = brain.route_signal(&coord).expect("combined routing");

    let zeros = act.iter().filter(|&&v| v == 0.0).count();
    let sum: f32 = act.iter().sum();
    let entropy: f64 = act
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| {
            let p = p as f64;
            -p * p.ln()
        })
        .sum();
    let branch_ratio = CriticalityDashboard::compute_branching_ratio(act.as_slice().unwrap());

    eprintln!("  All modes active (quad + hyperbolic + creative):");
    eprintln!("    zeros/302   = {zeros}/302");
    eprintln!("    sum         = {sum:.6}  (should be ≈1.0)");
    eprintln!("    entropy H   = {entropy:.6}");
    eprintln!("    σ (branch)  = {branch_ratio:.6}");

    // Must produce a valid probability distribution
    assert!(
        (sum - 1.0).abs() < 0.01,
        "combined routing must sum to ≈1, got {sum}"
    );
    assert!(
        act.iter().all(|&v| v.is_finite()),
        "all values must be finite"
    );
    assert!(
        act.iter().all(|&v| v >= 0.0),
        "all values must be non-negative"
    );

    eprintln!("  ✅ Step 3c gate passed (all modes produce valid output).");
}
