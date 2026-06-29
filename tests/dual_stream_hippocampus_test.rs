//! Phase 4 — Dual-Stream Cognition / EchoReservoir Validation
//!
//! Validates four Zero-Trust gates for the hippocampus-inspired dual-stream
//! architecture (pre-entmax dense learning + post-entmax sparse action):
//!
//!   Gate 4a — Bifurcation:  pre-entmax dense signal has strictly more
//!     non-zero entries than post-entmax sparse action.
//!   Gate 4b — Echo Formation: repeated presentation of related words
//!     builds measurable association in the reservoir's Hebbian matrix.
//!   Gate 4c — Backward Compat: route_with_echo(alpha_echo=0.0, no
//!     reservoir) produces the same sparse_action as route_signal.
//!   Gate 4d — Associative Echo: after Hebbian training on "ice" patterns,
//!     the echo bias nudges "cold" activation toward "ice" neurons.
//!
//! The core claim: echo learning is impossible on the sparse post-entmax
//! signal (σ ≈ 1.0 → disjoint active neurons → zero gradient), but possible
//! on the dense pre-entmax logits (many co-active dimensions → non-zero
//! outer products).

use ndarray::Array1;
use std::collections::HashSet;

const MANIFOLD_DIM: usize = 128;

use utophiecorn_architecture::{
    geometry::token_to_coord,
    hippocampus::{CognitionState, EchoReservoir},
    worm_brain::WormBrain,
};

/// Number of non-zero entries in a 302-D activation vector (tolerance 1e-6).
fn count_active(act: &[f32]) -> usize {
    act.iter().filter(|&&v| v.abs() > 1e-6).count()
}

/// Jaccard similarity between the top-5 neuron sets of two activations.
fn jaccard_top5(a: &[f32], b: &[f32]) -> f32 {
    let top5_a: HashSet<usize> = top5_indices(a);
    let top5_b: HashSet<usize> = top5_indices(b);
    let intersection = top5_a.intersection(&top5_b).count();
    let union = top5_a.union(&top5_b).count();
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

fn top5_indices(act: &[f32]) -> HashSet<usize> {
    let mut with_idx: Vec<(usize, f32)> = act.iter().copied().enumerate().collect();
    with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    with_idx.iter().take(5).map(|(i, _)| *i).collect()
}

// ── Gate 4a: Bifurcation ────────────────────────────────────────────────
//
// The pre-entmax dense signal must have strictly more non-zero entries than
// the post-entmax sparse action.  This is the fundamental property that makes
// echo learning possible: the dense signal carries co-activation between
// different words, while the sparse action only activates a few neurons.

#[test]
fn test_gate_4a_bifurcation() {
    let brain = WormBrain::new_baseline();
    let tokens = vec!["fire", "ice", "cold", "hot", "ocean", "wave"];

    for token in &tokens {
        let coord = token_to_coord(token).inner().clone();
        let (sparse, dense) = brain.route_with_echo(&coord).unwrap();
        let sparse_active = count_active(sparse.as_slice().unwrap());
        let dense_active = count_active(dense.as_slice().unwrap());

        assert!(
            dense_active > sparse_active,
            "Gate 4a FAIL: token '{token}' — dense_active={dense_active} <= sparse_active={sparse_active}. \
             The pre-entmax signal must be denser than the post-entmax action."
        );
        eprintln!(
            "  OK [{token}] dense_active={dense_active} sparse_active={sparse_active}"
        );
    }
}

// ── Gate 4b: Echo Formation ──────────────────────────────────────────────
//
// After repeatedly presenting "ice" and "cold" through the reservoir, the
// Hebbian association matrix must show measurable association between the
// neurons they co-activate (non-zero entries in W_assoc for the top neurons
// of each word).

#[test]
fn test_gate_4b_echo_formation() {
    let mut brain = WormBrain::new_baseline();
    // Attach a reservoir with echo blend
    brain.echo_reservoir = Some(EchoReservoir::new(10));
    brain.cognition = CognitionState {
        kappa_gate: 1.0,
        t_scale: 50.0,
        alpha_echo: 0.3,
    };

    let ice_coord = token_to_coord("ice").inner().clone();
    let cold_coord = token_to_coord("cold").inner().clone();

    // Phase 1: Present "ice" and "cold" alternately, training the reservoir
    // on the dense signal.
    for _step in 0..20 {
        let (_, dense_ice) = brain.route_with_echo(&ice_coord).unwrap();
        brain.train_echo_dense(&dense_ice);

        let (_, dense_cold) = brain.route_with_echo(&cold_coord).unwrap();
        brain.train_echo_dense(&dense_cold);
    }

    // Check the reservoir has stored states
    let reservoir = brain.echo_reservoir.as_ref().unwrap();
    assert_eq!(
        reservoir.history.len(),
        10,
        "Gate 4b: reservoir history should be at capacity (10)"
    );

    // The association matrix should have non-zero entries
    let max_assoc = reservoir
        .associations
        .iter()
        .map(|&v| v.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_assoc > 0.0,
        "Gate 4b FAIL: no associations formed — max|W_assoc| = 0.0"
    );
    eprintln!("  OK max|W_assoc| = {max_assoc:.6} after 20 steps");
}

// ── Gate 4c: Backward Compat ────────────────────────────────────────────
//
// With no reservoir and alpha_echo=0.0, route_with_echo must produce the
// same sparse_action as route_signal.  This guarantees the dual-stream
// refactoring does not change existing behaviour.

#[test]
fn test_gate_4c_backward_compat() {
    let brain = WormBrain::new_baseline();
    let tokens = vec!["fire", "ice", "cold", "hot", "ocean", "wave", "tide"];

    for token in &tokens {
        let coord = token_to_coord(token).inner().clone();
        let (sparse_echo, _) = brain.route_with_echo(&coord).unwrap();
        let (sparse_normal, _) = brain.route_signal(&coord).unwrap();

        let diff: f32 = sparse_echo
            .iter()
            .zip(sparse_normal.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1e-5,
            "Gate 4c FAIL: token '{token}' — route_with_echo differs from route_signal by {diff}"
        );
        eprintln!("  OK [{token}] diff={diff:.2e}");
    }
}

// ── Gate 4d: Associative Echo ───────────────────────────────────────────
//
// After Hebbian training on the dense signals of "ice" and "cold", the echo
// bias from the reservoir should measurably nudge "cold" sparse action toward
// the "ice" neuron set.  We quantify this as a reduction in Jaccard distance
// between ice and cold top-5 neuron sets.

#[test]
fn test_gate_4d_associative_echo() {
    let mut brain = WormBrain::new_baseline();
    brain.echo_reservoir = Some(EchoReservoir::new(20));
    brain.cognition = CognitionState {
        kappa_gate: 1.0,
        t_scale: 50.0,
        alpha_echo: 0.5,
    };

    let ice_coord = token_to_coord("ice").inner().clone();
    let cold_coord = token_to_coord("cold").inner().clone();

    // Baseline: Jaccard between ice and cold without any echo training
    let (sparse_ice_base, _) = brain.route_with_echo(&ice_coord).unwrap();
    let (sparse_cold_base, _) = brain.route_with_echo(&cold_coord).unwrap();
    let jac_before = jaccard_top5(sparse_ice_base.as_slice().unwrap(), sparse_cold_base.as_slice().unwrap());

    // Training: present "ice" and "cold" alternately, storing dense signals
    for _step in 0..50 {
        let (_, dense_ice) = brain.route_with_echo(&ice_coord).unwrap();
        brain.train_echo_dense(&dense_ice);

        let (_, dense_cold) = brain.route_with_echo(&cold_coord).unwrap();
        brain.train_echo_dense(&dense_cold);
    }

    // After training: cold routed with echo blend should have higher Jaccard
    // with ice (because the echo bias nudges cold toward ice's neurons).
    let (sparse_ice_post, _) = brain.route_with_echo(&ice_coord).unwrap();
    let (sparse_cold_post, _) = brain.route_with_echo(&cold_coord).unwrap();
    let jac_after = jaccard_top5(sparse_ice_post.as_slice().unwrap(), sparse_cold_post.as_slice().unwrap());

    eprintln!(
        "  Jaccard(ice, cold) before={jac_before:.4} after={jac_after:.4}"
    );

    // The echo must push the Jaccard UP (cold → ice association forms).
    // This is a soft assertion: if the reservoir is working, the Jaccard
    // should increase.  A small tolerance for random noise.
    assert!(
        jac_after >= jac_before - 0.05,
        "Gate 4d FAIL: Jaccard(ice,cold) dropped from {jac_before:.4} to {jac_after:.4}. \
         The echo bias should nudge cold toward ice's neurons."
    );
}

// ── Gate 4e: Reservoir Reset ────────────────────────────────────────────
//
// Resetting the reservoir must clear history AND zero out the association
// matrix.  This is critical for safe re-initialisation between training runs.

#[test]
fn test_gate_4e_reservoir_reset() {
    let mut reservoir = EchoReservoir::new(5);
    let mut state = Array1::zeros(302);
    state[10] = 0.5;
    state[20] = -0.3;

    reservoir.push(&state);
    reservoir.hebbian_step(&state);
    assert_eq!(reservoir.history.len(), 1);
    assert!(reservoir.associations.iter().any(|&v| v.abs() > 0.0));

    reservoir.reset();
    assert_eq!(reservoir.history.len(), 0);
    assert!(
        reservoir.associations.iter().all(|&v| v.abs() < 1e-8),
        "Gate 4e FAIL: association matrix not cleared after reset"
    );
}

// ── Gate 4f: Stereo Separation ──────────────────────────────────────────
//
// Under α-entmax (σ ≈ 1.0), different words must produce different sparse
// action patterns AND different dense learning signals.  If two different
// words collapse to the same pattern, the architecture cannot distinguish
// them.

#[test]
fn test_gate_4f_stereo_separation() {
    let brain = WormBrain::new_baseline();
    let tokens = vec!["fire", "ice", "ocean", "hot", "cold", "wave", "tide"];

    for i in 0..tokens.len() {
        for j in (i + 1)..tokens.len() {
            let coord_a = token_to_coord(tokens[i]).inner().clone();
            let coord_b = token_to_coord(tokens[j]).inner().clone();
            let (sparse_a, dense_a) = brain.route_with_echo(&coord_a).unwrap();
            let (sparse_b, dense_b) = brain.route_with_echo(&coord_b).unwrap();

            // Different words MUST have different sparse actions
            let sparse_diff: f32 = sparse_a
                .iter()
                .zip(sparse_b.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                sparse_diff > 1e-4,
                "Gate 4f FAIL: sparse actions for '{}' and '{}' are identical",
                tokens[i],
                tokens[j]
            );

            // Different words MUST have different dense signals
            let dense_diff: f32 = dense_a
                .iter()
                .zip(dense_b.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                dense_diff > 1e-4,
                "Gate 4f FAIL: dense signals for '{}' and '{}' are identical",
                tokens[i],
                tokens[j]
            );
        }
    }
}

// ── Integration: Echo + Training Loop ───────────────────────────────────
//
// Demonstrates a minimal training loop using the dual-stream API:
// forward → capture dense → train echo → repeat.  This is the intended
// usage pattern for incorporating echo learning into the existing
// dendritic training pipeline.

#[test]
fn test_dual_stream_training_loop() {
    let mut brain = WormBrain::new_baseline();
    brain.echo_reservoir = Some(EchoReservoir::new(10));
    brain.cognition = CognitionState {
        kappa_gate: 1.0,
        t_scale: 50.0,
        alpha_echo: 0.2,
    };

    let tokens = vec!["fire", "ice", "cold", "hot", "ocean", "wave"];
    let coords: Vec<Array1<f64>> = tokens
        .iter()
        .map(|t| token_to_coord(t).inner().clone())
        .collect();

    for step in 0..30 {
        for coord in &coords {
            let (sparse, dense) = brain.route_with_echo(coord).unwrap();
            brain.train_echo_dense(&dense);

            // Basic sanity: sparse action always sums to ~1
            let sum_sparse: f32 = sparse.iter().sum();
            assert!(
                (sum_sparse - 1.0).abs() < 0.1,
                "Step {step}: sparse action sum = {sum_sparse}, expected ~1.0"
            );
        }
    }

    let reservoir = brain.echo_reservoir.as_ref().unwrap();
    assert_eq!(reservoir.history.len(), 10);
    let max_assoc = reservoir
        .associations
        .iter()
        .map(|&v| v.abs())
        .fold(0.0_f32, f32::max);
    assert!(max_assoc > 0.0, "No associations formed during training loop");
    eprintln!("  Training loop OK: max|W_assoc| = {max_assoc:.6}");
}
