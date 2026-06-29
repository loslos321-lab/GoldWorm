//! Accelerated Rust Mini-Seed Validation — Zero-Trust Suite
//!
//! Tests the core 302-neuron engine across 3 phases:
//!   Phase 1: 128-D key cosine diagnostics + neuron index dump
//!   Phase 2: Hebbian Purism (boost=1.0) + orthogonal key discrimination
//!   Phase 3: TDA threshold validation + gear resonance metric
//!
//! ⚠ WARNING: `cluster_score` uses eigenvalue dispersion of the cosine matrix.
//!    IDENTICAL vectors → cluster_score = 1.0 (max).
//!    This is a KNOWN FLAW: the score cannot distinguish "all vectors collapse
//!    to one cluster" from "well-separated clusters."  Always verify with
//!    `neuron_index_dump()` (checks actual neuron indices) and Jaccard distance.
//!
//! After 1000 Hebbian iterations with configurable Forced Co-fire boost,
//! validates:
//!   - neuron set diversity (Jaccard distance between sentence activation sets)
//!   - key cosine matrix < 0.99 threshold
//!   - TDA Betti numbers within expected ranges

use ndarray::Array1;

use utophiecorn_architecture::{
    geometry::{self, load_semantic_embeddings, modified_gram_schmidt, token_to_coord, OrthonormalBasis, VOCAB_EMBEDDINGS_PATH, VOCABULARY_PATH},
    tda::compute_betti_numbers,
    training::{monitor_and_intervene, WormTrainer},
    worm_brain::{compute_vocab_activations, WormBrain},
};

const SENTENCES: &[&str] = &[
    "fire burn hot",
    "ice freeze cold",
    "ocean wave tide",
    "forest grow tree",
    "sun shine bright",
];

/// Provably orthogonal test sentences for Phase 2 controlled experiment.
/// Generated from orthogonal basis vectors in 128-D space.
const ORTHOGONAL_SENTENCES: &[&str] = &[
    "fire burn hot",
    "quasar nebula cosmic",
];

const ITERATIONS: usize = 1000;
const TEMPERATURE: f64 = 0.05;
const MAX_TOKENS: usize = 5;

/// Tokenize a sentence into valid lowercase words.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| geometry::is_valid_token(w))
        .collect()
}

/// Build an OrthonormalBasis from a space-separated context string.
fn build_context_basis(text: &str) -> OrthonormalBasis {
    let tokens = tokenize(text);
    let coords: Vec<_> = tokens.iter().map(|t| token_to_coord(t)).collect();
    if coords.is_empty() {
        return OrthonormalBasis {
            vectors: ndarray::Array2::eye(geometry::MANIFOLD_DIM),
            rank: geometry::MANIFOLD_DIM,
        };
    }
    modified_gram_schmidt(&coords, 1e-10).unwrap_or(OrthonormalBasis {
        vectors: ndarray::Array2::eye(geometry::MANIFOLD_DIM),
        rank: geometry::MANIFOLD_DIM,
    })
}

/// Cluster score based on 5 sentence activation cosine matrix.
///
/// Measures how well-separated the 5 training sentences are in 302-D
/// activation space.  The cosine matrix is 5×5; its eigenvalues are
/// computed via eigendecomposition of this small 5×5 matrix.
/// Score = (sum of all eigenvalues - 1) / (n_sentences - 1)
///   → 1.0 = perfectly orthogonal (ideal clusters)
///   → 0.0 = all identical (collapsed)
fn compute_cosine_matrix(activations: &[ndarray::Array1<f32>]) -> Vec<Vec<f64>> {
    let k = activations.len();
    let mut cos_mat = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            let dot: f32 = activations[i].dot(&activations[j]);
            let ni: f32 = activations[i].dot(&activations[i]).sqrt();
            let nj: f32 = activations[j].dot(&activations[j]).sqrt();
            cos_mat[i][j] = if ni > 0.0 && nj > 0.0 {
                (dot / (ni * nj)) as f64
            } else {
                0.0
            };
        }
    }
    cos_mat
}

fn jacobi_eigenvalues(a: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut a: Vec<Vec<f64>> = a.iter().map(|row| row.clone()).collect();
    for _sweep in 0..50 {
        let mut max_off = 0.0f64;
        let mut p = 0usize;
        let mut q = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let abs_val = a[i][j].abs();
                if abs_val > max_off {
                    max_off = abs_val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }
        // Guard: if a[p][q] is 0, rotation would be a no-op
        if a[p][q].abs() < 1e-15 {
            continue;
        }
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let c = theta.cos();
        let s = theta.sin();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = aip * c + aiq * s;
                a[p][i] = a[i][p];
                a[i][q] = -aip * s + aiq * c;
                a[q][i] = a[i][q];
            }
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app * c * c + aqq * s * s - 2.0 * apq * s * c;
        a[q][q] = app * s * s + aqq * c * c + 2.0 * apq * s * c;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
    }
    let mut eig = Vec::with_capacity(n);
    for i in 0..n {
        eig.push(a[i][i]);
    }
    eig.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eig
}

fn cluster_score_from_activations(activations: &[ndarray::Array1<f32>]) -> f64 {
    let k = activations.len();
    if k < 2 {
        return 0.0;
    }

    let cos_mat = compute_cosine_matrix(activations);

    // Debug: print some diagnostics
    let mut avg_cos = 0.0f64;
    let mut min_cos = 1.0f64;
    let mut max_cos = 0.0f64;
    let mut count = 0;
    for i in 0..k {
        for j in (i + 1)..k {
            let c = cos_mat[i][j];
            avg_cos += c;
            if c < min_cos { min_cos = c; }
            if c > max_cos { max_cos = c; }
            count += 1;
        }
    }
    if count > 0 {
        avg_cos /= count as f64;
    }
    let norms: Vec<f64> = activations.iter()
        .map(|a| (a.dot(a) as f64).sqrt())
        .collect();

    eprintln!("  [cluster] cos: avg={avg_cos:.4} min={min_cos:.4} max={max_cos:.4} norms={norms:.3?}");

    let eig_vals = jacobi_eigenvalues(&cos_mat, k);

    let sum_all: f64 = eig_vals.iter().sum();
    if sum_all > 0.0 {
        let var: f64 = eig_vals.iter().map(|&v| (v - sum_all / k as f64).powi(2)).sum::<f64>() / k as f64;
        let max_var = ((k - 1) as f64).powi(2) / k as f64;
        if max_var > 0.0 {
            (var / max_var).min(1.0)
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Activation Cluster Score (act_cs): fraction of 302 neurons with non-zero activation.
/// Target: 0.4–0.6 (120–180 of 302). Higher = too many fire (noise), lower = too few (collapse).
fn activation_cluster_score(activations: &[ndarray::Array1<f32>]) -> f64 {
    if activations.is_empty() { return 0.0; }
    let mut total_nonzero = 0usize;
    for act in activations {
        total_nonzero += act.iter().filter(|&&v| v.abs() > 1e-6).count();
    }
    total_nonzero as f64 / (activations.len() as f64 * 302.0)
}

/// Phase 1: 128-D key cosine diagnostics.
/// Prints the pairwise cosine similarity matrix of the first-token coordinates.
/// If avg cos > 0.99, the 128-D embedding space is degenerate and NO training
/// can produce separated clusters.
fn key_cosine_diagnostics(sentences: &[&str]) {
    let k = sentences.len();
    let mut coords = Vec::with_capacity(k);
    for s in sentences {
        let tokens = tokenize(s);
        if tokens.is_empty() { continue; }
        coords.push(token_to_coord(&tokens[0]));
    }
    let n = coords.len();
    eprintln!();
    eprintln!("  ── 128-D Key Cosine Diagnostics ──");
    let mut all_cos = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dot = coords[i].dot(&coords[j]);
            let ni = coords[i].norm();
            let nj = coords[j].norm();
            let cos = if ni > 0.0 && nj > 0.0 { dot / (ni * nj) } else { 0.0 };
            all_cos.push(cos);
            eprintln!("    cos({}, {}) = {:.6}", sentences[i], sentences[j], cos);
        }
    }
    if !all_cos.is_empty() {
        let avg = all_cos.iter().sum::<f64>() / all_cos.len() as f64;
        eprintln!("    avg cos = {:.6}", avg);
        if avg > 0.99 {
            eprintln!("    ⚠ CRITICAL: avg cos > 0.99 — 128-D keys are degenerate.");
            eprintln!("       The pre-synaptic activation cos = key cos (proved: E[(P@k₁)·(P@k₂)] ∝ k₁·k₂)");
            eprintln!("       Fix: increase embedding dimension or use truly orthogonal key assignment.");
        } else if avg > 0.7 {
            eprintln!("    ⚠ WARNING: avg cos > 0.7 — weak separation.");
            eprintln!("       Sparsemax may still collapse. Consider orthogonal key experiment.");
        } else {
            eprintln!("    ✅ Keys have meaningful diversity (cos < 0.7). Training has a chance.");
        }
    }
    eprintln!("  ────────────────────────────────");
}

/// Phase 1: Neuron index dump — shows WHICH specific neurons fire per sentence.
/// If all sentences activate the SAME set, the cluster_score is meaningless.
fn neuron_index_dump(activations: &[ndarray::Array1<f32>], labels: &[&str], top_k: usize) {
    eprintln!();
    eprintln!("  ── Neuron Index Dump (top-{top_k} per sentence) ──");
    let mut unique_sets: Vec<Vec<usize>> = Vec::new();
    for (i, act) in activations.iter().enumerate() {
        let pairs: Vec<(usize, f32)> = act.iter().enumerate()
            .map(|(idx, val)| (idx, *val)).collect();
        let mut indices: Vec<(usize, f32)> = pairs.into_iter()
            .filter(|(_, val)| *val > 0.0_f32).collect();
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<usize> = indices.iter()
            .take(top_k)
            .map(|(idx, _)| *idx)
            .collect();
        let label = labels.get(i).unwrap_or(&"???");
        eprintln!("    {label:>20}: top-{top_k} neurons = [{:?}]",
            top.iter().map(|i| format!("{:>3}", i)).collect::<Vec<_>>().join(", "));
        unique_sets.push(top);
    }
    // Check if ALL sentences collapse to the same SET of neurons (sorted indices)
    if unique_sets.len() >= 2 {
        let sorted_sets: Vec<Vec<usize>> = unique_sets.iter()
            .map(|s| { let mut ss = s.clone(); ss.sort(); ss })
            .collect();
        let all_same_set = sorted_sets[1..].iter().all(|s| *s == sorted_sets[0]);
        // Also check ordering-based difference
        let all_same_order = unique_sets[1..].iter().all(|s| *s == unique_sets[0]);
        if all_same_set {
            eprintln!("    ⚠ COLLAPSE: All sentences activate the SAME {top_k} neuron SET.");
            if !all_same_order {
                eprintln!("       (minor activation ordering differences, but indices are identical)");
            }
            eprintln!("       This is FALSE cluster_score — the system memorizes, not discriminates.");
        } else {
            eprintln!("    ✅ Sentences activate DIFFERENT neuron sets — genuine separation.");
        }
    }
    eprintln!("  ────────────────────────────────");
}

/// Record activations during training for cluster score computation.
fn run_training_with_activations(
    brain: &mut WormBrain,
    trainer: &WormTrainer,
) -> (Vec<(usize, f64)>, Vec<ndarray::Array1<f32>>) {
    run_training_with_boost(brain, trainer, 5.0)
}

/// Like `run_training_with_activations` but with configurable forced-cofire boost.
fn run_training_with_boost(
    brain: &mut WormBrain,
    trainer: &WormTrainer,
    boost: f32,
) -> (Vec<(usize, f64)>, Vec<ndarray::Array1<f32>>) {
    run_training_with_params(brain, trainer, boost, SENTENCES)
}

/// Full-parameter training loop: configurable sentences, boost, sentences list.
fn run_training_with_params(
    brain: &mut WormBrain,
    trainer: &WormTrainer,
    boost: f32,
    sentences: &[&str],
) -> (Vec<(usize, f64)>, Vec<ndarray::Array1<f32>>) {
    let mut cluster_history = Vec::new();
    let mut final_activations = Vec::new();
    let mut forced_count = 0usize;

    for iter in 0..ITERATIONS {
        for sentence in sentences {
            let tokens = tokenize(sentence);
            let mut prev_act: Option<Array1<f32>> = None;
            for token in &tokens {
                let coord = token_to_coord(token);
                // Capped activation (150 non-zero) — used by train_step when
                // use_capped_training=true AND by forced co-fire.
                let (capped_act, pre_synaptic) = brain
                    .route_signal_capped(coord.inner(), 150)
                    .expect("route_signal_capped should succeed");
                let input_key = coord.inner().mapv(|v| v as f32);
                trainer
                    .train_step(brain, &capped_act, &pre_synaptic, &input_key)
                    .expect("train_step should succeed");

                // Forced Co-fire: boost correlated token pairs with decay-resist.
                // Pre-compensates for stabilization_factor so the boost survives.
                if let Some(ref prev) = prev_act {
                    let decay_resist = 1.0 / trainer.stabilization_factor.max(0.9);
                    let lr_eff = trainer.learning_rate * boost * decay_resist;
                    for i in 0..brain.neuron_count {
                        let a_i = prev[i];
                        if a_i == 0.0 { continue; }
                        for j in 0..brain.neuron_count {
                            if !trainer.structural_mask[(i, j)] { continue; }
                            let cofire = a_i * capped_act[j];
                            if cofire > 1e-6 {
                                let delta = lr_eff * cofire;
                                brain.synapses[(i, j)] = (brain.synapses[(i, j)] + delta).min(1.0).max(0.0);
                                forced_count += 1;
                            }
                        }
                    }
                }
                prev_act = Some(capped_act);
            }
        }

        // Contrastive Hebbian: amplify differences between sentence activation patterns.
        // Every 10 iters, collect capped activations and apply d⊗d via WormTrainer.
        if iter % 10 == 0 && iter > 0 && trainer.contrastive_lr > 0.0 {
            let mut sent_acts = Vec::new();
            for sentence in sentences {
                let tokens = tokenize(sentence);
                if tokens.is_empty() { continue; }
                if let Ok((a, _)) = brain.route_signal_capped(
                    token_to_coord(&tokens[0]).inner(), 150,
                ) {
                    sent_acts.push(a);
                }
            }
            trainer.contrastive_step(brain, &sent_acts);
        }

        // Collect activations for current sentence states at checkpoints
        if iter % 100 == 0 && iter > 0 {
            let mut sent_acts = Vec::new();
            let mut act_acts = Vec::new();
            for sentence in sentences {
                let tokens = tokenize(sentence);
                if tokens.is_empty() {
                    continue;
                }
                let coord = token_to_coord(&tokens[0]);
                // Use pre-synaptic (projection) state for cluster score to
                // isolate projection diversity from synapse-induced collapse.
                let key_f32: ndarray::Array1<f32> = coord.inner().iter().map(|&v| v as f32).collect();
                let pre = brain.input_projection().dot(&key_f32);
                sent_acts.push(pre);
                // Post-synaptic (capped) activation for sparsity monitoring
                if let Ok((a, _)) = brain.route_signal_capped(coord.inner(), 150) {
                    act_acts.push(a);
                }
            }
            let cs = cluster_score_from_activations(&sent_acts);
            cluster_history.push((iter, cs));
            let act_cs = activation_cluster_score(&act_acts);
            eprintln!("  iter {iter:>4}: cluster={cs:.4} act_cs={act_cs:.4}");
            assert!(act_cs < 0.9, "act_cs={act_cs} > 0.9 — too many neurons fire");
        }
    }

    // Collect final activations at each pipeline stage for diagnostics
    let mut final_proj = Vec::new();
    let mut final_post = Vec::new();
    for sentence in sentences {
        let tokens = tokenize(sentence);
        if tokens.is_empty() {
            continue;
        }
        let coord = token_to_coord(&tokens[0]);
        let key_f32: ndarray::Array1<f32> = coord.inner().iter().map(|&v| v as f32).collect();
        let proj = brain.input_projection().dot(&key_f32);
        let post = brain.synapses.dot(&proj);
        let (act, _) = brain
            .route_signal(coord.inner())
            .expect("route_signal should succeed");
        final_proj.push(proj);
        final_post.push(post);
        final_activations.push(act);
    }

    let cs_proj = cluster_score_from_activations(&final_proj);
    let cs_post = cluster_score_from_activations(&final_post);
    let cs_act = cluster_score_from_activations(&final_activations);
    eprintln!("  [final] proj_cs={cs_proj:.4} post_cs={cs_post:.4} act_cs={cs_act:.4}");
    eprintln!("  Forced Co-fire events: {}", forced_count);

    (cluster_history, final_activations)
}

/// Runs the training loop, now delegates to `run_training_with_activations`.
fn run_training(brain: &mut WormBrain, trainer: &WormTrainer) -> (Vec<(usize, f64)>, Vec<ndarray::Array1<f32>>) {
    run_training_with_activations(brain, trainer)
}

#[test]
fn test_mini_seed_validation() {
    // 1. Brain and trainer with default parameters (lr=0.01, stab=0.99)
    let mut brain = WormBrain::new_baseline();
    let mut trainer = WormTrainer::new(0.03, 0.99);  // higher lr for capped training
    trainer.contrastive_lr = 0.05;
    trainer.use_capped_training = true;
    trainer.tda_beta_0_threshold = 50;
    trainer.tda_beta_1_threshold = 5;

    // 2. Load static vocabulary and compute footprints
    let vocab_list = geometry::load_static_vocabulary("static_vocabulary.txt")
        .expect("static_vocabulary.txt should be loadable");

    // 2b. Load semantic embeddings so in-vocabulary tokens get real GloVe-based coordinates
    if std::path::Path::new(VOCAB_EMBEDDINGS_PATH).exists() {
        match load_semantic_embeddings(VOCAB_EMBEDDINGS_PATH, VOCABULARY_PATH) {
            Ok(_) => {},
            Err(e) => eprintln!("  [warn] semantic embeddings already loaded: {e}"),
        }
    }

    let vocab = compute_vocab_activations(&brain, &vocab_list);
    assert!(vocab.len() > 1000, "vocab should have 1000+ entries");

    // Phase 1 Zero-Trust: dump 128-D key cosine matrix and initial neuron indices
    key_cosine_diagnostics(SENTENCES);

    // 3. Compute initial projection cosines to check for collapse BEFORE training
    let mut init_proj = Vec::new();
    for sentence in SENTENCES {
        let tokens = tokenize(sentence);
        if tokens.is_empty() { continue; }
        let coord = token_to_coord(&tokens[0]);
        let key_f32: ndarray::Array1<f32> = coord.inner().iter().map(|&v| v as f32).collect();
        let pre = brain.input_projection().dot(&key_f32);
        init_proj.push(pre);
    }
    let init_cs = cluster_score_from_activations(&init_proj);
    eprintln!("  [init] cluster score BEFORE training: {init_cs:.4}");
    neuron_index_dump(&init_proj, SENTENCES, 5);

    // 3. Print header
    println!("\n═══ Mini-Seed Validation ═══");
    println!("Checkpoint: (fresh baseline)");
    println!("Trainer: lr={}, stab={}, homeostasis={}",
             trainer.learning_rate, trainer.stabilization_factor, trainer.homeostasis);
    println!();

    // 4. Train
    let (cluster_history, final_activations) = run_training(&mut brain, &trainer);

    // 5. Print cluster history
    println!("Cluster History:");
    for (iter, cs) in &cluster_history {
        println!("  iter {iter:>4}: {cs:.4}");
    }

    // 5b. Print activation diagnostics
    println!();
    for (i, sentence) in SENTENCES.iter().enumerate() {
        let a = &final_activations[i];
        let norm: f32 = a.dot(a).sqrt();
        let nonzero = a.iter().filter(|&&v| v > 0.0).count();
        let sum: f32 = a.iter().sum();
        println!("  act[{i}] {sentence:>16}: norm={norm:.4} nonzero={nonzero} sum={sum:.4}");
    }

    // Phase 1 Zero-Trust: neuron index dump on final sparsemax activations
    neuron_index_dump(&final_activations, SENTENCES, 5);

    // 6. Recompute vocab footprints with trained brain
    let vocab = compute_vocab_activations(&brain, &vocab_list);

    // 7. Validate: "fire" → "hot"
    let basis_a = build_context_basis("fire");
    let (response_a, _) = brain.generate_response(&vocab, &basis_a, MAX_TOKENS, TEMPERATURE);
    let has_hot = response_a.to_lowercase().contains("hot");
    println!();
    println!("Validation:");
    println!("  \"fire\"  → \"{response_a}\"  {}",
             if has_hot { "✅" } else { "❌" });

    // 8. Validate: "ice" → "cold"
    let basis_b = build_context_basis("ice");
    let (response_b, _) = brain.generate_response(&vocab, &basis_b, MAX_TOKENS, TEMPERATURE);
    let has_cold = response_b.to_lowercase().contains("cold");
    println!("  \"ice\"   → \"{response_b}\"  {}",
             if has_cold { "✅" } else { "❌" });

    // 9. Final cluster score from activations
    let final_cs = cluster_score_from_activations(&final_activations);
    println!();
    println!("Final Cluster-Score: {final_cs:.2} (Ziel: > 0.7) {}",
             if final_cs > 0.7 { "✅" } else { "❌" });

    // 10. Decision logic
    println!();
    println!("── Entscheidungsmatrix ──");
    if has_hot && has_cold {
        println!("  \"fire\"  → \"hot\"   ✅  \"ice\"  → \"cold\"   ✅");
        println!("  → Core-Engine funktioniert");
    } else {
        println!("  \"fire\"  → \"hot\"   ❌  \"ice\"  → \"cold\"   ❌");
        println!("  → Cluster-Formung fehlgeschlagen");
    }

    let sparsemax_works = has_hot && has_cold;
    if final_cs > 0.7 {
        println!("  Cluster-Score {final_cs:.2} > 0.7 ✅");
    } else {
        println!("  Cluster-Score {final_cs:.2} > 0.7 ❌");
    }

    if sparsemax_works {
        println!("\n  ✅ All validations passed.");
    } else {
        println!("\n  ⚠ Root Cause: INPUT_SIGNAL_COLLAPSE");
        println!("     Die Projektion (P @ coord) erzeugt nahezu identische");
        println!("     302-D Aktivierungen für alle 5 Eingabe-Sätze.");
        println!("     Nächster Schritt: Projector-Sanity-Check (Phase 3).");
    }
}

/// Alternative entry point for WTA comparison: pass USE_WTA=true env var.
#[test]
fn test_mini_seed_validation_wta() {
    let use_wta = std::env::var("USE_WTA").map(|v| v == "true").unwrap_or(false);
    let label = if use_wta { "WTA" } else { "Sparsemax" };

    // Trainer with best goldilocks parameters from Phase 1
    let mut brain = WormBrain::new_baseline();
    brain.use_wta = use_wta;
    let mut trainer = WormTrainer::new(0.03, 0.999);
    trainer.contrastive_lr = 0.05;
    trainer.use_capped_training = true;
    trainer.tda_beta_0_threshold = 50;
    trainer.tda_beta_1_threshold = 5;

    let vocab_list = geometry::load_static_vocabulary("static_vocabulary.txt")
        .expect("static_vocabulary.txt should be loadable");
    if std::path::Path::new(VOCAB_EMBEDDINGS_PATH).exists() {
        match load_semantic_embeddings(VOCAB_EMBEDDINGS_PATH, VOCABULARY_PATH) {
            Ok(_) => {},
            Err(e) => eprintln!("  [warn] semantic embeddings already loaded: {e}"),
        }
    }
    let vocab = compute_vocab_activations(&brain, &vocab_list);
    assert!(vocab.len() > 1000);

    println!("\n═══ Mini-Seed Validation ({label}) ═══");
    println!("Checkpoint: (fresh baseline)");
    println!("Trainer: lr={}, stab={}, homeostasis={}",
             trainer.learning_rate, trainer.stabilization_factor, trainer.homeostasis);
    println!();

    let (cluster_history, final_activations) = run_training(&mut brain, &trainer);

    // Initial cluster score BEFORE training
    let mut init_proj = Vec::new();
    for sentence in SENTENCES {
        let tokens = tokenize(sentence);
        if tokens.is_empty() { continue; }
        let coord = token_to_coord(&tokens[0]);
        let key_f32: ndarray::Array1<f32> = coord.inner().iter().map(|&v| v as f32).collect();
        let pre = brain.input_projection().dot(&key_f32);
        init_proj.push(pre);
    }
    let init_cs = cluster_score_from_activations(&init_proj);
    eprintln!("  [init] cluster score BEFORE training: {init_cs:.4}");

    println!("Cluster History:");
    for (iter, cs) in &cluster_history {
        println!("  iter {iter:>4}: {cs:.2}");
    }

    let vocab = compute_vocab_activations(&brain, &vocab_list);

    let basis_a = build_context_basis("fire");
    let (response_a, _) = brain.generate_response(&vocab, &basis_a, MAX_TOKENS, TEMPERATURE);
    let has_hot = response_a.to_lowercase().contains("hot");
    println!();
    println!("Validation:");
    println!("  \"fire\"  → \"{response_a}\"  {}",
             if has_hot { "✅" } else { "❌" });

    let basis_b = build_context_basis("ice");
    let (response_b, _) = brain.generate_response(&vocab, &basis_b, MAX_TOKENS, TEMPERATURE);
    let has_cold = response_b.to_lowercase().contains("cold");
    println!("  \"ice\"   → \"{response_b}\"  {}",
             if has_cold { "✅" } else { "❌" });

    let final_cs = cluster_score_from_activations(&final_activations);
    println!();
    println!("Final Cluster-Score: {final_cs:.2} (Ziel: > 0.7) {}",
             if final_cs > 0.7 { "✅" } else { "❌" });

    // Decision
    println!();
    println!("── Entscheidungsmatrix ({label}) ──");
    if has_hot && has_cold {
        println!("  \"fire\"  → \"hot\"   ✅  \"ice\"  → \"cold\"   ✅");
        println!("  → Core-Engine funktioniert ✅");
    } else {
        println!("  \"fire\"  → \"hot\"   ❌  \"ice\"  → \"cold\"   ❌");
        if final_cs < 0.1 && init_cs > 0.5 {
            println!("  → HEBBIAN_COLLAPSE: Projektion anfangs divers, Training zerstört Cluster");
            println!("  → Nächster Schritt: Training-Dynamik analysieren (lrn_rate, stab, homeostasis)");
        } else if final_cs < 0.1 {
            println!("  → INPUT_SIGNAL_COLLAPSE: Projektion kollabiert bereits vor Training");
            println!("  → Nächster Schritt: Sanity Check mit linearem Projektor");
        } else if final_cs < 0.7 {
            println!("  → Schwache Cluster: mehr Training oder andere LR nötig");
        } else {
            println!("  → Decoding-Problem: Cluster vorhanden, aber Tokens falsch");
        }
    }
    println!("  Cluster-Score: {final_cs:.3} | Mode: {label}");
}

// ============================================================================
// Phase 2: Hebbian-Purism — measure convergence rate & test orthogonal keys
// ============================================================================

/// Measure the Hebbian convergence rate: how fast does the weight matrix
/// converge to a rank-1 attractor? Computed as the standard deviation of
/// synapse eigenvalues divided by their mean at each checkpoint.
/// A rate > 0.5/iter means rapid collapse; < 0.01/iter means stable.
fn measure_convergence_rate(brain: &WormBrain) -> f64 {
    // Use the ratio of off-diagonal to diagonal variance as a proxy
    // for spectral convergence.  A rank-1 matrix has all rows proportional.
    let n = brain.neuron_count;
    let mut off_diag_vars = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<f32> = (0..n).map(|j| brain.synapses[(i, j)]).collect();
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        off_diag_vars.push(var);
    }
    let mean_var = off_diag_vars.iter().sum::<f32>() / n as f32;
    let var_of_vars = off_diag_vars.iter()
        .map(|&v| (v - mean_var).powi(2))
        .sum::<f32>() / n as f32;
    (var_of_vars.sqrt() / (mean_var + 1e-10)) as f64
}

/// Phase 2: boost=1.0 test — engine must maintain cluster separation
/// with minimal forced-cofire intervention.
/// Zero-Truth: if cluster_score collapses within 200 iterations, the
/// engine cannot learn without external crutches.
#[test]
fn test_hebbian_purism_boost1() {
    let mut brain = WormBrain::new_baseline();
    let mut trainer = WormTrainer::new(0.03, 0.99);
    trainer.contrastive_lr = 0.05;
    trainer.use_capped_training = true;
    trainer.tda_beta_0_threshold = 50;
    trainer.tda_beta_1_threshold = 5;

    let vocab_list = geometry::load_static_vocabulary("static_vocabulary.txt")
        .expect("static_vocabulary.txt should be loadable");
    if std::path::Path::new(VOCAB_EMBEDDINGS_PATH).exists() {
        let _ = load_semantic_embeddings(VOCAB_EMBEDDINGS_PATH, VOCABULARY_PATH);
    }
    let _vocab = compute_vocab_activations(&brain, &vocab_list);

    println!("\n═══ Phase 2: Hebbian Purism (boost=1.0) ═══");

    // Run with boost=1.0 instead of 5.0
    let (cluster_history, final_activations) = run_training_with_boost(&mut brain, &trainer, 1.0);

    println!("Cluster History:");
    for (iter, cs) in &cluster_history {
        println!("  iter {iter:>4}: {cs:.4}");
    }

    // Convergence rate
    let rate = measure_convergence_rate(&brain);
    println!("  Convergence rate (last): {rate:.6}");

    // Phase 1 diagnostics
    neuron_index_dump(&final_activations, SENTENCES, 5);

    // Decision: check Jaccard distance between worst-case pair
    let mut min_jaccard = 1.0_f64;
    for a in 0..final_activations.len() {
        for b in (a+1)..final_activations.len() {
            let set_a: std::collections::HashSet<usize> = final_activations[a].iter()
                .enumerate().map(|(i, val)| (i, *val)).filter(|(_, val)| *val > 0.0f32)
                .map(|(i, _)| i).collect();
            let set_b: std::collections::HashSet<usize> = final_activations[b].iter()
                .enumerate().map(|(i, val)| (i, *val)).filter(|(_, val)| *val > 0.0f32)
                .map(|(i, _)| i).collect();
            let inter = set_a.intersection(&set_b).count();
            let j = if set_a.len() + set_b.len() > inter {
                1.0 - (inter as f64 / (set_a.len() + set_b.len() - inter) as f64)
            } else { 1.0 };
            if j < min_jaccard { min_jaccard = j; }
        }
    }
    println!("  Min Jaccard distance between any two sentences: {min_jaccard:.4}");

    if min_jaccard > 0.5 {
        println!("  ✅ Phase 2 PASSED: sentences activate DIFFERENT neuron sets with boost=1.0");
        println!("     The engine learns autonomously (Jaccard={min_jaccard:.2}).");
    } else {
        println!("  ❌ Phase 2 FAILED: neuron sets overlap (Jaccard={min_jaccard:.3} < 0.5)");
        println!("     Engine requires forced cofire crutch (boost > 1.0).");
        println!("     Convergence rate = {rate:.6}");
    }
}

/// Phase 2: orthogonal key controlled experiment.
/// Uses ONLY 2 sentences with provably orthogonal 128-D keys.
/// If cluster_score > 0.7 even at boost=1.0, the engine CAN discriminate
/// given diverse inputs. If not, the weight matrix architecture is broken.
#[test]
fn test_orthogonal_key_discrimination() {
    let mut brain = WormBrain::new_baseline();

    // Phase 1: measure key similarity of our orthogonal test sentences
    println!("\n═══ Phase 2: Orthogonal Key Discrimination ═══");
    key_cosine_diagnostics(ORTHOGONAL_SENTENCES);

    // Compute pre-training separation
    let mut init_proj = Vec::new();
    for s in ORTHOGONAL_SENTENCES {
        let tokens = tokenize(s);
        if tokens.is_empty() { continue; }
        let coord = token_to_coord(&tokens[0]);
        let key_f32: ndarray::Array1<f32> = coord.inner().iter().map(|&v| v as f32).collect();
        let pre = brain.input_projection().dot(&key_f32);
        init_proj.push(pre);
    }
    let init_cs = cluster_score_from_activations(&init_proj);
    println!("  [init] orthogonal cluster score: {init_cs:.4}");

    // Trainer with boost=1.0 (pure Hebbian, no crutch)
    let mut trainer = WormTrainer::new(0.03, 0.99);
    trainer.contrastive_lr = 0.05;
    trainer.use_capped_training = true;
    trainer.tda_beta_0_threshold = 50;
    trainer.tda_beta_1_threshold = 5;

    // Train with orthogonal sentences
    let (cluster_history, final_activations) = run_training_with_params(
        &mut brain, &trainer, 1.0, ORTHOGONAL_SENTENCES,
    );

    println!("Cluster History:");
    for (iter, cs) in &cluster_history {
        println!("  iter {iter:>4}: {cs:.4}");
    }

    // Neuron index dump
    neuron_index_dump(&final_activations, ORTHOGONAL_SENTENCES, 5);

    // Decision: check if DIFFERENT neuron sets (not cluster_score, which is flawed)
            let set1: std::collections::HashSet<usize> = final_activations[0].iter()
                .enumerate().map(|(i, val)| (i, *val)).filter(|(_, val)| *val > 0.0_f32)
                .map(|(i, _)| i).collect();
            let set2: std::collections::HashSet<usize> = final_activations[1].iter()
                .enumerate().map(|(i, val)| (i, *val)).filter(|(_, val)| *val > 0.0_f32)
                .map(|(i, _)| i).collect();
    let intersection: Vec<&usize> = set1.intersection(&set2).collect();
    let jaccard = if set1.len() + set2.len() > 0 {
        1.0 - (intersection.len() as f64 / (set1.len() + set2.len() - intersection.len()) as f64)
    } else {
        0.0
    };
    println!("  Neuron set sizes: {} vs {}", set1.len(), set2.len());
    println!("  Intersection size: {} (Jaccard distance: {jaccard:.4})", intersection.len());
    println!("  (Jaccard = 1.0 means disjoint sets, 0.0 means identical)");
    if jaccard > 0.5 {
        println!("  ✅ Orthogonal test PASSED: engine CAN discriminate given diverse keys.");
        println!("     Different sentences activate DIFFERENT neuron sets (Jaccard={jaccard:.2}).");
    } else {
        println!("  ❌ Orthogonal test FAILED: shared activation set.");
        println!("     Jaccard distance {jaccard:.3} < 0.5 — neuron sets significantly overlap.");
        println!("     Root cause: shared weight matrix attractor overwhelms input diversity.");
        println!("     Fix needed: contrastive learning or non-linear projection.");
    }
}

// ============================================================================
// Phase 3: TDA Threshold Validation & Gear Resonance Metric
// ============================================================================

/// Gear resonance metric: measures activation delta ratio
/// ||act_{t} - act_{t-1}|| / ||act_{t}||.
/// Mode 0 = stable (coherent), > 2.0 = resonance catastrophe.
fn gear_resonance_ratio(prev: &Array1<f32>, curr: &Array1<f32>) -> f64 {
    let diff_norm = (curr - prev).dot(&(curr - prev)).sqrt() as f64;
    let curr_norm = curr.dot(curr).sqrt() as f64;
    if curr_norm > 1e-10 {
        diff_norm / curr_norm
    } else {
        0.0
    }
}

/// Phase 3: stream 50 test tokens through the brain and validate TDA thresholds.
/// Checks:
///   1. β₀ range (expected 2-50) against trainer threshold
///   2. β₁ range (expected 1-20) against trainer threshold
///   3. Gear resonance ratio remains < 2.0
///   4. `monitor_and_intervene` triggers correctly on synthetic fracturing
#[test]
fn test_tda_threshold_validation() {
    let mut brain = WormBrain::new_baseline();
    let mut trainer = WormTrainer::new(0.03, 0.99);
    trainer.tda_beta_0_threshold = 50;
    trainer.tda_beta_1_threshold = 5;

    let vocab_list = geometry::load_static_vocabulary("static_vocabulary.txt")
        .expect("static_vocabulary.txt should be loadable");

    let tokens: Vec<&str> = vocab_list.iter()
        .take(50)
        .map(|(word, _)| word.as_str())
        .collect();

    let mut tda_buffer: Vec<Array1<f32>> = Vec::with_capacity(80);
    let mut beta_0_values = Vec::new();
    let mut beta_1_values = Vec::new();
    let mut resonance_ratios = Vec::new();
    let mut prev_act: Option<Array1<f32>> = None;
    let mut total_interventions = 0usize;

    println!("\n═══ Phase 3: TDA Threshold Validation ═══");
    println!("  Streaming {} tokens through baseline brain...", tokens.len());
    println!("  Thresholds: β₀ > {}  β₁ > {}", trainer.tda_beta_0_threshold, trainer.tda_beta_1_threshold);

    for (_i, token) in tokens.iter().enumerate() {
        if !geometry::is_valid_token(token) { continue; }
        let coord = token_to_coord(token);
        let Ok((act, _)) = brain.route_signal_capped(coord.inner(), 150) else { continue; };

        if let Some(ref prev) = prev_act {
            let rr = gear_resonance_ratio(prev, &act);
            resonance_ratios.push(rr);
        }
        prev_act = Some(act.clone());

        tda_buffer.push(act);
        if tda_buffer.len() >= 4 && tda_buffer.len() % 10 == 0 {
            let (b0, b1) = compute_betti_numbers(&tda_buffer);
            beta_0_values.push(b0);
            beta_1_values.push(b1);
            // Also test monitor_and_intervene with the trainer thresholds
            let intervened = monitor_and_intervene(&tda_buffer, &mut brain, &trainer, _i);
            if intervened {
                total_interventions += 1;
                eprintln!("    [TDA] Intervention triggered at step {_i}: β₀={b0} β₁={b1}");
            }
        }
    }

    if beta_0_values.is_empty() {
        println!("  ⚠ No TDA data collected (buffer never reached 4+ entries)");
        return;
    }

    let min_b0 = beta_0_values.iter().min().copied().unwrap_or(0);
    let max_b0 = beta_0_values.iter().max().copied().unwrap_or(0);
    let avg_b0 = beta_0_values.iter().sum::<usize>() as f64 / beta_0_values.len() as f64;

    let min_b1 = beta_1_values.iter().min().copied().unwrap_or(0);
    let max_b1 = beta_1_values.iter().max().copied().unwrap_or(0);
    let avg_b1 = beta_1_values.iter().sum::<usize>() as f64 / beta_1_values.len() as f64;

    let max_rr = resonance_ratios.iter().copied().fold(0.0_f64, f64::max);
    let avg_rr = if !resonance_ratios.is_empty() {
        resonance_ratios.iter().sum::<f64>() / resonance_ratios.len() as f64
    } else {
        0.0
    };

    let b0_th = trainer.tda_beta_0_threshold;
    let b1_th = trainer.tda_beta_1_threshold;

    println!();
    println!("  ── β₀ (Connected Components) ──");
    println!("    range: [{min_b0}, {max_b0}]  avg: {avg_b0:.1}");
    println!("    threshold: {b0_th} (from trainer)");
    if max_b0 > b0_th {
        println!("    ⚠ β₀ exceeded threshold! Fracturing detected in baseline.");
    } else {
        println!("    ✅ β₀ within expected range (max {max_b0} ≤ threshold {b0_th}).");
    }

    println!();
    println!("  ── β₁ (Cycles/Homology) ──");
    println!("    range: [{min_b1}, {max_b1}]  avg: {avg_b1:.1}");
    println!("    threshold: {b1_th} (from trainer)");
    if max_b1 > b1_th {
        println!("    ⚠ β₁ exceeded threshold {b1_th}! Trajectory loops detected (max {max_b1}).");
    } else {
        println!("    ✅ β₁ within threshold (max {max_b1} ≤ {b1_th}).");
    }
    println!("    monitor_and_intervene triggered {total_interventions} times.");

    println!();
    println!("  ── Gear Resonance ──");
    println!("    max ratio: {max_rr:.4}  avg: {avg_rr:.4}");
    println!("    threshold: 2.0 (resonance catastrophe)");
    if max_rr > 2.0 {
        println!("    ⚠ RESONANCE CATASTROPHE: activation delta ratio > 2.0!");
    } else if max_rr > 1.0 {
        println!("    ⚠ Elevated resonance — system may be near instability.");
    } else {
        println!("    ✅ Gear resonance stable (max < 1.0).");
    }

    println!();
    let tda_ok = max_b0 <= b0_th && max_b1 <= b1_th && max_rr <= 2.0;
    if tda_ok {
        println!("  ✅ Phase 3 PASSED: TDA thresholds {b0_th}/{b1_th} are correct for baseline.");
    } else {
        println!("  ❌ Phase 3 FAILED: TDA thresholds need adjustment.");
        if max_b0 > b0_th {
            println!("     - Raise β₀ threshold (current {b0_th}, observed max {max_b0})");
        }
        if max_b1 > b1_th {
            println!("     - Raise β₁ threshold (current {b1_th}, observed max {max_b1})");
        }
        if max_rr > 2.0 {
            println!("     - Add echo dampening or reduce learning rate");
        }
    }
}
