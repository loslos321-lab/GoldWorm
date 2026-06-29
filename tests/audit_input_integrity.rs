use utophiecorn_architecture::geometry::{load_semantic_embeddings, token_to_coord, MANIFOLD_DIM, VOCAB_EMBEDDINGS_PATH, VOCABULARY_PATH};
use utophiecorn_architecture::worm_brain::WormBrain;
use ndarray::Array1;
use std::io::Write;

fn print_matrix(label: &str, matrix: &[Vec<f64>], tokens: &[&str; 5]) {
    println!("--- {} ---", label);
    print!("{:>12}", "");
    for t in tokens {
        print!("{:>8}", t);
    }
    println!();
    for (i, t) in tokens.iter().enumerate() {
        print!("{:>12}", t);
        for j in 0..5 {
            print!("{:>8.4}", matrix[i][j]);
        }
        println!();
    }
    println!();
    let _ = std::io::stdout().flush();
}

#[test]
fn audit_input_integrity() {
    // Load semantic embeddings if available; fallback to char-ngram.
    if std::path::Path::new(VOCAB_EMBEDDINGS_PATH).exists() {
        load_semantic_embeddings(VOCAB_EMBEDDINGS_PATH, VOCABULARY_PATH).unwrap();
    }

    let brain = WormBrain::new_baseline();
    let tokens = ["quantum", "neuron", "memory", "signal", "synapse"];
    let n = tokens.len();

    // 128-D coordinates
    let coords: Vec<Array1<f64>> = tokens
        .iter()
        .map(|t| token_to_coord(t).into_inner())
        .collect();
    let norms_16d: Vec<f64> = coords.iter().map(|c| c.dot(c).sqrt()).collect();

    // 302-D activations
    let mut acts: Vec<Array1<f32>> = Vec::new();
    for c in &coords {
        acts.push(brain.route_signal(c).unwrap().0);
    }
    let norms_302d: Vec<f64> = acts.iter().map(|a| a.dot(a).sqrt() as f64).collect();

    // Pairwise cosine similarity (128-D)
    let mut cos_16d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            cos_16d[i][j] = coords[i].dot(&coords[j]) / (norms_16d[i] * norms_16d[j]);
        }
    }

    // Pairwise cosine similarity (302-D)
    let mut cos_302d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            cos_302d[i][j] = (acts[i].dot(&acts[j]) as f64) / (norms_302d[i] * norms_302d[j]);
        }
    }

    // Pairwise L2 distance (128-D)
    let mut l2_16d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            l2_16d[i][j] = (&coords[i] - &coords[j]).mapv(|x| x * x).sum().sqrt();
        }
    }

    // Pairwise L2 distance (302-D)
    let mut l2_302d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            l2_302d[i][j] =
                ((&acts[i] - &acts[j]).mapv(|x| x * x).sum() as f64).sqrt();
        }
    }

    // Variance per dimension (128-D)
    let mut variance = vec![0.0f64; MANIFOLD_DIM];
    for d in 0..MANIFOLD_DIM {
        let vals: Vec<f64> = coords.iter().map(|c| c[d]).collect();
        let mean: f64 = vals.iter().sum::<f64>() / n as f64;
        variance[d] = vals
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n as f64;
    }

    // Zero-input test
    let zero = Array1::<f64>::zeros(MANIFOLD_DIM);
    let (zero_act, _) = brain.route_signal(&zero).unwrap();
    let max_abs = zero_act
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let l2_norm = zero_act.dot(&zero_act).sqrt();
    let non_zero = zero_act.iter().filter(|&&x| x.abs() > 1e-6).count();

    // Input projection analysis
    let proj = brain.input_projection();
    let zero_proj = proj.dot(&Array1::<f32>::zeros(MANIFOLD_DIM));
    let zero_proj_inf = zero_proj
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    let count_f = (302 * MANIFOLD_DIM) as f64;
    let frob: f64 = proj
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let mean: f64 = proj.iter().map(|&x| x as f64).sum::<f64>() / count_f;
    let std: f64 = (proj
        .iter()
        .map(|&x| ((x as f64) - mean).powi(2))
        .sum::<f64>()
        / count_f)
        .sqrt();
    let min: f32 = proj.iter().copied().fold(f32::INFINITY, f32::min);
    let max: f32 = proj
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // ── OUTPUT ──
    println!("=== AUDIT 1: INPUT VECTOR INTEGRITY ===");
    println!("MODEL: new_baseline()");
    println!();

    // Token coordinates
    println!("--- Token Coordinates (128-D) ---");
    for (i, token) in tokens.iter().enumerate() {
        let vals: Vec<String> =
            coords[i].iter().take(4).map(|v| format!("{:.4}", v)).collect();
        println!(
            "TOKEN: {:?}  norm={:.4}  coords=[{}, ...]",
            token,
            norms_16d[i],
            vals.join(", ")
        );
    }
    println!();
    let _ = std::io::stdout().flush();

    // Matrices
    let tlabels = &["quantum", "neuron", "memory", "signal", "synapse"];
    print_matrix("Pairwise Cosine Similarity (128-D)", &cos_16d, tlabels);
    print_matrix(
        "Pairwise Cosine Similarity (302-D)",
        &cos_302d,
        tlabels,
    );
    print_matrix("Pairwise L2 Distance (128-D)", &l2_16d, tlabels);
    print_matrix("Pairwise L2 Distance (302-D)", &l2_302d, tlabels);

    // Variance
    println!("--- Variance per Dimension ({}-D) ---", MANIFOLD_DIM);
    for d in 0..MANIFOLD_DIM {
        println!("dim[{:02}]: var={:.4}", d, variance[d]);
    }
    println!();
    let _ = std::io::stdout().flush();

    // Zero-input test
    println!("--- Zero-Input Test ---");
    println!("route_signal(zeros(16)):");
    println!("  ‖output‖_∞ = {:.4}", max_abs);
    println!("  ‖output‖_2 = {:.4}", l2_norm);
    println!("  non-zero: {} / 302", non_zero);
    println!("  MAX_ABS_THRESHOLD: 0.01");
    let zero_pass = if max_abs > 0.01 { "FAIL" } else { "PASS" };
    println!("  RESULT: {}", zero_pass);
    println!();
    let _ = std::io::stdout().flush();

    // Input projection analysis
    println!("--- input_projection Analysis ---");
    println!("shape: [302, {}]", MANIFOLD_DIM);
    println!(
        "‖proj @ zeros(16)‖_∞ = {:.4} (expected: 0.0)",
        zero_proj_inf
    );
    println!("‖proj‖_F = {:.4}", frob);
    println!("mean = {:.4}", mean);
    println!("std  = {:.4}", std);
    println!("min  = {:.4}", min);
    println!("max  = {:.4}", max);
    println!();
    let _ = std::io::stdout().flush();

    // PASS/FAIL
    let mut collision = false;
    for i in 0..n {
        for j in (i + 1)..n {
            if cos_302d[i][j] > 0.95 {
                collision = true;
            }
        }
    }
    let collision_pass = if collision { "FAIL" } else { "PASS" };

    println!("--- PASS/FAIL ---");
    println!(
        "Zero-Input Integrity: {} (max_abs > 0.01 = FAIL)",
        zero_pass
    );
    println!(
        "Collision Check: {} (any cos_302d > 0.95 = FAIL = CRITICAL COLLISION)",
        collision_pass
    );
    let _ = std::io::stdout().flush();

    // Rust assertion
    assert!(
        max_abs <= 0.01,
        "Zero-Input Integrity FAIL: max_abs = {} > 0.01",
        max_abs
    );
    assert!(
        !collision,
        "Collision Check FAIL: some cos_302d > 0.95"
    );
}
