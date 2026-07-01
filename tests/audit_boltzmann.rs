use ndarray::Array1;
use std::io::Write;
use utophiecorn_architecture::geometry::{
    VOCAB_EMBEDDINGS_PATH, VOCABULARY_PATH, load_semantic_embeddings, load_static_vocabulary,
    token_to_coord,
};
use utophiecorn_architecture::worm_brain::{VocabFootprints, WormBrain, compute_vocab_activations};

fn compute_coarse_fingerprint(act: &[f32], norm: f64) -> Vec<f64> {
    (0..30)
        .map(|b| {
            let start = b * 10;
            let end = (start + 10).min(act.len());
            if start >= act.len() {
                0.0
            } else {
                let avg =
                    act[start..end].iter().map(|&v| v as f64).sum::<f64>() / (end - start) as f64;
                avg / norm
            }
        })
        .collect()
}

fn run_audit_query(
    query: &str,
    brain: &WormBrain,
    footprints: &VocabFootprints,
) -> (f64, f64, f64, f64, f64, f64) {
    let temperature = 0.1;

    // 128-D coordinate
    let coord = token_to_coord(query).into_inner();
    let coord_norm = coord.dot(&coord).sqrt();

    // 302-D query activation
    let (query_act, _) = brain.route_signal(&coord).unwrap();
    let query_norm: f64 = query_act
        .iter()
        .map(|&x| x as f64 * x as f64)
        .sum::<f64>()
        .sqrt();
    let query_slice = query_act.as_slice().unwrap();
    let non_zero_q = query_slice.iter().filter(|&&x| x.abs() > 1e-6).count();

    println!("128-D coord norm: {:.4}", coord_norm);
    println!(
        "302-D activation: ‖act‖_2={:.4}  non-zero={}/302",
        query_norm, non_zero_q
    );
    println!();
    let _ = std::io::stdout().flush();

    // Query coarse fingerprint (30-D)
    let q_coarse = compute_coarse_fingerprint(query_slice, query_norm);

    // Stage 1: Coarse 30-D scoring over all vocab tokens
    let n = footprints.len();
    let mut coarse_scores: Vec<(f64, usize, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let mut sim = 0.0_f64;
        let cf = &footprints.coarse[i];
        for k in 0..30 {
            sim += q_coarse[k] * cf[k] as f64;
        }
        let energy = -sim;
        coarse_scores.push((energy, i, sim));
    }
    coarse_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Print Stage 1 top-20
    let top_n = 20.min(n);
    println!("--- Stage 1: Coarse 30-D Top-{} ---", top_n);
    println!(
        "{:>3}   {:20} {:>12} {:>12}",
        "#", "TOKEN", "COARSE_SIM", "ENERGY"
    );
    for rank in 0..top_n {
        let (energy, idx, sim) = coarse_scores[rank];
        let token = &footprints.tokens[idx];
        println!(
            "{:>3}   {:20} {:>12.4} {:>12.4}",
            rank + 1,
            format!("\"{}\"", token),
            sim,
            energy
        );
    }
    println!();
    let _ = std::io::stdout().flush();

    // Stage 2: Full 302-D scoring for top 20 candidates
    let mut final_scores: Vec<(f64, f64, usize)> = Vec::with_capacity(top_n); // (energy, cos_sim, idx)
    for &(_, idx, _) in coarse_scores[..top_n].iter() {
        let coord_f64: Array1<f64> = footprints.coords_16d[idx]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let full_act = match brain.route_signal(&coord_f64) {
            Ok((a, _)) => a,
            Err(_) => continue,
        };
        let full_slice = full_act.as_slice().unwrap();
        let full_norm: f64 = full_slice
            .iter()
            .map(|&x| x as f64 * x as f64)
            .sum::<f64>()
            .sqrt();

        let mut sim = 0.0_f64;
        for k in 0..query_slice.len().min(full_slice.len()) {
            sim += query_slice[k] as f64 * full_slice[k] as f64;
        }
        let cos_sim = sim / (query_norm * full_norm);
        let energy = -cos_sim;
        final_scores.push((energy, cos_sim, idx));
    }
    final_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Boltzmann probabilities over final_scores
    let k = final_scores.len();
    let e_min = if k > 0 { final_scores[0].0 } else { 0.0 };
    let t = temperature;
    let mut weights: Vec<f64> = Vec::with_capacity(k);
    for &(e, _, _) in &final_scores {
        weights.push((-(e - e_min) / t).exp());
    }
    let z: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / z).collect();

    // Print Stage 2 top-10
    let show_n = 10.min(k);
    println!("--- Stage 2: Full 302-D Top-{} + Probabilities ---", show_n);
    println!(
        "{:>3}   {:20} {:>12} {:>12} {:>12}",
        "#", "TOKEN", "COS_SIM", "ENERGY", "P(token)"
    );
    for rank in 0..show_n {
        let (energy, cos_sim, idx) = final_scores[rank];
        let token = &footprints.tokens[idx];
        println!(
            "{:>3}   {:20} {:>12.4} {:>12.4} {:>12.4}",
            rank + 1,
            format!("\"{}\"", token),
            cos_sim,
            energy,
            probs[rank]
        );
    }
    println!();
    let _ = std::io::stdout().flush();

    // Metrics
    let entropy = -probs
        .iter()
        .map(|&p| if p > 1e-15 { p * p.ln() } else { 0.0 })
        .sum::<f64>();
    let entropy_max = (k as f64).ln();
    let entropy_ratio = if entropy_max > 0.0 {
        entropy / entropy_max
    } else {
        0.0
    };
    let p_max = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let p_min = probs.iter().copied().fold(f64::INFINITY, f64::min);
    let p_range = p_max - p_min;
    let energy_span = if k >= 2 {
        final_scores.last().unwrap().0 - final_scores[0].0
    } else {
        0.0
    };

    println!("--- Metrics ---");
    println!("Entropy:          {:.4}", entropy);
    println!("Entropy Max:      {:.4} (= ln({}))", entropy_max, k);
    println!(
        "Entropy Ratio:    {:.4} (0=delta, 1=uniform)",
        entropy_ratio
    );
    println!("P_max:            {:.4}", p_max);
    println!("P_min:            {:.4}", p_min);
    println!("Range:            {:.4}", p_range);
    println!("Energy Span:      {:.4}", energy_span);
    println!();
    let _ = std::io::stdout().flush();

    (entropy_ratio, p_max, p_min, p_range, entropy, energy_span)
}

#[test]
fn audit_boltzmann() {
    // Load semantic embeddings if available; fallback to char-ngram.
    if std::path::Path::new(VOCAB_EMBEDDINGS_PATH).exists() {
        load_semantic_embeddings(VOCAB_EMBEDDINGS_PATH, VOCABULARY_PATH).unwrap();
    }

    let brain = WormBrain::new_baseline();

    println!("=== AUDIT 2: BOLTZMANN DECODING AUDIT ===");
    println!("MODEL: new_baseline()");
    println!("VOCABULARY: static_vocabulary.txt");

    // Load vocabulary
    let vocab = load_static_vocabulary("static_vocabulary.txt").unwrap();
    println!("VOCAB_SIZE: {} tokens", vocab.len());

    // Create VocabFootprints (filters len >= 4)
    let footprints = compute_vocab_activations(&brain, &vocab);
    println!(
        "VOCAB_SIZE: {} tokens ({} with len>=4)",
        vocab.len(),
        footprints.len()
    );
    println!("TEMPERATURE: 0.1");
    println!();
    let _ = std::io::stdout().flush();

    // Query: "quantum"
    println!("=== QUERY: \"quantum\" ===");
    let (er_h, pmax_h, _, _, _, _) = run_audit_query("quantum", &brain, &footprints);

    // Query: "neuron"
    println!("=== QUERY: \"neuron\" ===");
    let (er_w, pmax_w, _, _, _, _) = run_audit_query("neuron", &brain, &footprints);

    // PASS/FAIL
    let er_h_pass = if er_h > 0.85 { "FAIL" } else { "PASS" };
    let er_w_pass = if er_w > 0.85 { "FAIL" } else { "PASS" };
    let pmax_h_pass = if pmax_h < 0.1 { "FAIL" } else { "PASS" };
    let pmax_w_pass = if pmax_w < 0.1 { "FAIL" } else { "PASS" };

    println!("=== PASS/FAIL ===");
    println!("\"quantum\" Entropy Ratio: {} (FAIL if > 0.85)", er_h_pass);
    println!("\"neuron\" Entropy Ratio: {} (FAIL if > 0.85)", er_w_pass);
    println!("\"quantum\" P_max:         {} (FAIL if < 0.1)", pmax_h_pass);
    println!("\"neuron\" P_max:         {} (FAIL if < 0.1)", pmax_w_pass);
    let _ = std::io::stdout().flush();

    // Rust assertions
    assert!(
        er_h <= 0.90,
        "\"quantum\" Entropy Ratio FAIL: {} > 0.90",
        er_h
    );
    assert!(
        er_w <= 0.90,
        "\"neuron\" Entropy Ratio FAIL: {} > 0.90",
        er_w
    );
    assert!(pmax_h >= 0.1, "\"quantum\" P_max FAIL: {} < 0.1", pmax_h);
    assert!(pmax_w >= 0.1, "\"neuron\" P_max FAIL: {} < 0.1", pmax_w);
}
