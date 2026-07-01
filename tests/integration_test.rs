use std::io::Write;
use std::path::Path;

use ndarray::Array1;

use utophiecorn_architecture::{
    CoreError,
    geometry::{self, MANIFOLD_DIM, ingest_reasoning_trace, token_to_coord},
    ingest::{self, ingest_jsonl_file},
    memory::{SynapticEchoBuffer, VAULT_PATH, consolidate_sleep, log_trajectory},
    neuro_symbolic::{
        OctetParityBuffer, classify_token, prune_by_parity, rational_closure, topological_witness,
    },
    storage::{load_brain_state, save_brain_state},
    tda::{PersistenceDiagram, PersistencePair},
    training::WormTrainer,
    worm_brain::{WORM_NEURON_COUNT, WormBrain},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_jsonl(path: &Path, lines: &[&str]) {
    let mut f = std::fs::File::create(path).unwrap();
    for line in lines {
        writeln!(f, "{line}").unwrap();
    }
    f.flush().unwrap();
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("utophiecorn_int_{label}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Back up the real vault, run a closure, then restore.
fn with_vault_backup<F: FnOnce()>(f: F) {
    let vault = Path::new(VAULT_PATH);
    let backup = std::env::temp_dir().join("utophiecorn_vault_backup.json");
    if vault.exists() {
        std::fs::copy(vault, &backup).unwrap();
    }
    f();
    if backup.exists() {
        let _ = std::fs::copy(&backup, vault);
        let _ = std::fs::remove_file(&backup);
    } else if vault.exists() {
        let _ = std::fs::remove_file(vault);
    }
}

// ---------------------------------------------------------------------------
// Pipeline: Ingest → binary save/load → training → brain save/load
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_ingest_train_save_load() {
    let dir = temp_dir("ingest_train");

    // --- Ingest JSONL ---
    let jsonl = dir.join("data.jsonl");
    write_jsonl(
        &jsonl,
        &[
            r#"{"system": "the worm moves through dark soil seeking food"}"#,
            r#"{"system": "quantum entanglement and superposition states"}"#,
        ],
    );

    let (trajs, report) = ingest_jsonl_file(&jsonl).unwrap();
    assert_eq!(report.total_lines, 2);
    assert_eq!(report.text_found, 2);
    assert_eq!(report.trajectories_created, 2);
    assert_eq!(trajs.len(), 2);
    assert!(trajs[0].path_length > 0.0);
    assert!(trajs[0].basis.rank >= 1);

    // --- Binary roundtrip ---
    let bin = dir.join("trajs.bin");
    ingest::save_trajectories(&trajs, &bin).unwrap();
    let loaded = ingest::load_trajectories(&bin).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].token_count, trajs[0].token_count);
    assert!((loaded[0].path_length - trajs[0].path_length).abs() < 1e-12);
    assert_eq!(loaded[0].basis.vectors, trajs[0].basis.vectors);

    // --- Train brain on traces ---
    let mut brain = WormBrain::new_baseline();
    let trainer = WormTrainer::new(0.01, 0.99);
    let before = brain.synapses.clone();

    // Re-tokenize the input text and route each token through the brain.
    for text in [
        "the worm moves through dark soil seeking food",
        "quantum entanglement and superposition states",
    ] {
        for raw in text.split_whitespace() {
            if !geometry::is_valid_token(raw) {
                continue;
            }
            let coord = token_to_coord(raw);
            let (activation, pre_synaptic) = brain.route_signal(coord.inner()).unwrap();
            let input_key = coord.inner().mapv(|v| v as f32);
            trainer
                .train_step(&mut brain, &activation, &pre_synaptic, &input_key)
                .unwrap();
        }
    }

    // Verify weights changed (should be different from initial after training).
    let before_clone = before.clone();
    let diff = brain.synapses.clone() - before;
    let max_change = diff.iter().copied().fold(0.0_f32, f32::max);
    assert!(
        max_change > 0.0,
        "training should modify at least one synapse"
    );

    // Verify structural zeros remain zero.
    for i in 0..WORM_NEURON_COUNT {
        for j in 0..WORM_NEURON_COUNT {
            if before_clone[(i, j)] == 0.0 {
                assert_eq!(
                    brain.synapses[(i, j)],
                    0.0,
                    "structural zero ({i},{j}) became non-zero"
                );
            }
        }
    }

    // --- Brain state safetensors roundtrip ---
    let st = dir.join("brain.safetensors");
    save_brain_state(&brain, &st).unwrap();
    let (loaded_syn, loaded_proj) = load_brain_state(&st).unwrap();
    assert_eq!(loaded_syn.shape(), &[WORM_NEURON_COUNT, WORM_NEURON_COUNT]);
    assert_eq!(loaded_syn, brain.synapses);
    assert_eq!(loaded_proj.shape(), &[WORM_NEURON_COUNT, MANIFOLD_DIM]);
    assert_eq!(loaded_proj, *brain.input_projection());
}

// ---------------------------------------------------------------------------
// Echo buffer integrated with real routing
// ---------------------------------------------------------------------------

#[test]
fn test_echo_buffer_with_routing() {
    let brain = WormBrain::new_baseline();
    let mut echo = SynapticEchoBuffer::new(0.75);

    // Route first token, inject echo.
    let coord_a = token_to_coord("worm");
    let (proj_a, _) = brain.route_signal(coord_a.inner()).unwrap(); // 302-D
    assert_eq!(proj_a.len(), WORM_NEURON_COUNT);

    echo.inject_echo(&proj_a);

    // Route second token with echo injected.
    let coord_b = token_to_coord("soil");
    let (proj_raw, _) = brain.route_signal(coord_b.inner()).unwrap();
    let mut proj_b = proj_raw.clone();
    echo.apply_and_decay(&mut proj_b).unwrap();

    // proj_b should differ from the raw projection (echo was added).
    let echo_diff = proj_b.clone() - &proj_raw;
    let echo_norm = echo_diff.dot(&echo_diff).sqrt();
    assert!(
        echo_norm > 0.0,
        "echo injection should modify the projection"
    );

    // Inject again without routing through brain — just echo state.
    let coord_c = token_to_coord("food");
    let (proj_c, _) = brain.route_signal(coord_c.inner()).unwrap();
    echo.inject_echo(&proj_c);

    // Apply-and-decay on a fresh vector.
    let mut fresh = Array1::zeros(WORM_NEURON_COUNT);
    echo.apply_and_decay(&mut fresh).unwrap();
    assert!(
        fresh.iter().any(|&v| v != 0.0),
        "echo should bleed into fresh input"
    );
}

// ---------------------------------------------------------------------------
// Vault: empty → log → consolidate → verify cleared, then empty no-op
// (single test to avoid races on the hardcoded VAULT_PATH)
// ---------------------------------------------------------------------------

#[test]
fn test_vault_lifecycle() {
    with_vault_backup(|| {
        // Phase 1: Empty vault → no-op consolidation.
        std::fs::write(Path::new(VAULT_PATH), "[]").unwrap();
        let mut brain = WormBrain::new_baseline();
        let before = brain.synapses.clone();
        let report = consolidate_sleep(&mut brain).unwrap();

        assert_eq!(report.entries_consolidated, 0);
        assert_eq!(report.total_steps, 0);
        assert!(report.status.is_ok());
        assert_eq!(
            brain.synapses, before,
            "empty vault should not change brain"
        );

        // Phase 2: Log a trajectory entry.
        let steps = vec![
            "food".to_string(),
            "energy".to_string(),
            "move".to_string(),
            "forward".to_string(),
        ];
        log_trajectory("worm", &steps).unwrap();

        // Phase 3: Consolidate with a baseline brain.
        let mut brain = WormBrain::new_baseline();
        let before = brain.synapses.clone();

        let report = consolidate_sleep(&mut brain).unwrap();

        assert_eq!(report.entries_consolidated, 1);
        assert_eq!(report.total_steps, 5); // "worm" + 4 steps
        assert!(report.status.is_ok());

        // Weights should have changed from baseline.
        let diff = brain.synapses.clone() - before;
        let max_change = diff.iter().copied().fold(0.0_f32, f32::max);
        assert!(max_change > 0.0, "consolidation should modify synapses");

        // Vault should be cleared after consolidation.
        let vault_content = std::fs::read_to_string(Path::new(VAULT_PATH)).unwrap();
        assert_eq!(
            vault_content.trim(),
            "[]",
            "vault should be empty after sleep"
        );
    });
}

// ---------------------------------------------------------------------------
// Similarity between ingested trajectories
// ---------------------------------------------------------------------------

#[test]
fn test_trajectory_similarity_pipeline() {
    let a = "the worm moves through dark soil";
    let b = "the worm moves through dark soil"; // identical
    let c = "quantum entanglement superposition wavefunction"; // different domain

    let traj_a = ingest_reasoning_trace(a).unwrap();
    let traj_b = ingest_reasoning_trace(b).unwrap();
    let traj_c = ingest_reasoning_trace(c).unwrap();

    let sim_ab = traj_a.similarity_to(&traj_b);
    assert!((sim_ab - 1.0).abs() < 1e-10, "identical→{sim_ab}");

    let sim_ac = traj_a.similarity_to(&traj_c);
    assert!(sim_ac < 1.0, "dissimilar→{sim_ac} must be < 1");

    let sim_ca = traj_c.similarity_to(&traj_a);
    assert!(
        (sim_ac - sim_ca).abs() < 1e-15,
        "asymmetry {sim_ac} vs {sim_ca}"
    );
}

// ---------------------------------------------------------------------------
// Decode token from vocabulary (requires static_vocabulary.txt in CWD)
// ---------------------------------------------------------------------------

#[test]
fn test_decode_token_from_vocabulary() {
    let vocab = geometry::load_static_vocabulary("static_vocabulary.txt")
        .expect("vocabulary file should load");
    assert!(
        vocab.len() > 9_000,
        "expected 9000+ words, got {}",
        vocab.len()
    );

    let brain = WormBrain::new_baseline();
    let coord = token_to_coord("worm");
    let (activation, _) = brain.route_signal(coord.inner()).unwrap();

    let token =
        geometry::decode_token(&activation, brain.input_projection(), &vocab, &[], 0.8, 3.0);
    assert!(
        !token.is_empty(),
        "decode_token should return a non-empty string"
    );
    assert!(
        token.len() >= 3,
        "decoded token '{token}' should have ≥3 chars"
    );
}

// ---------------------------------------------------------------------------
// Error path: dimension mismatch on echo buffer
// ---------------------------------------------------------------------------

#[test]
fn test_echo_dimension_error() {
    let mut buf = SynapticEchoBuffer::new(0.75);
    let mut bad = Array1::from_elem(100, 0.0);
    let result = buf.apply_and_decay(&mut bad);
    assert!(result.is_err());
    match result.unwrap_err() {
        CoreError::InvalidDimension { expected, got } => {
            assert_eq!(expected, WORM_NEURON_COUNT);
            assert_eq!(got, 100);
        }
        _ => panic!("expected InvalidDimension error"),
    }
}

// ---------------------------------------------------------------------------
// is_valid_token edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_is_valid_token_edge_cases() {
    assert!(geometry::is_valid_token("worm"), "≥3 alphabetic → valid");
    assert!(!geometry::is_valid_token("a"), "1 char → invalid");
    assert!(
        !geometry::is_valid_token("ab"),
        "2 chars, not whitelisted → invalid"
    );
    assert!(
        geometry::is_valid_token("in"),
        "whitelisted short word → valid"
    );
    assert!(geometry::is_valid_token("the"), "3 char word → valid");
    assert!(!geometry::is_valid_token(""), "empty → invalid");
    assert!(!geometry::is_valid_token("123"), "digits → invalid");
    assert!(!geometry::is_valid_token("foo-bar"), "hyphen → invalid");
    assert!(
        !geometry::is_valid_token("aby"),
        "archaic exclusion → invalid"
    );
}

// ---------------------------------------------------------------------------
// Neuro-Symbolic: Octet Parity, Topological Witness, Rational Closure
// ---------------------------------------------------------------------------

#[test]
fn test_neuro_symbolic_prune_by_parity() {
    let mut buf = OctetParityBuffer::new(8);
    buf.push(classify_token("water")); // noun-only mask = 0b0000_0001

    let tokens = vec!["the".to_string(), "fire".to_string(), "think".to_string()];
    let candidates = vec![(0.0, 0usize), (1.0, 1), (2.0, 2)];

    let filtered = prune_by_parity(&candidates, &tokens, &buf);

    assert!((filtered[0].0 - 0.0).abs() < 1e-6);
    assert!(
        (filtered[1].0 - 11.0).abs() < 1e-6,
        "fire should be penalized"
    );
    assert!((filtered[2].0 - 2.0).abs() < 1e-6);
}

#[test]
fn test_neuro_symbolic_rational_closure() {
    assert_eq!(rational_closure("hello world"), "Hello world.");
    assert_eq!(rational_closure("Hello world!"), "Hello world!");
    assert_eq!(rational_closure("Hello world."), "Hello world.");
    assert_eq!(rational_closure(""), "");
}

#[test]
fn test_neuro_symbolic_topological_witness() {
    let diagram = PersistenceDiagram {
        beta_0_pairs: vec![],
        beta_1_pairs: vec![PersistencePair {
            birth: 10,
            death: 250,
        }],
        beta_2_pairs: vec![],
        n_points: 20,
    };

    let witness = topological_witness(&diagram);
    assert!(
        witness > 0.0,
        "topological witness must be > 0 for persistent cycles"
    );
    assert!(witness < 1.0, "single cycle witness should not exceed 1.0");
    // 0.5 * (240/256) + 0.3 * (240/1000) = 0.46875 + 0.072 = 0.54075
    assert!(
        (witness - 0.54075).abs() < 0.001,
        "expected ~0.54075, got {witness}"
    );

    let empty = PersistenceDiagram {
        beta_0_pairs: vec![],
        beta_1_pairs: vec![],
        beta_2_pairs: vec![],
        n_points: 0,
    };
    assert!((topological_witness(&empty) - 0.0).abs() < 1e-6);
}
