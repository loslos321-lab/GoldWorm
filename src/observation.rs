use std::collections::VecDeque;

use ndarray::Array1;

use crate::criticality::CriticalityDashboard;
use crate::hippocampus::EchoReservoir;
use crate::worm_brain::{WORM_NEURON_COUNT, WormBrain};

/// Configuration for the observation CLI loop.
#[derive(Clone, Debug)]
pub struct ObservationConfig {
    pub model_path: String,
    pub vocab_path: String,
    pub reservoir_capacity: usize,
    pub hebbian_lr: f32,
    pub jaccard_window: usize,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            model_path: "trained_worm_v1.safetensors".to_string(),
            vocab_path: crate::geometry::VOCABULARY_PATH.to_string(),
            reservoir_capacity: 64,
            hebbian_lr: 0.01,
            jaccard_window: 10,
        }
    }
}

/// Sliding-window Jaccard drift monitor for the dense learning signal.
///
/// Tracks how rapidly the set of active neurons changes between consecutive
/// pre-entmax states. Low drift = stable associations forming.
#[derive(Clone, Debug)]
pub struct JaccardDrift {
    pub window: VecDeque<f64>,
    pub window_size: usize,
    prev_dense: Option<Array1<f32>>,
}

impl JaccardDrift {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            prev_dense: None,
        }
    }

    pub fn push(&mut self, curr: &Array1<f32>) {
        if let Some(ref prev) = self.prev_dense {
            let drift = compute_jaccard_drift(prev, curr, 1e-4);
            self.window.push_back(drift);
            while self.window.len() > self.window_size {
                self.window.pop_front();
            }
        }
        self.prev_dense = Some(curr.clone());
    }

    pub fn mean_drift(&self) -> f64 {
        let n = self.window.len();
        if n == 0 {
            return 0.0;
        }
        self.window.iter().sum::<f64>() / n as f64
    }

    pub fn std_drift(&self) -> f64 {
        let n = self.window.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean_drift();
        let var = self.window.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / n as f64;
        var.sqrt()
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.prev_dense = None;
    }
}

/// Jaccard distance between two dense state activation sets.
fn compute_jaccard_drift(prev: &Array1<f32>, curr: &Array1<f32>, threshold: f32) -> f64 {
    let active_prev: Vec<usize> = prev
        .iter()
        .enumerate()
        .filter(|&(_, v)| v.abs() > threshold)
        .map(|(i, _)| i)
        .collect();
    let active_curr: Vec<usize> = curr
        .iter()
        .enumerate()
        .filter(|&(_, v)| v.abs() > threshold)
        .map(|(i, _)| i)
        .collect();
    if active_prev.is_empty() && active_curr.is_empty() {
        return 0.0;
    }
    let intersection = active_prev
        .iter()
        .filter(|i| active_curr.contains(i))
        .count();
    let union = active_prev.len() + active_curr.len() - intersection;
    if union == 0 {
        0.0
    } else {
        1.0 - intersection as f64 / union as f64
    }
}

/// Compute the branching ratio σ for a probability distribution.
pub fn current_sigma(activation: &[f32]) -> f64 {
    CriticalityDashboard::compute_branching_ratio(activation)
}

/// Compute the mean absolute echo bias strength.
pub fn echo_bias_strength(reservoir: &EchoReservoir, current: &Array1<f32>) -> f32 {
    let bias = reservoir.query(current);
    bias.iter().map(|&v| v.abs()).sum::<f32>() / WORM_NEURON_COUNT as f32
}

/// Get the top-K active neuron indices and their values from a state.
pub fn top_k_active(state: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut with_idx: Vec<(usize, f32)> = state.iter().copied().enumerate().collect();
    with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    with_idx
        .into_iter()
        .take(k)
        .filter(|&(_, v)| v > 0.0)
        .collect()
}

/// Parsed observation command.
#[derive(Debug, PartialEq)]
pub enum ObsCommand {
    Alpha(f32),
    Kappa(f32),
    Temperature(f64),
    Reset,
    Vocab(String),
    Help,
    Quit,
    Step,
    Auto,
    None,
}

/// Parse a CLI input line into an `ObsCommand`.
pub fn parse_command(line: &str) -> ObsCommand {
    let line = line.trim();
    if let Some(val) = line.strip_prefix("/alpha ") {
        let v: f32 = val.trim().parse().unwrap_or(-1.0);
        if (0.0..=1.0).contains(&v) {
            return ObsCommand::Alpha(v);
        }
        return ObsCommand::None;
    }
    if let Some(val) = line.strip_prefix("/kappa ") {
        let v: f32 = val.trim().parse().unwrap_or(-1.0);
        if v >= 0.0 {
            return ObsCommand::Kappa(v);
        }
        return ObsCommand::None;
    }
    if let Some(val) = line.strip_prefix("/temp ") {
        let v: f64 = val.trim().parse().unwrap_or(-1.0);
        if (0.0..=5.0).contains(&v) {
            return ObsCommand::Temperature(v);
        }
        return ObsCommand::None;
    }
    if let Some(val) = line.strip_prefix("/vocab ") {
        return ObsCommand::Vocab(val.trim().to_string());
    }
    match line {
        "/reset" => ObsCommand::Reset,
        "/step" => ObsCommand::Step,
        "/auto" => ObsCommand::Auto,
        "/help" => ObsCommand::Help,
        "/quit" | "/exit" => ObsCommand::Quit,
        "" => ObsCommand::None,
        _ => ObsCommand::None,
    }
}

/// Move cursor up `n` lines, clear each line, and move back to top.
pub fn clear_lines(n: usize) {
    print!("\x1B[{}A", n);
    for i in 0..n {
        print!("\x1B[2K\r");
        if i + 1 < n {
            print!("\x1B[1B");
        }
    }
    print!("\x1B[{}A", n);
}

/// Print the 22-line dashboard (no cursor management — caller handles that).
pub fn render_dashboard(
    brain: &WormBrain,
    input_word: &str,
    sparse: Option<&Array1<f32>>,
    dense: Option<&Array1<f32>>,
    jaccard: &JaccardDrift,
    temperature: f64,
    resonance_trace: &[String],
    step_count: usize,
) {
    let sigma = sparse
        .map(|s| current_sigma(s.as_slice().unwrap_or(&[])))
        .unwrap_or(0.0);
    let echo_strength = brain
        .echo_reservoir
        .as_ref()
        .map(|r| {
            let dummy = Array1::zeros(WORM_NEURON_COUNT);
            echo_bias_strength(r, &dummy)
        })
        .unwrap_or(0.0);
    let reservoir_used = brain
        .echo_reservoir
        .as_ref()
        .map(|r| r.history.len())
        .unwrap_or(0);
    let reservoir_cap = brain
        .echo_reservoir
        .as_ref()
        .map(|r| r.capacity)
        .unwrap_or(0);

    // --- Panel 1 ---
    println!("┌─ SYSTEM STATUS ─────────────────────────────────────────────────┐");
    println!(
        "│ Σ(branch): {:<6.4}  │  κ_gate: {:<6.2}  │  t_scale: {:<6.1}  │",
        sigma, brain.cognition.kappa_gate, brain.cognition.t_scale,
    );
    println!(
        "│ α_echo: {:<6.3}     │  Echo bias: {:<8.5} │  Reservoir: {:>3}/{:>3} {:<10} │",
        brain.cognition.alpha_echo,
        echo_strength,
        reservoir_used,
        reservoir_cap,
        reservoir_bar(reservoir_used, reservoir_cap),
    );
    println!(
        "│ Jaccard drift: {:<6.4} (σ={:<6.4}) │  Entropy: {:<8.3} nats  │  Steps: {:<5}      │",
        jaccard.mean_drift(),
        jaccard.std_drift(),
        sparse
            .map(|s| CriticalityDashboard::compute_betti_entropy(s.as_slice().unwrap_or(&[])))
            .unwrap_or(0.0),
        step_count,
    );
    println!("├────────────────────────────────────────────────────────────────┤");

    // --- Panel 2: Dual-Stream ---
    println!("│ INPUT: {:54} │", input_word);
    match (sparse, dense) {
        (Some(s), Some(d)) => {
            let top = top_k_active(s.as_slice().unwrap(), 5);
            let active_sparse = s.iter().filter(|&&v| v > 0.0).count();
            let active_dense = d.iter().filter(|&&v| v.abs() > 1e-4).count();
            let dense_rms = (d.mapv(|v| v * v).sum() / WORM_NEURON_COUNT as f32).sqrt();

            if !top.is_empty() {
                let neurons_str: String = top
                    .iter()
                    .map(|&(i, v)| format!("{:>3}:{:.3}", i, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("│ SPARSE (action):  {:53} │", neurons_str);
            } else {
                println!("│ SPARSE (action):  (uniform / no dominant neuron)          │");
            }
            println!(
                "│                   {} active neurons                       │",
                format!("{:>3}", active_sparse)
            );
            println!(
                "│ DENSE (learning): {} active nodes  RMS={:.5}            │",
                format!("{:>3}", active_dense),
                dense_rms,
            );

            // Echo bias effect display
            if brain.cognition.alpha_echo > 0.0 {
                if let Some(ref reservoir) = brain.echo_reservoir {
                    let bias = reservoir.query(d);
                    let bias_rms = (bias.mapv(|v| v * v).sum() / WORM_NEURON_COUNT as f32).sqrt();
                    let bias_top = top_k_active(bias.as_slice().unwrap(), 3);
                    if !bias_top.is_empty() {
                        let bias_str: String = bias_top
                            .iter()
                            .map(|&(i, v)| format!("{:>3}:{:.4}", i, v))
                            .collect::<Vec<_>>()
                            .join(", ");
                        println!("│ ECHO bias:  RMS={:.5}  top: {:42} │", bias_rms, bias_str);
                    } else {
                        println!(
                            "│ ECHO bias:  RMS={:.5}  (diffuse)                      │",
                            bias_rms
                        );
                    }
                }
            }
        }
        _ => {
            println!("│ SPARSE (action):  —                                                 │");
            println!("│ DENSE (learning): —                                                 │");
        }
    }
    println!(
        "│ Temperature: {:<6.3}                                           │",
        temperature
    );
    println!("├────────────────────────────────────────────────────────────────┤");

    // --- Panel 3: Associative Resonance ---
    println!("│ HIPPOCAMPUS RESONANCE                                          │");

    // Association matrix fill percentage
    let assoc_pct = brain
        .echo_reservoir
        .as_ref()
        .map(|r| {
            let nonzero = r.associations.iter().filter(|&&v| v.abs() > 1e-4).count();
            let total = WORM_NEURON_COUNT * WORM_NEURON_COUNT;
            if total == 0 {
                0.0
            } else {
                nonzero as f64 / total as f64 * 100.0
            }
        })
        .unwrap_or(0.0);
    let hebbian_lr = brain
        .echo_reservoir
        .as_ref()
        .map(|r| r.hebbian_lr)
        .unwrap_or(0.0);
    let assoc_bar_len = assoc_pct as usize / 5;
    let assoc_bar = format!(
        "{}{}",
        "█".repeat(assoc_bar_len.min(20)),
        "░".repeat((20 - assoc_bar_len).max(0))
    );
    println!(
        "│ Associations: {} {:>5.1}% | Hebbian LR: {:<8.5}      │",
        assoc_bar, assoc_pct, hebbian_lr,
    );

    // Resonance trace (last few words)
    let trace_show = resonance_trace.len().min(4);
    if trace_show > 0 {
        let start = resonance_trace.len().saturating_sub(trace_show);
        let trace_line: String = resonance_trace[start..]
            .iter()
            .map(|t| format!(" \"{}\"", t))
            .collect::<Vec<_>>()
            .join(" →");
        println!("│ Active memory trace:{}              │", trace_line);
    }

    // Firing neurons bar (show top ~10)
    if let Some(s) = sparse {
        let top_firing = top_k_active(s.as_slice().unwrap(), 10);
        if !top_firing.is_empty() {
            let neuron_ids: String = top_firing
                .iter()
                .map(|&(i, _)| format!("{:>4}", i))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "│ Neurons firing:{}                                   │",
                neuron_ids
            );
            // ASCII bar visualization
            let max_v = top_firing.first().map(|&(_, v)| v).unwrap_or(1.0);
            let bars: String = top_firing
                .iter()
                .map(|&(_, v)| {
                    let n = ((v / max_v) * 6.0) as usize;
                    format!("{:>4}", "█".repeat(n.max(1).min(6)))
                })
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "│                  {}                                   │",
                bars
            );
        }
    }

    println!("└────────────────────────────────────────────────────────────────┘");
    println!();

    use std::io::{self, Write};
    let _ = io::stdout().flush();
}

fn reservoir_bar(used: usize, cap: usize) -> String {
    if cap == 0 {
        return String::new();
    }
    let filled = (used * 10) / cap;
    format!("{}{}", "█".repeat(filled), "░".repeat((10 - filled).max(0)))
}

/// Print the help message.
pub fn print_help() {
    println!();
    println!("  Commands:");
    println!("    /alpha <0.0-1.0>   Set echo permeability");
    println!("    /kappa <0.0-10.0>  Set NMDA gate sharpness");
    println!("    /temp  <0.0-5.0>   Set Boltzmann temperature");
    println!("    /reset             Clear hippocampus associations");
    println!("    /step              Single inference step");
    println!("    /auto              Continuous polling mode");
    println!("    /vocab <word>      Query specific word activation");
    println!("    /help              This menu");
    println!("    /quit              Exit");
    println!();
}
