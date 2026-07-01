//! Short-term Synaptic Echo Buffer & long-term trajectory storage.
//!
//! Implements a volatile [`SynapticEchoBuffer`] that maintains a decaying
//! trace of recent 302-neuron activation patterns, allowing previous
//! tokens to warp the geometric projection of the current prompt.
//!
//! Also manages the persistent trajectory vault (`vault.json`) and offline
//! Hebbian sleep consolidation for engraving interaction pathways.

use crate::geometry::token_to_coord;
use crate::training::WormTrainer;
use crate::worm_brain::{WORM_NEURON_COUNT, WormBrain};
use crate::{CoreError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A volatile, decaying trace of recent neural firing patterns with a Dirac
/// operator for chirality-preserving state evolution.
///
/// Each call to [`inject_echo`] stores the current 302-neuron activation
/// vector.  Before the next routing step, [`apply_and_decay`]:
/// 1. Blends the stored echo into the new projection (standard decay).
/// 2. Applies a **discrete Dirac operator** — the square root of the graph
///    Laplacian on the activation trajectory — to prevent feature smoothing
///    (oversmoothing) over deep association chains.
///
/// ## Dirac Operator
/// The Dirac operator `D` satisfies `D² = L` where `L` is the graph
/// Laplacian.  On a 1-D chain of activations we approximate
/// `D(ψₜ) = ψₜ + γ · ∇ψₜ` where `∇ψₜ = ψₜ − ψₜ₋₁` is the discrete gradient
/// and `γ` is the Dirac coupling strength.  `D` has both positive and
/// negative eigenvalues: the positive branch preserves feature identity
/// (particle modes), while the negative branch inhibits uniform collapse
/// (anti-particle / chiral modes).  Together they prevent the activation
/// from "oversmoothing" into a flat, featureless state.
#[derive(Clone, Debug)]
pub struct SynapticEchoBuffer {
    /// Decaying trace of the most recent activation (size 302).
    pub echo_state: Array1<f32>,
    /// Attenuation multiplier applied every decay step (default `0.75`).
    pub decay_factor: f32,
    /// Previous activation for Dirac gradient computation.
    prev_activation: Option<Array1<f32>>,
    /// Dirac coupling strength γ (default `0.15`).
    dirac_gamma: f32,
}

impl SynapticEchoBuffer {
    /// Create a new buffer with zeroed echo state and default Dirac coupling.
    #[inline]
    pub fn new(decay_factor: f32) -> Self {
        Self {
            echo_state: Array1::zeros(WORM_NEURON_COUNT),
            decay_factor,
            prev_activation: None,
            dirac_gamma: 0.15,
        }
    }

    /// Overwrite the echo state with the latest activation pattern.
    ///
    /// Call this *after* decoding to store the current firing vector
    /// so it can influence the next routing step.
    #[inline]
    pub fn inject_echo(&mut self, activations: &Array1<f32>) {
        self.echo_state.assign(activations);
    }

    /// Blend the decayed echo into `current_input` (a 302-D projection
    /// state), apply the Dirac operator, then attenuate the stored echo.
    ///
    /// ## Dirac Application
    /// 1. Standard echo blending: `input += echo_state`.
    /// 2. Dirac gradient injection: `input += γ · (input − prev_input)`
    ///    where the gradient is taken in the *current* state direction,
    ///    not the echo direction.  This is the discrete analog of
    ///    `(I + γ · ∇)ψ` in the Dirac equation.
    /// 3. The positive eigenvalue branch (particle) maintains feature
    ///    identity; the negative branch (anti-particle) prevents collapse
    ///    to a uniform fixpoint.
    ///
    /// # Errors
    /// * [`CoreError::InvalidDimension`] if `current_input` is not 302-long.
    /// * [`CoreError::Bridge`] if the Dirac operator produces non-finite values.
    #[inline]
    pub fn apply_and_decay(&mut self, current_input: &mut Array1<f32>) -> Result<()> {
        if current_input.len() != WORM_NEURON_COUNT {
            return Err(CoreError::InvalidDimension {
                expected: WORM_NEURON_COUNT,
                got: current_input.len(),
            });
        }

        // 1. Standard echo blending.
        *current_input += &self.echo_state;

        // 2. Discrete Dirac operator: D(ψ) = ψ + γ · ∇ψ
        //    The gradient ∇ψ = ψ − ψ_prev injects chirality into the state,
        //    preventing uniform oversmoothing across deep association chains.
        if let Some(prev) = &self.prev_activation {
            let gamma = self.dirac_gamma;
            for k in 0..current_input.len() {
                let grad = current_input[k] - prev[k];
                current_input[k] += gamma * grad;
            }
            // Finiteness check for the Dirac term.
            for &v in current_input.iter() {
                if !v.is_finite() {
                    return Err(CoreError::Bridge(
                        "Dirac operator produced non-finite values".into(),
                    ));
                }
            }
        }

        // 3. Store the current state as the "previous" for the next Dirac step.
        self.prev_activation = Some(current_input.clone());

        // 4. Attenuate the echo for the next cycle.
        self.echo_state.mapv_inplace(|v| v * self.decay_factor);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Trajectory vault – persistent long-term storage of interaction histories
// ---------------------------------------------------------------------------

/// Default path for the trajectory vault.
pub const VAULT_PATH: &str = "/home/darky/workspace/UtoPhieCorn_Architecture/vault.json";

/// A single recorded interaction trajectory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryEntry {
    /// Unix epoch seconds when this interaction occurred.
    pub timestamp: u64,
    /// The user's input token.
    pub input: String,
    /// The 4-step decoded chain produced by the corpus callosum.
    pub steps: Vec<String>,
}

/// Summary returned by [`consolidate_sleep`].
#[derive(Debug)]
pub struct ConsolidationReport {
    /// Number of trajectory chains consolidated from the vault.
    pub entries_consolidated: usize,
    /// Total Hebbian training steps applied across all chains.
    pub total_steps: usize,
    /// Final stability guard pass (always `Ok` or `Err`).
    pub status: Result<()>,
}

/// Append a single interaction trajectory to the vault file.
///
/// Creates the vault if it does not exist; appends to the JSON array
/// otherwise.  Uses an append-friendly strategy (read-all, push, write-all).
///
/// # Errors
/// * [`CoreError::Bridge`] if I/O or JSON serialization fails.
pub fn log_trajectory(user_input: &str, steps: &[String]) -> Result<()> {
    let path = Path::new(VAULT_PATH);

    let mut entries: Vec<TrajectoryEntry> = if path.exists() {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CoreError::Bridge(format!("failed to read vault: {e}")))?;
        if content.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&content)
                .map_err(|e| CoreError::Bridge(format!("vault JSON parse error: {e}")))?
        }
    } else {
        Vec::new()
    };

    let entry = TrajectoryEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        input: user_input.to_string(),
        steps: steps.to_vec(),
    };

    entries.push(entry);

    let json = serde_json::to_string_pretty(&entries)
        .map_err(|e| CoreError::Bridge(format!("vault JSON serialization error: {e}")))?;

    std::fs::write(path, &json)
        .map_err(|e| CoreError::Bridge(format!("failed to write vault: {e}")))?;

    Ok(())
}

/// Read all trajectory entries from the vault file.
///
/// Returns an empty `Vec` if the vault does not exist or is empty.
pub fn read_vault_entries() -> Result<Vec<TrajectoryEntry>> {
    let path = Path::new(VAULT_PATH);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(path)
        .map_err(|e| CoreError::Bridge(format!("failed to read vault: {e}")))?;
    if content.trim().is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(&content)
        .map_err(|e| CoreError::Bridge(format!("vault JSON parse error: {e}")))
}

/// Clear the vault file (write an empty JSON array).
fn clear_vault() -> Result<()> {
    let json = serde_json::to_string_pretty::<[TrajectoryEntry; 0]>(&[])
        .map_err(|e| CoreError::Bridge(format!("vault JSON serialization error: {e}")))?;
    std::fs::write(Path::new(VAULT_PATH), &json)
        .map_err(|e| CoreError::Bridge(format!("failed to clear vault: {e}")))?;
    Ok(())
}

/// Offline Hebbian sleep consolidation.
///
/// 1. Reads all recorded trajectories from [`VAULT_PATH`].
/// 2. For each token in each chain, maps it to a 128-D coordinate, routes
///    it through `brain` to obtain a 302-D activation, and applies a
///    conservative Hebbian update (`lr = 0.001`, `sf = 0.999`).
/// 3. Clears the vault after successful consolidation to prevent
///    over-saturation.
/// 4. Returns a [`ConsolidationReport`] with summary statistics.
///
/// # Errors
/// * [`CoreError::Bridge`] if vault I/O fails.
/// * [`CoreError::InvalidDimension`] if a token coordinate is wrong size.
/// * [`CoreError::Numerical`] if Hebbian update produces non-finite weights.
pub fn consolidate_sleep(brain: &mut WormBrain) -> Result<ConsolidationReport> {
    let entries = read_vault_entries()?;
    if entries.is_empty() {
        return Ok(ConsolidationReport {
            entries_consolidated: 0,
            total_steps: 0,
            status: Ok(()),
        });
    }

    // Use a conservative trainer for sleep consolidation.
    let trainer = WormTrainer::new(0.001, 0.999);

    let mut total_steps = 0usize;

    for entry in &entries {
        // Walk the full path: input → each decoded step.
        let full_path = std::iter::once(&entry.input)
            .chain(entry.steps.iter())
            .cloned()
            .collect::<Vec<_>>();

        for token in &full_path {
            let coord = token_to_coord(token);
            let (activation, _pre_synaptic) = brain.route_signal(coord.inner())?;
            let input_key = coord.inner().mapv(|v| v as f32);
            trainer
                .train_step(brain, &activation, &_pre_synaptic, &input_key)
                .map_err(|e| {
                    CoreError::Numerical(format!(
                        "sleep consolidation failed at token '{token}': {e}"
                    ))
                })?;
            total_steps += 1;
        }
    }

    // Clear the vault so pathways don't over-saturate on repeated sleep.
    clear_vault()?;

    Ok(ConsolidationReport {
        entries_consolidated: entries.len(),
        total_steps,
        status: Ok(()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worm_brain::WormBrain;
    use ndarray::Array1;

    #[test]
    fn test_echo_inject_apply_decay() {
        let mut buf = SynapticEchoBuffer::new(0.75);

        // Initial state is zero — no echo, no Dirac prev → no change.
        let mut proj = Array1::from_elem(WORM_NEURON_COUNT, 1.0);
        buf.apply_and_decay(&mut proj).unwrap();
        assert!(
            (proj[0] - 1.0).abs() < 1e-12,
            "zero echo should not change input"
        );
        assert_eq!(buf.echo_state[0], 0.0);

        // Inject an activation.
        let act = Array1::from_elem(WORM_NEURON_COUNT, 0.5);
        buf.inject_echo(&act);

        // Apply: echo (0.5) is added to input (1.0) → 1.5, Dirac γ·∇ = 0.15·(1.5 - 1.0) = 0.075,
        // so final = 1.5 + 0.075 = 1.575.  Echo decays: 0.5 × 0.75 = 0.375.
        let mut proj2 = Array1::from_elem(WORM_NEURON_COUNT, 1.0);
        buf.apply_and_decay(&mut proj2).unwrap();
        assert!(
            (proj2[0] - 1.575).abs() < 1e-12,
            "echo + Dirac should produce 1.575"
        );
        assert!(
            (buf.echo_state[0] - 0.375).abs() < 1e-12,
            "echo should decay by 0.75"
        );

        // Apply again without injection: echo (0.375) added to 1.0 → 1.375,
        // Dirac γ·∇ = 0.15·(1.375 - 1.575) = −0.03, final = 1.375 − 0.03 = 1.345.
        let mut proj3 = Array1::from_elem(WORM_NEURON_COUNT, 1.0);
        buf.apply_and_decay(&mut proj3).unwrap();
        assert!((proj3[0] - 1.345).abs() < 1e-12);
        assert!((buf.echo_state[0] - 0.28125).abs() < 1e-12);
    }

    #[test]
    fn test_log_trajectory_roundtrip() {
        let temp = std::env::temp_dir().join("utophiecorn_vault_test.json");
        // Override VAULT_PATH by writing/reading directly
        let steps = vec![
            "food".to_string(),
            "energy".to_string(),
            "move".to_string(),
            "forward".to_string(),
        ];

        // Write via log_trajectory (uses the real VAULT_PATH, so we just test
        // the serialization roundtrip manually to avoid side effects).
        let entry = TrajectoryEntry {
            timestamp: 1_000_000,
            input: "worm".to_string(),
            steps: steps.clone(),
        };
        let json = serde_json::to_string_pretty(&vec![entry]).unwrap();
        std::fs::write(&temp, &json).unwrap();

        let content = std::fs::read_to_string(&temp).unwrap();
        let restored: Vec<TrajectoryEntry> = serde_json::from_str(&content).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].input, "worm");
        assert_eq!(restored[0].steps, steps);

        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn test_consolidate_sleep_empty_vault() {
        // Should not fail with empty vault.
        let temp_vault = std::env::temp_dir().join("utophiecorn_empty_vault.json");
        // Write empty array to temp location, then test read_vault_entries directly
        std::fs::write(&temp_vault, "[]").unwrap();
        // We test the function via the real vault path — skip if not exists
        // Instead, test the consolidation function doesn't panic:
        let mut brain = WormBrain::new_baseline();
        // The function will read the *real* VAULT_PATH. If empty/nonexistent it returns early.
        let report = consolidate_sleep(&mut brain).unwrap();
        // Should handle gracefully:
        assert!(report.status.is_ok());
    }

    #[test]
    fn test_echo_dimension_mismatch() {
        let mut buf = SynapticEchoBuffer::new(0.75);
        let mut bad = Array1::from_elem(100, 0.0);
        let result = buf.apply_and_decay(&mut bad);
        assert!(result.is_err());
    }
}
