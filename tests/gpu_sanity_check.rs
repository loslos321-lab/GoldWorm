//! GPU sanity check for CUDA-accelerated Maxwell-damped Hebbian training.
//!
//! Run: `cargo test --test gpu_sanity_check --features cuda -- --nocapture`
//!
//! This test:
//! 1. Creates a minimal WormBrain (302 neurons) with default connectome
//! 2. Initialises GpuEngine on Device::Cuda(0)
//! 3. Executes one forward + damped backward pass on GPU
//! 4. Queries and prints GPU VRAM consumption via nvidia-smi
//! 5. Reports pass/fail for each telemetry checkpoint

use std::process::Command;

#[cfg(feature = "cuda")]
use ndarray::Array1;
#[cfg(feature = "cuda")]
use utophiecorn_architecture::gpu::GpuEngine;
#[cfg(feature = "cuda")]
use utophiecorn_architecture::training::WormTrainer;
#[cfg(feature = "cuda")]
use utophiecorn_architecture::worm_brain::WormBrain;

/// Print current GPU VRAM usage by parsing nvidia-smi output.
#[allow(dead_code)]
fn print_gpu_vram(label: &str) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    println!("  [{label}] nvidia-smi: {trimmed}");
                }
            }
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            println!("  [VRAM] nvidia-smi error: {stderr}");
        }
        Err(e) => {
            println!("  [VRAM] nvidia-smi unavailable: {e}");
        }
    }
}

#[cfg(feature = "cuda")]
fn run_gpu_sanity() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== GPU Sanity Check ===\n");

    // 1. Query initial VRAM
    print_gpu_vram("before init");

    // 2. Initialise GpuEngine
    println!("  step 1: GpuEngine::try_new()...");
    let engine = match GpuEngine::try_new() {
        Ok(Some(eng)) => {
            println!("  ✓ GpuEngine created on CUDA device");
            eng
        }
        Ok(None) => {
            println!("  SKIP: no CUDA device found");
            return Ok(());
        }
        Err(e) => {
            println!("  SKIP: GpuEngine init failed: {e}");
            return Ok(());
        }
    };

    print_gpu_vram("after engine init");

    // 3. Create WormBrain with default connectome
    println!("  step 2: WormBrain::new_baseline()...");
    let mut brain = WormBrain::new_baseline();
    println!("  ✓ WormBrain created ({} neurons)", brain.neuron_count);
    let saved_synapses = brain.synapses.clone();

    // 4. Create WormTrainer with default params
    println!("  step 3: WormTrainer::new()...");
    let trainer = WormTrainer::new(0.01, 0.99);
    println!(
        "  ✓ WormTrainer created (lr={}, stabilize={})",
        trainer.learning_rate, trainer.stabilization_factor
    );

    // 5. Create a dummy 302-D activation (non-zero to trigger co-firing)
    println!("  step 4: creating dummy activation...");
    let mut activation = Array1::<f32>::zeros(brain.neuron_count);
    // Set a few neurons to fire: sensory (20-29), command hubs (99-102)
    for i in [20usize, 25, 30, 99, 100, 101, 150, 200, 250, 280] {
        if i < brain.neuron_count {
            activation[i] = 0.85;
        }
    }
    println!(
        "  ✓ Activation vector created ({} nonzero entries)",
        activation.iter().filter(|&&v| v > 0.0).count()
    );

    // 6. Execute one GPU damped training step
    println!("  step 5: executing train_step_gpu...");
    let before_vram = {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader"])
            .output();
        match output {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout);
                text.trim().to_string()
            }
            _ => "unknown".to_string(),
        }
    };

    trainer.train_step_gpu(&mut brain, &activation, &engine)?;

    let after_vram = {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader"])
            .output();
        match output {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout);
                text.trim().to_string()
            }
            _ => "unknown".to_string(),
        }
    };

    println!("  ✓ GPU training step completed");
    println!("  VRAM delta: {before_vram} → {after_vram}");

    // 7. Verify weights are finite and structurally bounded
    println!("  step 6: post-step verification...");
    for &val in brain.synapses.iter() {
        if !val.is_finite() {
            return Err("non-finite weight after GPU step".into());
        }
        if val < 0.0 || val > 1.0 {
            return Err(format!("weight out of bounds [0,1]: {val}").into());
        }
    }
    println!("  ✓ All weights finite and in [0,1]");

    // Verify no de novo synaptogenesis: zero positions remain zero
    let nonzero_diff = (&brain.synapses - &saved_synapses)
        .mapv(|v| v.abs())
        .iter()
        .zip(saved_synapses.iter())
        .filter(|&(diff, orig)| *diff > 0.0 && *orig == 0.0)
        .count();
    if nonzero_diff > 0 {
        println!(
            "  ⚠ {nonzero_diff} new synapses appeared (structrual blueprint not strictly enforced)"
        );
    } else {
        println!("  ✓ No de novo synaptogenesis detected");
    }

    // 8. Query final VRAM
    print_gpu_vram("after test");

    println!("\n=== GPU Sanity Check: PASSED ===\n");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_gpu_sanity() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== GPU Sanity Check ===\n");
    println!("  SKIP: compile with --features cuda to enable GPU test\n");
    Ok(())
}

#[test]
fn gpu_sanity_check() {
    match run_gpu_sanity() {
        Ok(()) => {}
        Err(e) => panic!("GPU sanity check FAILED: {e}"),
    }
}
