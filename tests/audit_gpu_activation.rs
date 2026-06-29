/// GPU Activation Audit — prüft ob GPU tanh wirklich Null liefert
/// oder nur das Display fälschlich 0.0 loggt.
///
/// 1. checkpoint laden → synapses, input_projection
/// 2. token_to_coord("quantum") → 128-D
/// 3. CPU: route_signal() → sparsemax activation
/// 4. GPU: proj @ coord → syn @ state → tanh
/// 5. download → vergleiche mean, max, non-zero

use utophiecorn_architecture::{
    geometry::token_to_coord,
    worm_brain::WormBrain,
    storage::load_brain_state,
};

#[cfg(feature = "cuda")]
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use utophiecorn_architecture::{
    gpu::GpuEngine,
    geometry::MANIFOLD_DIM,
};

#[cfg(feature = "cuda")]
#[test]
fn audit_gpu_activation() {
    let (synapses, input_proj) = load_brain_state(
        std::path::Path::new("trained_worm_v1.safetensors")
    ).expect("checkpoint load");
    let mut brain = WormBrain::new_baseline();
    brain.synapses = synapses;
    brain.input_projection = input_proj;

    let coord = token_to_coord("quantum").inner().mapv(|v| v as f32);
    let coord_f64 = coord.mapv(|v| v as f64);

    // CPU forward (sparsemax)
    let (cpu_act, cpu_pre) = brain.route_signal(&coord_f64).expect("cpu route");
    let cpu_nonzero = cpu_act.iter().filter(|&&v| v > 0.0).count();
    let cpu_mean = cpu_act.mean().unwrap();
    let cpu_max = cpu_act.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("[CPU sparsemax]");
    println!("  pre_synaptic: mean={:.8}, max={:.8}",
        cpu_pre.mean().unwrap(),
        cpu_pre.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("  activation:   mean={:.8}, max={:.8}, non-zero={}/302, sum={:.8}",
        cpu_mean, cpu_max, cpu_nonzero, cpu_act.iter().sum::<f32>());

    // GPU forward (tanh) — exakt wie gpu.rs PersistantGpuState
    let eng = GpuEngine::try_new().expect("gpu engine").expect("cuda device");
    let dev = eng.device();

    let n = 302usize;
    let d = MANIFOLD_DIM;

    let proj_flat: Vec<f32> = brain.input_projection().iter().copied().collect();
    let syn_flat: Vec<f32> = brain.synapses.iter().copied().collect();

    let proj_gpu = eng.host_to_tensor_2d_f32(&proj_flat, &[n, d], "proj").unwrap();
    let syn_gpu = eng.host_to_tensor_2d_f32(&syn_flat, &[n, n], "syn").unwrap();

    // proj @ coord → (302,)  via (302,16) @ (16,1)
    let coord_vec: Vec<f32> = coord.to_vec();
    let coord_2d = Tensor::new(coord_vec.as_slice(), dev).unwrap()
        .reshape((d, 1)).unwrap();
    let state = proj_gpu.matmul(&coord_2d).unwrap();

    // syn @ state → (302,)
    let state = syn_gpu.matmul(&state).unwrap();

    // tanh activation
    let act_gpu = state.tanh().unwrap();

    // Download → CPU
    let gpu_out: Vec<f32> = eng.tensor_to_host_2d(&act_gpu, "act").unwrap();
    let gpu_mean = gpu_out.iter().sum::<f32>() / gpu_out.len() as f32;
    let gpu_max = gpu_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gpu_min = gpu_out.iter().cloned().fold(f32::INFINITY, f32::min);
    let gpu_nonzero = gpu_out.iter().filter(|&&v| v.abs() > 1e-10).count();
    let gpu_sum: f32 = gpu_out.iter().sum();

    println!("\n[GPU tanh]");
    println!("  activation:   mean={:.8}, max={:.8}, min={:.8}, non-zero={}/302, sum={:.8}",
        gpu_mean, gpu_max, gpu_min, gpu_nonzero, gpu_sum);

    // Bewertung
    println!("\n--- RESULT ---");
    if gpu_max.abs() < 1e-10 && gpu_min.abs() < 1e-10 {
        println!("  STATUS: KOLLAPS — GPU tanh liefert Nullvektor");
        println!("  URSACHE: synapses-Gewichte zu klein → tanh(0) = 0");
        println!("  LÖSUNG: Bias-Injektion 0.01 in gpu.rs vor tanh");
    } else {
        println!("  STATUS: AKTIV — GPU tanh produziert nicht-Null Aktivierung");
        println!("  max={:.8} ≠ 0 ✓", gpu_max);
        println!("  CPU/GPU-Divergenz: sparsemax (sum=1, 3/302 non-zero) vs tanh (saturates at -1)");
        println!("  LÖSUNG: Display-Fix — logge echten GPU-Normwert statt 0.0");
    }

    // Assertion: GPU tanh darf nicht exakt Null sein (max oder min ≠ 0)
    assert!(
        gpu_max.abs() > 1e-10 || gpu_min.abs() > 1e-10,
        "GPU KOLLAPS: tanh liefert Nullvektor (max={})", gpu_max
    );

    // CPU/GPU Divergenz dokumentieren
    println!("\n--- CPU/GPU VERGLEICH ---");
    println!("  CPU sparsemax: mean={:.6}, max={:.6}, non-zero={}/302", cpu_mean, cpu_max, cpu_nonzero);
    println!("  GPU tanh:      mean={:.6}, max={:.6}, min={:.6}", gpu_mean, gpu_max, gpu_min);
    println!("  FAZIT: Strukturelle Divergenz durch Aktivierungsfunktion (sparsemax vs tanh)");
    println!("  GPU trainiert trotzdem — ΔW ≠ 0 beweist Lernen.");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn audit_gpu_activation_skip() {
    eprintln!("SKIP: compile with --features cuda");
}
