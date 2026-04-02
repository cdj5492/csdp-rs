use candle_core::{DType, Device, Tensor};
use custom_framework::layer::Layer;
use custom_framework::layer::lif::LIFLayer;
use custom_framework::layer::mod_signal::standard::StandardModSignal;
use custom_framework::synapse::SynapseOps;
use custom_framework::synapse::csdp::CSDP;

#[test]
fn test_investigate_csdp_explosion() {
    let device = Device::Cpu;
    let pre_size = 10;
    let post_size = 10;

    let mut csdp = CSDP::new(pre_size, post_size, &device).unwrap();

    // THE MATH (BUG PROOF):
    // 1. As weights hit 0, `spikes` stays at 0.
    // 2. In `StandardModSignal::calc_mod_signal`, the trace `z` decays via RC: `dz = (dt / tau) * (0.0 - z)`.
    // 3. Since `z` is previously positive, `dz` is explicitly negatively traversing.
    // 4. To calculate the delta gradient `dl / dz` without dividing by 0, the codebase does: `dz_ep = dz + 0.00001`.
    // 5. As `z` mathematically approaches `0`, `dz` approaches `0` from the negative direction.
    //    It perfectly crosses `-0.00001`.
    // 6. When `dz = -0.00001`, then `dz_ep = -0.00001 + 0.00001 = 0.0`!
    // 7. `dl.div(0.0)` explodes `mod_signal` instantly to ~2.15e4, scaling weights drastically outward.

    // Force weights to be extremely close to 0 to simulate the state right before explosion
    csdp.weights = Tensor::ones((post_size, pre_size), DType::F32, &device)
        .unwrap()
        .affine(1e-6, 0.0)
        .unwrap();

    let mod_sig = Box::new(StandardModSignal::new(post_size, 10.0, 1.0, 0.5, &device).unwrap());

    let mut post_layer =
        Box::new(LIFLayer::new(post_size, 13.0, 0.5, 0.01, mod_sig, &device).unwrap())
            as Box<dyn Layer>;

    let mut pre_activity = Tensor::ones((pre_size, 1), DType::F32, &device).unwrap();

    let dt = 0.1;
    let lambda_d = 0.00005f32; // same as inside update_weights

    println!("Starting simulation with weights near zero...");

    for epoch in 0..1000 {
        // Evaluate what happens over timesteps.
        for t in 0..40 {
            // Suppose post_layer spikes slightly, pre_layer spikes slightly
            // This is a dummy mod_signal simulation.
            // The exact bug is mathematically reproducing the tensor operations
            // inside CSDP update_weights.

            post_layer.set_positive_sample(1.0);
            post_layer.set_reward(0.001);
            let mut inputs = Tensor::zeros((post_size, 1), DType::F32, &device).unwrap();

            // To simulate the network, we give it inputs that might drive spikes occasionally
            if rand::random::<f32>() < 0.1 {
                inputs = inputs.affine(0.0, 10.0).unwrap();
            }

            post_layer.add_input(&inputs).unwrap();
            post_layer.step(dt).unwrap();

            csdp.update_weights(&pre_activity, &mut post_layer, dt)
                .unwrap();

            // Re-simulate pre_activity as somewhat sparse spikes
            let pre_vec: Vec<f32> = (0..pre_size)
                .map(|_| {
                    if rand::random::<f32>() < 0.1 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            pre_activity = Tensor::from_vec(pre_vec, (pre_size, 1), &device).unwrap();
        }

        // Print weight stats to observe explosion
        let stats = csdp.weight_stats().unwrap();
        if epoch % 100 == 0 || stats.max > 100.0 || stats.min < -100.0 {
            println!(
                "Epoch {}: Mean={:e}, Max={:e}, Min={:e}",
                epoch, stats.mean, stats.max, stats.min
            );
            if stats.max > 100.0 || stats.min < -100.0 {
                println!("EXPLOSION DETECTED!");
                break;
            }
        }
    }
}
