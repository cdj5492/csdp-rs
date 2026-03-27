use candle_core::{Device, Tensor};
use custom_framework::models::rl_model3::RLModel3;

#[test]
fn test_investigate_runaway_firing() {
    let device = Device::Cpu;
    let dt = 0.1;
    let n_timesteps = 40;

    // Simulate Grid Environment
    let state_size = 4;
    let action_size = 5;

    // Create models using RLModel3, matching algorithm3.rs
    let mut model = RLModel3::new(
        state_size,
        action_size,
        vec![256, 128], // actor hidden
        vec![256, 128], // critic hidden
        &device,
        dt,
        Some(vec![50, 50, 50, 50]), // Input bounds for Actor
    )
    .expect("Failed to create RLModel3");

    println!("--- Testing Runaway Simulation Over Epochs ---");

    for epoch in 0..10 {
        println!("\n=========== EPOCH {} ===========", epoch);

        let state_f32 = vec![0.5f32, 0.5f32, 0.98f32, 0.2f32];
        let action_f32 = vec![0.5f32, 0.2f32, 0.1f32, 0.0f32, 0.9f32];

        let mut input_vec = Vec::with_capacity(state_size + action_size);
        input_vec.extend(state_f32.iter().map(|&x| x * 0.1));
        input_vec.extend(action_f32.iter().map(|&x| x * 0.1));

        let input_tensor =
            Tensor::from_vec(input_vec, (state_size + action_size, 1), &device).unwrap();

        // 1. Evaluate Critic
        model.critic.reset().unwrap();
        model.critic.is_learning = false;

        let mut crit_spikes = 0.0;
        for _ in 0..n_timesteps {
            model.critic.step(&input_tensor, None).unwrap();
            crit_spikes += model.critic.layers[4]
                .output()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
        }
        println!("Critic Baseline Q: {}", crit_spikes);

        // 2. Train Critic
        model.critic.is_learning = true;
        for layer in model.critic.layers.iter_mut() {
            layer.set_positive_sample(10.0); // Dummy Q target
            layer.set_reward(1.0); // Dummy reward
        }
        model.critic.reset().unwrap();
        for _ in 0..n_timesteps {
            model.critic.step(&input_tensor, None).unwrap();
        }

        // Check Critic Layers Activity
        let cr_in = model.critic.layers[0]
            .activity()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum::<f32>();
        let cr_h1 = model.critic.layers[2]
            .output()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum::<f32>();
        let cr_h2 = model.critic.layers[3]
            .output()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum::<f32>();
        let cr_out = model.critic.layers[4]
            .output()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum::<f32>();
        println!(
            "Critic Layer Final Activity (end of train step) -> In(p): {:.2}, H1(spikes): {}, H2(spikes): {}, Out(spikes): {}",
            cr_in, cr_h1, cr_h2, cr_out
        );

        // Print sample weights
        let stat = model.critic.synapses[4].synapse.weight_stats().unwrap();
        println!(
            "Critic Synapse H1->Out Weights: Mean={:.4}, Max={:.4}, Min={:.4}",
            stat.mean, stat.max, stat.min
        );

        let stat_out_h1 = model.critic.synapses[5].synapse.weight_stats().unwrap();
        println!(
            "Critic Synapse Out->H1 Weights: Mean={:.4}, Max={:.4}, Min={:.4}",
            stat_out_h1.mean, stat_out_h1.max, stat_out_h1.min
        );
    }
}
