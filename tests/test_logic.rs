use candle_core::{Device, Tensor};
use custom_framework::models::rl_model3::RLModel3;

/// Test the AC-CSDP Actor on a simple Logic Gate (XOR) by scoring perturbations manually.
#[test]
fn test_ac_csdp_actor_logic_learning() {
    let device = Device::Cpu;
    let dt = 0.1;
    let n_timesteps = 40;

    // 2 inputs for logic. 2 outputs for one-hot (True/False).
    // State bounds: [2, 2] means each input variable is a discrete integer {0, 1}.
    let mut model = RLModel3::new(2, 2, vec![64, 32], vec![32], &device, dt, Some(vec![2, 2]))
        .expect("Failed to construct RLModel3");

    let pairs = vec![
        (vec![0.0f32, 0.0f32], 0), // False
        (vec![0.0f32, 1.0f32], 1), // True
        (vec![1.0f32, 0.0f32], 1), // True
        (vec![1.0f32, 1.0f32], 0), // False
    ];

    // Test logic function training
    for epoch in 1..=50 {
        let mut total_accuracy = 0;

        for (input_vec, class_idx) in &pairs {
            let input_tensor = Tensor::from_vec(input_vec.clone(), (2, 1), &device).unwrap();

            // Baseline pass
            model.actor.is_learning = false;
            let mut z_base_vec = vec![0.0f32; 2];
            model.actor.reset().unwrap();
            for _ in 0..n_timesteps {
                for layer in model.actor.layers.iter_mut() {
                    layer.reset_input().unwrap();
                }
                model.actor.layers[0].add_input(&input_tensor).unwrap();
                model.actor.layers[0].step(dt).unwrap();
                model.actor.layers[1].step(dt).unwrap();
                for syn_conn in model.actor.synapses.iter_mut() {
                    let pre_act = model.actor.layers[syn_conn.metadata.pre_layer]
                        .output()
                        .unwrap()
                        .clone();
                    let post_in = syn_conn.synapse.forward(&pre_act).unwrap();
                    model.actor.layers[syn_conn.metadata.post_layer]
                        .add_input(&post_in)
                        .unwrap();
                }
                for layer in model.actor.layers.iter_mut().skip(2) {
                    layer.step(dt).unwrap();
                }

                let spikes = model.actor.layers.last().unwrap().output().unwrap();
                let spikes_vec = spikes.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                z_base_vec[0] += spikes_vec[0];
                z_base_vec[1] += spikes_vec[1];
            }

            // Perturbation pass
            let mut z_pert_vec = vec![0.0f32; 2];
            model.actor.reset().unwrap();
            let mut noise_sequence = vec![];
            for _ in 0..n_timesteps {
                noise_sequence.push(Tensor::randn(0.0f32, 15.0f32, (2, 1), &device).unwrap());
            }
            for t in 0..n_timesteps {
                for layer in model.actor.layers.iter_mut() {
                    layer.reset_input().unwrap();
                }
                model.actor.layers[0].add_input(&input_tensor).unwrap();
                model.actor.layers[0].step(dt).unwrap();
                model.actor.layers[1].step(dt).unwrap();
                for syn_conn in model.actor.synapses.iter_mut() {
                    let pre_act = model.actor.layers[syn_conn.metadata.pre_layer]
                        .output()
                        .unwrap()
                        .clone();
                    let post_in = syn_conn.synapse.forward(&pre_act).unwrap();
                    model.actor.layers[syn_conn.metadata.post_layer]
                        .add_input(&post_in)
                        .unwrap();
                }
                model
                    .actor
                    .layers
                    .last_mut()
                    .unwrap()
                    .add_input(&noise_sequence[t])
                    .unwrap();
                for layer in model.actor.layers.iter_mut().skip(2) {
                    layer.step(dt).unwrap();
                }

                let spikes = model.actor.layers.last().unwrap().output().unwrap();
                let spikes_vec = spikes.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                z_pert_vec[0] += spikes_vec[0];
                z_pert_vec[1] += spikes_vec[1];
            }

            // Evaluate proxy score for specific one-hot target
            let target_one_hot = if *class_idx == 1 {
                [0.0f32, 1.0f32]
            } else {
                [1.0f32, 0.0f32]
            };
            let dist_base = (z_base_vec[0] / 40.0 - target_one_hot[0]).abs()
                + (z_base_vec[1] / 40.0 - target_one_hot[1]).abs();
            let dist_pert = (z_pert_vec[0] / 40.0 - target_one_hot[0]).abs()
                + (z_pert_vec[1] / 40.0 - target_one_hot[1]).abs();

            let pert_better = dist_pert < dist_base;
            let best_z = if pert_better {
                z_pert_vec.clone()
            } else {
                z_base_vec.clone()
            };
            if epoch == 1 || epoch == 50 || epoch % 10 == 0 {
                println!(
                    "Epoch {} - Pair {:?} -> Base: {:?}, Pert: {:?}",
                    epoch, input_vec, z_base_vec, z_pert_vec
                );
            }
            if best_z[0] < best_z[1] && *class_idx == 1 {
                total_accuracy += 1;
            } else if best_z[0] > best_z[1] && *class_idx == 0 {
                total_accuracy += 1;
            }

            // Train
            model.actor.is_learning = true;
            for _pass in 0..2 {
                let use_pert = (pert_better && _pass == 0) || (!pert_better && _pass == 1);
                let label = if _pass == 0 { 1.0 } else { -1.0 };

                for layer in model.actor.layers.iter_mut() {
                    layer.set_positive_sample(label);
                    // Use reward explicitly as the missing structural learning rate alpha = 0.001
                    // Without this, outer products add >40.0 weights to synapses intrinsically in 1 epoch!
                    layer.set_reward(0.001);
                }
                model.actor.reset().unwrap();

                for t in 0..n_timesteps {
                    for layer in model.actor.layers.iter_mut() {
                        layer.reset_input().unwrap();
                    }
                    model.actor.layers[0].add_input(&input_tensor).unwrap();
                    model.actor.layers[0].step(dt).unwrap();
                    model.actor.layers[1].step(dt).unwrap();
                    for syn_conn in model.actor.synapses.iter_mut() {
                        let pre_act = model.actor.layers[syn_conn.metadata.pre_layer]
                            .output()
                            .unwrap()
                            .clone();
                        let post_in = syn_conn.synapse.forward(&pre_act).unwrap();
                        model.actor.layers[syn_conn.metadata.post_layer]
                            .add_input(&post_in)
                            .unwrap();
                    }
                    if use_pert {
                        model
                            .actor
                            .layers
                            .last_mut()
                            .unwrap()
                            .add_input(&noise_sequence[t])
                            .unwrap();
                    }
                    for layer in model.actor.layers.iter_mut().skip(2) {
                        layer.step(dt).unwrap();
                    }

                    for syn_conn in model.actor.synapses.iter_mut() {
                        let pre_act = model.actor.layers[syn_conn.metadata.pre_layer]
                            .output()
                            .unwrap()
                            .clone();
                        syn_conn
                            .synapse
                            .update_weights(
                                &pre_act,
                                &mut model.actor.layers[syn_conn.metadata.post_layer],
                                dt,
                            )
                            .unwrap();
                    }
                }
            }
        }

        println!("Epoch {}: Accuracy {}/4", epoch, total_accuracy);

        if epoch == 50 {
            assert_eq!(
                total_accuracy, 4,
                "Model failed to learn logic gate within 50 epochs."
            );
        }
    }
}
