use super::Algorithm;
use crate::environment::Environment;
use crate::models::rl_model2::RLModel2;
use crate::visualization::{RuntimeStats, VisualizationState};
use candle_core::{Device, Tensor};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Algorithm2 {
    pub model: RLModel2,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub n_timesteps: usize,
    pub device: Device,
}

impl Algorithm2 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        dt: f32,
        device: Device,
        state_bounds: Option<Vec<usize>>,
    ) -> Result<Self, Box<dyn Error>> {
        let input_size = state_size * 2 + 1;
        let context_size = 1;

        // Construct input_bounds for RLModel2: [s_t_bounds, s_t_plus_m_bounds, action_bound]
        let input_bounds = state_bounds.map(|bounds| {
            let mut combined = Vec::with_capacity(state_size * 2 + 1);
            combined.extend(bounds.iter());
            combined.extend(bounds.iter());
            combined.push(action_size); // Action input is a single integer 0..action_size
            combined
        });

        let model = RLModel2::new(input_size, context_size, hidden_sizes, &device, dt, input_bounds)
            .expect("Failed to create RLModel2");

        Ok(Self {
            model,
            n_episodes: 100,
            n_steps_per_episode: 50,
            epochs_per_episode: 5,
            n_timesteps: 40,
            device,
        })
    }
}

impl Algorithm for Algorithm2 {
    fn run(
        &mut self,
        env: &mut dyn Environment,
        _visualize: bool,
        vis_state: Option<Arc<Mutex<VisualizationState>>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut total_iteration = 0;
        let start_time = Instant::now();
        let mut total_inference_time = Duration::new(0, 0);
        let mut total_inference_actions: usize = 0;
        let mut total_training_time = Duration::new(0, 0);
        let mut total_epochs: usize = 0;
        let action_size = env.action_size();
        let state_size = env.state_size();

        for episode in 1..=self.n_episodes {
            println!("starting episode {}", episode);

            env.reset()?;
            std::thread::sleep(Duration::from_millis(500)); // wait for environment to settle

            let mut episode_states = Vec::new();
            let mut episode_actions = Vec::new();
            let mut episode_rewards = Vec::new();

            self.model.disable_learning(); // inference mode

            let mut total_reward = 0.0;

            let inference_start = Instant::now();

            for _step in 0..self.n_steps_per_episode {
                let current_state = env.get_state()?;

                let state_f32: Vec<f32> = current_state.iter().map(|&x| x as f32).collect();
                episode_states.push(state_f32.clone());

                let mut best_activity = -1.0;
                let mut best_action = 0;

                for a in 0..action_size {
                    total_inference_actions += 1;
                    // Create combined input: [x_t, x_{t+M}, a]
                    let mut input_vec = Vec::with_capacity(state_size * 2 + 1);
                    input_vec.extend(state_f32.iter());
                    input_vec.extend(vec![0.0f32; state_size]); // x_{t+M} is zeroed out during inference
                    input_vec.push(a as f32);

                    let input_tensor = Tensor::from_vec(
                        input_vec,
                        (state_size * 2 + 1, 1),
                        &self.device,
                    )?;
                    let activity = self.model.process(&input_tensor, self.n_timesteps)?;
                    println!("activity for {}: {}", a, activity);

                    if activity > best_activity {
                        best_activity = activity;
                        best_action = a;
                    }
                }

                println!("best action: {}", best_action);
                let reward = env.evaluate_action(&current_state, best_action);

                episode_actions.push(best_action);
                episode_rewards.push(reward);

                total_reward += reward;

                env.apply_action(best_action)?;
                std::thread::sleep(Duration::from_millis(100)); // give some time for action to take effect

                // Update visualization state for environment
                if let Some(ref vis_state_arc) = vis_state {
                    if let Ok(mut state) = vis_state_arc.try_lock() {
                        let env_state = env.get_state()?;
                        state.environment_state = Some(env_state);
                    }
                }

                // Visualization check during inference
                if let Some(ref vis_state_arc) = vis_state {
                    let mut should_break = false;
                    loop {
                        let (is_paused, should_close) = vis_state_arc
                            .try_lock()
                            .map(|state| (state.is_paused, state.should_close))
                            .unwrap_or((false, false));
                        if should_close {
                            should_break = true;
                            break;
                        }
                        if !is_paused {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    if should_break {
                        break;
                    }
                }
            } // end of inference steps

            let inference_elapsed = inference_start.elapsed();
            total_inference_time += inference_elapsed;

            // Check if save or load was requested
            if let Some(state_arc) = &vis_state {
                if let Ok(mut lock) = state_arc.lock() {
                    if lock.save_requested {
                        println!("Manual save requested...");
                        let checkpoints_dir = std::path::Path::new("checkpoints");
                        if !checkpoints_dir.exists() {
                            std::fs::create_dir_all(checkpoints_dir)?;
                        }
                        let path =
                            checkpoints_dir.join(format!("model_epoch_{}.safetensors", episode));
                        match self.model.save(&path) {
                            Ok(_) => println!("Successfully saved model to {:?}", path),
                            Err(e) => println!("Failed to save model: {}", e),
                        }
                        lock.save_requested = false;
                    }

                    if lock.load_requested {
                        println!("Manual load requested...");
                        let checkpoints_dir = std::path::Path::new("checkpoints");
                        if checkpoints_dir.exists() {
                            // Find the most recently modified safetensors file
                            let mut latest_file = None;
                            let mut latest_time = std::time::SystemTime::UNIX_EPOCH;

                            for entry in std::fs::read_dir(checkpoints_dir)? {
                                let entry = entry?;
                                let path = entry.path();
                                if path.is_file()
                                    && path.extension().and_then(|s| s.to_str())
                                        == Some("safetensors")
                                {
                                    if let Ok(metadata) = entry.metadata() {
                                        if let Ok(modified) = metadata.modified() {
                                            if modified > latest_time {
                                                latest_time = modified;
                                                latest_file = Some(path);
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(path) = latest_file {
                                println!("Loading model from {:?}...", path);
                                match self.model.load(&path) {
                                    Ok(_) => println!("Successfully loaded model."),
                                    Err(e) => println!(
                                        "Failed to load model (possibly shape mismatch): {}",
                                        e
                                    ),
                                }
                            } else {
                                println!("No .safetensors file found in checkpoints/");
                            }
                        } else {
                            println!("checkpoints directory does not exist.");
                        }
                        lock.load_requested = false;
                    }
                }
            }

            // Update epoch rewards
            if let Some(ref vis_state_arc) = vis_state {
                if let Ok(mut state) = vis_state_arc.try_lock() {
                    state.epoch_rewards.push((episode, total_reward as f32));
                }
            }

            // 2. Data pairing and augmentation
            let mut train_data = Vec::new(); // (input_tensor_vec, label, reward_normalized)
            let mut rng = rand::thread_rng();
            let n_states = episode_states.len();

            use rand::Rng;

            println!("Pairing and augmenting data...");
            if n_states > 1 {
                for t in 0..n_states - 1 {
                    let max_m = n_states - t;
                    let m = rng.gen_range(1..max_m);

                    let s_t = &episode_states[t];
                    let s_t_plus_m = &episode_states[t + m];
                    let a_t = episode_actions[t];
                    let r_t = episode_rewards[t] as f32;

                    // Positive sample
                    let mut pos_input = Vec::with_capacity(state_size * 2 + 1);
                    pos_input.extend(s_t.clone());
                    pos_input.extend(s_t_plus_m.clone());
                    pos_input.push(a_t as f32);
                    train_data.push((pos_input, 1.0f32, r_t));

                    // Negative sample 1: random future state
                    let neg_m = rng.gen_range(0..n_states);
                    let mut neg_input1 = Vec::with_capacity(state_size * 2 + 1);
                    neg_input1.extend(s_t.clone());
                    neg_input1.extend(episode_states[neg_m].clone());
                    neg_input1.push(a_t as f32);
                    train_data.push((neg_input1, 0.0f32, 1.0f32)); // r_t ignored for Y=0

                    // Negative sample 2: random valid action
                    let mut a_star = rng.gen_range(0..action_size);
                    if a_star == a_t {
                        a_star = (a_star + 1) % action_size;
                    }
                    let mut neg_input2 = Vec::with_capacity(state_size * 2 + 1);
                    neg_input2.extend(s_t.clone());
                    neg_input2.extend(s_t_plus_m.clone());
                    neg_input2.push(a_star as f32);
                    train_data.push((neg_input2, 0.0f32, 1.0f32));
                }

                // 3. Reward scaling (Sigmoid)
                for i in 0..train_data.len() {
                    let r = train_data[i].2;
                    if train_data[i].1 > 0.5 {
                        let sig_r = 1.0 / (1.0 + (-r).exp());
                        train_data[i].2 = sig_r;
                    }
                }
            }

            // Train on collected episode data
            self.model.enable_learning();

            let training_start = Instant::now();

            for _epoch in 0..self.epochs_per_episode {
                println!(
                    "Training epoch {} with {} samples",
                    _epoch,
                    train_data.len()
                );
                for (input_vec, label, reward) in &train_data {
                    total_iteration += 1;

                    let input_tensor = Tensor::from_vec(
                        input_vec.clone(),
                        (state_size * 2 + 1, 1),
                        &self.device,
                    )?;
                    let context_tensor = Tensor::from_vec(vec![*label], (1, 1), &self.device)?;

                    for layer in self.model.layers.iter_mut() {
                        layer.set_positive_sample(*label);
                    }
                    self.model.set_reward(*reward);

                    // For spike plot history
                    let mut spike_history = Vec::new();
                    let record_layer = vis_state
                        .as_ref()
                        .and_then(|vs| vs.try_lock().ok().and_then(|s| s.selected_layer_id));

                    self.model.reset()?;
                    for _ in 0..self.n_timesteps {
                        self.model.step(&input_tensor, Some(&context_tensor))?;

                        if let Some(layer_id) = record_layer {
                            if let Ok(activity) = self.model.get_layer_activity(layer_id) {
                                spike_history.push(activity);
                            }
                        }
                    }

                    // Update visualization snapshot every 20 iterations during training
                    if let Some(ref vis_state_arc) = vis_state {
                        if let Ok(mut state) = vis_state_arc.try_lock() {
                            if !spike_history.is_empty() {
                                state.epoch_spike_history = Some((episode, spike_history));
                            }
                            if total_iteration % 20 == 0 {
                                if let Ok(snapshot) = self.model.get_visualization_snapshot() {
                                    state.update_from_snapshot(snapshot);
                                }
                                let elapsed = start_time.elapsed().as_secs_f32();
                                let speed = if elapsed > 0.0 {
                                    total_iteration as f32 / elapsed
                                } else {
                                    0.0
                                };
                                state.runtime_stats = RuntimeStats {
                                    epoch: episode,
                                    iteration: total_iteration,
                                    timestep: total_iteration * self.n_timesteps,
                                    iterations_per_second: speed,
                                }
                            }
                        }
                    }
                }
            }

            let training_elapsed = training_start.elapsed();
            total_training_time += training_elapsed;
            total_epochs += self.epochs_per_episode;

            let inf_aps = if total_inference_time.as_secs_f32() > 0.0 {
                total_inference_actions as f32 / total_inference_time.as_secs_f32()
            } else {
                0.0
            };
            let ep_s = if total_training_time.as_secs_f32() > 0.0 {
                total_epochs as f32 / total_training_time.as_secs_f32()
            } else {
                0.0
            };
            println!(
                "[Episode {}] Actions/sec: {:.1} | Epochs/sec: {:.2}",
                episode, inf_aps, ep_s
            );
        } // end of episodes

        // Final auto-save
        println!("Training completed. Auto-saving final model...");
        let checkpoints_dir = std::path::Path::new("checkpoints");
        if !checkpoints_dir.exists() {
            std::fs::create_dir_all(checkpoints_dir)?;
        }
        let final_path = checkpoints_dir.join("model_final.safetensors");
        match self.model.save(&final_path) {
            Ok(_) => println!("Successfully saved final model to {:?}", final_path),
            Err(e) => println!("Failed to save final model: {}", e),
        }

        if let Some(ref vis_state_arc) = vis_state {
            if let Ok(state) = vis_state_arc.try_lock() {
                let csv_path = checkpoints_dir.join("epoch_rewards.csv");
                let _ = state.save_graphs_to_csv(&csv_path);
            }
        }

        Ok(())
    }
}
