use super::Algorithm;
use crate::environment::Environment;
use crate::models::rl_model1::RLModel1;
use crate::visualization::{RuntimeStats, VisualizationState};
use candle_core::{Device, Tensor};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Algorithm1 {
    pub model: RLModel1,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub n_timesteps: usize,
    pub device: Device,
}

impl Algorithm1 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        dt: f32,
        device: Device,
    ) -> Result<Self, Box<dyn Error>> {
        let model = RLModel1::new(state_size, action_size, hidden_sizes, &device, dt)
            .expect("Failed to create RLModel1");

        Ok(Self {
            model,
            n_episodes: 100,
            n_steps_per_episode: 50,
            epochs_per_episode: 5,
            n_timesteps: 40,
            device,
        })
    }

    fn get_action_tensor(
        &self,
        action_idx: usize,
        action_size: usize,
    ) -> Result<Tensor, Box<dyn Error>> {
        let mut data = vec![0.0f32; action_size];
        data[action_idx] = 1.0;
        Ok(Tensor::from_vec(data, (action_size, 1), &self.device)?)
    }
}

impl Algorithm for Algorithm1 {
    fn run(
        &mut self,
        env: &mut dyn Environment,
        visualize: bool,
        vis_state: Option<Arc<Mutex<VisualizationState>>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut total_iteration = 0;
        let start_time = Instant::now();
        let action_size = env.action_size();
        let state_size = env.state_size();

        for episode in 1..=self.n_episodes {
            println!("starting episode {}", episode);

            env.reset()?;
            std::thread::sleep(Duration::from_millis(500)); // wait for environment to settle

            let mut episode_data = Vec::new();

            self.model.disable_learning(); // inference mode

            let mut total_reward = 0.0;

            for _step in 0..self.n_steps_per_episode {
                let current_state = env.get_state()?;

                let state_f32: Vec<f32> = current_state.iter().map(|&x| x as f32).collect();
                let state_tensor = Tensor::from_vec(state_f32, (state_size, 1), &self.device)?;

                let mut step_actions = Vec::new();
                let mut best_activity = -1.0;
                let mut best_action = 0;

                for a in 0..action_size {
                    let action_tensor = self.get_action_tensor(a, action_size)?;
                    let activity =
                        self.model
                            .process(&state_tensor, &action_tensor, self.n_timesteps)?;
                    let reward = env.evaluate_action(&current_state, a);

                    step_actions.push((a, action_tensor, reward));

                    if activity > best_activity {
                        best_activity = activity;
                        best_action = a;
                    }
                }

                println!("best action: {}", best_action);

                // Add the true reward for the chosen action to the episode's total reward
                total_reward += env.evaluate_action(&current_state, best_action);

                env.apply_action(best_action)?;
                std::thread::sleep(Duration::from_millis(100)); // give some time for action to take effect

                // Update visualization state for environment
                if let Some(ref vis_state_arc) = vis_state {
                    if let Ok(mut state) = vis_state_arc.try_lock() {
                        let env_state = env.get_state()?;
                        state.environment_state = Some(env_state);
                    }
                }

                // Sort actions by reward (ascending)
                step_actions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

                // Assign Ytype proportional to position
                let num_choices = step_actions.len() as f32;
                for (i, (_a, action_tensor, _reward)) in step_actions.into_iter().enumerate() {
                    // Ytype between 0.0 (worst) and 1.0 (best)
                    let ytype = i as f32 / (num_choices - 1.0);
                    episode_data.push((state_tensor.clone(), action_tensor, ytype));
                }

                // Visualization check during inference
                if let Some(ref vis_state_arc) = vis_state {
                    loop {
                        let is_paused = vis_state_arc
                            .try_lock()
                            .map(|state| state.is_paused)
                            .unwrap_or(false);
                        if !is_paused {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            } // end of inference steps

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

            // Train on collected episode data
            self.model.enable_learning();

            for _epoch in 0..self.epochs_per_episode {
                for (state_t, action_t, ytype) in &episode_data {
                    total_iteration += 1;

                    for layer in self.model.layers.iter_mut() {
                        layer.set_positive_sample(*ytype);
                    }

                    // For spike plot history
                    let mut spike_history = Vec::new();
                    let record_layer = vis_state
                        .as_ref()
                        .and_then(|vs| vs.try_lock().ok().and_then(|s| s.selected_layer_id));

                    self.model.reset()?;
                    for _ in 0..self.n_timesteps {
                        self.model.step(state_t, action_t)?;
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

        Ok(())
    }
}
