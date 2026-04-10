use super::Algorithm;
use crate::environment::Environment;
use crate::models::ff_model::FFModel;
use crate::visualization::VisualizationState;
use candle_core::{Device, Tensor};
use rand::Rng;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct AlgorithmFF3 {
    pub model: FFModel,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub device: Device,
}

impl AlgorithmFF3 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        device: Device,
    ) -> Result<Self, Box<dyn Error>> {
        let input_size = state_size + action_size;
        let epochs_per_episode = 100;
        let mut dims = vec![input_size];
        dims.extend(hidden_sizes);
        let model = FFModel::new(&dims, &device, epochs_per_episode).expect("Failed to create FFModel");

        Ok(Self {
            model,
            n_episodes: 100,
            n_steps_per_episode: 50,
            epochs_per_episode,
            device,
        })
    }
}

impl Algorithm for AlgorithmFF3 {
    fn run(
        &mut self,
        env: &mut dyn Environment,
        _visualize: bool,
        vis_state: Option<Arc<Mutex<VisualizationState>>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut _total_iteration = 0;
        let mut total_inference_time = Duration::new(0, 0);
        let mut total_inference_actions: usize = 0;
        let mut total_training_time = Duration::new(0, 0);
        let mut total_epochs: usize = 0;
        let action_size = env.action_size();
        let state_size = env.state_size();
        
        let n_envs = 16;
        let mut envs: Vec<Box<dyn Environment>> = vec![env.clone_box()];
        for _ in 1..n_envs {
            envs.push(env.clone_box());
        }

        for episode in 1..=self.n_episodes {
            log::info!("starting vectorized episode {} (x{})", episode, n_envs);

            for e in envs.iter_mut() {
                e.reset()?;
            }
            std::thread::sleep(Duration::from_millis(50)); // wait for environment to settle
            
            // Format: episode_data[env_idx] = vec of (state_f32, action)
            let mut episode_data: Vec<Vec<(Vec<f32>, usize)>> = vec![Vec::new(); n_envs];
            let mut total_rewards = vec![0.0; n_envs];

            let inference_start = Instant::now();

            for _step in 0..self.n_steps_per_episode {
                let mut current_states = Vec::new();
                for e in envs.iter_mut() {
                    current_states.push(e.get_state()?);
                }

                let mut all_action_inputs = Vec::new();
                for state in current_states.iter() {
                    let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
                    for a in 0..action_size {
                        let mut input_vec = vec![0.0; action_size];
                        input_vec[a] = 1.0;
                        input_vec.extend(state_f32.iter().copied());
                        all_action_inputs.push(Tensor::from_vec(
                            input_vec,
                            (1, action_size + state_size),
                            &self.device,
                        )?);
                        total_inference_actions += 1;
                    }
                }

                let best_actions = self.model.predict(&all_action_inputs, action_size)?;

                for (env_idx, action) in best_actions.into_iter().enumerate() {
                    let e = &mut envs[env_idx];
                    let current_state = &current_states[env_idx];
                    
                    let state_f32: Vec<f32> = current_state.iter().map(|&x| x as f32).collect();
                    episode_data[env_idx].push((state_f32, action));

                    total_rewards[env_idx] += e.evaluate_action(current_state, action);
                    e.apply_action(action)?;
                }
                
                std::thread::sleep(Duration::from_millis(10));

                if let Some(ref vis_state_arc) = vis_state {
                    if let Ok(mut state) = vis_state_arc.try_lock() {
                        let env_state = envs[0].get_state()?;
                        state.environment_state = Some(env_state);
                    }
                }

                if let Some(ref vis_state_arc) = vis_state {
                    let mut should_break = false;
                    loop {
                        let (is_paused, should_close) = vis_state_arc
                            .try_lock()
                            .map(|state| (state.is_paused, state.should_close))
                            .unwrap_or((false, false));
                        if should_close {
                            return Ok(());
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

            if let Some(ref vis_state_arc) = vis_state {
                if let Ok(mut state) = vis_state_arc.try_lock() {
                    let avg_reward = total_rewards[0] as f32 / self.n_steps_per_episode as f32;
                    state.epoch_rewards.push((episode, avg_reward));
                    state.runtime_stats.epoch = episode;
                    state.total_epochs = self.n_episodes;
                }
            }

            log::info!("Training phase: Probabilistic CEM Ranking");

            let training_start = Instant::now();
            let mut pos_tensors = Vec::new();
            let mut neg_tensors = Vec::new();

            // 1. Sort environments by reward (lowest to highest)
            let mut env_ranks: Vec<(usize, f64)> = total_rewards.iter().copied().enumerate().collect();
            env_ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut rng = rand::thread_rng();
            let k = n_envs as f32;

            for (rank, &(env_idx, _r)) in env_ranks.iter().enumerate() {
                // r = 0..15 -> 0% .. 100%
                let p_pos = rank as f32 / (k - 1.0);
                
                for (state, action) in &episode_data[env_idx] {
                    let mut input_vec = vec![0.0; action_size];
                    input_vec[*action] = 1.0;
                    input_vec.extend(state.iter().copied());
                    
                    let tensor = Tensor::from_vec(input_vec, (1, action_size + state_size), &self.device)?;
                    
                    let rand_val: f32 = rng.r#gen();
                    if rand_val < p_pos {
                        pos_tensors.push(tensor);
                    } else {
                        neg_tensors.push(tensor);
                    }
                }
            }

            if !pos_tensors.is_empty() && !neg_tensors.is_empty() {
                log::info!(
                    "Rank partitioning created: {} positive / {} negative samples",
                    pos_tensors.len(), neg_tensors.len()
                );
                let pos_batch = Tensor::cat(&pos_tensors, 0)?;
                let neg_batch = Tensor::cat(&neg_tensors, 0)?;
                self.model.train(&pos_batch, &neg_batch)?;
                _total_iteration += 1;
            } else {
                log::info!("Warning: Skipping batch training (batch imbalance). Pos: {}, Neg: {}", pos_tensors.len(), neg_tensors.len());
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
            log::info!(
                "[Episode {} Tracking Env Reward: {}] Actions/sec: {:.1} | Epochs/sec: {:.2}",
                episode, total_rewards[0], inf_aps, ep_s
            );
        }

        log::info!("Training completed.");
        if let Some(ref vis_state_arc) = vis_state {
            if let Ok(state) = vis_state_arc.try_lock() {
                let checkpoints_dir = std::path::Path::new("checkpoints");
                if !checkpoints_dir.exists() {
                    let _ = std::fs::create_dir_all(checkpoints_dir);
                }
                let csv_path = checkpoints_dir.join("epoch_rewards.csv");
                let _ = state.save_graphs_to_csv(&csv_path);
            }
        }
        Ok(())
    }
}
