#![allow(clippy::needless_range_loop)]
use super::Algorithm;
use crate::environment::Environment;
use crate::models::rl_model2::RLModel2;
use crate::visualization::VisualizationState;
use candle_core::{Device, Tensor};
use rand::Rng;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Algorithm4 {
    pub model: RLModel2,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub n_timesteps: usize,
    pub device: Device,
    pub buffer: Vec<(Vec<f32>, usize, f32)>, // state, action, reward
    pub local_rewards: Vec<(usize, f32)>,
}

impl Algorithm4 {
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

        let input_bounds = state_bounds.map(|bounds| {
            let mut combined = Vec::with_capacity(state_size * 2 + 1);
            combined.extend(bounds.iter());
            combined.extend(bounds.iter());
            combined.push(action_size);
            combined
        });

        let model = RLModel2::new(
            input_size,
            context_size,
            hidden_sizes,
            &device,
            dt,
            input_bounds,
        )
        .ok_or("Failed to create RLModel2")?;

        Ok(Self {
            model,
            n_episodes: 500,
            n_steps_per_episode: 70,
            epochs_per_episode: 1,
            n_timesteps: 40,
            device,
            buffer: Vec::new(),
            local_rewards: Vec::new(),
        })
    }

    pub fn save_checkpoint(
        &self,
        dir: &std::path::Path,
        _completed_episode: usize,
        epoch_rewards: &[(usize, f32)],
    ) -> Result<(), Box<dyn Error>> {
        if !dir.exists() {
            std::fs::create_dir_all(dir)?;
        }
        let mut csv = String::from("epoch,reward\n");
        for (ep, r) in epoch_rewards {
            csv.push_str(&format!("{},{}\n", ep, r));
        }
        std::fs::write(dir.join("epoch_rewards.csv"), csv)?;
        Ok(())
    }
}

impl Algorithm for Algorithm4 {
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
        let input_dim = state_size * 2 + 1;

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

            let mut episode_data: Vec<Vec<(Vec<f32>, usize, f32)>> = vec![Vec::new(); n_envs];
            let mut raw_rewards = vec![0.0; n_envs];

            let inference_start = Instant::now();
            let mut rng = rand::thread_rng();

            let tau = 0.07;

            self.model.disable_learning();

            for _step in 0..self.n_steps_per_episode {
                let mut current_states = Vec::new();
                for e in envs.iter_mut() {
                    current_states.push(e.get_state()?);
                }

                let mut all_action_inputs = Vec::with_capacity(n_envs * action_size * input_dim);

                // Construct inputs such that the batch dimension is the SECOND dimension for SNN
                // In CSDP (Candle), tensors expect dimension (features, batch)
                // We will collect column by column
                for state in current_states.iter() {
                    let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
                    for a in 0..action_size {
                        all_action_inputs.extend(&state_f32);
                        all_action_inputs.extend(vec![0.0; state_size]); // x_{t+1} is 0 for inference
                        all_action_inputs.push(a as f32);
                        total_inference_actions += 1;
                    }
                }

                // reshape carefully: we have collected contiguous vectors of size `input_dim`.
                // meaning we have [action0_vec, action1_vec, action2_vec...]
                // To get (input_dim, batch_size) we need to pass the memory and then TRASPOSE it!
                let input_tensor_row_major = Tensor::from_vec(
                    all_action_inputs,
                    (n_envs * action_size, input_dim),
                    &self.device,
                )?;
                // now transpose to (input_dim, batch)
                let input_tensor = input_tensor_row_major.transpose(0, 1)?.contiguous()?;

                // batch size = n_envs * action_size
                let activity_tensor = self.model.process(&input_tensor, self.n_timesteps)?; // returns shape (1, batch)
                // flatten the results
                let goodness_scores = activity_tensor.flatten_all()?.to_vec1::<f32>()?;

                for (env_idx, env_state) in current_states.iter().enumerate() {
                    let e = &mut envs[env_idx];
                    let chunk_start = env_idx * action_size;
                    let chunk_scores = &goodness_scores[chunk_start..chunk_start + action_size];

                    if episode % 10 == 0 && _step == 0 && env_idx == 0 {
                        let chunk = &goodness_scores[0..action_size];
                        log::info!("Ep {} Step 0 | Goodness Scores: {:.4?}", episode, chunk);
                    }

                    let max_g = chunk_scores
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);

                    let mut exp_sum = 0.0;
                    let exps: Vec<f32> = chunk_scores
                        .iter()
                        .map(|&g| {
                            let e_val = ((g - max_g) / tau).exp();
                            exp_sum += e_val;
                            e_val
                        })
                        .collect();

                    let rand_val: f32 = rng.r#gen::<f32>() * exp_sum;
                    let mut cumulative = 0.0;
                    let mut selected_action = action_size - 1;

                    for (a, &e_val) in exps.iter().enumerate() {
                        cumulative += e_val;
                        if rand_val <= cumulative {
                            selected_action = a;
                            break;
                        }
                    }

                    let state_f32: Vec<f32> = env_state.iter().map(|&x| x as f32).collect();
                    let step_reward = e.evaluate_action(env_state, selected_action);

                    episode_data[env_idx].push((state_f32, selected_action, step_reward as f32));
                    raw_rewards[env_idx] += step_reward;

                    e.apply_action(selected_action)?;
                }

                if let Some(ref vis_state_arc) = vis_state
                    && let Ok(mut state) = vis_state_arc.try_lock() {
                        let env_state = envs[0].get_state()?;
                        if state.runtime_stats.epoch != episode {
                            state.render_trail.clear();
                        }
                        if env_state.len() == 4 {
                            state
                                .render_trail
                                .push((env_state[0] + env_state[2], env_state[1] + env_state[3]));
                        }
                        state.environment_state = Some(env_state);
                    }

                if let Some(ref vis_state_arc) = vis_state {
                    let should_break = false;
                    loop {
                        let (is_paused, should_close, delay) = vis_state_arc
                            .try_lock()
                            .map(|state| (state.is_paused, state.should_close, state.delay_ms))
                            .unwrap_or((false, false, 0));
                        if should_close {
                            return Ok(());
                        }
                        if !is_paused {
                            if delay > 0 {
                                std::thread::sleep(std::time::Duration::from_millis(delay));
                            }
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    if should_break {
                        break;
                    }
                }
            }

            let inference_elapsed = inference_start.elapsed();
            total_inference_time += inference_elapsed;

            if let Some(ref vis_state_arc) = vis_state
                && let Ok(mut state) = vis_state_arc.try_lock() {
                    let avg_reward =
                        raw_rewards.iter().sum::<f64>() as f32 / self.n_steps_per_episode as f32;
                    state.epoch_rewards.push((episode, avg_reward));
                    state.runtime_stats.epoch = episode;
                    state.total_epochs = self.n_episodes;
                }

            log::info!("Training phase: SNN Temporal Contrastive RL");

            let gamma = 0.99f32;

            for env_idx in 0..n_envs {
                let seq = &mut episode_data[env_idx];
                let mut current_r = 0.0;
                for t in (0..seq.len()).rev() {
                    current_r = seq[t].2 + gamma * current_r;
                    seq[t] = (seq[t].0.clone(), seq[t].1, current_r);
                }
                for item in seq.iter() {
                    self.buffer.push(item.clone());
                }
            }

            if self.buffer.len() > 200000 {
                log::info!("Draining replay buffer...");
                let drain_count = self.buffer.len() - 200000;
                self.buffer.drain(0..drain_count);
            }

            let training_start = Instant::now();

            self.model.enable_learning();

            if self.buffer.len() >= 1000 {
                let mut sorted_indices: Vec<usize> = (0..self.buffer.len()).collect();
                sorted_indices.sort_unstable_by(|&a, &b| {
                    self.buffer[a].2.partial_cmp(&self.buffer[b].2).unwrap()
                });

                let batch_size = std::cmp::min(256, self.buffer.len() / 4);
                let mut pos_tensors_flat = Vec::with_capacity(batch_size * input_dim);
                let mut neg_tensors_flat = Vec::with_capacity(batch_size * input_dim);
                let mut pos_rewards = Vec::with_capacity(batch_size);

                for i in 0..batch_size {
                    let best = &self.buffer[sorted_indices[sorted_indices.len() - 1 - i]];

                    // Positive Sample
                    pos_tensors_flat.extend(best.0.iter().copied());
                    pos_tensors_flat.extend(vec![0.0; state_size]); // placeholder for x_{t+m} or ignore
                    pos_tensors_flat.push(best.1 as f32);

                    // Sigmoid reward
                    let sig_r = 1.0 / (1.0 + (-best.2).exp());
                    pos_rewards.push(sig_r);

                    // Negative Sample
                    let mut a_random = rng.r#gen_range(0..action_size);
                    if a_random == best.1 {
                        a_random = (a_random + 1) % action_size;
                    }
                    neg_tensors_flat.extend(best.0.iter().copied());
                    neg_tensors_flat.extend(vec![0.0; state_size]);
                    neg_tensors_flat.push(a_random as f32);
                }

                let pos_batch =
                    Tensor::from_vec(pos_tensors_flat, (batch_size, input_dim), &self.device)?
                        .transpose(0, 1)?
                        .contiguous()?;
                let neg_batch =
                    Tensor::from_vec(neg_tensors_flat, (batch_size, input_dim), &self.device)?
                        .transpose(0, 1)?
                        .contiguous()?;

                let pos_rewards_tensor =
                    Tensor::from_vec(pos_rewards, (1, batch_size), &self.device)?;
                let neg_rewards_tensor =
                    Tensor::ones((1, batch_size), candle_core::DType::F32, &self.device)?;

                let pos_label =
                    Tensor::ones((1, batch_size), candle_core::DType::F32, &self.device)?;
                let neg_label =
                    Tensor::zeros((1, batch_size), candle_core::DType::F32, &self.device)?;

                // Train Positive
                for layer in self.model.layers.iter_mut() {
                    layer.set_positive_sample(&pos_label);
                }
                self.model.set_reward(&pos_rewards_tensor);
                self.model.reset(batch_size)?;
                for _ in 0..self.n_timesteps {
                    self.model.step(&pos_batch, Some(&pos_label))?;
                }

                // Train Negative
                for layer in self.model.layers.iter_mut() {
                    layer.set_positive_sample(&neg_label);
                }
                self.model.set_reward(&neg_rewards_tensor);
                self.model.reset(batch_size)?;
                for _ in 0..self.n_timesteps {
                    self.model.step(&neg_batch, Some(&neg_label))?;
                }

                _total_iteration += 1;
                total_epochs += 1;
            }

            let training_elapsed = training_start.elapsed();
            total_training_time += training_elapsed;

            let avg_reward = raw_rewards.iter().sum::<f64>() as f32 / n_envs as f32;
            self.local_rewards.push((episode, avg_reward));

            if let Some(ref vs_arc) = vis_state
                && let Ok(mut state) = vs_arc.try_lock() {
                    state.epoch_rewards.push((episode, avg_reward));
                    state.runtime_stats.epoch = episode;
                    state.total_epochs = self.n_episodes;
                }

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
                "[Episode {} Reward: {:.2}] Actions/sec: {:.1} | Epochs/sec: {:.2} | Buffer: {}",
                episode,
                raw_rewards.iter().sum::<f64>(),
                inf_aps,
                ep_s,
                self.buffer.len()
            );
        }

        log::info!("Training completed. Saving final checkpoint...");
        let checkpoint_dir = std::path::Path::new("checkpoints/csdp4");
        self.save_checkpoint(checkpoint_dir, self.n_episodes, &self.local_rewards)?;

        Ok(())
    }
}
