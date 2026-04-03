use super::Algorithm;
use crate::environment::Environment;
use crate::models::ff_model::FFModel;
use crate::visualization::VisualizationState;
use candle_core::{Device, Tensor};
use rand::Rng;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct AlgorithmFF4 {
    pub model: FFModel,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub device: Device,
    pub buffer: Vec<(Vec<f32>, usize, f32)>, // state, action, reward
}

impl AlgorithmFF4 {
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
        let model =
            FFModel::new(&dims, &device, epochs_per_episode).expect("Failed to create FFModel");

        Ok(Self {
            model,
            n_episodes: 500,
            n_steps_per_episode: 70,
            epochs_per_episode,
            device,
            buffer: Vec::new(),
        })
    }
}

impl Algorithm for AlgorithmFF4 {
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

        let n_envs = 50;
        let mut envs: Vec<Box<dyn Environment>> = vec![env.clone_box()];
        for _ in 1..n_envs {
            envs.push(env.clone_box());
        }

        for episode in 1..=self.n_episodes {
            println!("starting vectorized episode {} (x{})", episode, n_envs);

            for e in envs.iter_mut() {
                e.reset()?;
            }

            let mut episode_data: Vec<Vec<(Vec<f32>, usize, f32)>> = vec![Vec::new(); n_envs];
            let mut raw_rewards = vec![0.0; n_envs];

            let inference_start = Instant::now();
            let mut rng = rand::thread_rng();

            let tau = f32::max(0.05, 1.0 - (episode as f32 / self.n_episodes as f32) * 0.95);
            // let tau = 0.1;

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

                let goodness_scores = self.model.predict_scores(&all_action_inputs)?;

                for (env_idx, env_state) in current_states.iter().enumerate() {
                    let e = &mut envs[env_idx];
                    let chunk_start = env_idx * action_size;
                    let chunk_scores = &goodness_scores[chunk_start..chunk_start + action_size];

                    if episode % 10 == 0 && _step == 0 && env_idx == 0 {
                        let chunk = &goodness_scores[0..action_size];
                        println!("Ep {} Step 0 | Goodness Scores: {:.4?}", episode, chunk);
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

                    let state_f32: Vec<f32> = env_state.iter().map(|&x| x as f32 / 50.0).collect();
                    let step_reward = e.evaluate_action(env_state, selected_action);

                    episode_data[env_idx].push((state_f32, selected_action, step_reward as f32));
                    raw_rewards[env_idx] += step_reward;

                    e.apply_action(selected_action)?;
                }

                if let Some(ref vis_state_arc) = vis_state {
                    if let Ok(mut state) = vis_state_arc.try_lock() {
                        let env_state = envs[0].get_state()?;
                        state.environment_state = Some(env_state);
                    }
                }

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
            }

            let inference_elapsed = inference_start.elapsed();
            total_inference_time += inference_elapsed;

            if let Some(ref vis_state_arc) = vis_state {
                if let Ok(mut state) = vis_state_arc.try_lock() {
                    let avg_reward =
                        raw_rewards.iter().sum::<f64>() as f32 / self.n_steps_per_episode as f32;
                    state.epoch_rewards.push((episode, avg_reward));
                    state.runtime_stats.epoch = episode;
                    state.total_epochs = self.n_episodes;
                }
            }

            println!("Training phase: Temporal Contrastive RL");

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
                println!("Draining replay buffer...");
                let drain_count = self.buffer.len() - 200000;
                self.buffer.drain(0..drain_count);
            }

            let training_start = Instant::now();

            if self.buffer.len() >= 1000 {
                let mut sorted_indices: Vec<usize> = (0..self.buffer.len()).collect();
                // Sort ascending: index 0 is worst, index len-1 is best
                sorted_indices.sort_unstable_by(|&a, &b| self.buffer[a].2.partial_cmp(&self.buffer[b].2).unwrap());

                let batch_size = std::cmp::min(256, self.buffer.len() / 4);
                let mut pos_tensors = Vec::new();
                let mut neg_tensors = Vec::new();

                for i in 0..batch_size {
                    // Grab the best successful transitions
                    let best = &self.buffer[sorted_indices[sorted_indices.len() - 1 - i]];

                    // Positive Sample: The state + the action actually taken
                    let mut p_vec = vec![0.0; action_size];
                    p_vec[best.1] = 1.0;
                    p_vec.extend(best.0.iter().copied());
                    pos_tensors.push(Tensor::from_vec(p_vec, (1, action_size + state_size), &self.device)?);

                    // Negative Sample: The EXACT SAME state + a random alternative action
                    let mut a_random = rng.r#gen_range(0..action_size);
                    if a_random == best.1 {
                        a_random = (a_random + 1) % action_size;
                    }
                    let mut n_vec = vec![0.0; action_size];
                    n_vec[a_random] = 1.0;
                    n_vec.extend(best.0.iter().copied());
                    neg_tensors.push(Tensor::from_vec(n_vec, (1, action_size + state_size), &self.device)?);
                }

                let pos_batch = Tensor::cat(&pos_tensors, 0)?;
                let neg_batch = Tensor::cat(&neg_tensors, 0)?;
                
                self.model.train(&pos_batch, &neg_batch)?;
                _total_iteration += 1;
                total_epochs += self.epochs_per_episode;
            }

            let training_elapsed = training_start.elapsed();
            total_training_time += training_elapsed;

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
                "[Episode {} Tracking Env Reward: {:.2}] Actions/sec: {:.1} | Epochs/sec: {:.2} | Buffer: {}",
                episode,
                raw_rewards.iter().sum::<f64>(),
                inf_aps,
                ep_s,
                self.buffer.len()
            );
        }

        println!("Training completed.");
        Ok(())
    }
}
