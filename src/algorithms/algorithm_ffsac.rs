use super::Algorithm;
use crate::environment::Environment;
use crate::models::ff_model::FFModel;
use crate::visualization::VisualizationState;
use candle_core::{Device, Tensor};
use rand::Rng;
use rand::seq::SliceRandom;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct AlgorithmFFSAC {
    pub model: FFModel,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub device: Device,
    pub buffer: Vec<(Vec<f32>, usize, f32, Vec<f32>)>,
}

impl AlgorithmFFSAC {
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

impl Algorithm for AlgorithmFFSAC {
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

        let gamma = 0.99f32;
        let tau = 0.1f32;

        for episode in 1..=self.n_episodes {
            if let Some(ref vis_state_arc) = vis_state
                && vis_state_arc
                    .try_lock()
                    .map(|s| s.should_close)
                    .unwrap_or(false)
                {
                    return Ok(());
                }
            log::info!("starting vectorized episode {} (x{})", episode, n_envs);

            for e in envs.iter_mut() {
                e.reset()?;
            }
            std::thread::sleep(Duration::from_millis(50));

            let mut raw_rewards = vec![0.0; n_envs];
            let inference_start = Instant::now();
            let mut rng = rand::thread_rng();

            for _step in 0..self.n_steps_per_episode {
                if let Some(ref vis_state_arc) = vis_state {
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
                }

                let mut current_states = Vec::new();
                for e in envs.iter_mut() {
                    current_states.push(e.get_state()?);
                }

                let mut all_action_inputs = Vec::new();
                for state in current_states.iter() {
                    let state_f32: Vec<f32> = state.iter().map(|&x| (x as f32) / 50.0).collect();
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

                    let state_f32: Vec<f32> =
                        env_state.iter().map(|&x| (x as f32) / 50.0).collect();
                    let step_reward = e.evaluate_action(env_state, selected_action);

                    e.apply_action(selected_action)?;
                    let next_state = e.get_state()?;
                    let next_state_f32: Vec<f32> =
                        next_state.iter().map(|&x| (x as f32) / 50.0).collect();

                    self.buffer.push((
                        state_f32,
                        selected_action,
                        step_reward as f32,
                        next_state_f32,
                    ));
                    raw_rewards[env_idx] += step_reward;
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
            }

            let inference_elapsed = inference_start.elapsed();
            total_inference_time += inference_elapsed;

            if let Some(ref vis_state_arc) = vis_state
                && let Ok(mut state) = vis_state_arc.try_lock() {
                    let avg_reward = raw_rewards.iter().sum::<f64>() as f32
                        / (n_envs as f32 * self.n_steps_per_episode as f32);
                    state.epoch_rewards.push((episode, avg_reward));
                    state.runtime_stats.epoch = episode;
                    state.total_epochs = self.n_episodes;
                }

            if self.buffer.len() > 50000 {
                let drain_count = self.buffer.len() - 50000;
                self.buffer.drain(0..drain_count);
            }

            let training_start = Instant::now();
            let batch_size = 256;

            if self.buffer.len() >= batch_size {
                let mut pos_tensors = Vec::new();
                let mut neg_tensors = Vec::new();

                let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
                indices.shuffle(&mut rng);
                let batch_indices = &indices[0..batch_size];

                let mut current_eval_inputs = Vec::new();
                let mut next_eval_inputs = Vec::new();

                for &idx in batch_indices {
                    let (s, a, _r, s_next) = &self.buffer[idx];

                    let mut c_vec = vec![0.0; action_size];
                    c_vec[*a] = 1.0;
                    c_vec.extend(s.iter().copied());
                    current_eval_inputs.push(Tensor::from_vec(
                        c_vec,
                        (1, action_size + state_size),
                        &self.device,
                    )?);

                    for next_a in 0..action_size {
                        let mut n_vec = vec![0.0; action_size];
                        n_vec[next_a] = 1.0;
                        n_vec.extend(s_next.iter().copied());
                        next_eval_inputs.push(Tensor::from_vec(
                            n_vec,
                            (1, action_size + state_size),
                            &self.device,
                        )?);
                    }
                }

                let current_scores = self.model.predict_scores(&current_eval_inputs)?;
                let next_scores_flat = self.model.predict_scores(&next_eval_inputs)?;

                for (i, &idx) in batch_indices.iter().enumerate() {
                    let (s, a, r, _s_next) = &self.buffer[idx];
                    let current_g = current_scores[i];

                    let n_start = i * action_size;
                    let next_g_chunk = &next_scores_flat[n_start..n_start + action_size];

                    let max_next_g = next_g_chunk
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0;
                    for &g in next_g_chunk {
                        sum_exp += ((g - max_next_g) / tau).exp();
                    }
                    let v_next = tau * sum_exp.ln() + max_next_g;

                    let target_g = r + gamma * v_next;
                    let delta = target_g - current_g;

                    let mut a_random = rng.r#gen_range(0..action_size);
                    if a_random == *a {
                        a_random = (a_random + 1) % action_size;
                    }

                    let mut p_vec = vec![0.0; action_size];
                    p_vec[*a] = 1.0;
                    p_vec.extend(s.iter().copied());
                    let t_actual =
                        Tensor::from_vec(p_vec, (1, action_size + state_size), &self.device)?;

                    let mut n_vec = vec![0.0; action_size];
                    n_vec[a_random] = 1.0;
                    n_vec.extend(s.iter().copied());
                    let t_random =
                        Tensor::from_vec(n_vec, (1, action_size + state_size), &self.device)?;

                    if delta > 0.1 {
                        pos_tensors.push(t_actual);
                        neg_tensors.push(t_random);
                    } else if delta < -0.1 {
                        neg_tensors.push(t_actual);
                        pos_tensors.push(t_random);
                    }
                }

                if !pos_tensors.is_empty() {
                    let pos_batch = Tensor::cat(&pos_tensors, 0)?;
                    let neg_batch = Tensor::cat(&neg_tensors, 0)?;
                    self.model.train(&pos_batch, &neg_batch)?;
                    _total_iteration += 1;
                    total_epochs += self.epochs_per_episode;
                }
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
            log::info!(
                "[Episode {} Tracking Env Reward: {:.2}] Actions/sec: {:.1} | Epochs/sec: {:.2} | Buffer: {}",
                episode,
                raw_rewards.iter().sum::<f64>(),
                inf_aps,
                ep_s,
                self.buffer.len()
            );
        }

        log::info!("Training completed.");
        if let Some(ref vis_state_arc) = vis_state
            && let Ok(state) = vis_state_arc.try_lock() {
                let checkpoints_dir = std::path::Path::new("checkpoints");
                if !checkpoints_dir.exists() {
                    let _ = std::fs::create_dir_all(checkpoints_dir);
                }
                let csv_path = checkpoints_dir.join("epoch_rewards.csv");
                let _ = state.save_graphs_to_csv(&csv_path);
            }
        Ok(())
    }
}
