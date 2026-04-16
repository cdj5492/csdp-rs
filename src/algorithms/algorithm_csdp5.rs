use super::Algorithm;
use crate::environment::Environment;
use crate::models::csdp_multi_model::CSDPMultiModel;
use crate::visualization::VisualizationState;
use candle_core::{DType, Device, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const NUM_RETURN_CLASSES: usize = 50;
const ACTION_SCALE: f32 = 3.0; // Same action amplification map as FFMulti2
const AUTO_SAVE_INTERVAL: usize = 50;
const CHECKPOINT_DIR: &str = "checkpoints/csdp5";

#[derive(Serialize, Deserialize)]
struct TrainingState {
    buffer: Vec<(Vec<f32>, usize, f32)>, // state, action, MC return
    min_return: f32,
    max_return: f32,
    bounds_initialized: bool,
    completed_episode: usize,
    epoch_rewards: Vec<(usize, f32)>,
}

pub struct AlgorithmCSDP5 {
    pub model: CSDPMultiModel,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub device: Device,
    pub buffer: Vec<(Vec<f32>, usize, f32)>,
    pub num_classes: usize,
    pub min_return: f32,
    pub max_return: f32,
    pub bounds_initialized: bool,
    pub start_episode: usize,
    pub timesteps: usize, // SNN evaluation timesteps
}

impl AlgorithmCSDP5 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        dt: f32,
        device: Device,
        state_bounds: Option<Vec<usize>>,
    ) -> Result<Self, Box<dyn Error>> {
        let _ = state_bounds; // Unused 
        let input_size = state_size * 2 + action_size; // concatenated previous state, diff, one-hot action
        let num_classes = NUM_RETURN_CLASSES;
        let timesteps = 40; // Default CSDP simulation timesteps

        let mut dims = vec![];
        for &h in &hidden_sizes {
            let rounded = ((h + num_classes - 1) / num_classes) * num_classes;
            dims.push(rounded);
        }

        let model = CSDPMultiModel::new(input_size, &dims, num_classes, &device, dt, timesteps)
            .expect("Failed to create CSDPMultiModel");

        Ok(Self {
            model,
            n_episodes: 500,
            n_steps_per_episode: 70,
            device,
            buffer: Vec::new(),
            num_classes,
            min_return: 0.0,
            max_return: 0.0,
            bounds_initialized: false,
            start_episode: 0,
            timesteps,
        })
    }

    pub fn save_checkpoint(
        &self,
        dir: &std::path::Path,
        completed_episode: usize,
        epoch_rewards: &[(usize, f32)],
    ) -> Result<(), Box<dyn Error>> {
        if !dir.exists() {
            std::fs::create_dir_all(dir)?;
        }
        let model_dir = dir.join("model");
        std::fs::create_dir_all(&model_dir)?;
        self.model.save(&model_dir)?;

        let state = TrainingState {
            buffer: self.buffer.clone(),
            min_return: self.min_return,
            max_return: self.max_return,
            bounds_initialized: self.bounds_initialized,
            completed_episode,
            epoch_rewards: epoch_rewards.to_vec(),
        };
        let json = serde_json::to_string(&state)?;
        std::fs::write(dir.join("training_state.json"), json)?;
        Ok(())
    }

    pub fn load_checkpoint(
        &mut self,
        dir: &std::path::Path,
    ) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
        let model_dir = dir.join("model");
        self.model.load(&model_dir)?;

        let json = std::fs::read_to_string(dir.join("training_state.json"))?;
        let state: TrainingState = serde_json::from_str(&json)?;

        self.buffer = state.buffer;
        self.min_return = state.min_return;
        self.max_return = state.max_return;
        self.bounds_initialized = state.bounds_initialized;
        self.start_episode = state.completed_episode;

        Ok(state.epoch_rewards)
    }

    fn value_to_class(&self, value: f32) -> usize {
        if !self.bounds_initialized || self.min_return == self.max_return {
            return self.num_classes / 2;
        }
        let norm = (value - self.min_return) / (self.max_return - self.min_return);
        let c = (norm * (self.num_classes as f32)).floor() as usize;
        c.clamp(0, self.num_classes - 1)
    }

    fn class_to_value(&self, class_idx: usize) -> f32 {
        if !self.bounds_initialized {
            return 0.0;
        }
        let norm = (class_idx as f32 + 0.5) / (self.num_classes as f32);
        self.min_return + norm * (self.max_return - self.min_return)
    }
}

impl Algorithm for AlgorithmCSDP5 {
    fn run(
        &mut self,
        env: &mut dyn Environment,
        _visualize: bool,
        vis_state: Option<Arc<Mutex<VisualizationState>>>,
    ) -> Result<(), Box<dyn Error>> {
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

        let gamma = 0.9f32;
        let tau = 0.5f32; 

        let checkpoint_dir = std::path::Path::new(CHECKPOINT_DIR);

        // Episode range
        let mut episode = self.start_episode + 1;
        let mut episode_end = self.start_episode + self.n_episodes;

        while episode <= episode_end {
            if let Some(ref vs) = vis_state {
                if vs.try_lock().map(|s| s.should_close).unwrap_or(false) {
                    return Ok(());
                }
            }

            for e in envs.iter_mut() {
                e.reset()?;
            }

            let mut episode_data: Vec<Vec<(Vec<f32>, usize, f32)>> = vec![Vec::new(); n_envs];
            let mut raw_rewards = vec![0.0; n_envs];

            let mut state_history = vec![Vec::new(); n_envs];
            for (i, e) in envs.iter_mut().enumerate() {
                let s = e.get_state()?;
                state_history[i].push(s);
            }

            let inference_start = Instant::now();
            let mut rng = rand::thread_rng();
            let epsilon = (0.2f32 * (-(episode as f32) / 50.0).exp()).max(0.01);

            self.model.disable_learning();

            for step in 0..self.n_steps_per_episode {
                let mut current_states = Vec::new();
                for e in envs.iter_mut() {
                    current_states.push(e.get_state()?);
                }

                let mut best_actions = vec![0; n_envs];

                if self.buffer.is_empty() || rng.r#gen::<f32>() < epsilon {
                    for i in 0..n_envs {
                        best_actions[i] = rng.r#gen_range(0..action_size);
                    }
                } else {
                    let mut batch_inputs = Vec::with_capacity(n_envs * action_size);

                    for env_idx in 0..n_envs {
                        let prev_s = if step > 0 {
                            &state_history[env_idx][step - 1]
                        } else {
                            &current_states[env_idx]
                        };
                        let curr_s = &current_states[env_idx];

                        let mut state_f32 = Vec::with_capacity(state_size * 2);
                        for x in prev_s {
                            state_f32.push(*x as f32);
                        }
                        for (i, x) in curr_s.iter().enumerate() {
                            state_f32.push(*x as f32 - prev_s[i] as f32);
                        }

                        for a in 0..action_size {
                            let mut input_vec = state_f32.clone();
                            let mut action_one_hot = vec![0.0f32; action_size];
                            action_one_hot[a] = ACTION_SCALE; // amplify action parameter
                            input_vec.extend(action_one_hot);

                            let t = Tensor::from_vec(input_vec, (1, state_size * 2 + action_size), &self.device)?;
                            batch_inputs.push(t);
                            total_inference_actions += 1;
                        }
                    }

                    let scores_tensor = self.model.predict_scores(&batch_inputs)?;
                    let scores = scores_tensor.flatten_all()?.to_vec1::<f32>()?;

                    for env_idx in 0..n_envs {
                        let mut exp_vals = Vec::with_capacity(action_size);

                        for a in 0..action_size {
                            let start_idx = (env_idx * action_size + a) * self.num_classes;
                            let end_idx = start_idx + self.num_classes;
                            let class_scores = &scores[start_idx..end_idx];

                            let max_g = class_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mut exp_sum = 0.0;
                            let mut exps = Vec::with_capacity(self.num_classes);
                            for &g in class_scores {
                                let e_val = ((g - max_g) / tau).exp();
                                exp_sum += e_val;
                                exps.push(e_val);
                            }

                            let mut expected_value = 0.0;
                            for (c, &e_val) in exps.iter().enumerate() {
                                let prob = e_val / exp_sum;
                                let value = self.class_to_value(c);
                                expected_value += prob * value;
                            }
                            exp_vals.push(expected_value);
                        }

                        let mut best_a = 0;
                        let mut best_exp = f32::NEG_INFINITY;
                        for (a, &exp_v) in exp_vals.iter().enumerate() {
                            if exp_v > best_exp {
                                best_exp = exp_v;
                                best_a = a;
                            }
                        }
                        best_actions[env_idx] = best_a;
                    }
                }

                for env_idx in 0..n_envs {
                    let a = best_actions[env_idx];
                    let r = envs[env_idx].evaluate_action(&current_states[env_idx], a);
                    let state_f32: Vec<f32> = current_states[env_idx].iter().map(|&x| x as f32).collect();
                    episode_data[env_idx].push((state_f32, a, r as f32));
                    raw_rewards[env_idx] += r;

                    envs[env_idx].apply_action(a)?;
                    state_history[env_idx].push(envs[env_idx].get_state()?);
                }

                // Vis Hook
                if let Some(ref vs_arc) = vis_state {
                    if let Ok(mut state) = vs_arc.try_lock() {
                        let env_state = envs[0].get_state()?;
                        if state.runtime_stats.epoch != episode {
                            state.render_trail.clear();
                        }
                        if env_state.len() == 4 {
                            state.render_trail.push((env_state[0] + env_state[2], env_state[1] + env_state[3]));
                        }
                        state.environment_state = Some(env_state);
                    }
                    
                    let mut should_break = false;
                    loop {
                        let (is_paused, should_close, delay) = vs_arc
                            .try_lock()
                            .map(|state| (state.is_paused, state.should_close, state.delay_ms))
                            .unwrap_or((false, false, 0));
                        if should_close { return Ok(()); }
                        if !is_paused {
                            if delay > 0 { std::thread::sleep(std::time::Duration::from_millis(delay)); }
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    if should_break { break; }
                }
            }
            total_inference_time += inference_start.elapsed();

            // Store rewards and calculate MC Returns
            for env_idx in 0..n_envs {
                let seq = &mut episode_data[env_idx];
                let mut current_r = 0.0;
                for t in (0..seq.len()).rev() {
                    current_r = seq[t].2 + gamma * current_r;
                    seq[t].2 = current_r;
                }
                for item in seq.iter() {
                    self.buffer.push(item.clone());
                }
            }

            if self.buffer.len() > 100000 {
                let drain_count = self.buffer.len() - 100000;
                self.buffer.drain(0..drain_count);
            }

            // Update bounds based on recent returns in buffer
            if self.buffer.len() >= 1000 {
                let mut returns: Vec<f32> = self.buffer.iter().map(|x| x.2).collect();
                returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let min_idx = (returns.len() as f32 * 0.05) as usize;
                let max_idx = (returns.len() as f32 * 0.95) as usize;
                
                self.min_return = returns[min_idx];
                self.max_return = returns[max_idx];
                self.bounds_initialized = true;
            }

            // SNN Multi-Class Training
            let training_start = Instant::now();
            self.model.enable_learning();

            if self.buffer.len() >= 1000 && self.bounds_initialized {
                // Train batch
                let batch_size = 256.min(self.buffer.len() / 4);
                let mut batch_inputs = Vec::with_capacity(batch_size * (state_size * 2 + action_size));
                let mut batch_classes = Vec::with_capacity(batch_size);

                for _ in 0..batch_size {
                    let idx = rng.r#gen_range(0..self.buffer.len());
                    let (s, a, mc_ret) = &self.buffer[idx];
                    
                    let mut input_vec = Vec::with_capacity(state_size * 2 + action_size);
                    input_vec.extend(s.iter().copied());
                    input_vec.extend(vec![0.0; state_size]); // Zero diff for basic random sample? 
                                                             // FFMulti2 just pushed current block diff=0 or identical
                    
                    let mut action_one_hot = vec![0.0f32; action_size];
                    action_one_hot[*a] = ACTION_SCALE;
                    input_vec.extend(action_one_hot);
                    
                    batch_inputs.extend(input_vec);
                    
                    let target_class = self.value_to_class(*mc_ret);
                    batch_classes.push(target_class as f32);
                }

                let x = Tensor::from_vec(batch_inputs, (1, batch_size * (state_size * 2 + action_size)), &self.device)?
                    .reshape((batch_size, state_size * 2 + action_size))?;
                
                let y = Tensor::from_vec(batch_classes, (1, batch_size), &self.device)?;

                // Train the entire SNN for exactly batch_size using 40 timesteps!
                self.model.train(&x, &y)?;

                total_epochs += 1;
            }

            total_training_time += training_start.elapsed();

            // Update Visualization
            if let Some(ref vs_arc) = vis_state {
                if let Ok(mut state) = vs_arc.try_lock() {
                    let avg_reward = raw_rewards.iter().sum::<f64>() as f32 / n_envs as f32;
                    state.epoch_rewards.push((episode, avg_reward));
                    state.runtime_stats.epoch = episode;
                    state.total_epochs = self.n_episodes;
                }
            }

            // Checkpointing
            if let Some(ref vs) = vis_state {
                if let Ok(mut state) = vs.try_lock() {
                    if state.save_requested {
                        log::info!("Manual save requested...");
                        let epoch_rewards = state.epoch_rewards.clone();
                        state.save_requested = false;
                        drop(state);
                        if let Err(e) = self.save_checkpoint(checkpoint_dir, episode, &epoch_rewards) {
                            log::error!("Manual save failed: {}", e);
                        }
                    } else if state.load_requested {
                        log::info!("Manual load requested...");
                        state.load_requested = false;
                        drop(state);
                        match self.load_checkpoint(checkpoint_dir) {
                            Ok(epoch_rewards) => {
                                if let Some(ref vs2) = vis_state {
                                    if let Ok(mut s) = vs2.try_lock() {
                                        s.epoch_rewards = epoch_rewards;
                                    }
                                }
                                episode = self.start_episode;
                                episode_end = self.start_episode + self.n_episodes;
                                log::info!("Manual load succeeded. Continuing training.");
                            }
                            Err(e) => {
                                log::error!("Manual load failed: {}", e);
                            }
                        }
                    }
                }
            }

            if episode % AUTO_SAVE_INTERVAL == 0 {
                let epoch_rewards = vis_state
                    .as_ref()
                    .and_then(|vs| vs.try_lock().ok().map(|s| s.epoch_rewards.clone()))
                    .unwrap_or_default();
                if let Err(e) = self.save_checkpoint(checkpoint_dir, episode, &epoch_rewards) {
                    log::error!("Auto-save failed: {}", e);
                }
            }

            let inf_aps = if total_inference_time.as_secs_f32() > 0.0 {
                total_inference_actions as f32 / total_inference_time.as_secs_f32()
            } else { 0.0 };
            
            let ep_s = if total_training_time.as_secs_f32() > 0.0 {
                total_epochs as f32 / total_training_time.as_secs_f32()
            } else { 0.0 };

            log::info!(
                "[Episode {} Reward: {:.2}] Actions/sec: {:.1} | Epochs/sec: {:.2} | Buffer: {} | Bounds [{:.2}, {:.2}]",
                episode,
                // `raw_rewards.iter().sum::<f64>()` computes the sum over all ENVS.
                raw_rewards.iter().sum::<f64>(),
                inf_aps,
                ep_s,
                self.buffer.len(),
                self.min_return,
                self.max_return
            );

            episode += 1;
        }

        Ok(())
    }
}
