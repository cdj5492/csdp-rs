use super::Algorithm;
use crate::environment::Environment;
use crate::models::ff_multi_model::FFMultiModel;
use crate::visualization::VisualizationState;
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::Rng;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Number of discrete return classes the model predicts.
const NUM_RETURN_CLASSES: usize = 50;

// ─────────────────────────────────────────────────────────────
// Dynamic Class Mapper
// ─────────────────────────────────────────────────────────────

/// Maps a discrete class ID (0..N-1) to a continuous expected-return value
/// linearly interpolated within [min_ret, max_ret].
fn class_to_value(class_id: usize, min_ret: f32, max_ret: f32, n_classes: usize) -> f32 {
    if n_classes <= 1 || (max_ret - min_ret).abs() < 1e-8 {
        return (min_ret + max_ret) / 2.0;
    }
    min_ret + (class_id as f32 / (n_classes - 1) as f32) * (max_ret - min_ret)
}

/// Maps a continuous expected-return value to the nearest discrete class ID
/// in [0, N-1], clamped to the current [min_ret, max_ret] bounds.
fn value_to_class(value: f32, min_ret: f32, max_ret: f32, n_classes: usize) -> usize {
    if n_classes <= 1 {
        return 0;
    }
    let range = max_ret - min_ret;
    if range.abs() < 1e-8 {
        return n_classes / 2; // middle bucket when range is degenerate
    }
    let t = ((value - min_ret) / range).clamp(0.0, 1.0);
    (t * (n_classes - 1) as f32).round() as usize
}

// ─────────────────────────────────────────────────────────────
// Target Model Synchronization
// ─────────────────────────────────────────────────────────────

/// Hard-copies every tracked variable from `main` into `target`.
/// Uses `VarMap::set_one` for safe per-key overwrites.
fn sync_target_from_main(
    main: &FFMultiModel,
    target: &FFMultiModel,
) -> Result<(), Box<dyn Error>> {
    for (main_vm, target_vm) in main.varmaps.iter().zip(target.varmaps.iter()) {
        let main_data = main_vm.data().lock().unwrap();
        let target_data = target_vm.data().lock().unwrap();
        for (key, main_var) in main_data.iter() {
            if let Some(target_var) = target_data.get(key) {
                let t: &Tensor = main_var; // Deref<Target = Tensor>
                target_var.set(&t.detach())?;
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────
// Algorithm Struct
// ─────────────────────────────────────────────────────────────

pub struct AlgorithmFFMulti2 {
    pub main_model: FFMultiModel,
    pub target_model: FFMultiModel,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub device: Device,
    /// Replay buffer: (normalised_state, action_id, reward, normalised_next_state)
    pub buffer: Vec<(Vec<f32>, usize, f32, Vec<f32>)>,
    pub num_classes: usize,
    pub min_return: f32,
    pub max_return: f32,
    pub target_sync_interval: usize,
}

impl AlgorithmFFMulti2 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        device: Device,
    ) -> Result<Self, Box<dyn Error>> {
        let input_size = state_size + action_size; // [one-hot action ++ state]
        let epochs_per_episode = 20;
        let num_classes = NUM_RETURN_CLASSES;

        // Round each hidden size up to the nearest multiple of num_classes
        // so that FFMultiLayer's out_features % num_classes == 0 invariant holds.
        let mut dims = vec![input_size];
        for &h in &hidden_sizes {
            let rounded = ((h + num_classes - 1) / num_classes) * num_classes;
            dims.push(rounded);
        }

        let main_model =
            FFMultiModel::new(&dims, num_classes, &device, epochs_per_episode)
                .expect("Failed to create main FFMultiModel");
        // Target never trains – epoch count is irrelevant.
        let target_model =
            FFMultiModel::new(&dims, num_classes, &device, 1)
                .expect("Failed to create target FFMultiModel");

        // Initialise target weights to match main.
        sync_target_from_main(&main_model, &target_model)?;

        Ok(Self {
            main_model,
            target_model,
            n_episodes: 500,
            n_steps_per_episode: 70,
            epochs_per_episode,
            device,
            buffer: Vec::new(),
            num_classes,
            min_return: 0.0,
            max_return: 0.0,
            target_sync_interval: 10, // sync every N episodes
        })
    }
}

// ─────────────────────────────────────────────────────────────
// Algorithm Trait Implementation
// ─────────────────────────────────────────────────────────────

impl Algorithm for AlgorithmFFMulti2 {
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
        let input_size = state_size + action_size;

        // Vectorised environments.
        let n_envs = 16;
        let mut envs: Vec<Box<dyn Environment>> = vec![env.clone_box()];
        for _ in 1..n_envs {
            envs.push(env.clone_box());
        }

        let gamma = 0.99f32;
        let tau = 0.07f32; // Boltzmann temperature for action selection

        for episode in 1..=self.n_episodes {
            // ── early-exit on close ──
            if let Some(ref vs) = vis_state {
                if vs.try_lock().map(|s| s.should_close).unwrap_or(false) {
                    return Ok(());
                }
            }

            log::info!(
                "starting episode {} (x{} envs) | ReturnRange: [{:.4}, {:.4}]",
                episode, n_envs, self.min_return, self.max_return
            );

            for e in envs.iter_mut() {
                e.reset()?;
            }

            let mut raw_rewards = vec![0.0f64; n_envs];
            let inference_start = Instant::now();
            let mut rng = rand::thread_rng();

            // ═══════════════════════════════════════════════════
            // Phase 1: Environmental Interaction (Inference)
            // ═══════════════════════════════════════════════════
            for _step in 0..self.n_steps_per_episode {
                // Collect current states from all envs.
                let mut current_states = Vec::new();
                for e in envs.iter_mut() {
                    current_states.push(e.get_state()?);
                }

                // Build a single flat buffer for all [one_hot(action) ++ state] pairs.
                let n_inference = n_envs * action_size;
                let mut flat_inf = Vec::with_capacity(n_inference * input_size);
                for state in current_states.iter() {
                    let state_f32: Vec<f32> =
                        state.iter().map(|&x| (x as f32) / 50.0).collect();
                    for a in 0..action_size {
                        for j in 0..action_size {
                            flat_inf.push(if j == a { 1.0f32 } else { 0.0 });
                        }
                        flat_inf.extend(state_f32.iter());
                    }
                }
                total_inference_actions += n_inference;
                let inf_tensor = Tensor::from_vec(
                    flat_inf, (n_inference, input_size), &self.device,
                )?;

                // Main Model: predict return-class for each (state, action).
                let predicted_classes = self.main_model.predict(&[inf_tensor])?;

                // Select actions per env.
                for (env_idx, env_state) in current_states.iter().enumerate() {
                    let e = &mut envs[env_idx];
                    let chunk_start = env_idx * action_size;

                    // Map each action's predicted class → continuous expected return.
                    let expected_values: Vec<f32> = (0..action_size)
                        .map(|a| {
                            class_to_value(
                                predicted_classes[chunk_start + a],
                                self.min_return,
                                self.max_return,
                                self.num_classes,
                            )
                        })
                        .collect();

                    if episode % 10 == 0 && _step == 0 && env_idx == 0 {
                        let classes: Vec<usize> = (0..action_size)
                            .map(|a| predicted_classes[chunk_start + a])
                            .collect();
                        log::info!(
                            "Ep {} Step 0 | Classes: {:?} | Values: {:.4?}",
                            episode, classes, expected_values
                        );
                    }

                    // Boltzmann (softmax) action selection.
                    let max_v = expected_values
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    let exps: Vec<f32> = expected_values
                        .iter()
                        .map(|&v| {
                            let e_val = ((v - max_v) / tau).exp();
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

                    // Interact with environment.
                    let state_f32: Vec<f32> =
                        env_state.iter().map(|&x| (x as f32) / 50.0).collect();
                    let step_reward = e.evaluate_action(env_state, selected_action);
                    e.apply_action(selected_action)?;
                    let next_state = e.get_state()?;
                    let next_state_f32: Vec<f32> =
                        next_state.iter().map(|&x| (x as f32) / 50.0).collect();

                    // Store to replay buffer.
                    self.buffer.push((
                        state_f32,
                        selected_action,
                        step_reward as f32,
                        next_state_f32,
                    ));
                    raw_rewards[env_idx] += step_reward;
                }

                // ── Visualisation hooks ──
                if let Some(ref vs) = vis_state {
                    if let Ok(mut state) = vs.try_lock() {
                        let env_state = envs[0].get_state()?;
                        if state.runtime_stats.epoch != episode {
                            state.render_trail.clear();
                        }
                        if env_state.len() == 4 {
                            state.render_trail.push((
                                env_state[0] + env_state[2],
                                env_state[1] + env_state[3],
                            ));
                        }
                        state.environment_state = Some(env_state);
                    }
                }

                // ── Pause / delay / close ──
                if let Some(ref vs) = vis_state {
                    loop {
                        let (is_paused, should_close, delay) = vs
                            .try_lock()
                            .map(|s| (s.is_paused, s.should_close, s.delay_ms))
                            .unwrap_or((false, false, 0));
                        if should_close {
                            return Ok(());
                        }
                        if !is_paused {
                            if delay > 0 {
                                std::thread::sleep(Duration::from_millis(delay));
                            }
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            }

            let inference_elapsed = inference_start.elapsed();
            total_inference_time += inference_elapsed;

            // ── Record episode reward ──
            if let Some(ref vs) = vis_state {
                if let Ok(mut state) = vs.try_lock() {
                    let avg_reward = raw_rewards.iter().sum::<f64>() as f32
                        / (n_envs as f32 * self.n_steps_per_episode as f32);
                    state.epoch_rewards.push((episode, avg_reward));
                    state.runtime_stats.epoch = episode;
                    state.total_epochs = self.n_episodes;
                }
            }

            // ── Trim replay buffer ──
            if self.buffer.len() > 100_000 {
                let drain_count = self.buffer.len() - 100_000;
                self.buffer.drain(0..drain_count);
            }

            // ═══════════════════════════════════════════════════
            // Phase 2: Generating Discrete Training Labels
            // ═══════════════════════════════════════════════════
            let training_start = Instant::now();
            let batch_size = 1024.min(self.buffer.len());

            if self.buffer.len() >= 1024 {
                let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
                indices.shuffle(&mut rng);
                let batch_indices = &indices[0..batch_size];

                // ── 2a: Lookahead – find best next action via Main Model ──
                let n_lookahead = batch_size * action_size;
                let mut flat_look = Vec::with_capacity(n_lookahead * input_size);
                for &idx in batch_indices {
                    let (_, _, _, ref s_next) = self.buffer[idx];
                    for a in 0..action_size {
                        for j in 0..action_size {
                            flat_look.push(if j == a { 1.0f32 } else { 0.0 });
                        }
                        flat_look.extend(s_next.iter());
                    }
                }
                let look_tensor = Tensor::from_vec(
                    flat_look, (n_lookahead, input_size), &self.device,
                )?;
                let lookahead_classes = self.main_model.predict(&[look_tensor])?;

                let mut best_next_actions = Vec::with_capacity(batch_size);
                for i in 0..batch_size {
                    let chunk_start = i * action_size;
                    let mut best_a = 0;
                    let mut best_val = f32::NEG_INFINITY;
                    for a in 0..action_size {
                        let val = class_to_value(
                            lookahead_classes[chunk_start + a],
                            self.min_return,
                            self.max_return,
                            self.num_classes,
                        );
                        if val > best_val {
                            best_val = val;
                            best_a = a;
                        }
                    }
                    best_next_actions.push(best_a);
                }

                // ── 2b: Target evaluation – predict class via Target Model ──
                let mut flat_tgt = Vec::with_capacity(batch_size * input_size);
                for (i, &idx) in batch_indices.iter().enumerate() {
                    let (_, _, _, ref s_next) = self.buffer[idx];
                    let best_a = best_next_actions[i];
                    for j in 0..action_size {
                        flat_tgt.push(if j == best_a { 1.0f32 } else { 0.0 });
                    }
                    flat_tgt.extend(s_next.iter());
                }
                let tgt_tensor = Tensor::from_vec(
                    flat_tgt, (batch_size, input_size), &self.device,
                )?;
                let target_classes = self.target_model.predict(&[tgt_tensor])?;

                // ── 2c: Bellman target → discrete class labels ──
                let mut target_scores = Vec::with_capacity(batch_size);
                for (i, &idx) in batch_indices.iter().enumerate() {
                    let (_, _, reward, _) = &self.buffer[idx];
                    let v_next = class_to_value(
                        target_classes[i],
                        self.min_return,
                        self.max_return,
                        self.num_classes,
                    );
                    let target_score = reward + gamma * v_next;
                    target_scores.push(target_score);
                }

                // Update dynamic return bounds from first values & expand as needed.
                for &ts in &target_scores {
                    if self.min_return == 0.0 && self.max_return == 0.0 {
                        // Bootstrap from first observed target.
                        self.min_return = ts - 1.0;
                        self.max_return = ts + 1.0;
                    }
                    if ts < self.min_return {
                        self.min_return = ts;
                    }
                    if ts > self.max_return {
                        self.max_return = ts;
                    }
                }

                // Discretise targets into class labels.
                let training_labels: Vec<usize> = target_scores
                    .iter()
                    .map(|&ts| {
                        value_to_class(ts, self.min_return, self.max_return, self.num_classes)
                    })
                    .collect();

                // ═══════════════════════════════════════════════════
                // Phase 3: Model Training
                // ═══════════════════════════════════════════════════
                let mut flat_train = Vec::with_capacity(batch_size * input_size);
                for &idx in batch_indices {
                    let (ref s, action, _, _) = self.buffer[idx];
                    for j in 0..action_size {
                        flat_train.push(if j == action { 1.0f32 } else { 0.0 });
                    }
                    flat_train.extend(s.iter());
                }
                let batch_x = Tensor::from_vec(
                    flat_train, (batch_size, input_size), &self.device,
                )?;
                self.main_model.train(&batch_x, &training_labels)?;
                total_epochs += self.epochs_per_episode;
            }

            let training_elapsed = training_start.elapsed();
            total_training_time += training_elapsed;

            // ═══════════════════════════════════════════════════
            // Target Model Sync (every K episodes)
            // ═══════════════════════════════════════════════════
            if episode % self.target_sync_interval == 0 {
                log::info!(
                    ">>> Syncing Target Model at episode {} (interval={}) <<<",
                    episode,
                    self.target_sync_interval
                );
                sync_target_from_main(&self.main_model, &self.target_model)?;
            }

            // ── Logging ──
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
                "[Episode {} Reward: {:.2}] Actions/sec: {:.1} | Epochs/sec: {:.2} | Buffer: {} | ReturnRange: [{:.4}, {:.4}]",
                episode,
                raw_rewards.iter().sum::<f64>(),
                inf_aps,
                ep_s,
                self.buffer.len(),
                self.min_return,
                self.max_return
            );
        }

        log::info!("Training completed.");
        if let Some(ref vs) = vis_state {
            if let Ok(state) = vs.try_lock() {
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
