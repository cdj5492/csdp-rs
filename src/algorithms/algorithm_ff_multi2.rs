use super::Algorithm;
use crate::environment::Environment;
use crate::models::ff_multi_model::FFMultiModel;
use crate::visualization::VisualizationState;
use candle_core::{Device, Tensor};
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Number of discrete return classes the model predicts.
const NUM_RETURN_CLASSES: usize = 50;

/// How often (in episodes) to auto-save a checkpoint.
const AUTO_SAVE_INTERVAL: usize = 25;

/// Default checkpoint directory for this algorithm.
const CHECKPOINT_DIR: &str = "checkpoints/ff_multi2";

/// EMA smoothing factor for return bounds (0 = never update, 1 = jump instantly).
const BOUNDS_EMA_ALPHA: f32 = 0.1;

/// Minimum allowed range for return bounds to prevent degenerate class resolution.
const MIN_RETURN_RANGE: f32 = 2.0;

/// Percentile indices used for bounds estimation (e.g., 5th and 95th).
const BOUNDS_LOW_PERCENTILE: f32 = 0.05;
const BOUNDS_HIGH_PERCENTILE: f32 = 0.95;

/// Scale factor for one-hot action encoding.
/// After L2 normalization, the raw one-hot (1.0) is diluted by the state
/// dimensions, making it hard for the model to distinguish between actions.
/// Scaling up to 3.0 ensures the action is a dominant part of the input
/// direction after normalization.
const ACTION_SCALE: f32 = 3.0;

// ─────────────────────────────────────────────────────────────
// Checkpoint Metadata
// ─────────────────────────────────────────────────────────────

/// Everything that needs to be persisted *besides* the model weights
/// (which go into safetensors files via `FFMultiModel::save`).
#[derive(Serialize, Deserialize)]
struct TrainingState {
    /// Replay buffer entries: (normalised_state, action_id, monte_carlo_return).
    buffer: Vec<(Vec<f32>, usize, f32)>,
    min_return: f32,
    max_return: f32,
    bounds_initialized: bool,
    /// The last completed episode number, so we can resume from episode+1.
    completed_episode: usize,
    /// Epoch reward history for the training graph.
    epoch_rewards: Vec<(usize, f32)>,
}

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
fn sync_target_from_main(main: &FFMultiModel, target: &FFMultiModel) -> Result<(), Box<dyn Error>> {
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
    /// Replay buffer: (normalised_state, action_id, monte_carlo_return)
    pub buffer: Vec<(Vec<f32>, usize, f32)>,
    pub num_classes: usize,
    pub min_return: f32,
    pub max_return: f32,
    /// Whether min/max return have been initialized from real data.
    pub bounds_initialized: bool,
    pub target_sync_interval: usize,
    /// Episode number to start from (0 = fresh run, >0 = resumed).
    pub start_episode: usize,
}

impl AlgorithmFFMulti2 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        device: Device,
    ) -> Result<Self, Box<dyn Error>> {
        let input_size = state_size + action_size; // [one-hot action ++ state]
        let epochs_per_episode = 12;
        let num_classes = NUM_RETURN_CLASSES;

        // Round each hidden size up to the nearest multiple of num_classes
        // so that FFMultiLayer's out_features % num_classes == 0 invariant holds.
        let mut dims = vec![input_size];
        for &h in &hidden_sizes {
            let rounded = ((h + num_classes - 1) / num_classes) * num_classes;
            dims.push(rounded);
        }

        let main_model = FFMultiModel::new(&dims, num_classes, &device, epochs_per_episode)
            .expect("Failed to create main FFMultiModel");
        // Target never trains – epoch count is irrelevant.
        let target_model = FFMultiModel::new(&dims, num_classes, &device, 1)
            .expect("Failed to create target FFMultiModel");

        // Initialise target weights to match main.
        sync_target_from_main(&main_model, &target_model)?;

        Ok(Self {
            main_model,
            target_model,
            n_episodes: 100,
            n_steps_per_episode: 500,
            epochs_per_episode,
            device,
            buffer: Vec::new(),
            num_classes,
            min_return: 0.0,
            max_return: 0.0,
            bounds_initialized: false,
            target_sync_interval: 10, // sync every N episodes
            start_episode: 0,
        })
    }

    // ─────────────────────────────────────────────────────────
    // Checkpoint Save / Load
    // ─────────────────────────────────────────────────────────

    /// Save a full checkpoint: model weights (safetensors) + training metadata (JSON).
    pub fn save_checkpoint(
        &self,
        dir: &std::path::Path,
        completed_episode: usize,
        epoch_rewards: &[(usize, f32)],
    ) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(dir)?;

        // 1. Save model weights.
        let main_dir = dir.join("main_model");
        self.main_model.save(&main_dir)?;
        let target_dir = dir.join("target_model");
        self.target_model.save(&target_dir)?;

        // 2. Save training metadata as JSON.
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

        log::info!(
            "Checkpoint saved to {:?} (episode {}, buffer size {}, rewards history {})",
            dir,
            completed_episode,
            self.buffer.len(),
            epoch_rewards.len(),
        );
        Ok(())
    }

    /// Load a checkpoint, restoring model weights and all training state.
    /// Returns the epoch_rewards history so the caller can feed it to the
    /// visualisation state.
    pub fn load_checkpoint(
        &mut self,
        dir: &std::path::Path,
    ) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
        // 1. Load model weights.
        let main_dir = dir.join("main_model");
        self.main_model.load(&main_dir)?;
        let target_dir = dir.join("target_model");
        self.target_model.load(&target_dir)?;

        // 2. Load metadata.
        let json = std::fs::read_to_string(dir.join("training_state.json"))?;
        let state: TrainingState = serde_json::from_str(&json)?;

        self.buffer = state.buffer;
        self.min_return = state.min_return;
        self.max_return = state.max_return;
        self.bounds_initialized = state.bounds_initialized;
        self.start_episode = state.completed_episode;

        log::info!(
            "Checkpoint loaded from {:?} (resuming after episode {}, buffer size {}, return range [{:.4}, {:.4}])",
            dir,
            state.completed_episode,
            self.buffer.len(),
            self.min_return,
            self.max_return,
        );

        Ok(state.epoch_rewards)
    }
}

// ─────────────────────────────────────────────────────────────
// State Normalization
// ─────────────────────────────────────────────────────────────

/// Normalize a raw RocketSim / Grid state vector into roughly [-1, 1] range.
/// RocketSim states are 31-dimensional with heterogeneous scales:
///   [0..9]   ball pos/vel      (positions up to ~5000, velocities up to ~6000)
///   [9..18]  car pos/vel       (same ranges)
///   [18..27] car rotation mat  (already in [-1, 1])
///   [27]     boost amount      (0..100)
///   [28..30] boolean flags     (0 or 1)
/// Grid states are 8-dimensional with values in [0, 50).
fn normalize_state(raw: &[f64]) -> Vec<f32> {
    if raw.len() == 31 {
        // RocketSim
        let scales: [f32; 31] = [
            // ball pos xyz
            4096.0, 5120.0, 2048.0, // ball vel xyz
            6000.0, 6000.0, 6000.0, // ball ang_vel xyz
            6.0, 6.0, 6.0, // car pos xyz
            4096.0, 5120.0, 2048.0, // car vel xyz
            2300.0, 2300.0, 2300.0, // car ang_vel xyz
            5.5, 5.5, 5.5, // car rotation matrix (9 elements, already [-1,1])
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   // boost amount
            100.0, // boolean flags
            1.0, 1.0, 1.0,
        ];
        raw.iter()
            .zip(scales.iter())
            .map(|(&v, &s)| (v as f32) / s)
            .collect()
    } else {
        // Grid or other environments — use generic normalization
        raw.iter().map(|&x| (x as f32) / 50.0).collect()
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

        let gamma = 0.9f32;
        let tau = 0.5f32; // Boltzmann temperature for action selection

        let checkpoint_dir = std::path::Path::new(CHECKPOINT_DIR);

        // Episode range: if resuming, start after the last completed episode.
        let mut episode = self.start_episode + 1;
        let mut episode_end = self.start_episode + self.n_episodes;

        while episode <= episode_end {
            // ── early-exit on close ──
            if let Some(ref vs) = vis_state {
                if vs.try_lock().map(|s| s.should_close).unwrap_or(false) {
                    return Ok(());
                }
            }

            log::info!(
                "starting episode {} (x{} envs) | ReturnRange: [{:.4}, {:.4}]",
                episode,
                n_envs,
                self.min_return,
                self.max_return
            );

            // ── Episode-local trajectory buffers for MC return computation ──
            let mut episode_trajectories: Vec<Vec<(Vec<f32>, usize, f32)>> = (0..n_envs)
                .map(|_| Vec::with_capacity(self.n_steps_per_episode))
                .collect();

            // Reset all environments at the start of each episode.
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
                    let state_f32 = normalize_state(state);
                    for a in 0..action_size {
                        for j in 0..action_size {
                            flat_inf.push(if j == a { ACTION_SCALE } else { 0.0 });
                        }
                        flat_inf.extend(state_f32.iter());
                    }
                }
                total_inference_actions += n_inference;
                let inf_tensor =
                    Tensor::from_vec(flat_inf, (n_inference, input_size), &self.device)?;

                // Main Model: get raw goodness scores for each (state, action).
                // Using predict_scores() instead of predict() gives continuous
                // action values via soft expected returns—even when the argmax
                // class is the same for all actions, the goodness distribution
                // still differs, producing differentiable action preferences.
                let score_tensor = self.main_model.predict_scores(&[inf_tensor])?;
                let scores_flat: Vec<f32> = score_tensor.flatten_all()?.to_vec1()?;

                // Select actions per env.
                for (env_idx, env_state) in current_states.iter().enumerate() {
                    let e = &mut envs[env_idx];
                    let chunk_start = env_idx * action_size;

                    // Compute soft expected value for each action.
                    let expected_values: Vec<f32> = (0..action_size)
                        .map(|a| {
                            let row_start = (chunk_start + a) * self.num_classes;
                            let row_end = row_start + self.num_classes;
                            let goodnesses = &scores_flat[row_start..row_end];

                            // Softmax over goodnesses → probability distribution over classes.
                            let max_g =
                                goodnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let exps: Vec<f32> =
                                goodnesses.iter().map(|&g| (g - max_g).exp()).collect();
                            let exp_sum: f32 = exps.iter().sum();

                            // Weighted average return = Σ p(class) * value(class).
                            let mut expected_val = 0.0f32;
                            for (c, &e_val) in exps.iter().enumerate() {
                                let p = e_val / exp_sum;
                                expected_val += p * class_to_value(
                                    c,
                                    self.min_return,
                                    self.max_return,
                                    self.num_classes,
                                );
                            }
                            expected_val
                        })
                        .collect();

                    if episode % 10 == 0 && _step == 0 && env_idx == 0 {
                        log::info!(
                            "Ep {} Step 0 | SoftValues: {:.4?}",
                            episode,
                            expected_values
                        );
                    }

                    // ε-greedy on top of Boltzmann — ensures diverse actions
                    // in the buffer even when the model is highly confident.
                    let epsilon_start = 0.20f32;
                    let epsilon_end = 0.05f32;
                    let decay_episodes = 200.0f32;
                    let epsilon = epsilon_end
                        + (epsilon_start - epsilon_end)
                            * (1.0 - (episode as f32 / decay_episodes).min(1.0));

                    let selected_action = if rng.r#gen::<f32>() < epsilon {
                        // Random exploration
                        rng.gen_range(0..action_size)
                    } else {
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
                        let mut sel = action_size - 1;
                        for (a, &e_val) in exps.iter().enumerate() {
                            cumulative += e_val;
                            if rand_val <= cumulative {
                                sel = a;
                                break;
                            }
                        }
                        sel
                    };

                    // Record step data for MC return computation at episode end.
                    let state_f32 = normalize_state(env_state);
                    let step_reward = e.evaluate_action(env_state, selected_action);
                    e.apply_action(selected_action)?;

                    episode_trajectories[env_idx].push((
                        state_f32,
                        selected_action,
                        step_reward as f32,
                    ));
                    raw_rewards[env_idx] += step_reward;
                }

                // ── Visualisation hooks ──
                if let Some(ref vs) = vis_state {
                    let mut main_probs = Vec::new();
                    let mut target_probs = Vec::new();

                    if let Some(&(ref state_f32, selected_action, _)) = episode_trajectories[0].last() {
                        let mut input_vec = Vec::with_capacity(input_size);
                        for j in 0..action_size {
                            input_vec.push(if j == selected_action { ACTION_SCALE } else { 0.0 });
                        }
                        input_vec.extend(state_f32.iter());

                        if let Ok(inf_tensor) = Tensor::from_vec(input_vec, (1, input_size), &self.device) {
                            if let Ok(main_scores) = self.main_model.predict_scores(&[inf_tensor.clone()]) {
                                if let Ok(main_flat) = main_scores.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                                    let main_max = main_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let main_exps: Vec<f32> = main_flat.iter().map(|&v| (v - main_max).exp()).collect();
                                    let main_sum: f32 = main_exps.iter().sum();
                                    main_probs = main_exps.iter().map(|v| v / main_sum).collect();
                                }
                            }
                            if let Ok(target_scores) = self.target_model.predict_scores(&[inf_tensor]) {
                                if let Ok(target_flat) = target_scores.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                                    let target_max = target_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let target_exps: Vec<f32> = target_flat.iter().map(|&v| (v - target_max).exp()).collect();
                                    let target_sum: f32 = target_exps.iter().sum();
                                    target_probs = target_exps.iter().map(|v| v / target_sum).collect();
                                }
                            }
                        }
                    }

                    if let Ok(mut state) = vs.try_lock() {
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
                        
                        if !main_probs.is_empty() && !target_probs.is_empty() {
                            state.model_probabilities = Some(vec![
                                ("Main P(ret|s,a)".to_string(), main_probs),
                                ("Target P(ret|s,a)".to_string(), target_probs),
                            ]);
                        }
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
                    state.total_epochs = episode_end;
                }
            }

            // ── Compute Monte Carlo returns and add to replay buffer ──
            // For each environment's episode trajectory, compute discounted
            // return-to-go from actual rewards (no bootstrapping).
            for traj in &episode_trajectories {
                let n_steps = traj.len();
                if n_steps == 0 {
                    continue;
                }
                let mut returns = vec![0.0f32; n_steps];
                returns[n_steps - 1] = traj[n_steps - 1].2;
                for t in (0..n_steps - 1).rev() {
                    returns[t] = traj[t].2 + gamma * returns[t + 1];
                }
                for (t, (state, action, _reward)) in traj.iter().enumerate() {
                    self.buffer.push((state.clone(), *action, returns[t]));
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

                // ── Compute bounds from actual MC returns in the batch ──
                {
                    let mut batch_returns: Vec<f32> = batch_indices
                        .iter()
                        .map(|&idx| self.buffer[idx].2)
                        .collect();
                    batch_returns
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = batch_returns.len();
                    let lo_idx = ((n as f32 * BOUNDS_LOW_PERCENTILE) as usize).min(n - 1);
                    let hi_idx = ((n as f32 * BOUNDS_HIGH_PERCENTILE) as usize).min(n - 1);
                    let r_lo = batch_returns[lo_idx];
                    let r_hi = batch_returns[hi_idx];

                    if !self.bounds_initialized {
                        self.min_return = r_lo - 1.0;
                        self.max_return = r_hi + 1.0;
                        self.bounds_initialized = true;
                    } else {
                        self.min_return =
                            (1.0 - BOUNDS_EMA_ALPHA) * self.min_return + BOUNDS_EMA_ALPHA * r_lo;
                        self.max_return =
                            (1.0 - BOUNDS_EMA_ALPHA) * self.max_return + BOUNDS_EMA_ALPHA * r_hi;
                    }

                    if self.max_return - self.min_return < MIN_RETURN_RANGE {
                        let mid = (self.min_return + self.max_return) / 2.0;
                        self.min_return = mid - MIN_RETURN_RANGE / 2.0;
                        self.max_return = mid + MIN_RETURN_RANGE / 2.0;
                    }
                }

                // Discretise MC returns into class labels.
                let training_labels: Vec<usize> = batch_indices
                    .iter()
                    .map(|&idx| {
                        value_to_class(
                            self.buffer[idx].2,
                            self.min_return,
                            self.max_return,
                            self.num_classes,
                        )
                    })
                    .collect();

                // ── Diagnostic: log class distribution every 10 episodes ──
                if episode % 10 == 0 {
                    let mut class_counts = vec![0usize; self.num_classes];
                    for &c in &training_labels {
                        class_counts[c] += 1;
                    }
                    let distinct = class_counts.iter().filter(|&&c| c > 0).count();
                    let max_count = class_counts.iter().max().copied().unwrap_or(0);
                    log::info!(
                        "ClassDist: distinct={}/{} | max_bucket={}/{} | range=[{:.4}, {:.4}]",
                        distinct,
                        self.num_classes,
                        max_count,
                        training_labels.len(),
                        self.min_return,
                        self.max_return,
                    );
                }

                // ═══════════════════════════════════════════════════
                // Phase 3: Model Training
                // ═══════════════════════════════════════════════════
                let mut flat_train = Vec::with_capacity(batch_size * input_size);
                for &idx in batch_indices {
                    let (ref s, action, _) = self.buffer[idx];
                    for j in 0..action_size {
                        flat_train.push(if j == action { ACTION_SCALE } else { 0.0 });
                    }
                    flat_train.extend(s.iter());
                }
                let batch_x = Tensor::from_vec(flat_train, (batch_size, input_size), &self.device)?;
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

            // ═══════════════════════════════════════════════════
            // Checkpointing
            // ═══════════════════════════════════════════════════

            // ── Manual save/load via visualisation UI ──
            if let Some(ref vs) = vis_state {
                if let Ok(mut state) = vs.try_lock() {
                    if state.save_requested {
                        log::info!("Manual save requested...");
                        let epoch_rewards = state.epoch_rewards.clone();
                        // Drop the lock before doing I/O.
                        state.save_requested = false;
                        drop(state);
                        if let Err(e) =
                            self.save_checkpoint(checkpoint_dir, episode, &epoch_rewards)
                        {
                            log::error!("Manual save failed: {}", e);
                        }
                    } else if state.load_requested {
                        log::info!("Manual load requested...");
                        state.load_requested = false;
                        drop(state);
                        match self.load_checkpoint(checkpoint_dir) {
                            Ok(epoch_rewards) => {
                                // Push restored rewards into vis state.
                                if let Some(ref vs2) = vis_state {
                                    if let Ok(mut s) = vs2.try_lock() {
                                        s.epoch_rewards = epoch_rewards;
                                    }
                                }
                                // Re-sync target model after loading.
                                sync_target_from_main(&self.main_model, &self.target_model)?;

                                // Jump the loop counter so training resumes properly
                                // instead of continuing the old loop index over loaded data.
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

            // ── Periodic auto-save ──
            if episode % AUTO_SAVE_INTERVAL == 0 {
                let epoch_rewards = vis_state
                    .as_ref()
                    .and_then(|vs| vs.try_lock().ok().map(|s| s.epoch_rewards.clone()))
                    .unwrap_or_default();
                if let Err(e) = self.save_checkpoint(checkpoint_dir, episode, &epoch_rewards) {
                    log::error!("Auto-save failed at episode {}: {}", episode, e);
                }
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

            episode += 1;
        }

        // ═══════════════════════════════════════════════════
        // Final save on completion
        // ═══════════════════════════════════════════════════
        log::info!("Training completed. Saving final checkpoint...");
        let epoch_rewards = vis_state
            .as_ref()
            .and_then(|vs| vs.try_lock().ok().map(|s| s.epoch_rewards.clone()))
            .unwrap_or_default();
        if let Err(e) = self.save_checkpoint(checkpoint_dir, episode_end, &epoch_rewards) {
            log::error!("Final save failed: {}", e);
        }

        if let Some(ref vs) = vis_state {
            if let Ok(state) = vs.try_lock() {
                let csv_path = std::path::Path::new(CHECKPOINT_DIR).join("epoch_rewards.csv");
                let _ = state.save_graphs_to_csv(&csv_path);
            }
        }
        Ok(())
    }
}
