#![allow(clippy::needless_range_loop)]
use super::Algorithm;
use crate::environment::Environment;
use crate::models::csdp_multi_model::CSDPMultiModel;
use crate::visualization::VisualizationState;
use candle_core::{Device, Tensor};
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Direct translation of ff_ppo constants. CSDP SNN training is more expensive
// per sample (40 timesteps per forward), so TRAIN_BATCH_SIZE caps mini-batch
// size inside the PPO update loops to avoid OOM on large rollouts.
const NUM_VALUE_CLASSES: usize = 50;
const NUM_ENVS: usize = 16;
const ROLLOUT_STEPS: usize = 512;
const PPO_EPOCHS: usize = 2;
const TRAIN_BATCH_SIZE: usize = 256;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const AUTO_SAVE_INTERVAL: usize = 25;
const CHECKPOINT_DIR: &str = "checkpoints/csdp_ppo";
const BOUNDS_EMA_ALPHA: f32 = 0.1;
const MIN_RETURN_RANGE: f32 = 2.0;
const BOUNDS_LOW_PERCENTILE: f32 = 0.05;
const BOUNDS_HIGH_PERCENTILE: f32 = 0.95;
const EPSILON_START: f32 = 0.3;
const EPSILON_END: f32 = 0.05;
const EPSILON_DECAY_EPISODES: f32 = 200.0;
const GOODNESS_CLAMP: f32 = 6.0;

// ── Exact copies of ff_ppo helper functions ──────────────────────────────────

#[derive(Serialize, Deserialize)]
struct TrainingState {
    min_return: f32,
    max_return: f32,
    bounds_initialized: bool,
    completed_episode: usize,
    epoch_rewards: Vec<(usize, f32)>,
}

fn class_to_value(class_id: usize, min_ret: f32, max_ret: f32, n_classes: usize) -> f32 {
    if n_classes <= 1 || (max_ret - min_ret).abs() < 1e-8 {
        return (min_ret + max_ret) / 2.0;
    }
    min_ret + (class_id as f32 / (n_classes - 1) as f32) * (max_ret - min_ret)
}

fn value_to_class(value: f32, min_ret: f32, max_ret: f32, n_classes: usize) -> usize {
    if n_classes <= 1 {
        return 0;
    }
    let range = max_ret - min_ret;
    if range.abs() < 1e-8 {
        return n_classes / 2;
    }
    let t = ((value - min_ret) / range).clamp(0.0, 1.0);
    (t * (n_classes - 1) as f32).round() as usize
}

fn expected_value_from_goodness(
    goodness: &[f32],
    min_ret: f32,
    max_ret: f32,
    n_classes: usize,
) -> f32 {
    if goodness.is_empty() {
        return 0.0;
    }
    let max_g = goodness.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = goodness.iter().map(|&g| (g - max_g).exp()).collect();
    let exp_sum: f32 = exps.iter().sum();
    if exp_sum < 1e-8 {
        return (min_ret + max_ret) / 2.0;
    }
    let mut v = 0.0f32;
    for (c, &e) in exps.iter().enumerate() {
        v += (e / exp_sum) * class_to_value(c, min_ret, max_ret, n_classes);
    }
    v
}

fn softmax_probs(goodness: &[f32]) -> Vec<f32> {
    if goodness.is_empty() {
        return Vec::new();
    }
    let max_g = goodness.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = goodness.iter().map(|&g| (g - max_g).exp()).collect();
    let exp_sum: f32 = exps.iter().sum();
    if exp_sum < 1e-8 {
        let n = goodness.len();
        return vec![1.0 / n as f32; n];
    }
    exps.iter().map(|&e| e / exp_sum).collect()
}

fn goodness_to_probs(goodness: &[f32]) -> Vec<f32> {
    let n = goodness.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = goodness.iter().sum::<f32>() / n as f32;
    let var = goodness.iter().map(|&g| (g - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt().max(1e-6);
    let normalized: Vec<f32> = goodness
        .iter()
        .map(|&g| ((g - mean) / std).clamp(-GOODNESS_CLAMP, GOODNESS_CLAMP))
        .collect();
    softmax_probs(&normalized)
}

// Same normalization logic as ff_ppo, but clamped to [0, 1] because CSDP's
// BernoulliLayer interprets inputs as spike probabilities.
fn normalize_state(raw: &[f64]) -> Vec<f32> {
    if raw.len() == 31 {
        let scales: [f32; 31] = [
            4096.0, 5120.0, 2048.0, 6000.0, 6000.0, 6000.0, 6.0, 6.0, 6.0,
            4096.0, 5120.0, 2048.0, 2300.0, 2300.0, 2300.0, 5.5, 5.5, 5.5,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            100.0, 1.0, 1.0, 1.0,
        ];
        raw.iter()
            .zip(scales.iter())
            .map(|(&v, &s)| ((v as f32 / s) * 0.5 + 0.5).clamp(0.0, 1.0))
            .collect()
    } else {
        raw.iter().map(|&x| ((x as f32) / 50.0 * 0.5 + 0.5).clamp(0.0, 1.0)).collect()
    }
}

// ── Struct ───────────────────────────────────────────────────────────────────

/// Direct CSDP translation of AlgorithmFFPPO.
///
/// Policy model:  input=state, num_classes=action_size → goodness per action.
/// Value model:   input=state, num_classes=NUM_VALUE_CLASSES → goodness per return bin.
///
/// Inference is identical to ff_ppo: predict_scores → goodness_to_probs → argmax.
/// Value estimation: expected_value_from_goodness over 50 return classes.
///
/// Policy training replaces train_binary with a CSDP equivalent:
///   • adv > 0 → label = taken_action   (push that action's neuron group up)
///   • adv < 0 → label = random_other_action (push taken_action's group down indirectly)
/// Value training uses CSDP train with return-class labels, identical in intent
/// to ff_ppo's value_model.train().
pub struct AlgorithmCSDPPPO {
    pub policy_model: CSDPMultiModel,
    pub value_model: CSDPMultiModel,
    pub n_episodes: usize,
    pub device: Device,
    pub num_actions: usize,
    pub num_value_classes: usize,
    pub min_return: f32,
    pub max_return: f32,
    pub bounds_initialized: bool,
    pub start_episode: usize,
}

impl AlgorithmCSDPPPO {
    pub fn new(
        state_size: usize,
        action_size: usize,
        device: Device,
        dt: f32,
    ) -> Result<Self, Box<dyn Error>> {
        let timesteps = 40;

        // Mirror ff_ppo's sizing: neurons_per_class × action_size, rounded to
        // multiples of action_size so the class-chunk invariant holds.
        let neurons_per_class = if action_size > 20 { 32usize } else { 64 };
        let base = neurons_per_class * action_size;
        let policy_hidden: Vec<usize> = [base * 2, base, base / 2]
            .iter()
            .map(|&h| h.div_ceil(action_size) * action_size)
            .collect();

        // Value sizing mirrors ff_ppo's [512, 256, 128] rounded to NUM_VALUE_CLASSES.
        let value_hidden: Vec<usize> = [512usize, 256, 128]
            .iter()
            .map(|&h| h.div_ceil(NUM_VALUE_CLASSES) * NUM_VALUE_CLASSES)
            .collect();

        let policy_model = CSDPMultiModel::new(
            state_size,
            &policy_hidden,
            action_size,
            &device,
            dt,
            timesteps,
        )?;

        let value_model = CSDPMultiModel::new(
            state_size,
            &value_hidden,
            NUM_VALUE_CLASSES,
            &device,
            dt,
            timesteps,
        )?;

        Ok(Self {
            policy_model,
            value_model,
            n_episodes: 1000,
            device,
            num_actions: action_size,
            num_value_classes: NUM_VALUE_CLASSES,
            min_return: 0.0,
            max_return: 0.0,
            bounds_initialized: false,
            start_episode: 0,
        })
    }

    pub fn save_checkpoint(
        &self,
        dir: &std::path::Path,
        completed_episode: usize,
        epoch_rewards: &[(usize, f32)],
    ) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(dir)?;
        self.policy_model.save(dir.join("policy_model.safetensors"))?;
        self.value_model.save(dir.join("value_model.safetensors"))?;
        let state = TrainingState {
            min_return: self.min_return,
            max_return: self.max_return,
            bounds_initialized: self.bounds_initialized,
            completed_episode,
            epoch_rewards: epoch_rewards.to_vec(),
        };
        std::fs::write(
            dir.join("training_state.json"),
            serde_json::to_string(&state)?,
        )?;
        log::info!(
            "Checkpoint saved to {:?} (episode {}, return range [{:.4}, {:.4}])",
            dir,
            completed_episode,
            self.min_return,
            self.max_return
        );
        Ok(())
    }

    pub fn load_checkpoint(
        &mut self,
        dir: &std::path::Path,
    ) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
        self.policy_model.load(dir.join("policy_model.safetensors"))?;
        self.value_model.load(dir.join("value_model.safetensors"))?;
        let json = std::fs::read_to_string(dir.join("training_state.json"))?;
        let state: TrainingState = serde_json::from_str(&json)?;
        self.min_return = state.min_return;
        self.max_return = state.max_return;
        self.bounds_initialized = state.bounds_initialized;
        self.start_episode = state.completed_episode;
        log::info!(
            "Checkpoint loaded from {:?} (resuming after episode {}, return range [{:.4}, {:.4}])",
            dir,
            state.completed_episode,
            self.min_return,
            self.max_return
        );
        Ok(state.epoch_rewards)
    }
}

impl Algorithm for AlgorithmCSDPPPO {
    fn run(
        &mut self,
        env: &mut dyn Environment,
        _visualize: bool,
        vis_state: Option<Arc<Mutex<VisualizationState>>>,
    ) -> Result<(), Box<dyn Error>> {
        let state_size = env.state_size();
        let action_size = self.num_actions;

        let mut envs: Vec<Box<dyn Environment>> = vec![env.clone_box()];
        for _ in 1..NUM_ENVS {
            envs.push(env.clone_box());
        }

        let checkpoint_dir = std::path::Path::new(CHECKPOINT_DIR);
        let mut rng = rand::thread_rng();

        let mut episode = self.start_episode + 1;
        let mut episode_end = self.start_episode.saturating_add(self.n_episodes);

        let mut total_inference_time = Duration::new(0, 0);
        let mut total_inference_actions: usize = 0;
        let mut total_training_time = Duration::new(0, 0);
        let mut total_training_calls: usize = 0;

        while episode <= episode_end {
            if let Some(ref vs) = vis_state
                && vs.try_lock().map(|s| s.should_close).unwrap_or(false)
            {
                return Ok(());
            }

            log::info!(
                "starting episode {} | ReturnRange: [{:.4}, {:.4}]",
                episode,
                self.min_return,
                self.max_return
            );

            for e in envs.iter_mut() {
                e.reset()?;
            }

            // ═══════════════════════════════════════════════════
            // Phase 1: Rollout Collection
            // ═══════════════════════════════════════════════════
            let n_total = ROLLOUT_STEPS * NUM_ENVS;
            let mut all_states: Vec<Vec<f32>> = Vec::with_capacity(n_total);
            let mut all_actions: Vec<usize> = Vec::with_capacity(n_total);
            let mut all_rewards: Vec<f32> = Vec::with_capacity(n_total);
            let mut all_values: Vec<f32> = Vec::with_capacity(n_total);
            let mut all_old_probs: Vec<f32> = Vec::with_capacity(n_total);
            let mut raw_total_reward = 0.0f32;

            let inference_start = Instant::now();

            self.policy_model.disable_learning();
            self.value_model.disable_learning();

            for _step in 0..ROLLOUT_STEPS {
                let mut raw_states: Vec<Vec<f64>> = Vec::with_capacity(NUM_ENVS);
                for e in envs.iter_mut() {
                    raw_states.push(e.get_state()?);
                }

                // Batch all envs into one tensor (NUM_ENVS, state_size).
                let mut flat = Vec::with_capacity(NUM_ENVS * state_size);
                for raw in &raw_states {
                    flat.extend(normalize_state(raw));
                }
                let state_tensor = Tensor::from_vec(flat, (NUM_ENVS, state_size), &self.device)?;

                // Identical to ff_ppo: two batched predict_scores calls.
                let policy_scores = self.policy_model.predict_scores(&[state_tensor.clone()])?;
                let value_scores = self.value_model.predict_scores(&[state_tensor])?;

                let policy_flat: Vec<f32> = policy_scores.flatten_all()?.to_vec1()?;
                let value_flat: Vec<f32> = value_scores.flatten_all()?.to_vec1()?;
                total_inference_actions += NUM_ENVS;

                let epsilon = EPSILON_END
                    + (EPSILON_START - EPSILON_END)
                        * (1.0 - (episode as f32 / EPSILON_DECAY_EPISODES).min(1.0));

                for env_idx in 0..NUM_ENVS {
                    let pol_start = env_idx * action_size;
                    let goodness = &policy_flat[pol_start..pol_start + action_size];
                    let probs = goodness_to_probs(goodness);

                    let selected_action = if rng.r#gen::<f32>() < epsilon {
                        rng.gen_range(0..action_size)
                    } else {
                        probs.iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                            .unwrap_or(0)
                    };

                    let val_start = env_idx * self.num_value_classes;
                    let val_goodness = &value_flat[val_start..val_start + self.num_value_classes];
                    let v_s = expected_value_from_goodness(
                        val_goodness,
                        self.min_return,
                        self.max_return,
                        self.num_value_classes,
                    );

                    let reward =
                        envs[env_idx].evaluate_action(&raw_states[env_idx], selected_action) as f32;
                    envs[env_idx].apply_action(selected_action)?;

                    raw_total_reward += reward;
                    all_states.push(normalize_state(&raw_states[env_idx]));
                    all_actions.push(selected_action);
                    all_rewards.push(reward);
                    all_values.push(v_s);
                    all_old_probs.push(probs[selected_action].max(1e-8));
                }

                if let Some(ref vs) = vis_state {
                    if let Ok(mut vis) = vs.try_lock() {
                        if vis.runtime_stats.epoch != episode {
                            vis.render_trail.clear();
                        }
                        let env0_state = &raw_states[0];
                        if env0_state.len() == 4 {
                            vis.render_trail.push((
                                env0_state[0] + env0_state[2],
                                env0_state[1] + env0_state[3],
                            ));
                        }
                        vis.environment_state = Some(env0_state.clone());

                        let pol_probs = goodness_to_probs(&policy_flat[0..action_size]);
                        let val_goodness = &value_flat[0..self.num_value_classes];
                        let val_probs = softmax_probs(val_goodness);
                        let v_scalar = expected_value_from_goodness(
                            val_goodness,
                            self.min_return,
                            self.max_return,
                            self.num_value_classes,
                        );
                        vis.model_probabilities = Some(vec![
                            ("Policy π(a|s)".to_string(), pol_probs),
                            (format!("Value P(v|s)  E[V]={:.3}", v_scalar), val_probs),
                        ]);
                    }

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

            total_inference_time += inference_start.elapsed();

            // ═══════════════════════════════════════════════════
            // Phase 2: Bootstrap + GAE
            // ═══════════════════════════════════════════════════
            let mut bootstrap_flat = Vec::with_capacity(NUM_ENVS * state_size);
            for e in envs.iter_mut() {
                bootstrap_flat.extend(normalize_state(&e.get_state()?));
            }
            let bootstrap_tensor =
                Tensor::from_vec(bootstrap_flat, (NUM_ENVS, state_size), &self.device)?;
            let bootstrap_scores = self.value_model.predict_scores(&[bootstrap_tensor])?;
            let bootstrap_flat_scores: Vec<f32> = bootstrap_scores.flatten_all()?.to_vec1()?;

            let bootstrap_values: Vec<f32> = (0..NUM_ENVS)
                .map(|env_idx| {
                    let v_start = env_idx * self.num_value_classes;
                    expected_value_from_goodness(
                        &bootstrap_flat_scores[v_start..v_start + self.num_value_classes],
                        self.min_return,
                        self.max_return,
                        self.num_value_classes,
                    )
                })
                .collect();

            let mut all_advantages = vec![0.0f32; n_total];
            let mut all_return_targets = vec![0.0f32; n_total];

            for env_idx in 0..NUM_ENVS {
                let mut gae = 0.0f32;
                let mut next_v = bootstrap_values[env_idx];
                for step in (0..ROLLOUT_STEPS).rev() {
                    let idx = step * NUM_ENVS + env_idx;
                    let delta = all_rewards[idx] + GAMMA * next_v - all_values[idx];
                    gae = delta + GAMMA * GAE_LAMBDA * gae;
                    all_advantages[idx] = gae;
                    all_return_targets[idx] = gae + all_values[idx];
                    next_v = all_values[idx];
                }
            }

            let adv_mean: f32 = all_advantages.iter().sum::<f32>() / n_total as f32;
            let adv_var: f32 = all_advantages
                .iter()
                .map(|&a| (a - adv_mean).powi(2))
                .sum::<f32>()
                / n_total as f32;
            let adv_std = adv_var.sqrt().max(1e-8);
            for a in all_advantages.iter_mut() {
                *a = (*a - adv_mean) / adv_std;
            }

            // ═══════════════════════════════════════════════════
            // Phase 3: Update Dynamic Return Bounds
            // ═══════════════════════════════════════════════════
            let training_start = Instant::now();
            {
                let mut sorted = all_return_targets.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                let lo_idx = ((n as f32 * BOUNDS_LOW_PERCENTILE) as usize).min(n - 1);
                let hi_idx = ((n as f32 * BOUNDS_HIGH_PERCENTILE) as usize).min(n - 1);
                let r_lo = sorted[lo_idx];
                let r_hi = sorted[hi_idx];
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

            // ═══════════════════════════════════════════════════
            // Phase 4: PPO Update Epochs
            // ═══════════════════════════════════════════════════
            self.policy_model.enable_learning();
            self.value_model.enable_learning();

            let mut indices: Vec<usize> = (0..n_total).collect();
            let record_layer_id = vis_state
                .as_ref()
                .and_then(|vs| vs.try_lock().ok().and_then(|s| s.selected_layer_id));

            for ppo_epoch in 0..PPO_EPOCHS {
                indices.shuffle(&mut rng);

                // ── Value Update ──
                // Identical intent to ff_ppo: map return targets to class labels,
                // train value model to predict the return class for each state.
                {
                    let mut offset = 0;
                    while offset < n_total {
                        let end = (offset + TRAIN_BATCH_SIZE).min(n_total);
                        let batch_len = end - offset;

                        let mut flat_states = Vec::with_capacity(batch_len * state_size);
                        let mut class_labels = Vec::with_capacity(batch_len);
                        for &i in &indices[offset..end] {
                            flat_states.extend_from_slice(&all_states[i]);
                            class_labels.push(value_to_class(
                                all_return_targets[i],
                                self.min_return,
                                self.max_return,
                                self.num_value_classes,
                            ) as f32);
                        }
                        let x = Tensor::from_vec(
                            flat_states,
                            (batch_len, state_size),
                            &self.device,
                        )?;
                        let y = Tensor::from_vec(class_labels, (1, batch_len), &self.device)?;
                        self.value_model.train(&x, &y, None)?;
                        offset = end;
                    }
                }

                // ── Policy Update ──
                // CSDP translation of ff_ppo's train_binary:
                //   adv > 0 → label = taken_action   (MultiClassModSignal strengthens that group)
                //   adv < 0 → label = random_other   (taken_action's group weakened by contrast)
                // Processed in mini-batches so the SNN doesn't OOM on n_total samples.
                {
                    // Pre-sample "other" actions for negative transitions.
                    let neg_actions: Vec<usize> = indices
                        .iter()
                        .map(|&i| {
                            let a = all_actions[i];
                            if action_size > 1 {
                                let mut other = rng.gen_range(0..action_size - 1);
                                if other >= a {
                                    other += 1;
                                }
                                other
                            } else {
                                0
                            }
                        })
                        .collect();

                    let mut offset = 0;
                    while offset < n_total {
                        let end = (offset + TRAIN_BATCH_SIZE).min(n_total);
                        let batch_len = end - offset;

                        let mut flat_states = Vec::with_capacity(batch_len * state_size);
                        let mut class_labels = Vec::with_capacity(batch_len);
                        let mut has_any = false;

                        for (batch_pos, &i) in indices[offset..end].iter().enumerate() {
                            let adv = all_advantages[i];
                            if adv.abs() < 1e-6 {
                                // Zero advantage — skip; don't pollute with a meaningless label.
                                // We still need to emit a row so the tensor stays rectangular,
                                // so we duplicate the last label (won't matter if we skip below).
                                flat_states.extend_from_slice(&all_states[i]);
                                class_labels.push(all_actions[i] as f32);
                                continue;
                            }
                            has_any = true;
                            flat_states.extend_from_slice(&all_states[i]);
                            let label = if adv > 0.0 {
                                all_actions[i] as f32
                            } else {
                                neg_actions[offset + batch_pos] as f32
                            };
                            class_labels.push(label);
                        }

                        if !has_any {
                            offset = end;
                            continue;
                        }

                        let x = Tensor::from_vec(
                            flat_states,
                            (batch_len, state_size),
                            &self.device,
                        )?;
                        let y = Tensor::from_vec(class_labels, (1, batch_len), &self.device)?;
                        self.policy_model.train(&x, &y, record_layer_id)?;
                        offset = end;
                    }

                    total_training_calls += 1;
                    log::info!(
                        "Ep {} PPO epoch {}: policy + value update on {} transitions",
                        episode, ppo_epoch, n_total
                    );
                }
            }

            total_training_time += training_start.elapsed();

            // ── Record episode reward ──
            let avg_reward = raw_total_reward / n_total as f32;
            if let Some(ref vs) = vis_state
                && let Ok(mut state) = vs.try_lock()
            {
                state.epoch_rewards.push((episode, avg_reward));
                state.runtime_stats.epoch = episode;
                state.total_epochs = episode_end;

                if let Ok(snapshot) = self.policy_model.get_visualization_snapshot() {
                    state.update_from_snapshot(snapshot);
                }
            }

            // ── Manual save / load via visualization UI ──
            if let Some(ref vs) = vis_state
                && let Ok(mut state) = vs.try_lock()
            {
                if state.save_requested {
                    let epoch_rewards = state.epoch_rewards.clone();
                    state.save_requested = false;
                    drop(state);
                    if let Err(e) = self.save_checkpoint(checkpoint_dir, episode, &epoch_rewards) {
                        log::error!("Manual save failed: {}", e);
                    }
                } else if state.load_requested {
                    state.load_requested = false;
                    drop(state);
                    match self.load_checkpoint(checkpoint_dir) {
                        Ok(epoch_rewards) => {
                            if let Some(ref vs2) = vis_state
                                && let Ok(mut s) = vs2.try_lock()
                            {
                                s.epoch_rewards = epoch_rewards;
                            }
                            episode = self.start_episode;
                            episode_end = self.start_episode.saturating_add(self.n_episodes);
                            log::info!("Manual load succeeded. Continuing training.");
                        }
                        Err(e) => log::error!("Manual load failed: {}", e),
                    }
                }
            }

            // ── Periodic auto-save ──
            if episode.is_multiple_of(AUTO_SAVE_INTERVAL) {
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
            let train_ps = if total_training_time.as_secs_f32() > 0.0 {
                total_training_calls as f32 / total_training_time.as_secs_f32()
            } else {
                0.0
            };
            log::info!(
                "[Episode {} Reward: {:.4}] Actions/sec: {:.1} | Train calls/sec: {:.2} | ReturnRange: [{:.4}, {:.4}]",
                episode,
                avg_reward,
                inf_aps,
                train_ps,
                self.min_return,
                self.max_return
            );

            episode += 1;
        }

        // Final checkpoint.
        log::info!("Training completed. Saving final checkpoint...");
        let epoch_rewards = vis_state
            .as_ref()
            .and_then(|vs| vs.try_lock().ok().map(|s| s.epoch_rewards.clone()))
            .unwrap_or_default();
        if let Err(e) = self.save_checkpoint(checkpoint_dir, episode_end, &epoch_rewards) {
            log::error!("Final save failed: {}", e);
        }
        if let Some(ref vs) = vis_state
            && let Ok(state) = vs.try_lock()
        {
            let csv_path = std::path::Path::new(CHECKPOINT_DIR).join("epoch_rewards.csv");
            let _ = state.save_graphs_to_csv(&csv_path);
        }
        Ok(())
    }
}
