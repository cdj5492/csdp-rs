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

const NUM_VALUE_CLASSES: usize = 50;
const NUM_ENVS: usize = 16;
const ROLLOUT_STEPS: usize = 512;
const PPO_EPOCHS: usize = 2;
// Entropy coefficient applied to each policy layer's train_binary call (RLGymPPO pattern).
// Maximizing action-distribution entropy prevents collapse without requiring
// explicit exploration noise.
const POLICY_ENT_COEF: f64 = 0.005;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const AUTO_SAVE_INTERVAL: usize = 25;
const CHECKPOINT_DIR: &str = "checkpoints/ff_ppo";
const BOUNDS_EMA_ALPHA: f32 = 0.1;
const MIN_RETURN_RANGE: f32 = 2.0;
const BOUNDS_LOW_PERCENTILE: f32 = 0.05;
const BOUNDS_HIGH_PERCENTILE: f32 = 0.95;
// Epsilon-greedy exploration schedule.
const EPSILON_START: f32 = 0.4;
const EPSILON_END: f32 = 0.05;
const EPSILON_DECAY_EPISODES: f32 = 300.0;
// Clamp applied to z-scored goodness before softmax. Must be large enough that
// the z-scores of reinforced actions are NOT truncated, otherwise goodness
// differences between similar-quality actions collapse to zero and all appear
// equally probable. With typical FF goodness spreads and 96 actions, reinforced
// action z-scores land around 4–5, so the clamp must be larger than that.
// A value of 6.0 lets the policy express graded preferences among its best
// actions while still bounding extreme single-action dominance.
const GOODNESS_CLAMP: f32 = 6.0;

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

/// Converts raw goodness scores to a policy probability distribution.
///
/// Raw goodness scores grow without bound as the FF contrastive loss trains,
/// which collapses a plain softmax to 100% on one action within a few episodes.
/// This function z-scores the goodness values and clamps them to ±GOODNESS_CLAMP
/// before softmax so the policy can never fully collapse regardless of how large
/// the underlying goodness magnitudes become.
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

fn sync_model(src: &FFMultiModel, dst: &FFMultiModel) -> Result<(), Box<dyn Error>> {
    for (src_vm, dst_vm) in src.varmaps.iter().zip(dst.varmaps.iter()) {
        let src_data = src_vm.data().lock().unwrap();
        let dst_data = dst_vm.data().lock().unwrap();
        for (key, src_var) in src_data.iter() {
            if let Some(dst_var) = dst_data.get(key) {
                let t: &Tensor = src_var;
                dst_var.set(&t.detach())?;
            }
        }
    }
    Ok(())
}

fn normalize_state(raw: &[f64]) -> Vec<f32> {
    if raw.len() == 31 {
        let scales: [f32; 31] = [
            4096.0, 5120.0, 2048.0, 6000.0, 6000.0, 6000.0, 6.0, 6.0, 6.0, 4096.0, 5120.0,
            2048.0, 2300.0, 2300.0, 2300.0, 5.5, 5.5, 5.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 100.0, 1.0, 1.0, 1.0,
        ];
        raw.iter().zip(scales.iter()).map(|(&v, &s)| (v as f32) / s).collect()
    } else {
        raw.iter().map(|&x| (x as f32) / 50.0).collect()
    }
}

pub struct AlgorithmFFPPO {
    pub policy_model: FFMultiModel,
    pub policy_model_old: FFMultiModel,
    pub value_model: FFMultiModel,
    pub n_episodes: usize,
    pub device: Device,
    pub num_actions: usize,
    pub num_value_classes: usize,
    pub min_return: f32,
    pub max_return: f32,
    pub bounds_initialized: bool,
    pub start_episode: usize,
}

impl AlgorithmFFPPO {
    pub fn new(
        state_size: usize,
        action_size: usize,
        device: Device,
    ) -> Result<Self, Box<dyn Error>> {
        // Policy dims: choose neurons-per-class based on action space size,
        // then round each hidden size up to a multiple of action_size so the
        // FFMultiLayer chunk invariant is satisfied.
        let neurons_per_class = if action_size > 20 { 6usize } else { 64 };
        let base = neurons_per_class * action_size;
        let policy_hidden = [base * 2, base, base / 2];
        let mut policy_dims = vec![state_size];
        for &h in &policy_hidden {
            policy_dims.push(h.div_ceil(action_size) * action_size);
        }

        // Value dims: same rounding logic for NUM_VALUE_CLASSES.
        let mut value_dims = vec![state_size];
        for &h in &[512usize, 256, 128] {
            value_dims.push(h.div_ceil(NUM_VALUE_CLASSES) * NUM_VALUE_CLASSES);
        }

        // 1 internal epoch per train() call; PPO outer loop provides multiple passes.
        let mut policy_model = FFMultiModel::new(&policy_dims, action_size, &device, 1)?;
        let policy_model_old = FFMultiModel::new(&policy_dims, action_size, &device, 1)?;
        let value_model = FFMultiModel::new(&value_dims, NUM_VALUE_CLASSES, &device, 1)?;

        // Apply entropy coefficient to every policy layer so train_binary maximizes
        // action-distribution entropy alongside the advantage-based binary loss.
        for layer in &mut policy_model.layers {
            layer.ent_coef = POLICY_ENT_COEF;
        }

        sync_model(&policy_model, &policy_model_old)?;

        Ok(Self {
            policy_model,
            policy_model_old,
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
        self.policy_model.save(dir.join("policy_model"))?;
        self.value_model.save(dir.join("value_model"))?;
        let state = TrainingState {
            min_return: self.min_return,
            max_return: self.max_return,
            bounds_initialized: self.bounds_initialized,
            completed_episode,
            epoch_rewards: epoch_rewards.to_vec(),
        };
        std::fs::write(dir.join("training_state.json"), serde_json::to_string(&state)?)?;
        log::info!(
            "Checkpoint saved to {:?} (episode {}, return range [{:.4}, {:.4}])",
            dir, completed_episode, self.min_return, self.max_return
        );
        Ok(())
    }

    pub fn load_checkpoint(
        &mut self,
        dir: &std::path::Path,
    ) -> Result<Vec<(usize, f32)>, Box<dyn Error>> {
        self.policy_model.load(dir.join("policy_model"))?;
        self.value_model.load(dir.join("value_model"))?;
        sync_model(&self.policy_model, &self.policy_model_old)?;
        let json = std::fs::read_to_string(dir.join("training_state.json"))?;
        let state: TrainingState = serde_json::from_str(&json)?;
        self.min_return = state.min_return;
        self.max_return = state.max_return;
        self.bounds_initialized = state.bounds_initialized;
        self.start_episode = state.completed_episode;
        log::info!(
            "Checkpoint loaded from {:?} (resuming after episode {}, return range [{:.4}, {:.4}])",
            dir, state.completed_episode, self.min_return, self.max_return
        );
        Ok(state.epoch_rewards)
    }
}

impl Algorithm for AlgorithmFFPPO {
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
        let mut episode_end = self.start_episode + self.n_episodes;

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
            // Per-transition storage indexed as [step * NUM_ENVS + env_idx].
            let n_total = ROLLOUT_STEPS * NUM_ENVS;
            let mut all_states: Vec<Vec<f32>> = Vec::with_capacity(n_total);
            let mut all_actions: Vec<usize> = Vec::with_capacity(n_total);
            let mut all_rewards: Vec<f32> = Vec::with_capacity(n_total);
            let mut all_values: Vec<f32> = Vec::with_capacity(n_total);
            let mut all_old_probs: Vec<f32> = Vec::with_capacity(n_total);
            let mut raw_total_reward = 0.0f32;

            let inference_start = Instant::now();

            for _step in 0..ROLLOUT_STEPS {
                // Collect raw states, then build a batched tensor once.
                let mut raw_states: Vec<Vec<f64>> = Vec::with_capacity(NUM_ENVS);
                for e in envs.iter_mut() {
                    raw_states.push(e.get_state()?);
                }

                let mut flat = Vec::with_capacity(NUM_ENVS * state_size);
                for raw in &raw_states {
                    flat.extend(normalize_state(raw));
                }
                let state_tensor =
                    Tensor::from_vec(flat, (NUM_ENVS, state_size), &self.device)?;

                let policy_scores =
                    self.policy_model.predict_scores(&[state_tensor.clone()])?;
                let value_scores =
                    self.value_model.predict_scores(&[state_tensor])?;

                let policy_flat: Vec<f32> = policy_scores.flatten_all()?.to_vec1()?;
                let value_flat: Vec<f32> = value_scores.flatten_all()?.to_vec1()?;
                total_inference_actions += NUM_ENVS;

                let epsilon = EPSILON_END
                    + (EPSILON_START - EPSILON_END)
                        * (1.0 - (episode as f32 / EPSILON_DECAY_EPISODES).min(1.0));

                for env_idx in 0..NUM_ENVS {
                    let pol_start = env_idx * action_size;
                    let goodness = &policy_flat[pol_start..pol_start + action_size];
                    // Use normalize+clamp softmax so the distribution cannot collapse
                    // to 100% on one action regardless of raw goodness magnitude.
                    let probs = goodness_to_probs(goodness);

                    let selected_action = if rng.r#gen::<f32>() < epsilon {
                        // Random exploration to ensure diverse actions even after
                        // partial policy collapse.
                        rng.gen_range(0..action_size)
                    } else {
                        // Sample proportionally from the bounded policy distribution.
                        let r: f32 = rng.r#gen::<f32>();
                        let mut cumulative = 0.0f32;
                        let mut sel = action_size - 1;
                        for (a, &p) in probs.iter().enumerate() {
                            cumulative += p;
                            if r <= cumulative {
                                sel = a;
                                break;
                            }
                        }
                        sel
                    };

                    let val_start = env_idx * self.num_value_classes;
                    let val_goodness =
                        &value_flat[val_start..val_start + self.num_value_classes];
                    let v_s = expected_value_from_goodness(
                        val_goodness,
                        self.min_return,
                        self.max_return,
                        self.num_value_classes,
                    );

                    let reward =
                        envs[env_idx].evaluate_action(&raw_states[env_idx], selected_action)
                            as f32;
                    envs[env_idx].apply_action(selected_action)?;

                    raw_total_reward += reward;
                    all_states.push(normalize_state(&raw_states[env_idx]));
                    all_actions.push(selected_action);
                    all_rewards.push(reward);
                    all_values.push(v_s);
                    all_old_probs.push(probs[selected_action].max(1e-8));
                }

                // Visualization hooks (env 0 only).
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
                        let val_probs = softmax_probs(&value_flat[0..self.num_value_classes]);
                        vis.model_probabilities = Some(vec![
                            ("Policy π(a|s)".to_string(), pol_probs),
                            ("Value P(v|s)".to_string(), val_probs),
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
            let bootstrap_scores =
                self.value_model.predict_scores(&[bootstrap_tensor])?;
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

            // GAE per env, then flatten into all_advantages / all_return_targets.
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

            // Normalize advantages.
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
            let mut indices: Vec<usize> = (0..n_total).collect();

            for ppo_epoch in 0..PPO_EPOCHS {
                indices.shuffle(&mut rng);

                // ── Value Update ──
                {
                    let value_labels: Vec<usize> = indices
                        .iter()
                        .map(|&i| {
                            value_to_class(
                                all_return_targets[i],
                                self.min_return,
                                self.max_return,
                                self.num_value_classes,
                            )
                        })
                        .collect();
                    let mut flat_states = Vec::with_capacity(n_total * state_size);
                    for &i in &indices {
                        flat_states.extend_from_slice(&all_states[i]);
                    }
                    let states_tensor =
                        Tensor::from_vec(flat_states, (n_total, state_size), &self.device)?;
                    self.value_model.train(&states_tensor, &value_labels)?;
                }

                // ── Policy Update (per-action binary classification) ──
                // Uses train_policy rather than train. Each action chunk is trained
                // independently as a binary classifier: positive advantage → push
                // goodness up, negative → push goodness down. No other chunks are
                // touched by a given sample, so there is no cross-action suppression
                // and no winner-take-all collapse.
                // Both positive and negative advantage transitions are included;
                // the sign of the advantage determines the push direction.
                {
                    let mut flat_states = Vec::with_capacity(n_total * state_size);
                    let mut batch_actions: Vec<usize> = Vec::with_capacity(n_total);
                    let mut batch_advantages: Vec<f32> = Vec::with_capacity(n_total);

                    for &i in &indices {
                        flat_states.extend_from_slice(&all_states[i]);
                        batch_actions.push(all_actions[i]);
                        batch_advantages.push(all_advantages[i]);
                    }

                    let states_tensor = Tensor::from_vec(
                        flat_states,
                        (n_total, state_size),
                        &self.device,
                    )?;
                    self.policy_model
                        .train_policy(&states_tensor, &batch_actions, &batch_advantages)?;
                    total_training_calls += 1;

                    log::info!(
                        "Ep {} PPO epoch {}: policy binary update on {} transitions",
                        episode, ppo_epoch, n_total
                    );
                }
            }

            // Sync old policy snapshot for the next episode's ratio computation.
            sync_model(&self.policy_model, &self.policy_model_old)?;
            total_training_time += training_start.elapsed();

            // ── Record episode reward ──
            let avg_reward = raw_total_reward / n_total as f32;
            if let Some(ref vs) = vis_state
                && let Ok(mut state) = vs.try_lock()
            {
                state.epoch_rewards.push((episode, avg_reward));
                state.runtime_stats.epoch = episode;
                state.total_epochs = episode_end;
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
                            episode_end = self.start_episode + self.n_episodes;
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
                episode, avg_reward, inf_aps, train_ps, self.min_return, self.max_return
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
