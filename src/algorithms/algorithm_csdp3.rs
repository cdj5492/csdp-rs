use super::Algorithm;
use crate::environment::Environment;
use crate::models::rl_model3::RLModel3;
use crate::visualization::{RuntimeStats, VisualizationState};
use candle_core::{Device, Tensor};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct Algorithm3 {
    pub model: RLModel3,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub n_timesteps: usize,
    pub noise_std: f32,
    pub gamma: f32,
    pub critic_baseline: f32,
    pub critic_baseline_alpha: f32,
    pub device: Device,
}

impl Algorithm3 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        dt: f32,
        device: Device,
        state_bounds: Option<Vec<usize>>,
    ) -> Result<Self, Box<dyn Error>> {
        let model = RLModel3::new(
            state_size,
            action_size,
            hidden_sizes.clone(),
            hidden_sizes,
            &device,
            dt,
            state_bounds,
        )
        .ok_or("Failed to create RLModel3")?;

        Ok(Self {
            model,
            n_episodes: 100,
            n_steps_per_episode: 50,
            n_timesteps: 40,
            noise_std: 0.5,
            gamma: 0.95,
            critic_baseline: 0.0,
            critic_baseline_alpha: 0.1,
            device,
        })
    }

    fn run_actor_pass(
        &mut self,
        state_tensor: &Tensor,
        noise_sequence: Option<&Vec<Tensor>>,
        collect_z: bool,
        label: Option<f32>,
        record_layer: Option<usize>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>), Box<dyn Error>> {
        if let Some(lbl) = label {
            let label_tensor = Tensor::from_vec(vec![lbl], (1, 1), &self.device)?;
            for layer in self.model.actor.layers.iter_mut() {
                layer.set_positive_sample(&label_tensor);
            }
        }

        self.model.actor.reset(1)?;
        let action_size = self.model.actor.layers.last().unwrap().size();
        let mut z_vec = vec![0.0; action_size];
        let mut spike_history = Vec::new();

        for t in 0..self.n_timesteps {
            // Replicate Model::step but with optional noise on last layer
            for layer in self.model.actor.layers.iter_mut() {
                layer.reset_input()?;
            }

            self.model.actor.layers[0].add_input(state_tensor)?;
            self.model.actor.layers[0].step(self.model.dt)?;

            // Context layer is index 1, skip stepping if not used, or step with zero
            // Model::new adds a Bernoulli context layer.
            self.model.actor.layers[1].step(self.model.dt)?;

            for syn_conn in self.model.actor.synapses.iter_mut() {
                let pre_id = syn_conn.metadata.pre_layer;
                let post_id = syn_conn.metadata.post_layer;
                let pre_act = self.model.actor.layers[pre_id].output()?.clone();
                let post_in = syn_conn.synapse.forward(&pre_act)?;
                self.model.actor.layers[post_id].add_input(&post_in)?;
            }

            if let Some(seq) = noise_sequence {
                self.model
                    .actor
                    .layers
                    .last_mut()
                    .unwrap()
                    .add_input(&seq[t])?;
            }

            for layer in self.model.actor.layers.iter_mut().skip(2) {
                layer.step(self.model.dt)?;
            }

            // Synapse updates
            for syn_conn in self.model.actor.synapses.iter_mut() {
                if self.model.actor.is_learning && syn_conn.metadata.is_learning {
                    let pre_id = syn_conn.metadata.pre_layer;
                    let post_id = syn_conn.metadata.post_layer;
                    let pre_act = self.model.actor.layers[pre_id].output()?.clone();
                    syn_conn.synapse.update_weights(
                        &pre_act,
                        &mut self.model.actor.layers[post_id],
                        self.model.dt,
                    )?;
                }
            }

            if collect_z {
                let spikes = self.model.actor.layers.last().unwrap().output()?;
                let spikes_vec = spikes.flatten_all()?.to_vec1::<f32>()?;
                for i in 0..action_size {
                    z_vec[i] += spikes_vec[i];
                }
            }
            if let Some(layer_id) = record_layer {
                if let Ok(activity) = self.model.actor.get_layer_activity(layer_id) {
                    spike_history.push(activity);
                }
            }
        }

        Ok((z_vec, spike_history))
    }

    fn evaluate_critic(
        &mut self,
        state_vec: &[f32],
        action_vec: &[f32],
        label: Option<f32>,
    ) -> Result<f32, Box<dyn Error>> {
        if let Some(lbl) = label {
            let label_tensor = Tensor::from_vec(vec![lbl], (1, 1), &self.device)?;
            for layer in self.model.critic.layers.iter_mut() {
                layer.set_positive_sample(&label_tensor);
            }
        }

        let mut input_vec = Vec::with_capacity(state_vec.len() + action_vec.len());
        // Scale inputs removed: the true cause of runaway was the adaptive threshold bug.
        input_vec.extend_from_slice(state_vec);
        input_vec.extend_from_slice(action_vec);
        let input_tensor = Tensor::from_vec(
            input_vec,
            (state_vec.len() + action_vec.len(), 1),
            &self.device,
        )?;

        self.model.critic.reset(1)?;
        let mut q_value = 0.0;

        for _ in 0..self.n_timesteps {
            self.model.critic.step(&input_tensor, None)?;

            // Sum activity of the Critic's output layer as the Q-value
            if label.is_none() {
                let spikes = self.model.critic.layers.last().unwrap().output()?;
                let spikes_vec = spikes.flatten_all()?.to_vec1::<f32>()?;
                q_value += spikes_vec.iter().sum::<f32>();
            }
        }

        Ok(q_value)
    }
}

impl Algorithm for Algorithm3 {
    fn run(
        &mut self,
        env: &mut dyn Environment,
        _visualize: bool,
        vis_state: Option<Arc<Mutex<VisualizationState>>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut total_iteration = 0;
        let start_time = Instant::now();
        let action_size = env.action_size();
        let state_size = env.state_size();

        for episode in 1..=self.n_episodes {
            if let Some(ref vis_state_arc) = vis_state {
                if vis_state_arc.try_lock().map(|s| s.should_close).unwrap_or(false) {
                    return Ok(());
                }
            }
            log::info!("starting episode {}", episode);
            env.reset()?;

            let mut total_reward = 0.0;

            for step in 0..self.n_steps_per_episode {
                if let Some(ref vis_state_arc) = vis_state {
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
                }
                
                total_iteration += 1;
                let current_state = env.get_state()?;
                let state_raw_f32: Vec<f32> = current_state.iter().map(|&x| x as f32).collect();
                let state_f32: Vec<f32> = if let Some(bounds) = env.state_bounds() {
                    current_state
                        .iter()
                        .zip(bounds.iter())
                        .map(|(&val, &max)| (val as f32) / (max as f32))
                        .collect()
                } else {
                    current_state.iter().map(|&x| x as f32).collect()
                };
                // Actor now uses OneHotLayer if bounds are passed, so it expects raw values
                let state_tensor =
                    Tensor::from_vec(state_raw_f32.clone(), (state_size, 1), &self.device)?;

                let record_layer = vis_state
                    .as_ref()
                    .and_then(|vs| vs.try_lock().ok().and_then(|s| s.selected_layer_id));

                // 1. Baseline & Perturbed Generation
                self.model.disable_learning();

                let (z_base, spike_history) =
                    self.run_actor_pass(&state_tensor, None, true, None, record_layer)?;

                let mut noise_sequence = Vec::with_capacity(self.n_timesteps);
                for _ in 0..self.n_timesteps {
                    noise_sequence.push(Tensor::randn(
                        0.0f32,
                        self.noise_std,
                        (action_size, 1),
                        &self.device,
                    )?);
                }

                let (z_pert, _) =
                    self.run_actor_pass(&state_tensor, Some(&noise_sequence), true, None, None)?;

                // Normalize z factors to probability ratios [0, 1] for the critic input
                let z_base_norm: Vec<f32> = z_base
                    .iter()
                    .map(|&z| z / self.n_timesteps as f32)
                    .collect();
                let z_pert_norm: Vec<f32> = z_pert
                    .iter()
                    .map(|&z| z / self.n_timesteps as f32)
                    .collect();

                // 2. Critic Evaluation
                let q_base = self.evaluate_critic(&state_f32, &z_base_norm, None)?;
                let q_pert = self.evaluate_critic(&state_f32, &z_pert_norm, None)?;

                log::info!(
                    "Step {}: q_base = {:.2}, q_pert = {:.2}",
                    step, q_base, q_pert
                );

                // 3. Contrastive Label Assignment for Actor
                self.model.actor.is_learning = true;

                let (best_z, _was_pert_better) = if q_pert > q_base {
                    // Perturbation was better
                    self.run_actor_pass(
                        &state_tensor,
                        Some(&noise_sequence),
                        false,
                        Some(1.0),
                        None,
                    )?; // Positive
                    self.run_actor_pass(&state_tensor, None, false, Some(0.0), None)?; // Negative
                    (z_pert, true)
                } else {
                    // Baseline was better
                    self.run_actor_pass(&state_tensor, None, false, Some(1.0), None)?; // Positive
                    self.run_actor_pass(
                        &state_tensor,
                        Some(&noise_sequence),
                        false,
                        Some(0.0),
                        None,
                    )?; // Negative
                    (z_base, false)
                };

                // Pick action based on the chosen z (continuous -> discrete)
                let best_action = best_z
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap_or(0);

                let reward = env.evaluate_action(&current_state, best_action);
                total_reward += reward;
                env.apply_action(best_action)?;

                // 4. Critic Training
                // To train Critic, we need TD Target. We estimate Q(s_{t+1}, a_{t+1})
                let next_state_opt = if step < self.n_steps_per_episode - 1 {
                    let ns = env.get_state()?;
                    let ns_raw_f32: Vec<f32> = ns.iter().map(|&x| x as f32).collect();
                    let ns_f32: Vec<f32> = if let Some(bounds) = env.state_bounds() {
                        ns.iter()
                            .zip(bounds.iter())
                            .map(|(&val, &max)| (val as f32) / (max as f32))
                            .collect()
                    } else {
                        ns.iter().map(|&x| x as f32).collect()
                    };
                    let ns_tensor =
                        Tensor::from_vec(ns_raw_f32.clone(), (state_size, 1), &self.device)?;
                    self.model.disable_learning();
                    let (ns_z, _) = self.run_actor_pass(&ns_tensor, None, true, None, None)?;
                    let ns_z_norm: Vec<f32> =
                        ns_z.iter().map(|&z| z / self.n_timesteps as f32).collect();
                    let next_q = self.evaluate_critic(&ns_f32, &ns_z_norm, None)?;
                    Some(next_q)
                } else {
                    None
                };

                let td_target = reward as f32
                    + if let Some(nq) = next_state_opt {
                        self.gamma * nq
                    } else {
                        0.0
                    };

                let best_z_norm: Vec<f32> = best_z
                    .iter()
                    .map(|&z| z / self.n_timesteps as f32)
                    .collect();

                self.model.critic.is_learning = true;
                if td_target > self.critic_baseline {
                    self.evaluate_critic(&state_f32, &best_z_norm, Some(1.0))?;
                } else {
                    self.evaluate_critic(&state_f32, &best_z_norm, Some(0.0))?;
                }

                self.critic_baseline = (1.0 - self.critic_baseline_alpha) * self.critic_baseline
                    + self.critic_baseline_alpha * td_target;

                if let Some(ref vis_state_arc) = vis_state {
                    if let Ok(mut state) = vis_state_arc.try_lock() {
                        let env_state = env.get_state()?;
                        state.environment_state = Some(env_state);

                        if !spike_history.is_empty() {
                            state.epoch_spike_history = Some((episode, spike_history.clone()));
                        }

                        // Update visualization snapshot
                        if !state.positions_initialized || state.selected_layer_id.is_some() {
                            if let Ok(snapshot) = self.model.actor.get_visualization_snapshot() {
                                state.update_from_snapshot(snapshot);
                            }
                        }

                        // Always update stat speed
                        if total_iteration % 20 == 0 {
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

            if let Some(ref vis_state_arc) = vis_state {
                if let Ok(mut state) = vis_state_arc.try_lock() {
                    state.epoch_rewards.push((episode, total_reward as f32));
                }
            }
            log::info!("Episode {} total reward: {}", episode, total_reward);
        }

        let checkpoints_dir = std::path::Path::new("checkpoints");
        if !checkpoints_dir.exists() {
            std::fs::create_dir_all(checkpoints_dir)?;
        }
        let final_acc_path = checkpoints_dir.join("actor_final.safetensors");
        let final_crit_path = checkpoints_dir.join("critic_final.safetensors");
        let _ = self.model.actor.save(&final_acc_path);
        let _ = self.model.critic.save(&final_crit_path);

        if let Some(ref vis_state_arc) = vis_state {
            if let Ok(state) = vis_state_arc.try_lock() {
                let csv_path = checkpoints_dir.join("epoch_rewards.csv");
                let _ = state.save_graphs_to_csv(&csv_path);
            }
        }

        Ok(())
    }
}
