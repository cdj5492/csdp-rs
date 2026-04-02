use super::Algorithm;
use crate::environment::Environment;
use crate::models::ff_model::FFModel;
use crate::visualization::{RuntimeStats, VisualizationState};
use candle_core::{Device, Tensor};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use rand::Rng;

pub struct AlgorithmFF2 {
    pub model: FFModel,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub device: Device,
}

impl AlgorithmFF2 {
    pub fn new(
        state_size: usize,
        action_size: usize, // Required for method signature matching conceptually
        hidden_sizes: Vec<usize>,
        device: Device,
    ) -> Result<Self, Box<dyn Error>> {
        let epochs_per_episode = 100;
        let input_size = action_size + state_size * 2;
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

impl Algorithm for AlgorithmFF2 {
    fn run(
        &mut self,
        env: &mut dyn Environment,
        _visualize: bool,
        vis_state: Option<Arc<Mutex<VisualizationState>>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut total_iteration = 0;
        let start_time = Instant::now();
        let mut total_inference_time = Duration::new(0, 0);
        let mut total_inference_actions: usize = 0;
        let mut total_training_time = Duration::new(0, 0);
        let mut total_epochs: usize = 0;
        let action_size = env.action_size();
        let state_size = env.state_size();

        let _ = action_size;

        let n_envs = 16;
        let mut envs: Vec<Box<dyn Environment>> = vec![env.clone_box()];
        for _ in 1..n_envs {
            envs.push(env.clone_box());
        }

        for episode in 1..=self.n_episodes {
            println!("starting vectorized episode {} (x{})", episode, n_envs);

            for e in envs.iter_mut() {
                e.reset()?;
            }
            std::thread::sleep(Duration::from_millis(50));

            let mut episode_states = vec![Vec::new(); n_envs];
            let mut episode_actions = vec![Vec::new(); n_envs];
            let mut total_rewards = vec![0.0; n_envs];

            let inference_start = Instant::now();
            
            for _step in 0..self.n_steps_per_episode {
                let mut current_states = Vec::new();
                for e in envs.iter_mut() {
                    current_states.push(e.get_state()?);
                }

                let mut all_action_inputs = Vec::new();

                for (env_idx, state) in current_states.iter().enumerate() {
                    let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
                    episode_states[env_idx].push(state_f32.clone());

                    let goal_state: Vec<f32> = if let Some(bounds) = envs[env_idx].state_bounds() {
                        bounds.iter().map(|&x| x as f32).collect()
                    } else {
                        vec![1.0; state_size]
                    };

                    for a in 0..action_size {
                        total_inference_actions += 1;
                        let mut input_vec = vec![0.0f32; action_size];
                        input_vec[a] = 1.0f32;
                        input_vec.extend(state_f32.iter());
                        input_vec.extend(goal_state.iter());

                        all_action_inputs.push(Tensor::from_vec(
                            input_vec,
                            (1, action_size + state_size * 2),
                            &self.device,
                        )?);
                    }
                }

                let best_actions = self.model.predict(&all_action_inputs, action_size)?;

                for (env_idx, action) in best_actions.into_iter().enumerate() {
                    let reward = envs[env_idx].evaluate_action(&current_states[env_idx], action);
                    total_rewards[env_idx] += reward;
                    episode_actions[env_idx].push(action);
                    envs[env_idx].apply_action(action)?;
                }

                std::thread::sleep(Duration::from_millis(10));

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
                    state.epoch_rewards.push((episode, total_rewards[0] as f32));
                }
            }

            let mut train_data = Vec::new();
            let mut rng = rand::thread_rng();

            println!("Pairing and augmenting data across {} envs...", n_envs);
            for env_idx in 0..n_envs {
                let n_states = episode_states[env_idx].len();
                if n_states > 1 {
                    for t in 0..n_states - 1 {
                        let max_m = n_states - t;
                        let m = rng.gen_range(1..max_m);

                        let s_t = &episode_states[env_idx][t];
                        let s_t_plus_m = &episode_states[env_idx][t + m];
                        let a_t = episode_actions[env_idx][t];

                        let mut pos_input = vec![0.0f32; action_size];
                        pos_input[a_t] = 1.0f32;
                        pos_input.extend(s_t.clone());
                        pos_input.extend(s_t_plus_m.clone());

                        let neg_m = rng.gen_range(0..n_states);
                        let mut neg_input1 = vec![0.0f32; action_size];
                        neg_input1[a_t] = 1.0f32;
                        neg_input1.extend(s_t.clone());
                        neg_input1.extend(episode_states[env_idx][neg_m].clone());

                        let mut a_star = rng.gen_range(0..action_size);
                        if a_star == a_t {
                            a_star = (a_star + 1) % action_size;
                        }
                        let mut neg_input2 = vec![0.0f32; action_size];
                        neg_input2[a_star] = 1.0f32;
                        neg_input2.extend(s_t.clone());
                        neg_input2.extend(s_t_plus_m.clone());

                        let p_t = Tensor::from_vec(pos_input, (1, action_size + state_size * 2), &self.device)?;
                        let n1_t = Tensor::from_vec(neg_input1, (1, action_size + state_size * 2), &self.device)?;
                        let n2_t = Tensor::from_vec(neg_input2, (1, action_size + state_size * 2), &self.device)?;
                        
                        train_data.push((p_t.clone(), n1_t));
                        train_data.push((p_t.clone(), n2_t));
                    }
                }
            }

            let training_start = Instant::now();
            let mut pos_tensors = Vec::new();
            let mut neg_tensors = Vec::new();

            for (pos_t, neg_t) in &train_data {
                pos_tensors.push(pos_t.clone());
                neg_tensors.push(neg_t.clone());
            }

            if !pos_tensors.is_empty() {
                println!(
                    "Training with batch size {} over generated trajectories",
                    pos_tensors.len()
                );
                let pos_batch = Tensor::cat(&pos_tensors, 0)?;
                let neg_batch = Tensor::cat(&neg_tensors, 0)?;
                self.model.train(&pos_batch, &neg_batch)?;
                total_iteration += 1;
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
            println!(
                "[Episode {} Tracking Env Reward: {}] Actions/sec: {:.1} | Epochs/sec: {:.2}",
                episode, total_rewards[0], inf_aps, ep_s
            );
        }

        Ok(())
    }
}
