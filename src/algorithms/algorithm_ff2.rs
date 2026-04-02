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
        let input_size = state_size * 2 + 1;
        let mut dims = vec![input_size];
        dims.extend(hidden_sizes);
        let model = FFModel::new(&dims, &device).expect("Failed to create FFModel");

        Ok(Self {
            model,
            n_episodes: 100,
            n_steps_per_episode: 50,
            epochs_per_episode: 5,
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

        // Used internally for creating inference tensors
        let _ = action_size;

        for episode in 1..=self.n_episodes {
            println!("starting episode {}", episode);

            env.reset()?;
            std::thread::sleep(Duration::from_millis(500));

            let mut episode_states = Vec::new();
            let mut episode_actions = Vec::new();
            let mut episode_rewards = Vec::new();

            let mut total_reward = 0.0;
            let inference_start = Instant::now();

            for _step in 0..self.n_steps_per_episode {
                let current_state = env.get_state()?;

                let state_f32: Vec<f32> = current_state.iter().map(|&x| x as f32).collect();
                episode_states.push(state_f32.clone());

                let mut action_inputs = Vec::new();
                for a in 0..action_size {
                    total_inference_actions += 1;
                    let mut input_vec = Vec::with_capacity(state_size * 2 + 1);
                    input_vec.extend(state_f32.iter());
                    input_vec.extend(vec![0.0f32; state_size]);
                    input_vec.push(a as f32);

                    action_inputs.push(Tensor::from_vec(
                        input_vec,
                        (1, state_size * 2 + 1),
                        &self.device,
                    )?);
                }

                let best_action_t = self.model.predict(&action_inputs)?;
                let best_action = best_action_t.to_vec1::<u32>()?[0] as usize;

                println!("best action: {}", best_action);
                let reward = env.evaluate_action(&current_state, best_action);

                episode_actions.push(best_action);
                episode_rewards.push(reward);

                total_reward += reward;

                env.apply_action(best_action)?;
                std::thread::sleep(Duration::from_millis(100));

                if let Some(ref vis_state_arc) = vis_state {
                    if let Ok(mut state) = vis_state_arc.try_lock() {
                        let env_state = env.get_state()?;
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
                    state.epoch_rewards.push((episode, total_reward as f32));
                }
            }

            let mut train_data = Vec::new();
            let mut rng = rand::thread_rng();
            let n_states = episode_states.len();

            println!("Pairing and augmenting data...");
            if n_states > 1 {
                for t in 0..n_states - 1 {
                    let max_m = n_states - t;
                    let m = rng.gen_range(1..max_m);

                    let s_t = &episode_states[t];
                    let s_t_plus_m = &episode_states[t + m];
                    let a_t = episode_actions[t];

                    let mut pos_input = Vec::with_capacity(state_size * 2 + 1);
                    pos_input.extend(s_t.clone());
                    pos_input.extend(s_t_plus_m.clone());
                    pos_input.push(a_t as f32);

                    let neg_m = rng.gen_range(0..n_states);
                    let mut neg_input1 = Vec::with_capacity(state_size * 2 + 1);
                    neg_input1.extend(s_t.clone());
                    neg_input1.extend(episode_states[neg_m].clone());
                    neg_input1.push(a_t as f32);

                    let mut a_star = rng.gen_range(0..action_size);
                    if a_star == a_t {
                        a_star = (a_star + 1) % action_size;
                    }
                    let mut neg_input2 = Vec::with_capacity(state_size * 2 + 1);
                    neg_input2.extend(s_t.clone());
                    neg_input2.extend(s_t_plus_m.clone());
                    neg_input2.push(a_star as f32);

                    let p_t = Tensor::from_vec(pos_input, (1, state_size * 2 + 1), &self.device)?;
                    let n1_t = Tensor::from_vec(neg_input1, (1, state_size * 2 + 1), &self.device)?;
                    let n2_t = Tensor::from_vec(neg_input2, (1, state_size * 2 + 1), &self.device)?;
                    
                    train_data.push((p_t.clone(), n1_t));
                    train_data.push((p_t.clone(), n2_t));
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
                "[Episode {}] Actions/sec: {:.1} | Epochs/sec: {:.2}",
                episode, inf_aps, ep_s
            );
        }

        Ok(())
    }
}
