use super::Algorithm;
use crate::environment::Environment;
use crate::models::ff_model::FFModel;
use crate::visualization::{RuntimeStats, VisualizationState};
use candle_core::{Device, Tensor};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct AlgorithmFF1 {
    pub model: FFModel,
    pub n_episodes: usize,
    pub n_steps_per_episode: usize,
    pub epochs_per_episode: usize,
    pub device: Device,
}

impl AlgorithmFF1 {
    pub fn new(
        state_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
        device: Device,
    ) -> Result<Self, Box<dyn Error>> {
        let input_size = state_size + action_size;
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

impl Algorithm for AlgorithmFF1 {
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

        for episode in 1..=self.n_episodes {
            println!("starting episode {}", episode);

            env.reset()?;
            std::thread::sleep(Duration::from_millis(500)); // wait for environment to settle

            let mut episode_data = Vec::new();

            let mut total_reward = 0.0;

            let inference_start = Instant::now();

            for _step in 0..self.n_steps_per_episode {
                let current_state = env.get_state()?;

                let state_f32: Vec<f32> = current_state.iter().map(|&x| x as f32).collect();

                let mut action_inputs = Vec::new();
                for a in 0..action_size {
                    let mut input_vec = vec![0.0; action_size];
                    input_vec[a] = 1.0;
                    input_vec.extend(state_f32.iter().copied());
                    action_inputs.push(Tensor::from_vec(
                        input_vec,
                        (1, action_size + state_size),
                        &self.device,
                    )?);
                    total_inference_actions += 1;
                }

                let best_action_t = self.model.predict(&action_inputs)?;
                let best_action = best_action_t.to_vec1::<u32>()?[0] as usize;

                let mut step_actions = Vec::new();
                for a in 0..action_size {
                    let reward = env.evaluate_action(&current_state, a);
                    step_actions.push((a, action_inputs[a].clone(), reward));
                }

                println!("best action: {}", best_action);

                // Add the true reward for the chosen action to the episode's total reward
                total_reward += env.evaluate_action(&current_state, best_action);

                env.apply_action(best_action)?;
                std::thread::sleep(Duration::from_millis(100)); // give some time for action to take effect

                if let Some(ref vis_state_arc) = vis_state {
                    if let Ok(mut state) = vis_state_arc.try_lock() {
                        let env_state = env.get_state()?;
                        state.environment_state = Some(env_state);
                    }
                }

                // Sort actions by reward (ascending)
                step_actions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

                // Save pairs of (best, worst) to mimic positive vs negative inputs
                // For FF model we just need a positive tensor and negative tensor
                let worst_tensor = step_actions.first().unwrap().1.clone();
                let best_tensor = step_actions.last().unwrap().1.clone();
                episode_data.push((best_tensor, worst_tensor));

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
            } // end of inference steps

            let inference_elapsed = inference_start.elapsed();
            total_inference_time += inference_elapsed;

            if let Some(ref vis_state_arc) = vis_state {
                if let Ok(mut state) = vis_state_arc.try_lock() {
                    state.epoch_rewards.push((episode, total_reward as f32));
                }
            }

            println!("Training phase");

            // Train on collected episode data
            let training_start = Instant::now();
            let mut pos_tensors = Vec::new();
            let mut neg_tensors = Vec::new();

            for (pos_t, neg_t) in &episode_data {
                pos_tensors.push(pos_t.clone());
                neg_tensors.push(neg_t.clone());
            }

            if !pos_tensors.is_empty() {
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

        println!("Training completed.");
        Ok(())
    }
}
