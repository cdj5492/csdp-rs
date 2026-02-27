use candle_core::{Device, Tensor};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tqdm::Iter;

mod dataset;
mod layer;
mod models;
mod robot;
mod synapse;
mod utils;
mod visualization;

use models::rl_model1::RLModel1;
use visualization::{RuntimeStats, VisualizationState};

use crate::robot::real_lerobot::LeRobot;

fn parse_args() -> bool {
    let args: Vec<String> = std::env::args().collect();
    args.contains(&"--visualize".to_string()) || args.contains(&"-v".to_string())
}

// Action space: 12 discrete actions (2 for each of the 6 joints: +delta or -delta)
const NUM_ACTIONS: usize = 12;
const ACTION_DELTA: f64 = 0.05; // radians
const NUM_JOINTS: usize = 6;

// Target position we want the robot to reach
const TARGET_POSITION: [f64; NUM_JOINTS] = [0.0, -1.0, 1.0, 0.5, 0.0, 0.5];

fn get_action_tensor(action_idx: usize, device: &Device) -> Result<Tensor, Box<dyn Error>> {
    let mut data = vec![0.0f32; NUM_ACTIONS];
    data[action_idx] = 1.0;
    Ok(Tensor::from_vec(data, (NUM_ACTIONS, 1), device)?)
}

// Given a state and an action index, simulate what the next state would be
// Then compute the reward based on distance to TARGET_POSITION
// Actions 0-5 are +delta for joints 0-5
// Actions 6-11 are -delta for joints 0-5
fn compute_reward(state: &[f64], action_idx: usize) -> f64 {
    let mut next_state = state.to_vec();

    let joint_idx = action_idx % NUM_JOINTS;
    let sign = if action_idx < NUM_JOINTS { 1.0 } else { -1.0 };

    next_state[joint_idx] += sign * ACTION_DELTA;

    let mut dist_sq = 0.0;
    for i in 0..NUM_JOINTS {
        let diff = next_state[i] - TARGET_POSITION[i];
        dist_sq += diff * diff;
    }

    // Negative distance squared => higher reward is better (closer to target)
    -dist_sq
}

fn apply_action(
    robot: &mut Option<LeRobot>,
    state: &[f64],
    action_idx: usize,
) -> Result<(), Box<dyn Error>> {
    if let Some(r) = robot {
        let mut next_state = state.to_vec();
        let joint_idx = action_idx % NUM_JOINTS;
        let sign = if action_idx < NUM_JOINTS { 1.0 } else { -1.0 };
        next_state[joint_idx] += sign * ACTION_DELTA;
        r.set_goal_positions(&next_state)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let cpu = Device::Cpu;

    // robot stuff
    let mut follower = LeRobot::new(
        "/dev/ttyACM0",
        [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
        [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
        [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )
    .ok();

    if follower.is_none() {
        println!("No physical follower robot connected. Will run simulation mode.");
    }
    if let Some(r) = &mut follower {
        r.enable()?;
    }

    let n_episodes = 100;
    let n_steps_per_episode = 10;
    let epochs_per_episode = 5;
    let n_timesteps = 40; // network timesteps per forward pass
    let dt = 0.1;
    let visualize = parse_args();

    println!(
        "Visualization: {}",
        if visualize { "enabled" } else { "disabled" }
    );
    println!("Use --visualize or -v flag to enable visualization");

    // input size: 6 (joint positions), action size: 12 (one-hot)
    let mut model = RLModel1::new(NUM_JOINTS, NUM_ACTIONS, vec![256, 128], &device, dt)
        .expect("Failed to create RLModel1");

    println!(
        "layers len: {}, num_synapses: {}",
        model.layers.len(),
        model.synapses.len()
    );

    // Start visualization if requested
    let vis_handle = if visualize {
        let vis_state = Arc::new(Mutex::new(VisualizationState::new(n_episodes)));

        // Initialize model structure
        if let Ok(mut state) = vis_state.lock() {
            if let Ok(snapshot) = model.get_visualization_snapshot() {
                println!(
                    "Initial snapshot: {} layers, {} synapses",
                    snapshot.layers.len(),
                    snapshot.synapses.len()
                );
                state.update_from_snapshot(snapshot);
            } else {
                println!("Warning: Failed to get initial visualization snapshot");
            }
        }

        let handle = visualization::start_visualization(vis_state.clone());
        Some((handle, vis_state))
    } else {
        None
    };

    let mut total_iteration = 0;
    let start_time = Instant::now();

    let episode_iter: Box<dyn Iterator<Item = usize>> = if visualize {
        Box::new(1..=n_episodes)
    } else {
        Box::new((1..=n_episodes).tqdm())
    };

    // Keep track of our simulated state if robot is not connected
    let mut sim_state = vec![0.0; NUM_JOINTS];

    for episode in episode_iter {
        println!("starting episode {}", episode);

        // Reset environment
        if let Some(r) = &mut follower {
            r.go_to_home_positions()?;
        }
        sim_state.fill(0.0);
        std::thread::sleep(Duration::from_millis(500)); // let robot settle

        let mut episode_data = Vec::new();

        model.disable_learning(); // inference mode

        for step in 0..n_steps_per_episode {
            // Get state
            let current_state = if let Some(r) = &mut follower {
                r.get_motor_positions()?
            } else {
                sim_state.clone()
            };

            let state_f32: Vec<f32> = current_state.iter().map(|&x| x as f32).collect();
            let state_tensor = Tensor::from_vec(state_f32, (NUM_JOINTS, 1), &device)?;

            let mut step_actions = Vec::new();
            let mut best_activity = -1.0;
            let mut best_action = 0;

            // Try all actions
            for a in 0..NUM_ACTIONS {
                let action_tensor = get_action_tensor(a, &device)?;
                let activity = model.process(&state_tensor, &action_tensor, n_timesteps)?;
                let reward = compute_reward(&current_state, a);

                step_actions.push((a, action_tensor, reward));

                if activity > best_activity {
                    best_activity = activity;
                    best_action = a;
                }
            }

            println!("best action: {}", best_action);

            // Execute best action
            apply_action(&mut follower, &current_state, best_action)?;
            if follower.is_none() {
                // update sim state
                let sign = if best_action < NUM_JOINTS { 1.0 } else { -1.0 };
                sim_state[best_action % NUM_JOINTS] += sign * ACTION_DELTA;
            }
            std::thread::sleep(Duration::from_millis(100)); // give some time for action to take effect physically

            // Sort actions by reward (ascending)
            step_actions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

            // Assign Ytype proportional to position
            let num_choices = step_actions.len() as f32;
            for (i, (_a, action_tensor, _reward)) in step_actions.into_iter().enumerate() {
                // Ytype between 0.0 (worst) and 1.0 (best)
                let ytype = i as f32 / (num_choices - 1.0);
                episode_data.push((state_tensor.clone(), action_tensor, ytype));
            }

            // Visualization check during inference
            if let Some((_, ref vis_state)) = vis_handle {
                loop {
                    let is_paused = vis_state
                        .try_lock()
                        .map(|state| state.is_paused)
                        .unwrap_or(false);
                    if !is_paused {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        } // end of inference steps

        // Train on collected episode data
        model.enable_learning();

        for _epoch in 0..epochs_per_episode {
            for (state_t, action_t, ytype) in &episode_data {
                total_iteration += 1;

                for layer in model.layers.iter_mut() {
                    layer.set_positive_sample(*ytype);
                }

                model.process(state_t, action_t, n_timesteps)?;

                // Update visualization snapshot every 20 iterations during training
                if let Some((_, ref vis_state)) = vis_handle
                    && total_iteration % 20 == 0
                    && let Ok(mut state) = vis_state.try_lock()
                {
                    if let Ok(snapshot) = model.get_visualization_snapshot() {
                        state.update_from_snapshot(snapshot);
                    }
                    let elapsed = start_time.elapsed().as_secs_f32();
                    let speed = if elapsed > 0.0 {
                        total_iteration as f32 / elapsed
                    } else {
                        0.0
                    };
                    state.runtime_stats = RuntimeStats {
                        epoch: episode,
                        iteration: total_iteration,
                        timestep: total_iteration * n_timesteps,
                        iterations_per_second: speed,
                    }
                }
            }
        }
    } // end of episodes

    // Signal visualization to close and wait for thread
    if let Some((handle, vis_state)) = vis_handle {
        println!("Closing visualization...");
        if let Ok(mut state) = vis_state.lock() {
            state.should_close = true;
        }
        let _ = handle.join();
    }

    if let Some(mut r) = follower {
        r.disable().unwrap_or(());
    }

    println!("Done");
    Ok(())
}
