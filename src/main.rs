use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tqdm::Iter;

mod dataset;
mod layer;
mod model;
mod robot;
mod synapse;
mod utils;
mod visualization;

use candle_core::Device;
use dataset::xor::XorDataset;
use model::Model;
use visualization::{RuntimeStats, VisualizationState};

use crate::robot::real_lerobot::LeRobot;

fn parse_args() -> bool {
    let args: Vec<String> = std::env::args().collect();
    args.contains(&"--visualize".to_string()) || args.contains(&"-v".to_string())
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    // let device = Device::Cpu;
    let cpu = Device::Cpu;

    // robot stuff
    let follower = LeRobot::new(
        "/dev/ttyACM0",
        [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
        [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
        [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )
    .ok();
    let leader = LeRobot::new(
        "/dev/ttyACM1",
        [
            0.05982525072754008,
            -0.32366994624387013,
            0.08743690490948142,
            -0.018407769454627854,
            1.6659031356438065,
            -1.0676506283684062,
        ],
        [-1.77, -0.32, -3.0, -3.0, -3.0, -1.07],
        [2.22, 3.0, 0.085, -0.069, 3.0, 0.65],
    )
    .ok();

    if follower.is_none() {
        println!("No physical follower robot connected");
    }
    if leader.is_none() {
        println!("No physical leader robot connected");
    }

    let n_epochs = 100;
    let dt = 0.1;
    let visualize = parse_args();

    println!(
        "Visualization: {}",
        if visualize { "enabled" } else { "disabled" }
    );
    println!("Use --visualize or -v flag to enable visualization");

    // simple XOR dataset
    let ds = XorDataset::new(&device)?;

    let mut model = Model::new(vec![2, 256, 256, 1], &device, dt).unwrap();

    println!(
        "layers len: {}, num_synapses: {}",
        model.layers.len(),
        model.synapses.len()
    );

    // Start visualization if requested
    let vis_handle = if visualize {
        let vis_state = Arc::new(Mutex::new(VisualizationState::new(n_epochs)));

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

    let mut iteration = 0;
    let start_time = Instant::now();

    // training loop: unsupervised Hebbian run for a few epochs:
    // Use tqdm terminal loading bar only if visualization is disabled
    let epoch_iter: Box<dyn Iterator<Item = usize>> = if visualize {
        Box::new(1..=n_epochs)
    } else {
        Box::new((1..=n_epochs).tqdm())
    };

    for epoch in epoch_iter {
        let mut epoch_spike_history: Option<Vec<Vec<f32>>> = None;
        if let Some((_, ref vis_state)) = vis_handle {
            if let Ok(state) = vis_state.lock() {
                if state.selected_layer_id.is_some() && epoch % 5 == 0 {
                    epoch_spike_history = Some(Vec::new());
                }
            }
        }

        for (input, _label) in ds.iter() {
            // Check for pause state
            if let Some((_, ref vis_state)) = vis_handle {
                loop {
                    let is_paused = vis_state
                        .try_lock()
                        .map(|state| state.is_paused)
                        .unwrap_or(false);

                    if !is_paused {
                        break;
                    }

                    // Sleep briefly while paused to avoid busy-waiting
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }

            iteration += 1;

            // Run one processing cycle (40 timesteps)
            for _t in 0..40 {
                model.step(input)?;

                if let Some(history) = epoch_spike_history.as_mut() {
                    if let Some((_, ref vis_state)) = vis_handle {
                        if let Ok(state) = vis_state.lock() {
                            if let Some(layer_id) = state.selected_layer_id {
                                if let Ok(activity) = model.get_layer_activity(layer_id) {
                                    history.push(activity);
                                }
                            }
                        }
                    }
                }
            }

            // Update visualization snapshot every 10 iterations
            if let Some((_, ref vis_state)) = vis_handle
                && iteration % 10 == 0
                && let Ok(mut state) = vis_state.try_lock()
            {
                // Update model structure (preserving animated positions)
                if let Ok(snapshot) = model.get_visualization_snapshot() {
                    state.update_from_snapshot(snapshot);
                }

                // Update runtime stats
                let elapsed = start_time.elapsed().as_secs_f32();
                let speed = if elapsed > 0.0 {
                    iteration as f32 / elapsed
                } else {
                    0.0
                };

                state.runtime_stats = RuntimeStats {
                    epoch,
                    iteration,
                    timestep: iteration * 40,
                    iterations_per_second: speed,
                };
            }

            // Print progress
            if epoch % 50 == 0
                && iteration % 4 == 0
                && let Ok(out) = model.process(input, 1, false, &device)
            {
                println!(
                    "Epoch {}: Input: {:?}, Output: {}",
                    epoch,
                    input.to_device(&cpu),
                    out.final_output.to_device(&cpu)?
                );
            }
        }

        if let Some(history) = epoch_spike_history {
            if let Some((_, ref vis_state)) = vis_handle {
                if let Ok(mut state) = vis_state.lock() {
                    state.epoch_spike_history = Some((epoch, history));
                }
            }
        }
    }

    // Signal visualization to close and wait for thread
    if let Some((handle, vis_state)) = vis_handle {
        println!("Closing visualization...");
        if let Ok(mut state) = vis_state.lock() {
            state.should_close = true;
        }
        let _ = handle.join();
    }

    println!("Done");
    Ok(())
}
