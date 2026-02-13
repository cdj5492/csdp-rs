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
use model::Model;
use visualization::{RuntimeStats, VisualizationState};

use crate::dataset::andor::AndOrDataset;
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

    let n_epochs = 1000;
    let n_timesteps = 40; // number of simulated timesteps per iteration
    let dt = 0.1;
    let visualize = parse_args();

    println!(
        "Visualization: {}",
        if visualize { "enabled" } else { "disabled" }
    );
    println!("Use --visualize or -v flag to enable visualization");

    // simple XOR dataset
    // let ds = XorDataset::new(&device)?;

    // simple AND-OR dataset
    let ds = AndOrDataset::new(&device)?;

    let mut model = Model::new(2, 1, vec![256, 256], &device, dt).unwrap();

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
        if let Some((_, ref vis_state)) = vis_handle
            && let Ok(state) = vis_state.lock()
            && state.selected_layer_id.is_some()
            && epoch % 5 == 0
        {
            epoch_spike_history = Some(Vec::new());
        }

        for (input, label) in ds.iter() {
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

            let label_val = label.flatten_all()?.to_vec1::<f32>()?[0];
            // Create negative samples with 50% probability for inputs that should be positive
            let is_negative_sample = label_val == 1.0 && rand::random::<f32>() < 0.5;
            let final_label = if is_negative_sample { 0.0 } else { label_val };

            for layer in model.layers.iter_mut() {
                layer.set_current_label(final_label);
            }

            // Run one processing cycle
            model.reset()?;
            for _t in 0..n_timesteps {
                // println!("label: {}", label);
                model.step(input, Some(label))?;

                if let Some(history) = epoch_spike_history.as_mut()
                    && let Some((_, ref vis_state)) = vis_handle
                    && let Ok(state) = vis_state.lock()
                    && let Some(layer_id) = state.selected_layer_id
                    && let Ok(activity) = model.get_layer_activity(layer_id)
                {
                    history.push(activity);
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
                    timestep: iteration * n_timesteps,
                    iterations_per_second: speed,
                };
            }

            // Print progress
            if epoch % 50 == 0
                && let Ok(out) = model.process(input, n_timesteps, false, &device)
            {
                println!(
                    "Epoch {}: Input: {}, Output: {}",
                    epoch,
                    input.to_device(&cpu)?,
                    out.final_output.to_device(&cpu)?
                );
            }
        }

        if let Some(history) = epoch_spike_history
            && let Some((_, ref vis_state)) = vis_handle
            && let Ok(mut state) = vis_state.lock()
        {
            state.epoch_spike_history = Some((epoch, history));
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
