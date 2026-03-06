use candle_core::Device;
use std::error::Error;
use std::sync::{Arc, Mutex};

mod algorithms;
mod dataset;
mod environment;
mod layer;
mod models;
mod robot;
mod synapse;
mod utils;
mod visualization;

use algorithms::Algorithm;
use algorithms::algorithm2::Algorithm2;
use environment::Environment;
use visualization::VisualizationState;

fn parse_args() -> (bool, bool) {
    let args: Vec<String> = std::env::args().collect();
    let visualize = args.contains(&"--visualize".to_string()) || args.contains(&"-v".to_string());
    let grid = args.contains(&"--grid".to_string());
    (visualize, grid)
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::Cpu;

    let (visualize, use_grid) = parse_args();

    let mut env: Box<dyn Environment> = if use_grid {
        println!("Using Grid Environment.");
        Box::new(environment::grid::GridEnvironment::new())
    } else {
        match environment::robot::RobotEnvironment::new() {
            Ok(robot_env) => {
                println!("Using physical Robot Environment.");
                Box::new(robot_env)
            }
            Err(e) => {
                println!(
                    "Failed to construct RobotEnvironment: {}. Falling back to Grid Environment.",
                    e
                );
                Box::new(environment::grid::GridEnvironment::new())
            }
        }
    };

    println!(
        "Visualization: {}",
        if visualize { "enabled" } else { "disabled" }
    );
    println!("Use --visualize or -v flag to enable visualization");

    let state_size = env.state_size();
    let action_size = env.action_size();
    let dt = 0.1;

    let mut algo = Algorithm2::new(state_size, action_size, vec![256, 128], dt, device)
        .expect("Failed to create Algorithm2");

    println!(
        "layers len: {}, num_synapses: {}",
        algo.model.layers.len(),
        algo.model.synapses.len()
    );

    let n_episodes = algo.n_episodes;

    // Start visualization if requested
    let vis_handle = if visualize {
        let vis_state = Arc::new(Mutex::new(VisualizationState::new(n_episodes)));

        // Initialize model structure
        if let Ok(mut state) = vis_state.lock() {
            if let Ok(snapshot) = algo.model.get_visualization_snapshot() {
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

    let vis_state_arg = vis_handle.as_ref().map(|(_, state)| state.clone());

    algo.run(env.as_mut(), visualize, vis_state_arg)?;

    // Signal visualization to close and wait for thread
    if let Some((handle, vis_state)) = vis_handle {
        println!("Closing visualization...");
        if let Ok(mut state) = vis_state.lock() {
            state.should_close = true;
        }
        let _ = handle.join();
    }

    // drop env cleans up whatever depends on drops, e.g. RobotEnvironment::disable()
    drop(env);

    println!("Done");
    Ok(())
}
