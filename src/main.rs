#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_core::Device;
use std::error::Error;
use std::sync::{Arc, Mutex};

use custom_framework::algorithms;
use custom_framework::dataset;
use custom_framework::environment;
use custom_framework::layer;
use custom_framework::models;
use custom_framework::robot;
use custom_framework::synapse;
use custom_framework::utils;
use custom_framework::visualization;

use algorithms::Algorithm;
use algorithms::algorithm_ff1::AlgorithmFF1;
use algorithms::algorithm_ff2::AlgorithmFF2;
use algorithms::algorithm_ff3::AlgorithmFF3;
use algorithms::algorithm1::Algorithm1;
use algorithms::algorithm2::Algorithm2;
use algorithms::algorithm3::Algorithm3;
use environment::Environment;
use visualization::VisualizationState;

fn parse_args() -> (bool, bool, usize, bool) {
    let args: Vec<String> = std::env::args().collect();
    let visualize = args.contains(&"--visualize".to_string()) || args.contains(&"-v".to_string());
    let grid = args.contains(&"--grid".to_string());

    let infinite_epochs = args.contains(&"--infinite-epochs".to_string());

    let mut algo = 2;
    if let Some(idx) = args.iter().position(|r| r == "--algo") {
        if idx + 1 < args.len() {
            if let Ok(val) = args[idx + 1].parse::<usize>() {
                algo = val;
            }
        }
    }

    (visualize, grid, algo, infinite_epochs)
}

fn main() -> Result<(), Box<dyn Error>> {
    // let device = Device::Cpu;
    let device = Device::new_cuda(0)?;

    let (visualize, use_grid, algo_choice, infinite_epochs) = parse_args();

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
    let state_bounds = env.state_bounds();
    let dt = 0.1;

    let mut algo1_opt = None;
    let mut algo2_opt = None;
    let mut algo3_opt = None;
    let mut algo_ff1_opt = None;
    let mut algo_ff2_opt = None;
    let mut algo_ff3_opt = None;

    let (n_episodes, snapshot_result, num_layers, num_synapses) = if algo_choice == 1 {
        let mut algo = Algorithm1::new(
            state_size,
            action_size,
            vec![256, 128],
            dt,
            device,
            state_bounds,
        )
        .expect("Failed to create Algorithm1");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = algo.model.get_visualization_snapshot();
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = algo.model.synapses.len();
        algo1_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == 2 {
        let mut algo = Algorithm2::new(
            state_size,
            action_size,
            vec![256, 128],
            dt,
            device,
            state_bounds,
        )
        .expect("Failed to create Algorithm2");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = algo.model.get_visualization_snapshot();
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = algo.model.synapses.len();
        algo2_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == 3 {
        println!("Using Algorithm 3 (AC-CSDP)");
        let mut algo = Algorithm3::new(
            state_size,
            action_size,
            vec![256, 128],
            dt,
            device.clone(),
            state_bounds.clone(),
        )
        .expect("Failed to create Algorithm3");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = algo.model.actor.get_visualization_snapshot();
        let eps = algo.n_episodes;
        let layers = algo.model.actor.layers.len() + algo.model.critic.layers.len();
        let syns = algo.model.actor.synapses.len() + algo.model.critic.synapses.len();
        algo3_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == 4 {
        println!("Using Algorithm 4 (FF Model - State/Action Iterator)");
        let mut algo = AlgorithmFF1::new(state_size, action_size, vec![256, 128], device.clone())
            .expect("Failed to create AlgorithmFF1");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = Err(candle_core::Error::Msg(
            "FF Model has no visualization".to_string(),
        ));
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = 0;
        algo_ff1_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == 5 {
        println!("Using Algorithm 5 (FF Model - Transition Evaluator)");
        let mut algo = AlgorithmFF2::new(state_size, action_size, vec![256, 128], device.clone())
            .expect("Failed to create AlgorithmFF2");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = Err(candle_core::Error::Msg(
            "FF Model has no visualization".to_string(),
        ));
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = 0;
        algo_ff2_opt = Some(algo);
        (eps, snap, layers, syns)
    } else {
        println!("Using Algorithm 6 (FF Model - Probabilistic Rank Trajectory)");
        let mut algo = AlgorithmFF3::new(state_size, action_size, vec![256, 128], device.clone())
            .expect("Failed to create AlgorithmFF3");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = Err(candle_core::Error::Msg(
            "FF Model has no visualization".to_string(),
        ));
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = 0;
        algo_ff3_opt = Some(algo);
        (eps, snap, layers, syns)
    };

    println!("layers len: {}, num_synapses: {}", num_layers, num_synapses);

    // Start visualization if requested
    let vis_handle = if visualize {
        let vis_state = Arc::new(Mutex::new(VisualizationState::new(n_episodes)));

        // Initialize model structure
        if let Ok(mut state) = vis_state.lock() {
            if let Ok(snapshot) = snapshot_result {
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

    if let Some(mut algo) = algo1_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo2_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo3_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ff1_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ff2_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ff3_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    }

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
