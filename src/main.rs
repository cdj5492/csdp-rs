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
use algorithms::algorithm_csdp1::Algorithm1;
use algorithms::algorithm_csdp2::Algorithm2;
use algorithms::algorithm_csdp3::Algorithm3;
use algorithms::algorithm_csdp4::Algorithm4;
use algorithms::algorithm_csdp5::AlgorithmCSDP5;
use algorithms::algorithm_ff_multi1::AlgorithmFFMulti1;
use algorithms::algorithm_ff_multi2::AlgorithmFFMulti2;
use algorithms::algorithm_ff1::AlgorithmFF1;
use algorithms::algorithm_ff2::AlgorithmFF2;
use algorithms::algorithm_ff3::AlgorithmFF3;
use algorithms::algorithm_ff4::AlgorithmFF4;
use algorithms::algorithm_ffsac::AlgorithmFFSAC;
use environment::Environment;
use visualization::VisualizationState;

struct VisLogger;
impl log::Log for VisLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let msg = format!("[{}] {}", record.level(), record.args());
            if let Ok(mut logs) = custom_framework::visualization::GLOBAL_LOGS.lock() {
                logs.push(msg);
                if logs.len() > 100 {
                    logs.remove(0);
                }
            }
            // Optional: Also print to stderr if visualizing? It would disrupt Ratatui, so no.
        }
    }
    fn flush(&self) {}
}
static LOGGER: VisLogger = VisLogger;

fn parse_args() -> (bool, String, String, bool, bool) {
    let args: Vec<String> = std::env::args().collect();
    let visualize = args.contains(&"--visualize".to_string()) || args.contains(&"-v".to_string());

    let mut env_type = "robot".to_string();
    if args.contains(&"--grid".to_string()) {
        env_type = "grid".to_string();
    } else if args.contains(&"--rocketsim".to_string()) {
        env_type = "rocketsim".to_string();
    }

    let infinite_epochs = args.contains(&"--infinite-epochs".to_string());
    let resume = args.contains(&"--resume".to_string());

    let mut algo = "csdp2".to_string();
    if let Some(idx) = args.iter().position(|r| r == "--algo") {
        if idx + 1 < args.len() {
            algo = args[idx + 1].clone();
        }
    }

    (visualize, env_type, algo, infinite_epochs, resume)
}

fn main() -> Result<(), Box<dyn Error>> {
    // let device = Device::Cpu;
    let device = Device::new_cuda(0)?;

    let (visualize, env_type, algo_choice, infinite_epochs, resume) = parse_args();

    if visualize {
        let _ = log::set_logger(&LOGGER);
        log::set_max_level(log::LevelFilter::Info);
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    let mut env: Box<dyn Environment> = if env_type == "grid" {
        log::info!("Using Grid Environment.");
        Box::new(environment::grid::GridEnvironment::new())
    } else if env_type == "rocketsim" {
        log::info!("Using RocketSim Environment.");
        Box::new(environment::rocketsim::RocketSimEnvironment::new(5)) // tickskip=5
    } else {
        match environment::robot::RobotEnvironment::new() {
            Ok(robot_env) => {
                log::info!("Using physical Robot Environment.");
                Box::new(robot_env)
            }
            Err(e) => {
                log::info!(
                    "Failed to construct RobotEnvironment: {}. Falling back to Grid Environment.",
                    e
                );
                Box::new(environment::grid::GridEnvironment::new())
            }
        }
    };

    log::info!(
        "Visualization: {}",
        if visualize { "enabled" } else { "disabled" }
    );
    log::info!("Use --visualize or -v flag to enable visualization");

    let state_size = env.state_size();
    let action_size = env.action_size();
    let state_bounds = env.state_bounds();
    let dt = 0.1;

    let mut algo1_opt = None;
    let mut algo2_opt = None;
    let mut algo3_opt = None;
    let mut algo4_opt = None;
    let mut algo5_opt = None;
    let mut algo_ff1_opt = None;
    let mut algo_ff2_opt = None;
    let mut algo_ff3_opt = None;
    let mut algo_ff4_opt = None;
    let mut algo_ffsac_opt = None;
    let mut algo_ff_multi1_opt = None;
    let mut algo_ff_multi2_opt = None;

    let (n_episodes, snapshot_result, num_layers, num_synapses) = if algo_choice == "csdp1" {
        log::info!("Using Algorithm CSDP1");
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
    } else if algo_choice == "csdp2" {
        log::info!("Using Algorithm CSDP2");
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
    } else if algo_choice == "csdp3" {
        log::info!("Using Algorithm CSDP3 (AC-CSDP)");
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
    } else if algo_choice == "csdp4" {
        log::info!("Using Algorithm CSDP4");
        let mut algo = Algorithm4::new(
            state_size,
            action_size,
            vec![256, 128],
            dt,
            device.clone(),
            state_bounds.clone(),
        )
        .expect("Failed to create Algorithm4");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = algo.model.get_visualization_snapshot();
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = algo.model.synapses.len();
        algo4_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == "csdp5" {
        log::info!("Using Algorithm CSDP5 (Multi-Class MC SNN)");
        let mut algo = AlgorithmCSDP5::new(
            state_size,
            action_size,
            vec![2000],
            dt,
            device.clone(),
            state_bounds.clone(),
        )
        .expect("Failed to create AlgorithmCSDP5");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = algo.model.get_visualization_snapshot();
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = algo.model.synapses.len();
        algo5_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == "ff1" {
        log::info!("Using Algorithm FF1 (FF Model - State/Action Iterator)");
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
    } else if algo_choice == "ff2" {
        log::info!("Using Algorithm FF2 (FF Model - Transition Evaluator)");
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
    } else if algo_choice == "ff3" {
        log::info!("Using Algorithm FF3 (FF Model - Probabilistic Rank Trajectory)");
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
    } else if algo_choice == "ff4" {
        log::info!("Using Algorithm FF4 (FF Model - Temporal Contrastive RL)");
        let mut algo = AlgorithmFF4::new(state_size, action_size, vec![256, 128], device.clone())
            .expect("Failed to create AlgorithmFF4");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = Err(candle_core::Error::Msg(
            "FF Model has no visualization".to_string(),
        ));
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = 0;
        algo_ff4_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == "ffsac" {
        log::info!("Using Algorithm FFSAC (FF Model - Soft Actor-Critic)");
        let mut algo = AlgorithmFFSAC::new(state_size, action_size, vec![256, 128], device.clone())
            .expect("Failed to create AlgorithmFFSAC");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = Err(candle_core::Error::Msg(
            "FF Model has no visualization".to_string(),
        ));
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = 0;
        algo_ffsac_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == "ff_multi1" {
        log::info!("Using Algorithm FF Multi 1 (FF Multi Model - Temporal Contrastive RL)");
        let mut algo =
            AlgorithmFFMulti1::new(state_size, action_size, vec![256, 128], device.clone())
                .expect("Failed to create AlgorithmFFMulti1");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        let snap = Err(candle_core::Error::Msg(
            "FF Model has no visualization".to_string(),
        ));
        let eps = algo.n_episodes;
        let layers = algo.model.layers.len();
        let syns = 0;
        algo_ff_multi1_opt = Some(algo);
        (eps, snap, layers, syns)
    } else if algo_choice == "ff_multi2" {
        log::info!("Using Algorithm FF Multi 2 (Classification-Based RL)");
        let mut algo =
            AlgorithmFFMulti2::new(state_size, action_size, vec![512, 256, 128], device.clone())
                .expect("Failed to create AlgorithmFFMulti2");
        if infinite_epochs {
            algo.n_episodes = usize::MAX - 1;
        }
        // Resume from checkpoint if --resume and a checkpoint exists.
        let mut restored_rewards = None;
        if resume {
            let cp_dir = std::path::Path::new("checkpoints/ff_multi2");
            if cp_dir.join("training_state.json").exists() {
                match algo.load_checkpoint(cp_dir) {
                    Ok(rewards) => {
                        log::info!("Resumed from checkpoint (episode {})", algo.start_episode);
                        restored_rewards = Some(rewards);
                    }
                    Err(e) => {
                        log::error!("Failed to load checkpoint: {}. Starting fresh.", e);
                    }
                }
            } else {
                log::info!("No checkpoint found at {:?}. Starting fresh.", cp_dir);
            }
        }
        let snap = Err(candle_core::Error::Msg(
            "FF Model has no visualization".to_string(),
        ));
        let eps = algo.n_episodes;
        let layers = algo.main_model.layers.len();
        let syns = 0;
        algo_ff_multi2_opt = Some((algo, restored_rewards));
        (eps, snap, layers, syns)
    } else {
        panic!("Unknown algorithm choice: {}", algo_choice);
    };

    log::info!("layers len: {}, num_synapses: {}", num_layers, num_synapses);

    // Start visualization if requested
    let vis_handle = if visualize {
        let vis_state = Arc::new(Mutex::new(VisualizationState::new(n_episodes)));

        // Initialize model structure
        if let Ok(mut state) = vis_state.lock() {
            if let Ok(snapshot) = snapshot_result {
                log::info!(
                    "Initial snapshot: {} layers, {} synapses",
                    snapshot.layers.len(),
                    snapshot.synapses.len()
                );
                state.update_from_snapshot(snapshot);
            } else {
                log::info!("Warning: Failed to get initial visualization snapshot");
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
    } else if let Some(mut algo) = algo4_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo5_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ff1_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ff2_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ff3_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ff4_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ffsac_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some(mut algo) = algo_ff_multi1_opt {
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    } else if let Some((mut algo, restored_rewards)) = algo_ff_multi2_opt {
        // If we resumed from a checkpoint, inject the restored reward history
        // into the visualization state so the graph picks up where it left off.
        if let Some(rewards) = restored_rewards {
            if let Some(ref vs) = vis_state_arg {
                if let Ok(mut state) = vs.lock() {
                    state.epoch_rewards = rewards;
                }
            }
        }
        algo.run(env.as_mut(), visualize, vis_state_arg)?;
    }

    if let Some((_, ref vis_state_arc)) = vis_handle {
        loop {
            let should_close = vis_state_arc
                .try_lock()
                .map(|state| state.should_close)
                .unwrap_or(false);
            if should_close {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    // Signal visualization to close and wait for thread
    if let Some((handle, vis_state)) = vis_handle {
        log::info!("Closing visualization...");
        while let Ok(state) = vis_state.lock() {
            if state.should_close {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        let _ = handle.join();
    }

    // drop env cleans up whatever depends on drops, e.g. RobotEnvironment::disable()
    drop(env);

    log::info!("Done");
    Ok(())
}
