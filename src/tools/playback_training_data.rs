#[path = "../robot/mod.rs"]
mod robot;

use crate::robot::real_lerobot::LeRobot;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RobotFrame {
    timestamp_ms: u64,
    j1: f64,
    j2: f64,
    j3: f64,
    j4: f64,
    j5: f64,
    j6: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let default_path = "data/training_data.csv".to_string();
    let file_path = args.get(1).unwrap_or(&default_path);

    println!("Loading data from {}...", file_path);
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let records: Vec<RobotFrame> = rdr.deserialize().collect::<Result<_, _>>()?;

    if records.is_empty() {
        println!("No records found.");
        return Ok(());
    }

    let mut robot = LeRobot::new(
        "/dev/ttyACM0",
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
        // [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
        // [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
        // [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )?;

    println!("Moving to start position...");

    robot.enable()?;

    // Move to start
    let first = &records[0];
    robot.set_goal_positions(&[first.j1, first.j2, first.j3, first.j4, first.j5, first.j6])?;
    thread::sleep(Duration::from_millis(1500));

    println!("Playback started. Press ENTER to stop early.");

    // Spawn thread to listen for interrupt
    let keep_running = Arc::new(AtomicBool::new(true));
    let r_handle = keep_running.clone();

    thread::spawn(move || {
        let mut s = String::new();
        io::stdin().read_line(&mut s).ok();
        r_handle.store(false, Ordering::Relaxed);
    });

    let start_time = Instant::now();
    let initial_timestamp = records[0].timestamp_ms;

    for frame in &records {
        // Check interrupt
        if !keep_running.load(Ordering::Relaxed) {
            println!("Playback interrupted by user.");
            break;
        }

        let target_elapsed = Duration::from_millis(frame.timestamp_ms - initial_timestamp);
        let current_elapsed = start_time.elapsed();

        if target_elapsed > current_elapsed {
            thread::sleep(target_elapsed - current_elapsed);
        }

        robot.set_goal_positions(&[frame.j1, frame.j2, frame.j3, frame.j4, frame.j5, frame.j6])?;
    }

    robot.disable()?;
    println!("Motors disabled. Done.");

    Ok(())
}
