#[path = "../robot/mod.rs"]
mod robot;

use crate::robot::real_lerobot::LeRobot;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;

#[derive(Serialize)]
struct RobotFrame {
    timestamp_ms: u128,
    j1: f64,
    j2: f64,
    j3: f64,
    j4: f64,
    j5: f64,
    j6: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
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
    )
    .expect("Failed to initialize robot");

    robot.go_to_home_positions()?;

    thread::sleep(Duration::from_millis(1000));

    robot.disable()?;
    println!("Robot initialized and torque disabled.");

    // 1. Blocking wait for Start
    print!("Press ENTER to START recording...");
    io::stdout().flush()?;
    let mut input_buffer = String::new();
    io::stdin().read_line(&mut input_buffer)?;

    println!("Recording started... Press ENTER to STOP.");

    // 2. Spawn a thread to listen for the Stop signal (Enter key)
    let keep_running = Arc::new(AtomicBool::new(true));
    let r_handle = keep_running.clone();

    thread::spawn(move || {
        let mut s = String::new();
        io::stdin().read_line(&mut s).ok();
        r_handle.store(false, Ordering::Relaxed);
    });

    // 3. Recording Loop
    let mut records = Vec::new();
    let start_time = Instant::now();
    let target_frame_time = Duration::from_secs_f64(1.0 / 30.0);

    while keep_running.load(Ordering::Relaxed) {
        let frame_start = Instant::now();

        if let Ok(positions) = robot.get_motor_positions()
            && positions.len() == 6
        {
            records.push(RobotFrame {
                timestamp_ms: start_time.elapsed().as_millis(),
                j1: positions[0],
                j2: positions[1],
                j3: positions[2],
                j4: positions[3],
                j5: positions[4],
                j6: positions[5],
            });
        }

        let elapsed = frame_start.elapsed();
        if elapsed < target_frame_time {
            thread::sleep(target_frame_time - elapsed);
        }
    }

    // Save to CSV
    let args: Vec<String> = env::args().collect();
    let default_filename = "data/training_data.csv".to_string();
    let filename = args.get(1).unwrap_or(&default_filename);

    println!("Saving {} frames to {}...", records.len(), filename);

    let file = File::create(filename)?;
    let mut wtr = csv::Writer::from_writer(file);

    for record in records {
        wtr.serialize(record)?;
    }
    wtr.flush()?;

    println!("Done.");
    Ok(())
}
