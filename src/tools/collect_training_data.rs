#[path = "../robot/mod.rs"]
mod robot;

use crate::robot::real_lerobot::LeRobot;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::stdout;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use serde::Serialize;

// Structure to hold a single frame of data for CSV serialization
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
    // Initialize Robot
    let mut robot = LeRobot::new(
        "/dev/ttyACM0",
        [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
        [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
        [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )
    .expect("Failed to initialize robot");

    // Disable torque so user can manipulate the arm
    robot.disable()?;
    println!("Robot initialized and torque disabled.");

    // Setup terminal for raw input (to detect spacebar without Enter)
    stdout().execute(crossterm::cursor::Hide)?;
    enable_raw_mode()?;

    println!("Press SPACE to START recording.\r");

    // wait for start
    loop {
        if let Event::Key(key) = event::read()?
            && event::poll(Duration::from_millis(100))?
        {
            if key.kind == KeyEventKind::Press && key.code == KeyCode::Char(' ') {
                break;
            }
            if key.code == KeyCode::Esc {
                cleanup()?;
                return Ok(());
            }
        }
    }

    println!("Recording started... Press SPACE to STOP.\r");

    let mut records = Vec::new();
    let start_time = Instant::now();
    let target_frame_time = Duration::from_secs_f64(1.0 / 30.0); // 30 FPS

    loop {
        let frame_start = Instant::now();

        // Check for stop condition (non-blocking)
        if let Event::Key(key) = event::read()?
            && event::poll(Duration::from_micros(0))?
            && key.kind == KeyEventKind::Press
            && key.code == KeyCode::Char(' ')
        {
            break;
        }

        // Record Data
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

        // Maintain 30 FPS
        let elapsed = frame_start.elapsed();
        if elapsed < target_frame_time {
            thread::sleep(target_frame_time - elapsed);
        }
    }

    cleanup()?;

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

fn cleanup() -> Result<(), Box<dyn Error>> {
    disable_raw_mode()?;
    stdout().execute(crossterm::cursor::Show)?;
    Ok(())
}
