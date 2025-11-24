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
    // 1. Load Data
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

    // 2. Initialize Robot
    let mut robot = LeRobot::new(
        "/dev/ttyACM0",
        [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
        [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
        [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )?;

    // Setup terminal for input
    stdout().execute(crossterm::cursor::Hide)?;
    enable_raw_mode()?;

    println!("Moving to start position... (Press SPACE to emergency stop)");

    // 3. Enable and Move to Start
    robot.enable()?;

    // Move to first frame position slowly (1.5 seconds) to avoid jerking
    let first = &records[0];
    robot.set_goal_positions(&[first.j1, first.j2, first.j3, first.j4, first.j5, first.j6])?;
    thread::sleep(Duration::from_millis(1500));

    println!("Playback started. Press SPACE to stop and relax motors.\r");

    // 4. Playback Loop
    let start_time = Instant::now();
    let initial_timestamp = records[0].timestamp_ms;

    for frame in &records {
        // Check for user interrupt
        if let Event::Key(key) = event::read()?
            && event::poll(Duration::from_micros(0))?
            && key.kind == KeyEventKind::Press
            && key.code == KeyCode::Char(' ')
        {
            println!("\r\nPlayback interrupted by user.");
            break;
        }

        // Sync timing
        let target_elapsed = Duration::from_millis(frame.timestamp_ms - initial_timestamp);
        let current_elapsed = start_time.elapsed();

        if target_elapsed > current_elapsed {
            thread::sleep(target_elapsed - current_elapsed);
        }

        // Send command
        robot.set_goal_positions(&[frame.j1, frame.j2, frame.j3, frame.j4, frame.j5, frame.j6])?;
    }

    // 5. Cleanup
    robot.disable()?;
    disable_raw_mode()?;
    stdout().execute(crossterm::cursor::Show)?;
    println!("Motors disabled. Done.");

    Ok(())
}
