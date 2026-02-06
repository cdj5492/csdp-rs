#[path = "../robot/mod.rs"]
mod robot;

use crate::robot::real_lerobot::LeRobot;
use std::error::Error;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize Follower (Active Robot)
    // TTY: /dev/ttyACM1
    println!("Initializing Follower on /dev/ttyACM1...");
    let mut follower = LeRobot::new(
        "/dev/ttyACM1",
        [-0.0276, -1.6, 1.29, 1.1, 0.254, -0.02],
        [-1.3, -1.6, -1.94, -2.0, -1.5, -0.02],
        [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )
    .expect("Failed to initialize follower");

    // Initialize Leader (Passive Input Device)
    // TTY: /dev/ttyACM0
    println!("Initializing Leader on /dev/ttyACM0...");
    let mut leader = LeRobot::new(
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
    .expect("Failed to initialize leader");

    // Setup Robots
    println!("Enabling Follower torque...");
    follower.enable()?; // Active

    println!("Disabling Leader torque (ready for manual input)...");
    leader.disable()?; // Passive

    // Safety: Move Follower to match Leader's current position slowly
    // This prevents the follower from snapping violently if the leader is in a different pose
    println!("Syncing start positions...");
    if let Ok(start_pos) = leader.get_motor_positions()
        && start_pos.len() == 6
    {
        follower.set_goal_positions(&[
            start_pos[0],
            start_pos[1],
            start_pos[2],
            start_pos[3],
            start_pos[4],
            start_pos[5],
        ])?;
        // Give it time to move there safely
        thread::sleep(Duration::from_millis(2000));
    }

    // Blocking wait for Start
    print!("Synced. Press ENTER to START teleoperation...");
    io::stdout().flush()?;
    let mut input_buffer = String::new();
    io::stdin().read_line(&mut input_buffer)?;

    println!("Teleoperation active! Press ENTER to STOP.");

    // Spawn thread to listen for Stop signal
    let keep_running = Arc::new(AtomicBool::new(true));
    let r_handle = keep_running.clone();

    thread::spawn(move || {
        let mut s = String::new();
        io::stdin().read_line(&mut s).ok();
        r_handle.store(false, Ordering::Relaxed);
    });

    // Control Loop
    let target_frame_time = Duration::from_secs_f64(1.0 / 60.0); // 60Hz update rate

    while keep_running.load(Ordering::Relaxed) {
        let loop_start = Instant::now();

        // Read Leader
        if let Ok(positions) = leader.get_motor_positions()
            && positions.len() == 6
        {
            // Write Follower
            // We map 1:1, assuming the robots are physically identical or compatible
            follower.set_goal_positions(&[
                positions[0],
                positions[1],
                positions[2],
                positions[3],
                positions[4],
                positions[5],
            ])?;
        }

        // Maintain Loop Rate
        let elapsed = loop_start.elapsed();
        if elapsed < target_frame_time {
            thread::sleep(target_frame_time - elapsed);
        }
    }

    // Cleanup
    println!("Stopping...");
    follower.disable()?;
    leader.disable()?;
    println!("Both robots disabled.");

    Ok(())
}
