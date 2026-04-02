use super::Environment;
use crate::robot::real_lerobot::LeRobot;
use std::error::Error;

const NUM_ACTIONS: usize = 12;
const ACTION_DELTA: f64 = 0.05; // radians
const NUM_JOINTS: usize = 6;
const TARGET_POSITION: [f64; NUM_JOINTS] = [0.0, -1.0, 1.0, 0.5, 0.0, 0.5];

pub struct RobotEnvironment {
    follower: LeRobot,
    target_position: [f64; NUM_JOINTS],
}

impl RobotEnvironment {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let mut follower = LeRobot::new(
            "/dev/ttyACM0",
            [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
            [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
            [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
        )?;

        follower.enable()?;

        Ok(Self {
            follower,
            target_position: TARGET_POSITION,
        })
    }
}

impl Drop for RobotEnvironment {
    fn drop(&mut self) {
        let _ = self.follower.disable();
    }
}

impl Environment for RobotEnvironment {
    fn state_size(&self) -> usize {
        NUM_JOINTS
    }
    fn action_size(&self) -> usize {
        NUM_ACTIONS
    }

    fn clone_box(&self) -> Box<dyn Environment> {
        panic!("RobotEnvironment cannot be cloned securely across processes/vectors! Disable vectorization or utilize Simulator proxies.");
    }

    fn get_state(&mut self) -> Result<Vec<f64>, Box<dyn Error>> {
        let pos = self.follower.get_motor_positions()?;
        Ok(pos.to_vec())
    }

    fn evaluate_action(&self, state: &[f64], action_idx: usize) -> f64 {
        let mut next_state = state.to_vec();
        let joint_idx = action_idx % NUM_JOINTS;
        let sign = if action_idx < NUM_JOINTS { 1.0 } else { -1.0 };
        next_state[joint_idx] += sign * ACTION_DELTA;

        let mut dist_sq = 0.0;
        for i in 0..NUM_JOINTS {
            let diff = next_state[i] - self.target_position[i];
            dist_sq += diff * diff;
        }

        -dist_sq
    }

    fn apply_action(&mut self, action_idx: usize) -> Result<(), Box<dyn Error>> {
        let current_state = self.get_state()?;
        let mut next_state = current_state.clone();
        let joint_idx = action_idx % NUM_JOINTS;
        let sign = if action_idx < NUM_JOINTS { 1.0 } else { -1.0 };
        next_state[joint_idx] += sign * ACTION_DELTA;
        self.follower.set_goal_positions(&next_state)?;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), Box<dyn Error>> {
        self.follower.go_to_home_positions()?;
        Ok(())
    }
}
