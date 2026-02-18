use crate::robot::real_lerobot::LeRobot;
use candle_core::{Device, Result as CandleResult, Tensor};

pub struct RealtimeLeader {
    robot: LeRobot,
    device: Device,
    limit: usize,
}

impl RealtimeLeader {
    pub fn new(robot: LeRobot, limit: usize, device: Device) -> Self {
        Self {
            robot,
            device,
            limit,
        }
    }

    pub fn iter(&mut self) -> RobotIterator<'_> {
        RobotIterator {
            robot: &mut self.robot,
            device: self.device.clone(),
            limit: self.limit,
            current: 0,
        }
    }
}

pub struct RobotIterator<'a> {
    robot: &'a mut LeRobot,
    device: Device,
    limit: usize,
    current: usize,
}

impl<'a> Iterator for RobotIterator<'a> {
    type Item = CandleResult<(Tensor, Tensor, f32)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.limit {
            return None;
        }

        let positions = match self.robot.get_motor_positions() {
            Ok(pos) => pos,
            Err(e) => return Some(Err(candle_core::Error::Msg(e.to_string()))),
        };

        // Input: first 5 actuators
        let input_slice: Vec<f32> = positions[0..5].iter().map(|&v| v as f32).collect();
        let input_tensor = match Tensor::from_vec(input_slice, (5, 1), &self.device) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        // Label: always 1.0
        let label_tensor = match Tensor::from_vec(vec![1.0f32], (1, 1), &self.device) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        // Reward logic: last actuator (index 5) > 0.5
        let is_positive = if positions[5] > 0.5 { 1.0f32 } else { 0.0f32 };

        self.current += 1;
        Some(Ok((input_tensor, label_tensor, is_positive)))
    }
}
