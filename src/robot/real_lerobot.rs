use rustypot::servo::feetech::sts3215::Sts3215Controller;
use std::time::Duration;

/// Hardcoded IDs assumed
const MOTOR_IDS: [u8; 6] = [1, 2, 3, 4, 5, 6];
const HOME_POSITIONS: [f64; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

pub struct LeRobot {
    controller: Sts3215Controller,
}

impl LeRobot {
    pub fn new<'a>(path: impl Into<std::borrow::Cow<'a, str>>) -> Self {
        let serial_port = serialport::new(path, 1_000_000)
            .timeout(Duration::from_millis(100))
            .open()
            .unwrap();

        let controller = Sts3215Controller::new()
            .with_protocol_v1()
            .with_serial_port(serial_port);

        LeRobot { controller }
    }

    pub fn set_max_speed_all(&mut self, speed: f64) {
        let arr = [speed; 6];
        self.controller
            .sync_write_goal_speed(&MOTOR_IDS, &arr)
            .unwrap();
    }

    pub fn set_goal_positions(&mut self, positions: &[f64; 6]) {
        self.controller
            .sync_write_goal_position(&MOTOR_IDS, positions)
            .unwrap();
    }

    pub fn go_to_home_positions(&mut self) {
        self.set_goal_positions(&HOME_POSITIONS);
    }
}
