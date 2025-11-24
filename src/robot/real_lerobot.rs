use rustypot::servo::feetech::sts3215::Sts3215Controller;
use std::error::Error;
use std::time::Duration;

/// Hardcoded IDs assumed
const MOTOR_IDS: [u8; 6] = [1, 2, 3, 4, 5, 6];

// Type alias for concise return signatures
pub type RobotResult<T> = Result<T, Box<dyn Error>>;

pub struct LeRobot {
    pub controller: Sts3215Controller,
    home_positions: [f64; 6],
}

impl LeRobot {
    pub fn new<'a>(
        path: impl Into<std::borrow::Cow<'a, str>>,
        home_positions: [f64; 6],
        min_positions: [f64; 6],
        max_positions: [f64; 6],
    ) -> RobotResult<Self> {
        let serial_port = serialport::new(path, 1_000_000)
            .timeout(Duration::from_millis(100))
            .open()?;

        let mut controller = Sts3215Controller::new()
            .with_protocol_v1()
            .with_serial_port(serial_port);

        // Initialize limits and dead zones
        controller.sync_write_min_angle_limit(&MOTOR_IDS, &min_positions)?;
        controller.sync_write_max_angle_limit(&MOTOR_IDS, &max_positions)?;
        controller.sync_write_cw_dead_zone(&MOTOR_IDS, &[5; 6])?;
        controller.sync_write_ccw_dead_zone(&MOTOR_IDS, &[5; 6])?;

        // Set max torque limit
        controller.sync_write_torque_limit(&MOTOR_IDS, &[400; 6])?;

        Ok(LeRobot {
            controller,
            home_positions,
        })
    }

    pub fn enable(&mut self) -> RobotResult<()> {
        let arr = [true; 6];
        self.controller.sync_write_torque_enable(&MOTOR_IDS, &arr)?;
        Ok(())
    }

    pub fn disable(&mut self) -> RobotResult<()> {
        let arr = [false; 6];
        self.controller.sync_write_torque_enable(&MOTOR_IDS, &arr)?;
        Ok(())
    }

    pub fn set_max_speed_all(&mut self, speed: f64) -> RobotResult<()> {
        let arr = [speed; 6];
        self.controller.sync_write_goal_speed(&MOTOR_IDS, &arr)?;
        Ok(())
    }

    pub fn set_goal_positions(&mut self, positions: &[f64]) -> RobotResult<()> {
        // Note: This assumes input slice length matches home_positions length
        let adjusted_positions = positions
            .iter()
            .zip(self.home_positions.iter())
            .map(|(p, h)| p + h)
            .collect::<Vec<_>>();

        self.controller
            .sync_write_goal_position(&MOTOR_IDS, &adjusted_positions)?;
        Ok(())
    }

    pub fn go_to_home_positions(&mut self) -> RobotResult<()> {
        self.set_goal_positions(&[0.0; 6])
    }

    pub fn get_motor_positions(&mut self) -> RobotResult<Vec<f64>> {
        let positions = self.controller.sync_read_present_position(&MOTOR_IDS)?;

        let computed = positions
            .iter()
            .zip(self.home_positions.iter())
            .map(|(p, h)| p - h)
            .collect::<Vec<_>>();

        Ok(computed)
    }
}
