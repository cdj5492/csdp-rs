use crate::environment::Environment;
use rand::Rng;
use std::error::Error;

pub struct GridEnvironment {
    player_x: i32,
    player_y: i32,
    goal_x: i32,
    goal_y: i32,
}

impl GridEnvironment {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            player_x: rng.gen_range(0..50),
            player_y: rng.gen_range(0..50),
            goal_x: rng.gen_range(0..50),
            goal_y: rng.gen_range(0..50),
        }
    }
}

impl Environment for GridEnvironment {
    fn state_size(&self) -> usize {
        4
    }

    fn action_size(&self) -> usize {
        5
    }

    fn state_bounds(&self) -> Option<Vec<usize>> {
        Some(vec![50, 50, 50, 50])
    }

    fn get_state(&mut self) -> Result<Vec<f64>, Box<dyn Error>> {
        Ok(vec![
            self.player_x as f64,
            self.player_y as f64,
            self.goal_x as f64,
            self.goal_y as f64,
        ])
    }

    fn evaluate_action(&self, _state: &[f64], action_idx: usize) -> f64 {
        let mut nx = self.player_x;
        let mut ny = self.player_y;

        match action_idx {
            1 => ny -= 1, // Up
            2 => ny += 1, // Down
            3 => nx -= 1, // Left
            4 => nx += 1, // Right
            _ => {}       // None
        }

        nx = nx.clamp(0, 49);
        ny = ny.clamp(0, 49);

        let dx = (nx - self.goal_x) as f64;
        let dy = (ny - self.goal_y) as f64;
        let distance = (dx * dx + dy * dy).sqrt();
        
        let max_dist = (49.0f64.powi(2) + 49.0f64.powi(2)).sqrt();
        
        // Maps 0 distance to 5.0 and max_dist to -5.0
        5.0 - (distance / max_dist) * 10.0
    }

    fn apply_action(&mut self, action_idx: usize) -> Result<(), Box<dyn Error>> {
        match action_idx {
            1 => self.player_y -= 1, // Up
            2 => self.player_y += 1, // Down
            3 => self.player_x -= 1, // Left
            4 => self.player_x += 1, // Right
            _ => {}                  // None
        }

        self.player_x = self.player_x.clamp(0, 49);
        self.player_y = self.player_y.clamp(0, 49);

        Ok(())
    }

    fn reset(&mut self) -> Result<(), Box<dyn Error>> {
        let mut rng = rand::thread_rng();
        self.player_x = rng.gen_range(0..50);
        self.player_y = rng.gen_range(0..50);
        self.goal_x = rng.gen_range(0..50);
        self.goal_y = rng.gen_range(0..50);
        Ok(())
    }
}