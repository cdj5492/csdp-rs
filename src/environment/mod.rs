pub mod grid;
pub mod robot;

use std::error::Error;

pub trait Environment {
    fn state_size(&self) -> usize;
    fn action_size(&self) -> usize;

    fn get_state(&mut self) -> Result<Vec<f64>, Box<dyn Error>>;

    /// Compute the reward of applying an action from a given state (without mutating)
    fn evaluate_action(&self, state: &[f64], action_idx: usize) -> f64;

    /// Apply an action and mutate the environment
    fn apply_action(&mut self, action_idx: usize) -> Result<(), Box<dyn Error>>;

    /// Reset the environment to its initial state
    fn reset(&mut self) -> Result<(), Box<dyn Error>>;
}
