pub mod algorithm_csdp1;
pub mod algorithm_csdp2;
pub mod algorithm_csdp3;
pub mod algorithm_ff1;
pub mod algorithm_ff2;
pub mod algorithm_ff3;
pub mod algorithm_ff4;
pub mod algorithm_ffsac;

use crate::environment::Environment;
use crate::visualization::VisualizationState;
use std::error::Error;
use std::sync::{Arc, Mutex};

pub trait Algorithm {
    fn run(
        &mut self,
        env: &mut dyn Environment,
        visualize: bool,
        vis_state: Option<Arc<Mutex<VisualizationState>>>,
    ) -> Result<(), Box<dyn Error>>;
}
