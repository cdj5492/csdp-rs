pub mod algorithm1;

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
