pub mod algorithm_csdp1;
pub mod algorithm_csdp2;
pub mod algorithm_csdp3;
pub mod algorithm_csdp4;
pub mod algorithm_csdp5;
pub mod algorithm_csdp_ppo;
pub mod algorithm_ff1;
pub mod algorithm_ff2;
pub mod algorithm_ff3;
pub mod algorithm_ff4;
pub mod algorithm_ff_multi1;
pub mod algorithm_ff_multi2;
pub mod algorithm_ff_ppo;
pub mod algorithm_ffsac;

pub use algorithm_csdp1::Algorithm1;
pub use algorithm_csdp2::Algorithm2;
pub use algorithm_csdp3::Algorithm3;
pub use algorithm_csdp4::Algorithm4;
pub use algorithm_ff1::AlgorithmFF1;
pub use algorithm_ff2::AlgorithmFF2;

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
