use candle_core::{Device, Result as CandleResult, Tensor};

use crate::models::Model;
// wrapper around the general CSDP model specifically for controlling the robots

pub struct RobotModel {
    model: Model,
}

impl RobotModel {
    pub fn new(num_hidden: usize, hidden_size: usize, device: &Device, dt: f32) -> Self {
        RobotModel {
            // Inputs:
            //   - 6 neurons (1 for each motor's position)
            //     - Experiment with 'place neuron' style encoding using multiple neurons per motor
            // Outputs:
            //   - 18 neurons
            //     - Broken apart into 6 groups of 3 for each motor (do nothing, spin left, spin
            //     right)
            // TODO: image input neurons and handle option
            model: Model::new(6, 18, vec![hidden_size; num_hidden], device, dt).unwrap(),
        }
    }

    pub fn step(&mut self, input: &Tensor, context: Option<&Tensor>) -> CandleResult<()> {
        self.model.step(input, context)
    }
}
