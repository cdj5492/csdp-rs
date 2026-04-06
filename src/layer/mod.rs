pub mod bernoulli;
pub mod lif;
pub mod mod_signal;
pub mod one_hot;

use candle_core::{Result as CandleResult, Tensor};

pub trait Layer: Send + Sync {
    /// update internal state and calculated output
    fn step(&mut self, dt: f32) -> CandleResult<()>;

    /// internal activity getter
    #[allow(dead_code)]
    fn activity(&self) -> CandleResult<&Tensor>;

    /// calculates the modulatory goodness signal used for CSDP synapse adjustment
    fn get_mod_signal(&self) -> &Tensor;

    /// output getter
    fn output(&self) -> CandleResult<&Tensor>;

    /// how many neurons in this layer
    fn size(&self) -> usize;

    /// Adds to the input compartment of the layer
    fn add_input(&mut self, input: &Tensor) -> CandleResult<()>;

    /// resets input compartment to zero
    fn reset_input(&mut self) -> CandleResult<()>;

    /// resets internal state fully
    fn reset(&mut self, batch_size: usize) -> CandleResult<()>;

    /// sets the current sample type for the layer
    fn set_positive_sample(&mut self, label: &Tensor);

    /// sets the environmental reward for the layer
    fn set_reward(&mut self, reward: &Tensor);
}

/// Position of a layer in visualization space
#[derive(Debug, Clone, Copy)]
pub struct LayerPosition {
    pub x: f32,
    pub y: f32,
}

/// Metadata about a layer for visualization and configuration
#[derive(Debug, Clone)]
pub struct LayerMetadata {
    pub id: usize,
    pub name: String,
    pub layer_type: String,
    pub size: usize,
    pub position: LayerPosition,
}
