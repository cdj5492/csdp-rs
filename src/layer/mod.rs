pub mod bernoulli;
pub mod lif;
// pub mod goodness;

use candle_core::{Result as CandleResult, Tensor};

pub trait Layer: Send + Sync {
    /// update internal state and calculated output
    fn step(&mut self, dt: f32) -> CandleResult<()>;

    /// internal activity getter
    fn activity(&self) -> CandleResult<&Tensor>;

    /// output getter
    fn output(&self) -> CandleResult<&Tensor>;

    /// how many neurons in this layer
    fn size(&self) -> usize;

    /// Adds to the input compartment of the layer
    fn add_input(&mut self, input: &Tensor) -> CandleResult<()>;

    /// resets input compartment to zero
    fn reset_input(&mut self) -> CandleResult<()>;

    /// resets internal state fully
    fn reset(&mut self) -> CandleResult<()>;
}
