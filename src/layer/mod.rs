pub mod bernoulli;
pub mod lif;

use candle_core::{Result as CandleResult, Tensor};

pub trait Layer: Send + Sync {
    /// update internal state and calculated output
    fn step(&mut self, input: &Tensor, dt: f32) -> CandleResult<()>;

    /// internal activity getter
    fn activity(&self) -> CandleResult<&Tensor>;

    /// output getter
    fn output(&self) -> CandleResult<&Tensor>;

    /// how many neurons in this layer
    fn size(&self) -> usize;
}
