pub mod basic;

use candle_core::Tensor;
pub trait CellUpdate: Send + Sync {
    fn update(&self, state: &Tensor, input: &Tensor, dt: f32) -> candle_core::Result<Tensor>;
}
