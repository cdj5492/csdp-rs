use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::ops::sigmoid;

use crate::layer::Layer;


// pub struct GoodnessLayer {
//     pub state: Tensor,
//     pub loss: Tensor,
//     pub mod_signal: Tensor,
//     pub thresh: f32,
//     pub size: usize,
// }
//
// impl GoodnessLayer {
//     pub fn new(size: usize, thresh: f32, device: &candle_core::Device) -> CandleResult<Self> {
//         let loss = Tensor::zeros((1, 1), DType::F32, device)?;
//         let mod_signal = Tensor::zeros((1, 1), DType::F32, device)?;
//         Ok(Self {
//             thresh,
//             loss,
//             mod_signal,
//             size,
//         })
//     }
// }
//
// impl Layer for GoodnessLayer {
//     fn step(&mut self, input: &Tensor, dt: f32) -> CandleResult<()> {
//         // noop
//         Ok(())
//     }
//
//     fn activity(&self) -> CandleResult<&Tensor> {
//         Ok(&self.state)
//     }
//
//     fn output(&self) -> CandleResult<&Tensor> {
//         Ok(&self.state)
//     }
//
//     fn size(&self) -> usize {}
// }
