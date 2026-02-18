use crate::layer::Layer;

use super::{SynapseOps, WeightStats};
use candle_core::{Result as CandleResult, Tensor};

#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct CSDP {
    pub weights: Tensor,
    /// biases are kept in a separate tensor
    pub biases: Tensor,
}

impl CSDP {
    pub fn new(
        pre_size: usize,
        post_size: usize,
        device: &candle_core::Device,
    ) -> CandleResult<Self> {
        // TODO: tune initialization
        let weights = Tensor::rand(-1.0f32, 1.0, (post_size, pre_size), device)?;
        let biases = Tensor::zeros((post_size, 1), candle_core::DType::F32, device)?;
        // let biases = Tensor::rand(-1.0f32, 1.0, (post_size, 1), device)?;
        Ok(Self { weights, biases })
    }
}

impl SynapseOps for CSDP {
    fn forward(&self, pre: &Tensor) -> CandleResult<Tensor> {
        let w_pre = self.weights.matmul(pre)?;
        let out = w_pre.add(&self.biases)?;
        Ok(out)
    }

    fn update_weights(
        &mut self,
        pre_activity: &Tensor,
        post_layer: &mut Box<dyn Layer>,
        _dt: f32,
    ) -> CandleResult<()> {
        let pre = pre_activity;
        let pre_row = pre.reshape((1, pre.dims()[0]))?;

        let mod_signal = post_layer.get_mod_signal();

        // synaptic decay factor
        // TODO: put at top level
        let lambda_d = 0.05;

        // outer product (should be same shape as weight matrix)
        let delta = (mod_signal.matmul(&pre_row)?
            + lambda_d * post_layer.output()?.matmul(&(1.0 - pre_row)?)?)?;
        self.weights = self.weights.add(&delta)?;

        // biases are treated as connections to a neuron that is always firing every timestep
        // TODO: figure out why this doesnt work
        // self.biases = self.biases.add(mod_signal)?;

        Ok(())
    }

    fn weight_stats(&self) -> CandleResult<WeightStats> {
        let weights_vec = self.weights.flatten_all()?.to_vec1::<f32>()?;
        let num_weights = weights_vec.len();

        if num_weights == 0 {
            return Ok(WeightStats {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                num_weights: 0,
            });
        }

        let sum: f32 = weights_vec.iter().sum();
        let mean = sum / num_weights as f32;

        let variance: f32 =
            weights_vec.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / num_weights as f32;
        let std = variance.sqrt();

        let min = weights_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights_vec
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        Ok(WeightStats {
            mean,
            std,
            min,
            max,
            num_weights,
        })
    }
}
