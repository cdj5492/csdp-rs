use candle_core::{Result as CandleResult, Tensor};
use super::{SynapseOps, WeightStats};

#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct CSDP {
    pub weights: Tensor,
    /// biases are only applied in the forward direction
    pub biases: Tensor,
    pub lr: f32,
}

impl CSDP {
    pub fn new(
        pre_size: usize,
        post_size: usize,
        device: &candle_core::Device,
    ) -> CandleResult<Self> {
        // TODO: tune initialization
        let weights = Tensor::randn(0.0f32, 0.1, (post_size, pre_size), device)?;
        let biases = Tensor::zeros((post_size, 1), candle_core::DType::F32, device)?;
        let lr = 0.01;
        Ok(Self {
            weights,
            biases,
            lr,
        })
    }
}

impl CSDP {
    /// Forward pass (public method for backward compatibility with existing model code)
    pub fn forward(&self, pre: &Tensor) -> CandleResult<Tensor> {
        let w_pre = self.weights.matmul(pre)?;
        let out = w_pre.add(&self.biases)?;
        Ok(out)
    }

    /// Update weights (old interface that returns new weights without mutating self)
    /// This is for backward compatibility with the existing model code
    pub fn update_weights(&self, pre: &Tensor, post: &Tensor, dt: f32) -> CandleResult<Tensor> {
        // outer = post[:,None] @ pre[None,:]  -> shape (post, pre)
        let post_col = post.reshape((post.dims()[0], 1))?;
        let pre_row = pre.reshape((1, pre.dims()[0]))?;

        let outer = post_col.matmul(&pre_row)?;
        // delta = lr * dt * outer
        let delta = outer.affine((self.lr * dt) as f64, 0.0)?;
        let new_w = self.weights.add(&delta)?;
        Ok(new_w)
    }
}

impl SynapseOps for CSDP {
    fn forward(&self, pre: &Tensor) -> CandleResult<Tensor> {
        CSDP::forward(self, pre)
    }

    fn update_weights(&mut self, pre: &Tensor, post: &Tensor, dt: f32) -> CandleResult<()> {
        // Use the old update_weights method and assign the result
        self.weights = CSDP::update_weights(self, pre, post, dt)?;
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

        let variance: f32 = weights_vec.iter()
            .map(|&w| (w - mean).powi(2))
            .sum::<f32>() / num_weights as f32;
        let std = variance.sqrt();

        let min = weights_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        Ok(WeightStats {
            mean,
            std,
            min,
            max,
            num_weights,
        })
    }
}
