use candle_core::{Result as CandleResult, Tensor};

#[derive(Clone)]
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

    pub fn forward(&self, pre: &Tensor) -> CandleResult<Tensor> {
        let w_pre = self.weights.matmul(pre)?;
        let out = w_pre.add(&self.biases)?;
        Ok(out)
    }
}
