use crate::synapse::SynapseUpdate;
use candle_core::{Result as CandleResult, Tensor};

#[derive(Clone)]
pub struct Hebbian {
    pub lr: f32,
}

impl Hebbian {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl SynapseUpdate for Hebbian {
    fn update(
        &self,
        weight: &Tensor,
        pre: &Tensor,
        post: &Tensor,
        dt: f32,
    ) -> CandleResult<Tensor> {
        // outer = post[:,None] @ pre[None,:]  -> shape (post, pre)
        let post_col = post.reshape((post.dims()[0], 1))?;
        let pre_row = pre.reshape((1, pre.dims()[0]))?;

        let outer = post_col.matmul(&pre_row)?;
        // delta = lr * dt * outer
        let delta = outer.affine((self.lr * dt) as f64, 0.0)?;
        let new_w = weight.add(&delta)?;
        Ok(new_w)
    }
}
