pub mod hebbian;

use candle_core::{Result as CandleResult, Tensor};
use crate::layer::basic::Layer;

pub trait SynapseUpdate: Send + Sync {
    fn update(&self, weight: &Tensor, pre: &Tensor, post: &Tensor, dt: f32) -> CandleResult<Tensor>;
}

pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub weight: Tensor,
    pub rule: Box<dyn SynapseUpdate>,
}

impl Synapse {
    pub fn new(
        pre_idx: usize,
        post_idx: usize,
        pre_size: usize,
        post_size: usize,
        rule: Box<dyn SynapseUpdate>,
        device: &candle_core::Device,
    ) -> CandleResult<Self> {
        // initialize weights small random
        let w = Tensor::randn(0.0f32, 0.1, (post_size, pre_size), device)?;
        Ok(Self {
            pre: pre_idx,
            post: post_idx,
            weight: w,
            rule,
        })
    }

    // update weights from the rule, given references to layers vector
    pub fn update(&mut self, layers: &mut Vec<Layer>) -> CandleResult<()> {
        let pre_act = layers[self.pre].activity()?;
        let post_act = layers[self.post].activity()?;
        let new_w = self.rule.update(&self.weight, &pre_act, &post_act, 1.0)?;
        self.weight = new_w;
        Ok(())
    }
}
