pub mod csdp;

use crate::layer::Layer;
use candle_core::{Result as CandleResult, Tensor};

pub trait SynapseUpdate: Send + Sync {
    fn update(&self, weight: &Tensor, pre: &Tensor, post: &Tensor, dt: f32)
    -> CandleResult<Tensor>;
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
    pub fn update(&mut self, layers: &mut Vec<Box<dyn Layer>>, dt: f32) -> CandleResult<()> {
        let pre_s = layers[self.pre].output()?;
        let post_s = layers[self.post].output()?;
        let new_w = self.rule.update(&self.weight, &pre_s, &post_s, dt)?;
        self.weight = new_w;
        Ok(())
    }
}
