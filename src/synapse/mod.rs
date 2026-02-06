pub mod csdp;

use crate::layer::Layer;
use candle_core::{Result as CandleResult, Tensor};

#[allow(dead_code)]
pub trait SynapseUpdate: Send + Sync {
    fn update(
        &self,
        weight: &Tensor,
        pre_layer: &Box<dyn Layer>,
        post_layer: &Box<dyn Layer>,
        dt: f32,
    ) -> CandleResult<Tensor>;
}

/// New trait for generic synapse operations
pub trait SynapseOps: Send + Sync {
    /// Forward pass: compute post-synaptic input from pre-synaptic activity
    fn forward(&self, pre: &Tensor) -> CandleResult<Tensor>;

    /// Update weights based on pre and post activity
    fn update_weights(
        &mut self,
        pre_activity: &Tensor,
        post_layer: &mut Box<dyn Layer>,
        dt: f32,
    ) -> CandleResult<()>;

    /// Get weight statistics for visualization
    fn weight_stats(&self) -> CandleResult<WeightStats>;
}

/// Statistics about synapse weights for visualization
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WeightStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub num_weights: usize,
}

/// Type IDs for layers and synapses
pub type LayerId = usize;
pub type SynapseId = usize;

/// Metadata about a synapse connection
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SynapseMetadata {
    pub id: SynapseId,
    pub pre_layer: LayerId,
    pub post_layer: LayerId,
    pub synapse_type: String,
    pub is_learning: bool,
}

/// Wrapper for a synapse connection with metadata
pub struct SynapseConnection {
    pub metadata: SynapseMetadata,
    pub synapse: Box<dyn SynapseOps>,
}

#[allow(dead_code)]
pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub weight: Tensor,
    pub rule: Box<dyn SynapseUpdate>,
}

#[allow(dead_code)]
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
    pub fn update(&mut self, layers: &mut [Box<dyn Layer>], dt: f32) -> CandleResult<()> {
        // let pre_s = layers[self.pre].output()?;
        // let post_s = layers[self.post].output()?;
        let new_w = self
            .rule
            .update(&self.weight, &layers[self.pre], &layers[self.post], dt)?;
        self.weight = new_w;
        Ok(())
    }
}
