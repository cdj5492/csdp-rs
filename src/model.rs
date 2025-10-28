use crate::layer::basic::Layer;
use crate::synapse::{Synapse, SynapseUpdate};
use candle_core::{Device, DType, Tensor, Result as CandleResult};
use std::rc::Rc;

pub struct Model {
    pub device: Device,
    pub layers: Vec<Layer>,
    pub synapses: Vec<Synapse>,
}

pub struct ProcessOutput {
    // store activations at a few timesteps for inspection/plotting
    pub timestep_activity: Vec<Tensor>,
}

impl Model {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            layers: vec![],
            synapses: vec![],
        }
    }

    pub fn add_layer(&mut self, l: Layer) {
        self.layers.push(l);
    }

    // synapse rule is boxed; we create synapse between layer `pre` and `post`.
    pub fn add_synapse(&mut self, pre_idx: usize, post_idx: usize, rule: Box<dyn SynapseUpdate> ) -> CandleResult<()> {
        let pre = self.layers[pre_idx].size();
        let post = self.layers[post_idx].size();
        let s = Synapse::new(pre_idx, post_idx, pre, post, rule.into(), &self.device)?;
        self.synapses.push(s);
        Ok(())
    }

    // Run one timestep: update layers and synapses once.
    pub fn step(&mut self, external_inputs: Option<&Tensor>) -> CandleResult<()> {
        // 1) compute local global signals (none for minimal example)
        // 2) step each layer: (this computes new activities from internal state and optional clamped inputs)
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let input = if i == 0 { external_inputs } else { None };
            layer.step(input)?;
        }

        // 3) update synapses given pre/post activities
        for syn in self.synapses.iter_mut() {
            syn.update(&mut self.layers)?;
        }

        Ok(())
    }

    // run for T timesteps, optionally clamping inputs/outputs, and return collected outputs
    pub fn process(&mut self, clamp_input: Option<&Tensor>, _clamp_output: Option<&Tensor>, timesteps: usize) -> CandleResult<ProcessOutput> {
        let mut out = ProcessOutput {
            timestep_activity: vec![],
        };
        for _ in 0..timesteps {
            self.step(clamp_input)?;
            // store a snapshot (e.g., layer 0 activity) for inspection
            let a0 = self.layers[0].activity()?;
            out.timestep_activity.push(a0);
        }
        Ok(out)
    }
}
