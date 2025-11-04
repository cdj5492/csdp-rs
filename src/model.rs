use crate::layer::Layer;
use crate::synapse::{Synapse, SynapseUpdate};
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub struct Model {
    pub device: Device,
    pub layer_inputs: Vec<Tensor>,
    pub layers: Vec<Box<dyn Layer>>,
    pub synapses: Vec<Synapse>,
    pub dt: f32,
}

/// data returned as output from prcess function
pub struct ProcessOutput {
    pub output_activity: Vec<Tensor>,
    pub final_output: Tensor,
}

impl Model {
    pub fn new(device: Device, dt: f32) -> Self {
        Self {
            device,
            layer_inputs: vec![],
            layers: vec![],
            synapses: vec![],
            dt,
        }
    }

    pub fn add_layer(&mut self, l: Box<dyn Layer>) -> CandleResult<()> {
        self.layer_inputs
            .push(Tensor::zeros((l.size(), 1), DType::F32, &self.device)?);
        self.layers.push(l);
        Ok(())
    }

    // synapse rule is boxed; we create synapse between layer `pre` and `post`.
    pub fn add_synapse(
        &mut self,
        pre_idx: usize,
        post_idx: usize,
        rule: Box<dyn SynapseUpdate>,
    ) -> CandleResult<()> {
        let pre = self.layers[pre_idx].size();
        let post = self.layers[post_idx].size();
        let s = Synapse::new(pre_idx, post_idx, pre, post, rule.into(), &self.device)?;
        self.synapses.push(s);
        Ok(())
    }

    // Run one timestep: update layers and synapses once.
    pub fn step(&mut self, input: &Tensor) -> CandleResult<()> {
        // first layer is clamped to input
        self.layer_inputs[0] = input.clone();
        // clear other layer inputs
        for i in 1..self.layer_inputs.len() {
            self.layer_inputs[i] =
                Tensor::zeros((self.layers[i].size(), 1), DType::F32, &self.device)?;
        }

        // go through all synapses to accumulate inputs to layers
        for syn in self.synapses.iter() {
            let pre_out = self.layers[syn.pre].output()?;
            let w_pre = syn.weight.matmul(&pre_out)?;
            self.layer_inputs[syn.post] = self.layer_inputs[syn.post].add(&w_pre)?;
        }

        // calculate layer dynamics
        for (layer_input, layer) in self.layer_inputs.iter().zip(self.layers.iter_mut()) {
            layer.step(layer_input, self.dt)?;
        }

        // calculate synapse updates
        for syn in self.synapses.iter_mut() {
            syn.update(&mut self.layers, self.dt)?;
        }

        Ok(())
    }

    // run for T timesteps, and return collected outputs
    pub fn process(
        &mut self,
        input: &Tensor,
        _output: Option<&Tensor>,
        timesteps: usize,
        collect_data: bool,
    ) -> CandleResult<ProcessOutput> {
        let mut out = ProcessOutput {
            output_activity: vec![],
            final_output: Tensor::zeros((0, 0), DType::F32, &self.device)?,
        };
        for _ in 0..timesteps {
            self.step(&input)?;

            if collect_data {
                // inspection test
                let a0 = self.layers.last().unwrap().activity()?;
                out.output_activity.push(a0.clone());
            }
        }
        out.final_output = self.layers.last().unwrap().output()?.clone();
        Ok(out)
    }
}
