use crate::layer::Layer;
use crate::layer::bernoulli::BernoulliLayer;
use crate::layer::lif::LIFLayer;
use crate::synapse::csdp::CSDP;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub struct Model {
    pub input_layer: BernoulliLayer,
    pub hidden_layers: Vec<LIFLayer>,
    pub output_layer: LIFLayer,
    // pub output_layer: SoftmaxLayer,
    /// synapses between hidden layers going forward, including input
    pub hidden_synapses_forward: Vec<CSDP>,
    /// synapses between hidden layers going backward
    pub hidden_synapses_backward: Vec<CSDP>,
    /// synapses between hidden layers and output layer
    pub output_synapses: Vec<CSDP>,
    pub dt: f32,
}

/// data returned as output from prcess function
pub struct ProcessOutput {
    pub output_activity: Vec<Tensor>,
    pub final_output: Tensor,
}

impl Model {
    pub fn new(layer_sizes: Vec<usize>, device: &Device, dt: f32) -> Option<Self> {
        // TODO: tune all of these
        // goodness function threshold
        let g_thr = 2.0;
        // for LIF layers
        // let input_r = 0.1;
        // let inhibitory_r = 0.01;
        // called trace constant in paper (ms)
        let tau_lif = 13.0;
        let trace_tau = 5.0;
        let thresh_lambda = 0.01;

        // must have at least an input, one hidden layer, and an output layer
        if layer_sizes.len() < 3 {
            return None;
        }

        // input layer spikes are generated according to a Bernoulli process.
        // requires inputs to be in [0, 1]
        let input_layer = BernoulliLayer::new(layer_sizes[0], device).ok()?;

        // hidden layers
        let mut layers = vec![];
        for &size in layer_sizes.iter().skip(1).take(layer_sizes.len() - 2) {
            let lif_layer =
                LIFLayer::new(size, tau_lif, g_thr, thresh_lambda, trace_tau, device).ok()?;
            layers.push(lif_layer);
        }

        // synapses between every hidden layer, but not output layer
        let mut synapses_forward = vec![];
        let mut synapses_backward = vec![];
        
        synapses_forward.push(
            CSDP::new(input_layer.size(), layers[0].size(), device).ok()?
        );

        for i in 0..layers.len() - 1 {
            let pre = layers[i].size();
            let post = layers[i + 1].size();
            let s = CSDP::new(pre, post, device).ok()?;
            if i > 0 {
                let s_back = CSDP::new(post, pre, device).ok()?;
                synapses_backward.push(s_back);
            }
            synapses_forward.push(s);
        }

        // final layer. Connects to all hidden layers
        let output_layer = LIFLayer::new(
            *layer_sizes.last().unwrap(),
            tau_lif,
            g_thr,
            thresh_lambda,
            trace_tau,
            device,
        )
        .ok()?;

        // all layers to output layer
        let mut output_synapses = vec![];
        for layer in layers.iter() {
            let pre = layer.size();
            let post = output_layer.size();
            let s = CSDP::new(pre, post, device).ok()?;
            output_synapses.push(s);
        }

        Some(Self {
            input_layer,
            hidden_layers: layers,
            output_layer,
            hidden_synapses_forward: synapses_forward,
            hidden_synapses_backward: synapses_backward,
            output_synapses,
            dt,
        })
    }

    // Run one timestep: update layers and synapses once.
    pub fn step(&mut self, input: &Tensor) -> CandleResult<()> {
        for layer in self.hidden_layers.iter_mut() {
            layer.reset_input()?;
        }
        self.output_layer.reset_input()?;

        self.input_layer.add_input(input)?;
        self.input_layer.step(self.dt)?;
        let post_input = self.hidden_synapses_forward[0].forward(self.input_layer.output()?)?;

        // forward connections
        for i in 0..self.hidden_layers.len() {
            if i == 0 {
                self.hidden_layers[i].add_input(&post_input)?;
            } else {
                let pre_activity = self.hidden_layers[i - 1].output()?;
                let post_activity = self.hidden_synapses_forward[i].forward(pre_activity)?;
                self.hidden_layers[i].add_input(&post_activity)?;
            }
        }
        

        // TODO: backward connections
        // for (i, layer) in self.hidden_layers.iter_mut().rev().enumerate() {
        // }
        
        // output layer connections from all hidden layers
        for (i, layer) in self.hidden_layers.iter().enumerate() {
            let pre_activity = layer.output()?;
            let post_activity = self.output_synapses[i].forward(pre_activity)?;
            self.output_layer.add_input(&post_activity)?;
        }

        // step all hidden layers
        for layer in self.hidden_layers.iter_mut() {
            layer.step(self.dt)?;
        }
        
        self.output_layer.step(self.dt)?;

        Ok(())
    }
    
    fn reset(&mut self) -> CandleResult<()> {
        self.input_layer.reset()?;
        for layer in self.hidden_layers.iter_mut() {
            layer.reset()?;
        }
        self.output_layer.reset()?;
        Ok(())
    }

    // run for T timesteps, and return collected outputs
    pub fn process(
        &mut self,
        input: &Tensor,
        timesteps: usize,
        collect_data: bool,
        device: &Device,
    ) -> CandleResult<ProcessOutput> {
        let mut out = ProcessOutput {
            output_activity: vec![],
            final_output: Tensor::zeros((0, 0), DType::F32, device)?,
        };
        self.reset()?;
        for _ in 0..timesteps {
            self.step(&input)?;

            if collect_data {
                // inspection test
                let output = self.output_layer.output()?;
                out.output_activity.push(output.clone());
            }
        }
        out.final_output = self.hidden_layers.last().unwrap().output()?.clone();
        Ok(out)
    }
}
