use crate::layer::bernoulli::BernoulliLayer;
use crate::layer::lif::LIFLayer;
use crate::layer::{Layer, LayerMetadata, LayerPosition};
use crate::synapse::csdp::CSDP;
use crate::synapse::{LayerId, SynapseConnection, SynapseMetadata, SynapseOps};
use crate::visualization::{LayerVisInfo, SynapseVisInfo};
use candle_core::{DType, Device, Result as CandleResult, Tensor};

/// Configuration for creating a model
pub struct ModelConfig {
    pub layer_configs: Vec<LayerConfig>,
    pub synapse_configs: Vec<SynapseConfig>,
    pub dt: f32,
}

/// Configuration for a single layer
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum LayerConfig {
    Bernoulli {
        size: usize,
        name: Option<String>,
    },
    LIF {
        size: usize,
        tau: f32,
        g_thr: f32,
        thresh_lambda: f32,
        trace_tau: f32,
        name: Option<String>,
    },
}

/// Configuration for a synapse connection
#[derive(Debug, Clone)]
pub struct SynapseConfig {
    pub pre_layer: usize,
    pub post_layer: usize,
    pub synapse_type: SynapseType,
    pub bidirectional: bool,
}

/// Types of synapses available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum SynapseType {
    CSDP,
}

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
    pub layer_metadata: Vec<LayerMetadata>,
    pub synapses: Vec<SynapseConnection>,
    pub is_learning: bool,
    pub dt: f32,
    pub device: Device,
}

/// Legacy Model structure (kept for reference, can be removed later)
pub struct _OldModel {
    pub input_layer: BernoulliLayer,
    pub hidden_layers: Vec<LIFLayer>,
    pub output_layer: LIFLayer,
    pub hidden_synapses_forward: Vec<CSDP>,
    pub hidden_synapses_backward: Vec<CSDP>,
    pub output_synapses: Vec<CSDP>,
    pub is_learning: bool,
    pub dt: f32,
}

/// data returned as output from prcess function
pub struct ProcessOutput {
    pub output_activity: Vec<Tensor>,
    pub final_output: Tensor,
}

impl Model {
    /// Create a new CSDP model with default configuration (backward compatible).
    pub fn new(layer_sizes: Vec<usize>, device: &Device, dt: f32) -> Option<Self> {
        // Default LIF parameters
        let g_thr = 2.0;
        let tau_lif = 13.0;
        let trace_tau = 5.0;
        let thresh_lambda = 0.01;

        if layer_sizes.len() < 3 {
            return None;
        }

        // Build layer configs
        let mut layer_configs = vec![];

        // Input layer (Bernoulli)
        layer_configs.push(LayerConfig::Bernoulli {
            size: layer_sizes[0],
            name: Some("Input".to_string()),
        });

        // Hidden layers (LIF)
        for (i, &size) in layer_sizes
            .iter()
            .skip(1)
            .take(layer_sizes.len() - 2)
            .enumerate()
        {
            layer_configs.push(LayerConfig::LIF {
                size,
                tau: tau_lif,
                g_thr,
                thresh_lambda,
                trace_tau,
                name: Some(format!("Hidden_{}", i)),
            });
        }

        // Output layer (LIF)
        layer_configs.push(LayerConfig::LIF {
            size: *layer_sizes.last()?,
            tau: tau_lif,
            g_thr,
            thresh_lambda,
            trace_tau,
            name: Some("Output".to_string()),
        });

        // Build synapse configs (same topology as before)
        let mut synapse_configs = vec![];

        // Input to first hidden layer
        synapse_configs.push(SynapseConfig {
            pre_layer: 0,
            post_layer: 1,
            synapse_type: SynapseType::CSDP,
            bidirectional: false,
        });

        // Hidden layers with bidirectional connections
        for i in 1..layer_configs.len() - 2 {
            synapse_configs.push(SynapseConfig {
                pre_layer: i,
                post_layer: i + 1,
                synapse_type: SynapseType::CSDP,
                bidirectional: true,
            });
        }

        // All hidden layers to output layer
        for i in 1..layer_configs.len() - 1 {
            synapse_configs.push(SynapseConfig {
                pre_layer: i,
                post_layer: layer_configs.len() - 1,
                synapse_type: SynapseType::CSDP,
                bidirectional: false,
            });
        }

        let config = ModelConfig {
            layer_configs,
            synapse_configs,
            dt,
        };

        Self::from_config(config, device).ok()
    }

    /// Create a model from a configuration
    pub fn from_config(config: ModelConfig, device: &Device) -> CandleResult<Self> {
        let mut layers: Vec<Box<dyn Layer>> = vec![];
        let mut layer_metadata = vec![];

        // Create layers
        for (id, layer_config) in config.layer_configs.iter().enumerate() {
            let (layer, metadata) = Self::create_layer(id, layer_config, device)?;
            layers.push(layer);
            layer_metadata.push(metadata);
        }

        // Create synapses
        let mut synapses = vec![];
        let mut synapse_id = 0;

        for syn_config in config.synapse_configs.iter() {
            let pre_size = layers[syn_config.pre_layer].size();
            let post_size = layers[syn_config.post_layer].size();

            // Forward synapse
            let synapse =
                Self::create_synapse(syn_config.synapse_type, pre_size, post_size, device)?;
            let metadata = SynapseMetadata {
                id: synapse_id,
                pre_layer: syn_config.pre_layer,
                post_layer: syn_config.post_layer,
                synapse_type: format!("{:?}", syn_config.synapse_type),
                is_learning: true,
            };
            synapses.push(SynapseConnection { metadata, synapse });
            synapse_id += 1;

            // Backward synapse if bidirectional
            if syn_config.bidirectional {
                let synapse =
                    Self::create_synapse(syn_config.synapse_type, post_size, pre_size, device)?;
                let metadata = SynapseMetadata {
                    id: synapse_id,
                    pre_layer: syn_config.post_layer,
                    post_layer: syn_config.pre_layer,
                    synapse_type: format!("{:?}", syn_config.synapse_type),
                    is_learning: true,
                };
                synapses.push(SynapseConnection { metadata, synapse });
                synapse_id += 1;
            }
        }

        Ok(Self {
            layers,
            layer_metadata,
            synapses,
            is_learning: true,
            dt: config.dt,
            device: device.clone(),
        })
    }

    fn create_layer(
        id: usize,
        config: &LayerConfig,
        device: &Device,
    ) -> CandleResult<(Box<dyn Layer>, LayerMetadata)> {
        let (layer, layer_type, size, name) = match config {
            LayerConfig::Bernoulli { size, name } => {
                let layer = BernoulliLayer::new(*size, device)?;
                let name = name.clone().unwrap_or_else(|| format!("Layer_{}", id));
                (
                    Box::new(layer) as Box<dyn Layer>,
                    "Bernoulli".to_string(),
                    *size,
                    name,
                )
            }
            LayerConfig::LIF {
                size,
                tau,
                g_thr,
                thresh_lambda,
                trace_tau,
                name,
            } => {
                let layer = LIFLayer::new(*size, *tau, *g_thr, *thresh_lambda, *trace_tau, device)?;
                let name = name.clone().unwrap_or_else(|| format!("Layer_{}", id));
                (
                    Box::new(layer) as Box<dyn Layer>,
                    "LIF".to_string(),
                    *size,
                    name,
                )
            }
        };

        // Calculate position based on layer index
        let position = LayerPosition {
            x: id as f32 * 150.0 + 50.0,
            y: 200.0,
        };

        let metadata = LayerMetadata {
            id,
            name,
            layer_type,
            size,
            position,
        };

        Ok((layer, metadata))
    }

    fn create_synapse(
        synapse_type: SynapseType,
        pre_size: usize,
        post_size: usize,
        device: &Device,
    ) -> CandleResult<Box<dyn SynapseOps>> {
        match synapse_type {
            SynapseType::CSDP => {
                let csdp = CSDP::new(pre_size, post_size, device)?;
                Ok(Box::new(csdp))
            }
        }
    }

    #[allow(dead_code)]
    pub fn enable_learning(&mut self) {
        self.is_learning = true;
    }

    #[allow(dead_code)]
    pub fn disable_learning(&mut self) {
        self.is_learning = false;
    }

    /// Run one timestep: update layers and synapses once.
    pub fn step(&mut self, input: &Tensor) -> CandleResult<()> {
        // Reset inputs for all layers except input layer
        for layer in self.layers.iter_mut().skip(1) {
            layer.reset_input()?;
        }

        // Add input to first layer and step it
        if !self.layers.is_empty() {
            self.layers[0].add_input(input)?;
            self.layers[0].step(self.dt)?;
        }

        // Process synapses: forward pass and weight updates
        for syn_conn in self.synapses.iter_mut() {
            let pre_layer_id = syn_conn.metadata.pre_layer;
            let post_layer_id = syn_conn.metadata.post_layer;

            // Get pre-synaptic activity
            let pre_activity = self.layers[pre_layer_id].output()?.clone();

            // Forward pass
            let post_input = syn_conn.synapse.forward(&pre_activity)?;

            // Add to post-synaptic layer input
            self.layers[post_layer_id].add_input(&post_input)?;

            // Update weights if learning is enabled
            if self.is_learning && syn_conn.metadata.is_learning {
                let post_activity = self.layers[post_layer_id].output()?.clone();
                syn_conn
                    .synapse
                    .update_weights(&pre_activity, &post_activity, self.dt)?;
            }
        }

        // Step all layers except the input layer (already stepped)
        for layer in self.layers.iter_mut().skip(1) {
            layer.step(self.dt)?;
        }

        Ok(())
    }

    fn reset(&mut self) -> CandleResult<()> {
        for layer in self.layers.iter_mut() {
            layer.reset()?;
        }
        Ok(())
    }

    /// run for T timesteps, and return collected outputs
    pub fn process(
        &mut self,
        input: &Tensor,
        timesteps: usize,
        collect_data: bool,
        _device: &Device,
    ) -> CandleResult<ProcessOutput> {
        let mut out = ProcessOutput {
            output_activity: vec![],
            final_output: Tensor::zeros((0, 0), DType::F32, &self.device)?,
        };
        self.reset()?;
        for _ in 0..timesteps {
            self.step(input)?;

            if collect_data && !self.layers.is_empty() {
                let output = self.layers.last().unwrap().output()?;
                out.output_activity.push(output.clone());
            }
        }

        if !self.layers.is_empty() {
            out.final_output = self.layers.last().unwrap().output()?.clone();
        }

        Ok(out)
    }

    /// Get a specific neuron's output value for visualization
    pub fn get_neuron_output(&self, layer_id: LayerId, neuron_idx: usize) -> CandleResult<f32> {
        if layer_id >= self.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "Layer {} does not exist",
                layer_id
            )));
        }

        let output = self.layers[layer_id].output()?;
        let output_vec = output.to_vec1::<f32>()?;

        if neuron_idx >= output_vec.len() {
            return Err(candle_core::Error::Msg(format!(
                "Neuron {} does not exist in layer {} (size: {})",
                neuron_idx,
                layer_id,
                output_vec.len()
            )));
        }

        Ok(output_vec[neuron_idx])
    }

    /// Get a snapshot of the model structure for visualization
    pub fn get_visualization_snapshot(&self) -> CandleResult<crate::visualization::ModelStructure> {
        let mut layer_vis_infos = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let output = layer.output()?;
            let output_vec = output.flatten(0, 1)?.to_vec1::<f32>()?;

            // Count spikes (values > 0.5)
            let spike_count = output_vec.iter().filter(|&&v| v > 0.5).count();

            let layer_info = LayerVisInfo {
                id: i,
                name: self.layer_metadata[i].name.clone(),
                layer_type: self.layer_metadata[i].layer_type.clone(),
                size: layer.size(),
                position: self.layer_metadata[i].position,
                current_activity: output_vec,
                spike_count,
            };

            layer_vis_infos.push(layer_info);
        }

        let mut synapse_vis_infos = Vec::new();

        for syn_conn in &self.synapses {
            let weight_stats = syn_conn.synapse.weight_stats()?;

            let synapse_info = SynapseVisInfo {
                id: syn_conn.metadata.id,
                pre_layer: syn_conn.metadata.pre_layer,
                post_layer: syn_conn.metadata.post_layer,
                synapse_type: syn_conn.metadata.synapse_type.clone(),
                weight_stats,
            };

            synapse_vis_infos.push(synapse_info);
        }

        Ok(crate::visualization::ModelStructure {
            layers: layer_vis_infos,
            synapses: synapse_vis_infos,
        })
    }
}
