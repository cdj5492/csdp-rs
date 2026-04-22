use crate::layer::bernoulli::BernoulliLayer;
use crate::layer::lif::LIFLayer;
use crate::layer::mod_signal::multi_class::MultiClassModSignal;
use crate::layer::{Layer, LayerMetadata, LayerPosition};
use crate::synapse::csdp::CSDP;
use crate::synapse::{SynapseConnection, SynapseMetadata};
use candle_core::{Device, Result as CandleResult, Tensor};

pub struct CSDPMultiModel {
    pub layers: Vec<Box<dyn Layer>>,
    pub layer_metadata: Vec<LayerMetadata>,
    pub synapses: Vec<SynapseConnection>,
    pub is_learning: bool,
    pub dt: f32,
    pub device: Device,
    pub num_classes: usize,
    pub timesteps: usize,
}

impl CSDPMultiModel {
    pub fn new(
        input_size: usize,
        hidden_sizes: &[usize], // E.g. [2000]
        num_classes: usize,
        device: &Device,
        dt: f32,
        timesteps: usize,
    ) -> CandleResult<Self> {
        let g_thr = 0.5;
        let tau_lif = 13.0;
        let trace_tau = 5.0;
        let thresh_lambda = 0.01;

        let mut layers: Vec<Box<dyn Layer>> = vec![];
        let mut layer_metadata = vec![];

        // Input Layer (Bernoulli)
        let in_layer = BernoulliLayer::new(input_size, device)?;
        layers.push(Box::new(in_layer));
        layer_metadata.push(LayerMetadata {
            id: 0,
            name: "Input".to_string(),
            layer_type: "Bernoulli".to_string(),
            size: input_size,
            position: LayerPosition { x: 0.0, y: 0.0 },
        });

        // Hidden Layers (LIF with MultiClassModSignal)
        for (i, &size) in hidden_sizes.iter().enumerate() {
            assert!(
                size % num_classes == 0,
                "Hidden size {} must be divisible by num_classes {}",
                size,
                num_classes
            );

            let mod_signal = Box::new(MultiClassModSignal::new(
                size,
                trace_tau,
                1.0, // max_z roughly
                0.3, // threshold is compared to normalized mean goodness [0..1]
                num_classes,
                device,
            )?);

            let lif_layer = LIFLayer::new(size, tau_lif, g_thr, thresh_lambda, mod_signal, device)?;
            layers.push(Box::new(lif_layer));
            layer_metadata.push(LayerMetadata {
                id: i + 1,
                name: format!("Hidden_{}", i),
                layer_type: "LIF (MultiClass)".to_string(),
                size,
                position: LayerPosition {
                    x: 100.0 * (i as f32 + 1.0),
                    y: 100.0,
                },
            });
        }

        // Synapses (dense between consecutive layers)
        let mut synapses = vec![];
        for i in 0..layers.len() - 1 {
            let pre_size = layers[i].size();
            let post_size = layers[i + 1].size();

            // Forward synapse
            let synapse = Box::new(CSDP::new(pre_size, post_size, device)?);
            synapses.push(SynapseConnection {
                metadata: SynapseMetadata {
                    id: synapses.len(),
                    pre_layer: i,
                    post_layer: i + 1,
                    synapse_type: "CSDP".to_string(),
                    is_learning: true,
                },
                synapse,
            });

            // Backward synapse
            if i > 0 {
                // Don't do backward to Bernoulli Input layer
                let synapse_back = Box::new(CSDP::new(post_size, pre_size, device)?);
                synapses.push(SynapseConnection {
                    metadata: SynapseMetadata {
                        id: synapses.len(),
                        pre_layer: i + 1,
                        post_layer: i,
                        synapse_type: "CSDP".to_string(),
                        is_learning: true,
                    },
                    synapse: synapse_back,
                });
            }
        }

        Ok(Self {
            layers,
            layer_metadata,
            synapses,
            is_learning: true,
            dt,
            device: device.clone(),
            num_classes,
            timesteps,
        })
    }

    pub fn enable_learning(&mut self) {
        self.is_learning = true;
    }

    pub fn disable_learning(&mut self) {
        self.is_learning = false;
    }

    pub fn reset(&mut self, batch_size: usize) -> CandleResult<()> {
        for layer in self.layers.iter_mut() {
            layer.reset(batch_size)?;
        }
        Ok(())
    }

    /// Step the SNN by one timestep
    pub fn step(&mut self, input: &Tensor, truth_classes: Option<&Tensor>) -> CandleResult<()> {
        for layer in self.layers.iter_mut() {
            layer.reset_input()?;
        }

        // Add input to Layer 0
        self.layers[0].add_input(input)?;
        self.layers[0].step(self.dt)?;

        // If truth_classes are passed, we provide them to all multi-class layers as `positive_sample`
        if let Some(target) = truth_classes {
            for layer in self.layers.iter_mut().skip(1) {
                layer.set_positive_sample(target);
            }
        } else {
            let dummy = Tensor::zeros((1, 1), candle_core::DType::F32, &self.device)?;
            for layer in self.layers.iter_mut().skip(1) {
                layer.set_positive_sample(&dummy);
            }
        }

        // Forward synapse pass
        for syn_conn in self.synapses.iter_mut() {
            let pre_layer_id = syn_conn.metadata.pre_layer;
            let post_layer_id = syn_conn.metadata.post_layer;
            let pre_act = self.layers[pre_layer_id].output()?;
            let post_input = syn_conn.synapse.forward(&pre_act)?;
            self.layers[post_layer_id].add_input(&post_input)?;
        }

        // Step layers
        for layer in self.layers.iter_mut().skip(1) {
            layer.step(self.dt)?;
        }

        // STDP Weight Updates
        for syn_conn in self.synapses.iter_mut() {
            if self.is_learning && syn_conn.metadata.is_learning {
                let pre_layer_id = syn_conn.metadata.pre_layer;
                let post_layer_id = syn_conn.metadata.post_layer;
                let pre_act = self.layers[pre_layer_id].output()?;

                // Rust lifetime scope rule limitation requires us to extract pre_act,
                // but we also need mutable access to layers. pre_act is cloned for safety.
                let pre_act_cloned = pre_act.clone();
                syn_conn.synapse.update_weights(
                    &pre_act_cloned,
                    &mut self.layers[post_layer_id],
                    self.dt,
                )?;
            }
        }

        Ok(())
    }

    /// Run the model for T timesteps to accumulate goodness scores across classes
    pub fn predict_scores(&mut self, inputs: &[Tensor]) -> CandleResult<Tensor> {
        let batched_inputs = Tensor::cat(inputs, 0)?;
        // Shape of batched_inputs: (batch_size, input_dim).
        // SNN expects (input_dim, batch_size).
        let input_tensor = batched_inputs.transpose(0, 1)?.contiguous()?;

        let batch_size = input_tensor.dim(1)?;
        self.reset(batch_size)?;

        let was_learning = self.is_learning;
        self.disable_learning();

        let mut total_goodness = Tensor::zeros(
            (batch_size, self.num_classes),
            candle_core::DType::F32,
            &self.device,
        )?;

        // We evaluate T timesteps.
        for _ in 0..self.timesteps {
            self.step(&input_tensor, None)?;

            // Accumulate instantaneous goodness at each timestep
            for layer in self.layers.iter().skip(1) {
                // Skip input layer
                let h = layer.output()?; // Shape: (size, batch_size)
                let h_t = h.transpose(0, 1)?.contiguous()?; // (batch_size, size)

                let chunks = h_t.chunk(self.num_classes, 1)?;
                let mut layer_goodnesses = Vec::new();
                for chunk in chunks {
                    let g = chunk.sqr()?.mean_keepdim(1)?; // (batch_size, 1)
                    layer_goodnesses.push(g);
                }
                let layer_g_tensor = Tensor::cat(&layer_goodnesses, 1)?; // (batch_size, num_classes)
                total_goodness = total_goodness.broadcast_add(&layer_g_tensor)?;
            }
        }

        // Average the goodness across the timesteps
        let total_goodness = (total_goodness / self.timesteps as f64)?;

        if was_learning {
            self.enable_learning();
        }

        Ok(total_goodness)
    }

    pub fn train(
        &mut self,
        x: &Tensor,
        label_classes: &Tensor,
        record_layer_id: Option<usize>,
    ) -> CandleResult<Option<Vec<Vec<f32>>>> {
        let x_t = x.transpose(0, 1)?.contiguous()?;
        let batch_size = x_t.dim(1)?;
        self.reset(batch_size)?;

        let mut history = if record_layer_id.is_some() {
            Some(Vec::with_capacity(self.timesteps))
        } else {
            None
        };

        // Let the STDP rules train based on MultiClassModSignal over T timesteps
        for _ in 0..self.timesteps {
            self.step(&x_t, Some(label_classes))?;

            if let (Some(id), Some(hist)) = (record_layer_id, history.as_mut()) {
                if let Some(layer) = self.layers.get(id) {
                    let output = layer.output()?; // shape: (size, batch_size)
                    let output_b0 = output.narrow(1, 0, 1)?; // take first item in batch
                    let spikes = output_b0.flatten_all()?.to_vec1::<f32>()?;
                    hist.push(spikes);
                }
            }
        }

        Ok(history)
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> CandleResult<()> {
        let mut tensor_map = std::collections::HashMap::new();
        for syn_conn in &self.synapses {
            let state = syn_conn.synapse.get_state()?;
            let prefix = format!("synapse_{}_", syn_conn.metadata.id);
            for (key, tensor) in state {
                tensor_map.insert(format!("{}{}", prefix, key), tensor);
            }
        }
        candle_core::safetensors::save(&tensor_map, path)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> CandleResult<()> {
        let loaded_tensors = candle_core::safetensors::load(path, &self.device)?;
        for syn_conn in self.synapses.iter_mut() {
            let prefix = format!("synapse_{}_", syn_conn.metadata.id);
            let mut state = std::collections::HashMap::new();
            for (key, tensor) in &loaded_tensors {
                if key.starts_with(&prefix) {
                    let local_key = key.strip_prefix(&prefix).unwrap().to_string();
                    state.insert(local_key, tensor.clone());
                }
            }
            if !state.is_empty() {
                syn_conn.synapse.set_state(&state)?;
            }
        }
        Ok(())
    }

    pub fn get_visualization_snapshot(&self) -> CandleResult<crate::visualization::ModelStructure> {
        let mut layers = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let output = layer.output()?;
            let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
            let spike_count = output_vec.iter().filter(|&&v| v > 0.5).count();

            layers.push(crate::visualization::LayerVisInfo {
                id: i,
                name: self.layer_metadata[i].name.clone(),
                layer_type: self.layer_metadata[i].layer_type.clone(),
                size: layer.size(),
                position: self.layer_metadata[i].position,
                velocity: (0.0, 0.0),
                current_activity: output_vec,
                spike_count,
            });
        }

        let mut synapses = Vec::new();
        for syn_conn in &self.synapses {
            synapses.push(crate::visualization::SynapseVisInfo {
                id: syn_conn.metadata.id,
                pre_layer: syn_conn.metadata.pre_layer,
                post_layer: syn_conn.metadata.post_layer,
                synapse_type: syn_conn.metadata.synapse_type.clone(),
                weight_stats: syn_conn.synapse.weight_stats()?,
            });
        }

        Ok(crate::visualization::ModelStructure { layers, synapses })
    }
}
