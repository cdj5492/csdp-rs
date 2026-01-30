pub mod app;

use crate::layer::LayerPosition;
use crate::synapse::{LayerId, SynapseId, WeightStats};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// State shared between training loop and visualization thread
pub struct VisualizationState {
    pub model_structure: ModelStructure,
    pub runtime_stats: RuntimeStats,
    pub neuron_traces: NeuronTraceManager,
    pub should_close: bool,
    pub is_paused: bool,
    pub total_epochs: usize,
}

/// Structure of the model for visualization
#[derive(Clone, Debug)]
pub struct ModelStructure {
    pub layers: Vec<LayerVisInfo>,
    pub synapses: Vec<SynapseVisInfo>,
}

/// Visualization info for a layer
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LayerVisInfo {
    pub id: LayerId,
    pub name: String,
    pub layer_type: String,
    pub size: usize,
    pub position: LayerPosition,
    pub current_activity: Vec<f32>,
    pub spike_count: usize,
}

/// Visualization info for a synapse
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct SynapseVisInfo {
    pub id: SynapseId,
    pub pre_layer: LayerId,
    pub post_layer: LayerId,
    pub synapse_type: String,
    pub weight_stats: WeightStats,
}

/// Runtime statistics
#[derive(Clone, Default)]
pub struct RuntimeStats {
    pub epoch: usize,
    pub iteration: usize,
    pub timestep: usize,
    pub iterations_per_second: f32,
}

/// Manager for tracked neuron spike traces
pub struct NeuronTraceManager {
    pub tracked_neurons: Vec<TrackedNeuron>,
    pub max_history: usize,
}

/// A tracked neuron with its spike history
#[derive(Clone)]
pub struct TrackedNeuron {
    pub layer_id: LayerId,
    pub neuron_idx: usize,
    pub spike_history: VecDeque<f32>,
    pub timesteps: VecDeque<usize>,
    pub display_name: String,
}

impl TrackedNeuron {
    pub fn new(layer_id: LayerId, neuron_idx: usize, layer_name: &str) -> Self {
        Self {
            layer_id,
            neuron_idx,
            spike_history: VecDeque::new(),
            timesteps: VecDeque::new(),
            display_name: format!("{}[{}]", layer_name, neuron_idx),
        }
    }

    pub fn add_spike(&mut self, spike_value: f32, timestep: usize, max_history: usize) {
        self.spike_history.push_back(spike_value);
        self.timesteps.push_back(timestep);

        // Maintain max history size
        while self.spike_history.len() > max_history {
            self.spike_history.pop_front();
            self.timesteps.pop_front();
        }
    }
}

impl NeuronTraceManager {
    pub fn new(max_history: usize) -> Self {
        Self {
            tracked_neurons: Vec::new(),
            max_history,
        }
    }

    pub fn add_neuron(&mut self, layer_id: LayerId, neuron_idx: usize, layer_name: &str) {
        // Check if already tracking this neuron
        for neuron in &self.tracked_neurons {
            if neuron.layer_id == layer_id && neuron.neuron_idx == neuron_idx {
                return; // Already tracking
            }
        }

        let tracked = TrackedNeuron::new(layer_id, neuron_idx, layer_name);
        self.tracked_neurons.push(tracked);
    }

    pub fn clear(&mut self) {
        self.tracked_neurons.clear();
    }
}

impl VisualizationState {
    pub fn new(total_epochs: usize) -> Self {
        Self {
            model_structure: ModelStructure {
                layers: Vec::new(),
                synapses: Vec::new(),
            },
            runtime_stats: RuntimeStats::default(),
            neuron_traces: NeuronTraceManager::new(1000),
            should_close: false,
            is_paused: true,
            total_epochs,
        }
    }
}

impl Default for VisualizationState {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Start the visualization in a separate thread
pub fn start_visualization(state: Arc<Mutex<VisualizationState>>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 800.0])
                .with_title("Neural Network Visualization"),
            event_loop_builder: Some(Box::new(|builder| {
                // Enable any_thread for Linux to allow event loop on non-main thread
                // This is necessary when visualization runs in a separate thread
                #[cfg(target_os = "linux")]
                {
                    // Check which display server is in use
                    if std::env::var("WAYLAND_DISPLAY").is_ok() {
                        use winit::platform::wayland::EventLoopBuilderExtWayland;
                        builder.with_any_thread(true);
                    } else {
                        use winit::platform::x11::EventLoopBuilderExtX11;
                        builder.with_any_thread(true);
                    }
                }
            })),
            ..Default::default()
        };

        let _ = eframe::run_native(
            "Neural Network Visualization",
            options,
            Box::new(|_cc| Ok(Box::new(app::NeuralNetworkVisualizerApp::new(state)))),
        );
    })
}
