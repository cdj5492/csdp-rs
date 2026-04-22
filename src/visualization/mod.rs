pub mod app;

use crate::layer::LayerPosition;
use crate::synapse::{LayerId, SynapseId, WeightStats};
use std::sync::{Arc, Mutex};

pub static GLOBAL_LOGS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// State shared between training loop and visualization thread
pub struct VisualizationState {
    pub model_structure: ModelStructure,
    pub runtime_stats: RuntimeStats,
    pub should_close: bool,
    pub is_paused: bool,
    pub total_epochs: usize,
    pub positions_initialized: bool,
    pub selected_layer_id: Option<LayerId>,
    pub epoch_spike_history: Option<(usize, Vec<Vec<f32>>)>,
    pub environment_state: Option<Vec<f64>>,
    pub epoch_rewards: Vec<(usize, f32)>,
    pub save_requested: bool,
    pub load_requested: bool,
    pub delay_ms: u64,
    pub render_trail: Vec<(f64, f64)>,
    pub model_probabilities: Option<Vec<(String, Vec<f32>)>>,
    pub sort_probabilities: bool,
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
    pub velocity: (f32, f32), // For force-directed layout
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

impl VisualizationState {
    pub fn new(total_epochs: usize) -> Self {
        Self {
            model_structure: ModelStructure {
                layers: Vec::new(),
                synapses: Vec::new(),
            },
            runtime_stats: RuntimeStats::default(),
            should_close: false,
            is_paused: true, // Start unpaused so data begins collecting immediately
            total_epochs,
            positions_initialized: false,
            selected_layer_id: None,
            epoch_spike_history: None,
            environment_state: None,
            epoch_rewards: Vec::new(),
            save_requested: false,
            load_requested: false,
            delay_ms: 0,
            render_trail: Vec::new(),
            model_probabilities: None,
            sort_probabilities: false,
        }
    }
}

impl Default for VisualizationState {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl VisualizationState {
    /// Update model structure from snapshot, preserving animated positions
    pub fn update_from_snapshot(&mut self, snapshot: ModelStructure) {
        // If positions haven't been initialized yet, just use the snapshot as-is
        if !self.positions_initialized {
            self.model_structure = snapshot;
            self.positions_initialized = true;
            return;
        }

        // Update activity and spike counts, but preserve animated positions
        for new_layer in &snapshot.layers {
            if let Some(existing_layer) = self
                .model_structure
                .layers
                .iter_mut()
                .find(|l| l.id == new_layer.id)
            {
                // Update activity data but keep position and velocity
                existing_layer.name = new_layer.name.clone();
                existing_layer.layer_type = new_layer.layer_type.clone();
                existing_layer.size = new_layer.size;
                existing_layer.current_activity = new_layer.current_activity.clone();
                existing_layer.spike_count = new_layer.spike_count;
                // Position and velocity are preserved
            } else {
                // New layer - add it (clone since we're iterating by reference)
                self.model_structure.layers.push(new_layer.clone());
            }
        }

        // Remove layers that no longer exist
        self.model_structure
            .layers
            .retain(|l| snapshot.layers.iter().any(|nl| nl.id == l.id));

        // Update synapses completely (they don't have animated state)
        self.model_structure.synapses = snapshot.synapses;
    }

    /// Save the epoch rewards (graph values) to a CSV file for later analysis
    pub fn save_graphs_to_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        if let Some(parent) = path.parent()
            && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "epoch,reward")?;
        for (epoch, reward) in &self.epoch_rewards {
            writeln!(file, "{},{}", epoch, reward)?;
        }
        Ok(())
    }
}

/// Start the visualization in a separate thread
pub fn start_visualization(state: Arc<Mutex<VisualizationState>>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        use crossterm::{
            event::{DisableMouseCapture, EnableMouseCapture},
            execute,
            terminal::{
                EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
            },
        };
        use ratatui::Terminal;
        use ratatui::backend::CrosstermBackend;
        use std::io;

        // Setup terminal
        enable_raw_mode().unwrap();
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture).unwrap();
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).unwrap();

        // Ensure cleanup on panic
        let original_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            let mut stdout = io::stdout();
            let _ = execute!(stdout, LeaveAlternateScreen, DisableMouseCapture);
            let _ = disable_raw_mode();
            original_hook(panic);
        }));

        let mut app = app::NeuralNetworkVisualizerApp::new(state);
        let res = app.run(&mut terminal);

        // Cleanup terminal
        disable_raw_mode().unwrap();
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )
        .unwrap();
        terminal.show_cursor().unwrap();

        if let Err(err) = res {
            log::error!("{:?}", err);
        }
    })
}
