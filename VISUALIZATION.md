# Interactive Neural Network Visualization

## Overview

This framework now includes an interactive egui-based visualization system that allows you to observe your spiking neural network in real-time during training. The visualization provides:

- **Network Topology View**: Visual representation of layers and synapses
- **Real-time Activity**: Live updates of neuron firing patterns
- **Interactive Neuron Tracking**: Click on layers to track individual neurons
- **Spike Train Visualization**: Time-series plots of selected neuron outputs
- **Layer Details**: Detailed information about each layer
- **Performance Metrics**: Training speed and progress tracking

## Usage

### Basic Usage

Run training with visualization enabled:

```bash
cargo run --release -- --visualize
# or
cargo run --release -- -v
```

Run without visualization (default):

```bash
cargo run --release
```

### Interacting with the Visualization

#### Main Window Layout

The visualization window is divided into several panels:

1. **Top Panel** (Stats): Shows training metrics
   - Current epoch and iteration
   - Total timesteps processed
   - Training speed (iterations/second)

2. **Central Panel** (Network View): Interactive network topology
   - Layers displayed as colored circles
   - Circle size indicates number of neurons (logarithmic scale)
   - Circle color intensity indicates spike activity (red = more active)
   - Synapses shown as lines connecting layers
   - Line thickness represents average weight magnitude

3. **Right Panel** (Layer Details): Collapsible info for each layer
   - Layer type (Bernoulli, LIF, etc.)
   - Number of neurons
   - Current spike count
   - Activity percentage

4. **Bottom Panel** (Spike Traces): Appears when neurons are tracked
   - Time-series plots of neuron outputs
   - Multiple neurons can be tracked simultaneously
   - Each trace is labeled with layer and neuron index

#### Tracking Individual Neurons

1. **Click on a layer** circle in the network view
2. A dialog will appear asking for a neuron index
3. Enter the index (0 to N-1 where N is the layer size)
4. Click "Add" to start tracking
5. The neuron's spike trace will appear in the bottom panel

**Example**: To track neuron 42 in Hidden_0 layer:
- Click on the "Hidden_0" circle
- Enter "42" in the dialog
- Click "Add"

#### Hover Tooltips

Hover over any layer circle to see detailed information:
- Layer name and type
- Total neuron count
- Currently active neurons
- Activity percentage

#### Managing Tracked Neurons

- **View multiple neurons**: Add as many neurons as you want from different layers
- **Clear all traces**: Click the "Clear All" button in the bottom panel
- **Each trace** is displayed with a unique color and labeled

## Architecture Features

### Generic Model Architecture

The visualization system is built on top of a completely refactored model architecture that supports:

- **Flexible Layer Types**: Not limited to Bernoulli and LIF
- **Generic Synapse Types**: Easy to add new learning rules (currently CSDP)
- **Runtime Configuration**: Models can be configured via `ModelConfig` structures
- **Trait-based Design**: Uses `Layer` and `SynapseOps` traits for extensibility

### Model Configuration Example

```rust
use custom_framework::model::{ModelConfig, LayerConfig, SynapseConfig, SynapseType};

let config = ModelConfig {
    layer_configs: vec![
        LayerConfig::Bernoulli {
            size: 2,
            name: Some("Input".to_string()),
        },
        LayerConfig::LIF {
            size: 256,
            tau: 13.0,
            g_thr: 2.0,
            thresh_lambda: 0.01,
            trace_tau: 5.0,
            name: Some("Hidden_0".to_string()),
        },
        LayerConfig::LIF {
            size: 1,
            tau: 13.0,
            g_thr: 2.0,
            thresh_lambda: 0.01,
            trace_tau: 5.0,
            name: Some("Output".to_string()),
        },
    ],
    synapse_configs: vec![
        SynapseConfig {
            pre_layer: 0,
            post_layer: 1,
            synapse_type: SynapseType::CSDP,
            bidirectional: false,
        },
        SynapseConfig {
            pre_layer: 1,
            post_layer: 2,
            synapse_type: SynapseType::CSDP,
            bidirectional: false,
        },
    ],
    dt: 0.1,
};

let model = Model::from_config(config, &device)?;
```

### Backward Compatibility

The original `Model::new()` API still works:

```rust
let model = Model::new(vec![2, 256, 256, 1], &device, dt).unwrap();
```

This is now implemented internally using `from_config()` with default parameters.

## Implementation Details

### Threading Model

- **Main Thread**: Runs the training loop
- **Visualization Thread**: Runs the egui application independently
- **Shared State**: `Arc<Mutex<VisualizationState>>` synchronizes data
- **Non-blocking**: Training continues even if visualization locks are contested

### Update Frequency

- **Neuron Spikes**: Collected at every timestep (high resolution)
- **Model Structure**: Updated every 10 iterations (optimized for performance)
- **UI Refresh**: Continuous repaint (smooth animation)

### Data Flow

1. Training loop calls `model.step()` for each timestep
2. After each step, tracked neuron outputs are collected via `model.get_neuron_output()`
3. Every 10 iterations, a full snapshot is captured via `model.get_visualization_snapshot()`
4. Visualization thread reads from shared state and renders UI
5. User interactions (clicks, button presses) update shared state
6. Training loop responds to changes (e.g., adding/removing tracked neurons)

### Performance Optimization

- **Try-lock pattern**: Uses `try_lock()` instead of blocking locks
- **Minimal critical sections**: Locks are held for minimal time
- **Data cloning**: Clones only necessary data before rendering
- **Spike history limiting**: Fixed buffer size (1000 timesteps by default)
- **Update throttling**: Full model snapshots only every 10 iterations

## Technical Architecture

### Key Components

**Visualization State**:
- `ModelStructure`: Current network topology and activity
- `RuntimeStats`: Training metrics (epoch, iteration, speed)
- `NeuronTraceManager`: Tracked neurons and their spike histories
- `should_close`: Flag for clean shutdown

**Layer Visualization**:
- `LayerVisInfo`: Display name, type, size, position, current activity
- Automatic layout algorithm (horizontal spacing)
- Activity-based coloring (red intensity = spike rate)
- Size-based circle scaling (logarithmic)

**Synapse Visualization**:
- `SynapseVisInfo`: Connection endpoints, type, weight statistics
- Line drawing between layers
- Thickness based on weight magnitude
- Supports arbitrary connectivity patterns

**Neuron Tracking**:
- `TrackedNeuron`: Layer ID, neuron index, spike history, timesteps
- Ring buffer implementation (fixed memory)
- High-resolution timestep tracking
- Display name formatting

### Code Organization

```
src/
├── model.rs                    # Generic model architecture
│   ├── ModelConfig             # Configuration structures
│   ├── Model::from_config()    # Flexible model creation
│   ├── Model::step()           # Generic step function
│   └── get_visualization_snapshot()  # Extract visualization data
│
├── layer/
│   ├── mod.rs                  # Layer trait and metadata
│   └── ...                     # Layer implementations
│
├── synapse/
│   ├── mod.rs                  # SynapseOps trait, metadata
│   ├── csdp.rs                 # CSDP implementation
│   └── ...                     # Other synapse types
│
└── visualization/
    ├── mod.rs                  # State structures, threading
    │   ├── VisualizationState  # Shared state
    │   ├── ModelStructure      # Snapshot data
    │   ├── NeuronTraceManager  # Spike tracking
    │   └── start_visualization()  # Thread spawner
    │
    └── app.rs                  # egui application
        ├── NeuralNetworkVisualizerApp  # Main app struct
        ├── draw_network()      # Network topology rendering
        ├── draw_layer()        # Layer circle rendering
        ├── draw_spike_traces() # Time-series plotting
        └── update()            # Main UI loop
```

## Extending the System

### Adding New Layer Types

1. Implement the `Layer` trait
2. Add a variant to `LayerConfig` enum
3. Update `Model::create_layer()` to handle the new type

### Adding New Synapse Types

1. Implement the `SynapseOps` trait
2. Add a variant to `SynapseType` enum
3. Update `Model::create_synapse()` to handle the new type

### Customizing Visualization

The visualization can be customized by modifying:
- **Layout**: Change `LayerPosition` calculations in `Model::create_layer()`
- **Colors**: Modify color mapping in `draw_layer()`
- **Update frequency**: Adjust the `iteration % 10` condition in main.rs
- **History size**: Change `NeuronTraceManager::max_history` default

## Dependencies

The visualization system requires:
- `eframe = "0.31"` - Native GUI framework
- `egui = "0.31"` - Immediate mode GUI library
- `egui_plot = "0.31"` - Plotting widgets

These are automatically included in Cargo.toml.

## Known Limitations

1. **Display Required**: Visualization requires a display server (X11, Wayland, or native windowing)
2. **Single Window**: Only one visualization window at a time
3. **Memory Usage**: Long-running training with many tracked neurons uses more memory
4. **Update Latency**: Visualization may lag slightly behind training on very fast hardware

## Future Enhancements

Potential improvements for future versions:
- **Weight Matrix Heatmaps**: Visualize full weight matrices
- **3D Network Layouts**: For spatial networks
- **Configuration Editor**: Runtime modification of learning parameters
- **Export Functionality**: Save spike traces to CSV
- **Playback Mode**: Replay saved training sessions
- **Additional Synapse Types**: STDP, BCM, Oja's rule
- **Custom Color Schemes**: User-selectable visualization themes

## Troubleshooting

**Visualization won't start**:
- Ensure you have a display server running (X11 or Wayland on Linux)
- Check that egui dependencies are installed
- On Linux, the event loop uses `with_any_thread()` to allow GUI on separate thread
- If you see winit threading errors, see `THREADING_FIX.md` for details

**Training is slow with visualization**:
- Reduce the number of tracked neurons
- Increase the update frequency threshold (change `iteration % 10` to `iteration % 20`)
- Run without visualization for maximum performance

**Can't see small layers**:
- Layers with few neurons may be hard to click
- The click detection area is 2x the visual radius
- Hover tooltips work from farther away than clicks

**Spike traces not updating**:
- Ensure neurons are actively firing (check layer details panel)
- Verify the neuron index is valid (0 to size-1)
- Check that training is progressing (iteration counter should increase)
