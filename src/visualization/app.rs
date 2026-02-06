use super::{
    LayerVisInfo, ModelStructure, NeuronTraceManager, RuntimeStats, SynapseVisInfo,
    VisualizationState,
};
use crate::synapse::LayerId;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::{Arc, Mutex};

pub struct NeuralNetworkVisualizerApp {
    vis_state: Arc<Mutex<VisualizationState>>,
    neuron_selector_open: Option<LayerId>,
    neuron_input_text: String,
    layer_velocities: std::collections::HashMap<LayerId, (f32, f32)>,
    // Force-directed layout hyperparameters
    repel_force: f32,
    link_force: f32,
    center_force: f32,
    link_distance: f32,
}

impl NeuralNetworkVisualizerApp {
    pub fn new(vis_state: Arc<Mutex<VisualizationState>>) -> Self {
        Self {
            vis_state,
            neuron_selector_open: None,
            neuron_input_text: String::new(),
            layer_velocities: std::collections::HashMap::new(),
            repel_force: 5000.0,
            link_force: 0.01,
            center_force: 0.005,
            link_distance: 150.0,
        }
    }

    fn draw_stats_panel(
        &mut self,
        ui: &mut egui::Ui,
        stats: &RuntimeStats,
        total_epochs: usize,
        is_paused: bool,
    ) {
        ui.horizontal(|ui| {
            // Pause/Resume button
            let button_text = if is_paused { "▶ Resume" } else { "⏸ Pause" };
            if ui.button(button_text).clicked()
                && let Ok(mut state) = self.vis_state.try_lock()
            {
                state.is_paused = !state.is_paused;
            }

            ui.separator();
            ui.label(format!("Epoch: {}/{}", stats.epoch, total_epochs));
            ui.separator();
            ui.label(format!("Iteration: {}", stats.iteration));
            ui.separator();
            ui.label(format!("Timestep: {}", stats.timestep));
            ui.separator();
            ui.label(format!("Speed: {:.1} it/s", stats.iterations_per_second));
        });

        // Progress bar
        ui.add_space(5.0);
        let progress = if total_epochs > 0 {
            stats.epoch as f32 / total_epochs as f32
        } else {
            0.0
        };

        ui.add(
            egui::ProgressBar::new(progress)
                .show_percentage()
                .text(format!("Training Progress: {:.1}%", progress * 100.0)),
        );
    }

    /// Apply force-directed layout to update layer positions
    fn update_force_layout(&mut self, model: &mut crate::visualization::ModelStructure) {
        const DAMPING: f32 = 0.8;
        const MIN_DISTANCE: f32 = 50.0;

        let num_layers = model.layers.len();
        if num_layers == 0 {
            return;
        }

        // Initialize velocities if needed
        for layer in &model.layers {
            self.layer_velocities.entry(layer.id).or_insert((0.0, 0.0));
        }

        let mut forces: Vec<(f32, f32)> = vec![(0.0, 0.0); num_layers];

        // Center of the canvas
        let center_x = 500.0;
        let center_y = 200.0;

        // Repulsion between all layers
        for i in 0..num_layers {
            for j in (i + 1)..num_layers {
                let dx = model.layers[j].position.x - model.layers[i].position.x;
                let dy = model.layers[j].position.y - model.layers[i].position.y;
                let dist_sq = dx * dx + dy * dy;
                let dist = dist_sq.sqrt().max(MIN_DISTANCE);

                let force = self.repel_force / dist_sq;
                let fx = force * dx / dist;
                let fy = force * dy / dist;

                forces[i].0 -= fx;
                forces[i].1 -= fy;
                forces[j].0 += fx;
                forces[j].1 += fy;
            }
        }

        // Attraction along synapse connections
        for synapse in &model.synapses {
            if let (Some(pre_idx), Some(post_idx)) = (
                model.layers.iter().position(|l| l.id == synapse.pre_layer),
                model.layers.iter().position(|l| l.id == synapse.post_layer),
            ) {
                let dx = model.layers[post_idx].position.x - model.layers[pre_idx].position.x;
                let dy = model.layers[post_idx].position.y - model.layers[pre_idx].position.y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Spring force: F = k * (distance - rest_length)
                let force = self.link_force * (dist - self.link_distance);
                let fx = force * dx / dist.max(1.0);
                let fy = force * dy / dist.max(1.0);

                forces[pre_idx].0 += fx;
                forces[pre_idx].1 += fy;
                forces[post_idx].0 -= fx;
                forces[post_idx].1 -= fy;
            }
        }

        // Center force: pull all nodes toward center
        for (i, force) in forces.iter_mut().enumerate().take(num_layers) {
            let dx = center_x - model.layers[i].position.x;
            let dy = center_y - model.layers[i].position.y;
            force.0 += dx * self.center_force;
            force.1 += dy * self.center_force;
        }

        // Update velocities and positions
        for (i, layer) in model.layers.iter_mut().enumerate() {
            let vel = self.layer_velocities.get_mut(&layer.id).unwrap();
            vel.0 = (vel.0 + forces[i].0) * DAMPING;
            vel.1 = (vel.1 + forces[i].1) * DAMPING;

            layer.position.x += vel.0;
            layer.position.y += vel.1;

            // Keep within bounds
            layer.position.x = layer.position.x.clamp(50.0, 950.0);
            layer.position.y = layer.position.y.clamp(50.0, 350.0);
        }
    }

    /// Draw a curved arrow between two points
    fn draw_curved_arrow(
        &self,
        painter: &egui::Painter,
        from: Pos2,
        to: Pos2,
        curvature: f32,
        thickness: f32,
        color: Color32,
    ) {
        // Calculate control point for quadratic Bezier curve
        let mid = Pos2::new((from.x + to.x) / 2.0, (from.y + to.y) / 2.0);
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        // Perpendicular offset
        let perp_x = -dy;
        let perp_y = dx;
        let len = (perp_x * perp_x + perp_y * perp_y).sqrt();
        let control = if len > 0.0 {
            Pos2::new(
                mid.x + (perp_x / len) * curvature,
                mid.y + (perp_y / len) * curvature,
            )
        } else {
            mid
        };

        // Draw curve with multiple segments
        let segments = 20;
        let mut points = Vec::new();
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let t2 = 1.0 - t;
            let x = t2 * t2 * from.x + 2.0 * t2 * t * control.x + t * t * to.x;
            let y = t2 * t2 * from.y + 2.0 * t2 * t * control.y + t * t * to.y;
            points.push(Pos2::new(x, y));
        }

        // Draw the curve
        for i in 0..segments {
            painter.line_segment([points[i], points[i + 1]], Stroke::new(thickness, color));
        }

        // Draw arrowhead at the end
        let arrow_len: f32 = 10.0;
        let arrow_angle: f32 = 0.5;

        // Direction at end of curve
        let end_t = 0.95; // Slightly before end for better arrow placement
        let t2 = 1.0 - end_t;
        let arrow_base_x = t2 * t2 * from.x + 2.0 * t2 * end_t * control.x + end_t * end_t * to.x;
        let arrow_base_y = t2 * t2 * from.y + 2.0 * t2 * end_t * control.y + end_t * end_t * to.y;
        let arrow_base = Pos2::new(arrow_base_x, arrow_base_y);

        let dir_x = to.x - arrow_base.x;
        let dir_y = to.y - arrow_base.y;
        let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();

        if dir_len > 0.0 {
            let dx = dir_x / dir_len;
            let dy = dir_y / dir_len;

            let left = Pos2::new(
                to.x - arrow_len * (dx * arrow_angle.cos() - dy * arrow_angle.sin()),
                to.y - arrow_len * (dx * arrow_angle.sin() + dy * arrow_angle.cos()),
            );
            let right = Pos2::new(
                to.x - arrow_len * (dx * arrow_angle.cos() + dy * arrow_angle.sin()),
                to.y - arrow_len * (dy * arrow_angle.cos() - dx * arrow_angle.sin()),
            );

            painter.line_segment([to, left], Stroke::new(thickness, color));
            painter.line_segment([to, right], Stroke::new(thickness, color));
        }
    }

    fn draw_network(
        &mut self,
        ui: &mut egui::Ui,
        model: &mut crate::visualization::ModelStructure,
    ) {
        // Debug: Show layer count and info
        ui.label(format!(
            "Layers: {}, Synapses: {}",
            model.layers.len(),
            model.synapses.len()
        ));

        let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click());

        // Clear background explicitly
        painter.rect_filled(response.rect, 0.0, ui.style().visuals.extreme_bg_color);

        if model.layers.is_empty() {
            painter.text(
                response.rect.center(),
                egui::Align2::CENTER_CENTER,
                "Waiting for model data...",
                egui::FontId::proportional(16.0),
                Color32::WHITE,
            );
            return;
        }

        // Update force-directed layout
        self.update_force_layout(model);

        let to_screen = egui::emath::RectTransform::from_to(
            Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 400.0)),
            response.rect,
        );

        // Group synapses by connection pairs to detect bidirectional connections
        let mut synapse_pairs: std::collections::HashMap<(LayerId, LayerId), Vec<&SynapseVisInfo>> =
            std::collections::HashMap::new();

        for synapse in &model.synapses {
            let key = (
                synapse.pre_layer.min(synapse.post_layer),
                synapse.pre_layer.max(synapse.post_layer),
            );
            synapse_pairs.entry(key).or_default().push(synapse);
        }

        // Draw synapses first (so they appear behind layers)
        for synapse in &model.synapses {
            if let (Some(pre_layer), Some(post_layer)) = (
                model.layers.iter().find(|l| l.id == synapse.pre_layer),
                model.layers.iter().find(|l| l.id == synapse.post_layer),
            ) {
                let pre_pos =
                    to_screen.transform_pos(Pos2::new(pre_layer.position.x, pre_layer.position.y));
                let post_pos = to_screen
                    .transform_pos(Pos2::new(post_layer.position.x, post_layer.position.y));

                // Line thickness based on mean weight
                let thickness = (synapse.weight_stats.mean.abs() * 2.0).clamp(0.5, 5.0);

                // Check if there's a reverse connection
                let key = (
                    synapse.pre_layer.min(synapse.post_layer),
                    synapse.pre_layer.max(synapse.post_layer),
                );
                let has_bidirectional = synapse_pairs
                    .get(&key)
                    .map(|v| v.len() > 1)
                    .unwrap_or(false);

                // Curvature: offset for bidirectional connections
                let curvature = if has_bidirectional {
                    // Determine which direction this synapse goes
                    if synapse.pre_layer < synapse.post_layer {
                        30.0
                    } else {
                        -30.0
                    }
                } else {
                    15.0 // Slight curve for better visibility
                };

                self.draw_curved_arrow(
                    &painter,
                    pre_pos,
                    post_pos,
                    curvature,
                    thickness,
                    Color32::from_gray(150),
                );
            }
        }

        // Draw layers
        for layer in &model.layers {
            self.draw_layer(&painter, layer, &to_screen, &response);
        }

        // Debug: Draw a border around the drawable area
        painter.rect_stroke(
            response.rect,
            0.0,
            Stroke::new(1.0, Color32::from_gray(100)),
            egui::epaint::StrokeKind::Outside,
        );
    }

    fn draw_layer(
        &mut self,
        painter: &egui::Painter,
        layer: &LayerVisInfo,
        transform: &egui::emath::RectTransform,
        response: &egui::Response,
    ) {
        let world_pos = Pos2::new(layer.position.x, layer.position.y);
        let pos = transform.transform_pos(world_pos);

        // Debug: Print first time we draw each layer
        static mut DEBUG_PRINTED: bool = false;
        unsafe {
            if !DEBUG_PRINTED {
                eprintln!(
                    "Drawing layer '{}' at world ({}, {}) -> screen ({}, {})",
                    layer.name, world_pos.x, world_pos.y, pos.x, pos.y
                );
                if layer.id == 3 {
                    // Print only first 4 layers
                    DEBUG_PRINTED = true;
                }
            }
        }

        // Size based on neuron count (logarithmic scale for better visualization)
        let base_size = 20.0 + (layer.size as f32).log10() * 15.0;

        // Color based on spike activity
        let activity_ratio = if layer.size > 0 {
            layer.spike_count as f32 / layer.size as f32
        } else {
            0.0
        };

        let color = Color32::from_rgb(
            (activity_ratio * 255.0) as u8,
            100,
            (255.0 - activity_ratio * 200.0) as u8,
        );

        // Draw circle
        painter.circle_filled(pos, base_size, color);
        painter.circle_stroke(pos, base_size, Stroke::new(2.0, Color32::BLACK));

        // Draw label
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            &layer.name,
            egui::FontId::proportional(12.0),
            Color32::WHITE,
        );

        // Click detection (larger hit area)
        let click_radius = base_size;
        let rect = Rect::from_center_size(pos, Vec2::splat(click_radius * 2.0));

        if response.clicked()
            && let Some(click_pos) = response.interact_pointer_pos()
            && rect.contains(click_pos)
        {
            self.neuron_selector_open = Some(layer.id);
            self.neuron_input_text.clear();
        }

        // Hover tooltip
        if response.hovered()
            && let Some(hover_pos) = response.hover_pos()
            && rect.contains(hover_pos)
        {
            egui::Area::new(egui::Id::new(format!("layer_tooltip_{}", layer.id)))
                .fixed_pos(hover_pos + Vec2::new(10.0, 10.0))
                .show(&response.ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.label(format!("Layer: {}", layer.name));
                        ui.label(format!("Type: {}", layer.layer_type));
                        ui.label(format!("Size: {} neurons", layer.size));
                        ui.label(format!("Active: {} neurons", layer.spike_count));
                        ui.label(format!("Activity: {:.1}%", activity_ratio * 100.0));
                    });
                });
        }
    }

    fn draw_layer_details(&self, ui: &mut egui::Ui, model: &ModelStructure) {
        ui.heading("Layer Details");
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            for layer in &model.layers {
                ui.collapsing(&layer.name, |ui| {
                    ui.label(format!("Type: {}", layer.layer_type));
                    ui.label(format!("Neurons: {}", layer.size));
                    ui.label(format!("Active: {}", layer.spike_count));
                    let activity_ratio = if layer.size > 0 {
                        layer.spike_count as f32 / layer.size as f32 * 100.0
                    } else {
                        0.0
                    };
                    ui.label(format!("Activity: {:.1}%", activity_ratio));
                });
            }
        });
    }

    #[allow(dead_code)]
    fn draw_spike_traces(&mut self, ui: &mut egui::Ui, traces: &NeuronTraceManager) {
        ui.horizontal(|ui| {
            ui.heading("Neuron Spike Traces");
            if ui.button("Clear All").clicked() {
                // Signal to clear traces (will be handled in update)
                if let Ok(mut state) = self.vis_state.try_lock() {
                    state.neuron_traces.clear();
                }
            }
        });

        if traces.tracked_neurons.is_empty() {
            ui.label("No neurons tracked. Click on a layer to select neurons.");
            return;
        }

        Plot::new("spike_traces")
            .height(200.0)
            .show_axes([true, true])
            .show_grid([true, true])
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                for neuron in &traces.tracked_neurons {
                    if neuron.timesteps.is_empty() {
                        continue;
                    }

                    let points: PlotPoints = neuron
                        .timesteps
                        .iter()
                        .zip(neuron.spike_history.iter())
                        .map(|(&t, &spike)| [t as f64, spike as f64])
                        .collect();

                    plot_ui.line(Line::new(points).name(&neuron.display_name));
                }
            });
    }

    fn show_neuron_selector(&mut self, ctx: &egui::Context, layer: &LayerVisInfo) {
        egui::Window::new("Select Neuron")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!(
                    "Layer: {} (size: {} neurons)",
                    layer.name, layer.size
                ));
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Neuron index (0-{}):",
                        layer.size.saturating_sub(1)
                    ));
                    ui.text_edit_singleline(&mut self.neuron_input_text);
                });

                ui.horizontal(|ui| {
                    if ui.button("Add").clicked()
                        && let Ok(idx) = self.neuron_input_text.parse::<usize>()
                    {
                        if idx < layer.size {
                            if let Ok(mut state) = self.vis_state.try_lock() {
                                state.neuron_traces.add_neuron(layer.id, idx, &layer.name);
                            }
                            self.neuron_selector_open = None;
                        } else {
                            // Show error - index out of bounds
                        }
                    }

                    if ui.button("Cancel").clicked() {
                        self.neuron_selector_open = None;
                    }
                });
            });
    }
}

impl eframe::App for NeuralNetworkVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request continuous repaint for animation
        ctx.request_repaint();

        // Try to lock state (non-blocking)
        let state = match self.vis_state.try_lock() {
            Ok(s) => s,
            Err(_) => return, // Skip this frame if state is locked
        };

        // Clone data we need (to release lock quickly)
        let mut model_structure = state.model_structure.clone();
        let runtime_stats = state.runtime_stats.clone();
        let has_tracked_neurons = !state.neuron_traces.tracked_neurons.is_empty();
        let should_close = state.should_close;
        let total_epochs = state.total_epochs;
        let is_paused = state.is_paused;

        // Release lock before UI rendering
        drop(state);

        // Check if we should close
        if should_close {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        // Top panel: stats and controls
        egui::TopBottomPanel::top("stats_panel").show(ctx, |ui| {
            self.draw_stats_panel(ui, &runtime_stats, total_epochs, is_paused);
        });

        // Bottom panel: spike traces (if any tracked)
        if has_tracked_neurons {
            let vis_state_clone = self.vis_state.clone();

            // Clone the data we need before UI rendering
            let tracked_neurons = if let Ok(state) = vis_state_clone.try_lock() {
                state.neuron_traces.tracked_neurons.clone()
            } else {
                Vec::new()
            };

            egui::TopBottomPanel::bottom("traces_panel")
                .min_height(250.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Neuron Spike Traces");
                        if ui.button("Clear All").clicked()
                            && let Ok(mut state) = vis_state_clone.try_lock()
                        {
                            state.neuron_traces.clear();
                        }
                    });

                    if tracked_neurons.is_empty() {
                        ui.label("No neurons tracked. Click on a layer to select neurons.");
                    } else {
                        // Debug info
                        let total_points: usize = tracked_neurons.iter().map(|n| n.timesteps.len()).sum();
                        ui.label(format!(
                            "Tracking {} neuron(s), {} total data points",
                            tracked_neurons.len(),
                            total_points
                        ));

                        // Check if any neuron has data
                        let has_data = tracked_neurons.iter().any(|n| !n.timesteps.is_empty());

                        if !has_data {
                            ui.colored_label(
                                Color32::YELLOW,
                                "⚠ No data yet. Make sure to unpause the simulation (▶ Resume button above)."
                            );
                        } else {
                            ui.colored_label(Color32::GREEN, "✓ Receiving data");
                        }

                        Plot::new("spike_traces")
                            .height(200.0)
                            .show_axes([true, true])
                            .show_grid([true, true])
                            .legend(egui_plot::Legend::default())
                            .auto_bounds([true, true])
                            .allow_zoom(true)
                            .allow_drag(true)
                            .include_y(0.0)
                            .include_y(1.0)
                            .show(ui, |plot_ui| {
                                for neuron in &tracked_neurons {
                                    if neuron.timesteps.is_empty() {
                                        continue;
                                    }

                                    let points: PlotPoints = neuron
                                        .timesteps
                                        .iter()
                                        .zip(neuron.spike_history.iter())
                                        .map(|(&t, &spike)| [t as f64, spike as f64])
                                        .collect();

                                    plot_ui.line(
                                        Line::new(points)
                                            .name(&neuron.display_name)
                                            .width(2.0)
                                    );
                                }
                            });
                    }
                });
        }

        // Right side panel: layer details
        egui::SidePanel::right("details_panel")
            .min_width(200.0)
            .show(ctx, |ui| {
                self.draw_layer_details(ui, &model_structure);
            });

        // Central panel: network visualization
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Network Topology");
            ui.separator();
            self.draw_network(ui, &mut model_structure);
        });

        // Show neuron selector modal if open
        if let Some(layer_id) = self.neuron_selector_open
            && let Some(layer) = model_structure.layers.iter().find(|l| l.id == layer_id)
        {
            self.show_neuron_selector(ctx, layer);
        }

        // Write updated positions and velocities back to shared state
        if let Ok(mut state) = self.vis_state.try_lock() {
            // Update layer positions and velocities in the shared state
            for layer in &model_structure.layers {
                if let Some(state_layer) = state
                    .model_structure
                    .layers
                    .iter_mut()
                    .find(|l| l.id == layer.id)
                {
                    state_layer.position = layer.position;
                    state_layer.velocity = layer.velocity;
                }
            }
        }
    }
}
