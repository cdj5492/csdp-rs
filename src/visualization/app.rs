use super::{LayerVisInfo, ModelStructure, NeuronTraceManager, RuntimeStats, VisualizationState};
use crate::synapse::LayerId;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::{Arc, Mutex};

pub struct NeuralNetworkVisualizerApp {
    vis_state: Arc<Mutex<VisualizationState>>,
    neuron_selector_open: Option<LayerId>,
    neuron_input_text: String,
}

impl NeuralNetworkVisualizerApp {
    pub fn new(vis_state: Arc<Mutex<VisualizationState>>) -> Self {
        Self {
            vis_state,
            neuron_selector_open: None,
            neuron_input_text: String::new(),
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

    fn draw_network(&mut self, ui: &mut egui::Ui, model: &ModelStructure) {
        // Debug: Show layer count and info
        ui.label(format!(
            "Layers: {}, Synapses: {}",
            model.layers.len(),
            model.synapses.len()
        ));

        let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click());

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

        let to_screen = egui::emath::RectTransform::from_to(
            Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 400.0)),
            response.rect,
        );

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

                painter.line_segment(
                    [pre_pos, post_pos],
                    Stroke::new(thickness, Color32::from_gray(150)),
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
        let model_structure = state.model_structure.clone();
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
                        Plot::new("spike_traces")
                            .height(200.0)
                            .show_axes([true, true])
                            .show_grid([true, true])
                            .legend(egui_plot::Legend::default())
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

                                    plot_ui.line(Line::new(points).name(&neuron.display_name));
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
            self.draw_network(ui, &model_structure);
        });

        // Show neuron selector modal if open
        if let Some(layer_id) = self.neuron_selector_open
            && let Some(layer) = model_structure.layers.iter().find(|l| l.id == layer_id)
        {
            self.show_neuron_selector(ctx, layer);
        }
    }
}
