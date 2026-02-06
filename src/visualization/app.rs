use super::{LayerVisInfo, ModelStructure, RuntimeStats, SynapseVisInfo, VisualizationState};
use crate::synapse::LayerId;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use std::sync::{Arc, Mutex};

pub struct NeuralNetworkVisualizerApp {
    vis_state: Arc<Mutex<VisualizationState>>,
    layer_velocities: std::collections::HashMap<LayerId, (f32, f32)>,
    repel_force: f32,
    link_force: f32,
    center_force: f32,
    link_distance: f32,
    selected_layer_id: Option<LayerId>,
    spike_history: Vec<Vec<f32>>,
    raster_texture: Option<egui::TextureHandle>,
    displayed_epoch: usize,
    zoom: f32,
}

impl NeuralNetworkVisualizerApp {
    pub fn new(vis_state: Arc<Mutex<VisualizationState>>) -> Self {
        Self {
            vis_state,
            layer_velocities: std::collections::HashMap::new(),
            repel_force: 5000.0,
            link_force: 0.01,
            center_force: 0.005,
            link_distance: 150.0,
            selected_layer_id: None,
            spike_history: Vec::new(),
            raster_texture: None,
            displayed_epoch: 0,
            zoom: 1.0,
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
            let button_text = if is_paused { "▶ Resume" } else { "⏸ Pause" };
            if ui.button(button_text).clicked() {
                if let Ok(mut state) = self.vis_state.try_lock() {
                    state.is_paused = !state.is_paused;
                }
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

    fn update_force_layout(&mut self, model: &mut crate::visualization::ModelStructure) {
        const DAMPING: f32 = 0.8;
        const MIN_DISTANCE: f32 = 50.0;

        let num_layers = model.layers.len();
        if num_layers == 0 {
            return;
        }

        for layer in &model.layers {
            self.layer_velocities.entry(layer.id).or_insert((0.0, 0.0));
        }

        let mut forces: Vec<(f32, f32)> = vec![(0.0, 0.0); num_layers];
        let center_x = 500.0;
        let center_y = 200.0;

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

        for synapse in &model.synapses {
            if let (Some(pre_idx), Some(post_idx)) = (
                model.layers.iter().position(|l| l.id == synapse.pre_layer),
                model.layers.iter().position(|l| l.id == synapse.post_layer),
            ) {
                let dx = model.layers[post_idx].position.x - model.layers[pre_idx].position.x;
                let dy = model.layers[post_idx].position.y - model.layers[pre_idx].position.y;
                let dist = (dx * dx + dy * dy).sqrt();
                let force = self.link_force * (dist - self.link_distance);
                let fx = force * dx / dist.max(1.0);
                let fy = force * dy / dist.max(1.0);
                forces[pre_idx].0 += fx;
                forces[pre_idx].1 += fy;
                forces[post_idx].0 -= fx;
                forces[post_idx].1 -= fy;
            }
        }

        for (i, force) in forces.iter_mut().enumerate().take(num_layers) {
            let dx = center_x - model.layers[i].position.x;
            let dy = center_y - model.layers[i].position.y;
            force.0 += dx * self.center_force;
            force.1 += dy * self.center_force;
        }

        for (i, layer) in model.layers.iter_mut().enumerate() {
            let vel = self.layer_velocities.get_mut(&layer.id).unwrap();
            vel.0 = (vel.0 + forces[i].0) * DAMPING;
            vel.1 = (vel.1 + forces[i].1) * DAMPING;
            layer.position.x += vel.0;
            layer.position.y += vel.1;
            layer.position.x = layer.position.x.clamp(50.0, 950.0);
            layer.position.y = layer.position.y.clamp(50.0, 350.0);
        }
    }

    fn draw_curved_arrow(
        &self,
        painter: &egui::Painter,
        from: Pos2,
        to: Pos2,
        curvature: f32,
        thickness: f32,
        color: Color32,
    ) {
        let mid = Pos2::new((from.x + to.x) / 2.0, (from.y + to.y) / 2.0);
        let dx = to.x - from.x;
        let dy = to.y - from.y;
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

        let segments = 20;
        let mut points = Vec::new();
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let t2 = 1.0 - t;
            let x = t2 * t2 * from.x + 2.0 * t2 * t * control.x + t * t * to.x;
            let y = t2 * t2 * from.y + 2.0 * t2 * t * control.y + t * t * to.y;
            points.push(Pos2::new(x, y));
        }

        for i in 0..segments {
            painter.line_segment([points[i], points[i + 1]], Stroke::new(thickness, color));
        }

        let arrow_len: f32 = 10.0;
        let arrow_angle: f32 = 0.5;
        let end_t = 0.95;
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
        let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click());
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

        self.update_force_layout(model);

        let to_screen = egui::emath::RectTransform::from_to(
            Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 400.0)),
            response.rect,
        );

        let mut synapse_pairs: std::collections::HashMap<(LayerId, LayerId), Vec<&SynapseVisInfo>> =
            std::collections::HashMap::new();
        for synapse in &model.synapses {
            let key = (
                synapse.pre_layer.min(synapse.post_layer),
                synapse.pre_layer.max(synapse.post_layer),
            );
            synapse_pairs.entry(key).or_default().push(synapse);
        }

        for synapse in &model.synapses {
            if let (Some(pre_layer), Some(post_layer)) = (
                model.layers.iter().find(|l| l.id == synapse.pre_layer),
                model.layers.iter().find(|l| l.id == synapse.post_layer),
            ) {
                let pre_pos =
                    to_screen.transform_pos(Pos2::new(pre_layer.position.x, pre_layer.position.y));
                let post_pos = to_screen
                    .transform_pos(Pos2::new(post_layer.position.x, post_layer.position.y));
                let thickness = (synapse.weight_stats.mean.abs() * 2.0).clamp(0.5, 5.0);
                let key = (
                    synapse.pre_layer.min(synapse.post_layer),
                    synapse.pre_layer.max(synapse.post_layer),
                );
                let has_bidirectional =
                    synapse_pairs.get(&key).map(|v| v.len() > 1).unwrap_or(false);
                let curvature = if has_bidirectional {
                    if synapse.pre_layer < synapse.post_layer {
                        30.0
                    } else {
                        -30.0
                    }
                } else {
                    15.0
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

        for layer in &model.layers {
            self.draw_layer(&painter, layer, &to_screen, &response);
        }

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
        let base_size = 20.0 + (layer.size as f32).log10() * 15.0;

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

        painter.circle_filled(pos, base_size, color);
        painter.circle_stroke(pos, base_size, Stroke::new(2.0, Color32::BLACK));

        if self.selected_layer_id == Some(layer.id) {
            painter.circle_stroke(pos, base_size + 4.0, Stroke::new(2.0, Color32::YELLOW));
        }

        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            &layer.name,
            egui::FontId::proportional(12.0),
            Color32::WHITE,
        );

        let click_radius = base_size;
        let rect = Rect::from_center_size(pos, Vec2::splat(click_radius * 2.0));

        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                if rect.contains(click_pos) {
                    if let Ok(mut state) = self.vis_state.lock() {
                        if state.selected_layer_id == Some(layer.id) {
                            state.selected_layer_id = None;
                            self.spike_history.clear();
                        } else {
                            state.selected_layer_id = Some(layer.id);
                            self.spike_history.clear();
                        }
                    }
                }
            }
        }

        if response.hovered() {
            if let Some(hover_pos) = response.hover_pos() {
                if rect.contains(hover_pos) {
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

    fn draw_raster_panel(&mut self, ui: &mut egui::Ui, model: &ModelStructure) {
        if let Ok(mut state) = self.vis_state.lock() {
            // Update local selected_layer_id from shared state
            self.selected_layer_id = state.selected_layer_id;
            // Consume new epoch data if available
            if let Some((epoch, history)) = state.epoch_spike_history.take() {
                if self.displayed_epoch != epoch {
                    self.spike_history = history;
                    self.displayed_epoch = epoch;
                }
            }
        }

        ui.heading(format!("Spike Raster Plot (Epoch {})", self.displayed_epoch));
        if let Some(layer_id) = self.selected_layer_id {
            if let Some(layer) = model.layers.iter().find(|l| l.id == layer_id) {
                ui.label(format!(
                    "Showing spikes for layer: '{}' ({} neurons)",
                    layer.name, layer.size
                ));

                if self.spike_history.is_empty() {
                    ui.label("Waiting for epoch data...");
                    return;
                }

                let num_neurons = layer.size;
                let num_timesteps = self.spike_history.len();

                let mut image_data = Vec::with_capacity(num_neurons * num_timesteps * 4);
                for n in 0..num_neurons {
                    for t in 0..num_timesteps {
                        let spike_val = self.spike_history.get(t).and_then(|s| s.get(n)).cloned().unwrap_or(0.0);
                        let color = if spike_val > 0.0 { Color32::WHITE } else { Color32::BLACK };
                        image_data.extend_from_slice(&color.to_array());
                    }
                }
                let image = egui::ColorImage::from_rgba_unmultiplied([num_timesteps, num_neurons], &image_data);

                let texture_options = egui::TextureOptions {
                    magnification: egui::TextureFilter::Nearest,
                    minification: egui::TextureFilter::Nearest,
                    ..Default::default()
                };

                let texture = self.raster_texture.get_or_insert_with(|| {
                    ui.ctx().load_texture("raster_plot", image.clone(), texture_options)
                });
                texture.set(image, texture_options);

                egui::ScrollArea::both().show(ui, |ui| {
                    let image_original_width = num_timesteps as f32;
                    let image_original_height = num_neurons as f32;

                    let mut displayed_width = image_original_width;
                    let mut displayed_height = image_original_height;

                    let available_panel_width = ui.available_width();
                    let available_panel_height = ui.available_height();

                    // Scale to fit available width, maintaining aspect ratio
                    if displayed_width > available_panel_width {
                        let scale = available_panel_width / displayed_width;
                        displayed_width *= scale;
                        displayed_height *= scale;
                    }

                    // Then scale to fit available height, maintaining aspect ratio
                    if displayed_height > available_panel_height {
                        let scale = available_panel_height / displayed_height;
                        displayed_width *= scale;
                        displayed_height *= scale;
                    }

                    // Ensure a minimum size for visibility if highly compressed, but avoid making it too big
                    let min_display_height = (num_neurons as f32).min(50.0); // At least 1 pixel per neuron, or 50px
                    if displayed_height < min_display_height && num_neurons > 0 {
                        let current_scale = displayed_height / image_original_height;
                        let target_scale = min_display_height / image_original_height;
                        if target_scale > current_scale { // Only scale up if smaller than min_display_height
                            displayed_width = image_original_width * target_scale;
                            displayed_height = image_original_height * target_scale;
                        }
                    }

                    // Set a maximum displayed height to prevent it from dominating the UI
                    let max_rendered_height = 600.0;
                    if displayed_height > max_rendered_height {
                         let scale = max_rendered_height / displayed_height;
                         displayed_width *= scale;
                         displayed_height *= scale;
                    }

                    let displayed_size = Vec2::new(displayed_width, displayed_height) * self.zoom;
                    let response = ui.image((texture.id(), displayed_size));

                    if response.hovered() {
                        let scroll = ui.input(|i| i.raw_scroll_delta);
                        if scroll.y != 0.0 {
                            self.zoom = (self.zoom + scroll.y * 0.01 * self.zoom).max(0.1);
                        }
                    }
                });

            } else {
                ui.label("Selected layer not found.");
            }
        } else {
            ui.label("Click on a layer in the network topology to display its spike raster plot.");
        }
    }
}

impl eframe::App for NeuralNetworkVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        let state = match self.vis_state.try_lock() {
            Ok(s) => s,
            Err(_) => return,
        };

        let mut model_structure = state.model_structure.clone();
        let runtime_stats = state.runtime_stats.clone();
        let should_close = state.should_close;
        let total_epochs = state.total_epochs;
        let is_paused = state.is_paused;
        let selected_layer_id = state.selected_layer_id;

        drop(state);

        if should_close {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        egui::TopBottomPanel::top("stats_panel").show(ctx, |ui| {
            self.draw_stats_panel(ui, &runtime_stats, total_epochs, is_paused);
        });

        if selected_layer_id.is_some() {
            egui::TopBottomPanel::bottom("raster_panel")
                .min_height(150.0)
                .max_height(400.0)
                .resizable(true)
                .show(ctx, |ui| {
                    self.draw_raster_panel(ui, &model_structure);
                });
        }

        egui::SidePanel::right("details_panel")
            .min_width(200.0)
            .show(ctx, |ui| {
                self.draw_layer_details(ui, &model_structure);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Network Topology");
            ui.separator();
            self.draw_network(ui, &mut model_structure);
        });

        if let Ok(mut state) = self.vis_state.try_lock() {
            for layer in &model_structure.layers {
                if let Some(state_layer) =
                    state.model_structure.layers.iter_mut().find(|l| l.id == layer.id)
                {
                    state_layer.position = layer.position;
                }
            }
            // Update local selected_layer_id from shared state
             self.selected_layer_id = state.selected_layer_id;
        }
    }
}
