use super::{ModelStructure, VisualizationState};
use crate::synapse::LayerId;
use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout, Rect, Alignment},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        canvas::{Canvas, Line as CanvasLine, Circle, Points},
        Block, Borders, Gauge, List, ListItem, Paragraph, ListState,
        Chart, Axis, Dataset, GraphType, Tabs,
    },
    Frame,
};
use crossterm::event::{self, Event, KeyCode, MouseEventKind};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct NeuralNetworkVisualizerApp {
    vis_state: Arc<Mutex<VisualizationState>>,
    layer_velocities: std::collections::HashMap<LayerId, (f32, f32)>,
    repel_force: f32,
    link_force: f32,
    center_force: f32,
    link_distance: f32,
    
    selected_layer_id: Option<LayerId>,
    spike_history: Vec<Vec<f32>>,
    displayed_epoch: usize,
    
    log_state: ListState,
    show_help: bool,
    active_tab: usize,
    user_scrolled: bool,
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
            displayed_epoch: 0,
            log_state: ListState::default(),
            show_help: false,
            active_tab: 0,
            user_scrolled: false,
        }
    }

    pub fn run<B: Backend>(&mut self, terminal: &mut ratatui::Terminal<B>) -> Result<(), std::io::Error>
    where
        std::io::Error: From<B::Error>,
    {
        let tick_rate = Duration::from_millis(50);
        let mut last_tick = std::time::Instant::now();

        loop {
            // Check state for should_close
            let should_close = if let Ok(state) = self.vis_state.lock() {
                state.should_close
            } else { false };

            if should_close {
                return Ok(());
            }

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if crossterm::event::poll(timeout)? {
                let event = event::read()?;
                self.handle_event(event);
            }

            if last_tick.elapsed() >= tick_rate {
                let vis_state = Arc::clone(&self.vis_state);
                if let Ok(mut state) = vis_state.lock() {
                    self.update_force_layout(&mut state.model_structure);
                }
                terminal.draw(|f| self.draw(f))?;
                last_tick = std::time::Instant::now();
            }
        }
    }

    fn cycle_selection(&mut self, dir: isize) {
        if let Ok(mut state) = self.vis_state.lock() {
            let layers = &state.model_structure.layers;
            if layers.is_empty() { return; }
            
            let current_idx = if let Some(id) = state.selected_layer_id {
                layers.iter().position(|l| l.id == id).unwrap_or(0) as isize
            } else {
                -1
            };
            
            let mut next_idx = current_idx + dir;
            if next_idx < 0 {
                next_idx = layers.len() as isize - 1;
            } else if next_idx >= layers.len() as isize {
                next_idx = 0;
            }
            
            state.selected_layer_id = Some(layers[next_idx as usize].id);
            self.selected_layer_id = state.selected_layer_id;
            state.epoch_spike_history = None;
            self.spike_history.clear();
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Key(key) => {
                match key.code {
                    KeyCode::Char('q') => {
                        if let Ok(mut state) = self.vis_state.lock() {
                            state.should_close = true;
                        }
                    }
                    KeyCode::Char('p') => {
                        if let Ok(mut state) = self.vis_state.lock() {
                            state.is_paused = !state.is_paused;
                        }
                    }
                    KeyCode::Char('s') => {
                        if let Ok(mut state) = self.vis_state.lock() {
                            state.save_requested = true;
                            let path = std::path::Path::new("checkpoints/epoch_rewards.csv");
                            let _ = state.save_graphs_to_csv(path);
                        }
                    }
                    KeyCode::Char('l') => {
                        if let Ok(mut state) = self.vis_state.lock() {
                            state.load_requested = true;
                        }
                    }
                    KeyCode::Esc => {
                        self.show_help = false;
                    }
                    KeyCode::Char('?') => {
                        self.show_help = !self.show_help;
                    }
                    KeyCode::Tab => {
                        self.active_tab = (self.active_tab + 1) % 3;
                    }
                    KeyCode::Right => {
                        self.cycle_selection(1);
                    }
                    KeyCode::Left => {
                        self.cycle_selection(-1);
                    }
                    KeyCode::Up => {
                        let i = self.log_state.selected().unwrap_or(0);
                        self.log_state.select(Some(i.saturating_sub(1)));
                        self.user_scrolled = true;
                    }
                    KeyCode::Down => {
                        let i = self.log_state.selected().unwrap_or(0);
                        self.log_state.select(Some(i + 1));
                        self.user_scrolled = true;
                    }
                    _ => {}
                }
            }
            Event::Mouse(mouse) => {
                match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        let i = self.log_state.selected().unwrap_or(0);
                        self.log_state.select(Some(i.saturating_sub(1)));
                        self.user_scrolled = true;
                    }
                    MouseEventKind::ScrollDown => {
                        let i = self.log_state.selected().unwrap_or(0);
                        self.log_state.select(Some(i + 1));
                        self.user_scrolled = true;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // Force layout physics for beautiful representation
    fn update_force_layout(&mut self, model: &mut ModelStructure) {
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
            // Bound within Canvas space (0..1000, 0..400)
            layer.position.x = layer.position.x.clamp(50.0, 950.0);
            layer.position.y = layer.position.y.clamp(50.0, 350.0);
        }
    }

    fn draw(&mut self, f: &mut Frame) {
        let size = f.area();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(3), // Header/Stats
                Constraint::Length(3), // Tabs
                Constraint::Min(0),    // Main content
                Constraint::Length(10), // Logs
            ])
            .split(size);

        let vis_state = Arc::clone(&self.vis_state);
        if let Ok(mut state) = vis_state.try_lock() {
            self.selected_layer_id = state.selected_layer_id;

            if let Some((epoch, history)) = state.epoch_spike_history.take() {
                if self.displayed_epoch != epoch {
                    self.spike_history = history;
                    self.displayed_epoch = epoch;
                }
            }
            
            self.draw_header(f, chunks[0], &state);

            let titles = vec!["Network Topology", "Environment", "Analysis"]
                .into_iter()
                .map(|t| Line::from(Span::styled(t, Style::default().fg(Color::White))))
                .collect::<Vec<_>>();
                
            let tabs = Tabs::new(titles)
                .block(Block::default().borders(Borders::ALL).title("Views"))
                .select(self.active_tab)
                .highlight_style(Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow));
                
            f.render_widget(tabs, chunks[1]);
            
            match self.active_tab {
                0 => {
                    let main_chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
                        .split(chunks[2]);
                    self.draw_network(f, main_chunks[0], &state.model_structure);
                    self.draw_details(f, main_chunks[1], &state.model_structure);
                }
                1 => {
                    self.draw_env_state(f, chunks[2], &state);
                }
                2 => {
                    let side_chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(chunks[2]);
                    self.draw_rewards(f, side_chunks[0], &state);
                    self.draw_raster(f, side_chunks[1], &state.model_structure);
                }
                _ => {}
            }
        }

        self.draw_logs(f, chunks[3]);

        if self.show_help {
            self.draw_help(f, size);
        }
    }
    
    fn draw_env_state(&self, f: &mut Frame, area: Rect, state: &VisualizationState) {
        if let Some(env_state) = &state.environment_state {
            if env_state.len() == 4 {
                // Grid visualization
                let px = env_state[0] + env_state[2];
                let py = env_state[1] + env_state[3];
                let gx = env_state[2];
                let gy = env_state[3];
                let grid_size = 50.0;
                
                let canvas = Canvas::default()
                    .block(Block::default().borders(Borders::ALL).title("Env: Grid 50x50 (Yellow=Player, Green=Goal)"))
                    .marker(ratatui::symbols::Marker::HalfBlock)
                    .x_bounds([0.0, grid_size])
                    .y_bounds([0.0, grid_size])
                    .paint(move |ctx| {
                        ctx.draw(&Points { coords: &[(gx as f64, grid_size - gy as f64)], color: Color::Green });
                        ctx.draw(&Points { coords: &[(px as f64, grid_size - py as f64)], color: Color::Yellow });
                    });
                f.render_widget(canvas, area);
            } else if env_state.len() == 6 {
                // Robot arm visualization
                let text = env_state.iter().enumerate().map(|(i, &v)| format!("J{}:{:.1}", i, v)).collect::<Vec<_>>().join(" ");
                let p = Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("Env State: Robot"));
                f.render_widget(p, area);
            } else {
                let text = env_state.iter().enumerate().map(|(i, &v)| format!("D{}:{:.1}", i, v)).collect::<Vec<_>>().join(" ");
                let p = Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("Env State"));
                f.render_widget(p, area);
            }
        } else {
            let p = Paragraph::new("No Environment Data").block(Block::default().borders(Borders::ALL).title("Env State"));
            f.render_widget(p, area);
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect, state: &VisualizationState) {
        let text = format!(
            "Epoch: {}/{} | Iter: {} | Speed: {:.1} it/s | State: {} | Press '?' for Help",
            state.runtime_stats.epoch,
            state.total_epochs,
            state.runtime_stats.iteration,
            state.runtime_stats.iterations_per_second,
            if state.is_paused { "PAUSED" } else { "RUNNING" }
        );

        let progress = if state.total_epochs > 0 {
            (state.runtime_stats.epoch as f32 / state.total_epochs as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Training Progress"))
            .gauge_style(Style::default().fg(Color::Green).bg(Color::Black).add_modifier(Modifier::BOLD))
            .ratio(progress as f64)
            .label(text);
        
        f.render_widget(gauge, area);
    }

    fn draw_network(&self, f: &mut Frame, area: Rect, model: &ModelStructure) {
        let canvas = Canvas::default()
            .block(Block::default().borders(Borders::ALL).title("Network Topology"))
            .x_bounds([0.0, 1000.0])
            .y_bounds([0.0, 400.0])
            .paint(|ctx| {
                // Draw Curved Synapses 
                for synapse in &model.synapses {
                    if let (Some(pre), Some(post)) = (
                        model.layers.iter().find(|l| l.id == synapse.pre_layer),
                        model.layers.iter().find(|l| l.id == synapse.post_layer),
                    ) {
                        let mid_x = (pre.position.x + post.position.x) / 2.0;
                        let mid_y = (pre.position.y + post.position.y) / 2.0;
                        let dx = post.position.x - pre.position.x;
                        let dy = post.position.y - pre.position.y;
                        
                        // Normal vector to bend the connection
                        let nx = -dy * 0.15;
                        let ny = dx * 0.15;
                        let ctrl_x = mid_x + nx;
                        let ctrl_y = mid_y + ny;
                        
                        let mut prev_x = pre.position.x as f64;
                        let mut prev_y = pre.position.y as f64;
                        
                        for i in 1..=10 {
                            let t = i as f64 / 10.0;
                            let mt = 1.0 - t;
                            let cur_x = mt * mt * pre.position.x as f64 + 2.0 * mt * t * ctrl_x as f64 + t * t * post.position.x as f64;
                            let cur_y = mt * mt * pre.position.y as f64 + 2.0 * mt * t * ctrl_y as f64 + t * t * post.position.y as f64;
                            
                            ctx.draw(&CanvasLine {
                                x1: prev_x,
                                y1: prev_y,
                                x2: cur_x,
                                y2: cur_y,
                                color: Color::DarkGray,
                            });
                            prev_x = cur_x;
                            prev_y = cur_y;
                        }
                    }
                }

                // Draw Circle Layers
                for layer in &model.layers {
                    let activity_ratio = if layer.size > 0 {
                        layer.spike_count as f32 / layer.size as f32
                    } else { 0.0 };

                    let color = if self.selected_layer_id == Some(layer.id) {
                        Color::Yellow
                    } else if activity_ratio > 0.0 {
                        let intensity = (232.0 + activity_ratio * 23.0) as u8;
                        Color::Indexed(intensity)
                    } else {
                        Color::LightBlue
                    };

                    let radius = 10.0 + (layer.size.max(1) as f64).log10() * 5.0;
                    ctx.draw(&Circle {
                        x: layer.position.x as f64,
                        y: layer.position.y as f64,
                        radius,
                        color,
                    });
                    
                    ctx.print(
                        layer.position.x as f64,
                        layer.position.y as f64 - 20.0,
                        Span::styled(layer.name.clone(), Style::default().fg(Color::White))
                    );
                }
            });

        f.render_widget(canvas, area);
    }

    fn draw_details(&self, f: &mut Frame, area: Rect, model: &ModelStructure) {
        let mut items = Vec::new();

        if let Some(layer_id) = self.selected_layer_id {
            if let Some(layer) = model.layers.iter().find(|l| l.id == layer_id) {
                items.push(ListItem::new(format!("Selected Layer: {}", layer.name)));
                items.push(ListItem::new(format!("Type: {}", layer.layer_type)));
                items.push(ListItem::new(format!("Size: {} neurons", layer.size)));
                let activity_ratio = if layer.size > 0 {
                    layer.spike_count as f32 / layer.size as f32 * 100.0
                } else { 0.0 };
                items.push(ListItem::new(format!("Activity: {:.1}%", activity_ratio)));
            }
        } else {
            items.push(ListItem::new("No layer selected. Use Left/Right keys targeting."));
            items.push(ListItem::new(format!("Total Layers: {}", model.layers.len())));
            items.push(ListItem::new(format!("Total Synapses: {}", model.synapses.len())));
        }

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Details"));
        f.render_widget(list, area);
    }

    fn draw_raster(&self, f: &mut Frame, area: Rect, model: &ModelStructure) {
        let title = format!("Spike Raster (Epoch {})", self.displayed_epoch);
        if let Some(layer_id) = self.selected_layer_id {
            if let Some(layer) = model.layers.iter().find(|l| l.id == layer_id) {
                if self.spike_history.is_empty() {
                    let p = Paragraph::new("Waiting for epoch data...")
                        .block(Block::default().borders(Borders::ALL).title(title));
                    f.render_widget(p, area);
                    return;
                }

                let num_neurons = layer.size as f64;
                let num_timesteps = self.spike_history.len() as f64;
                let mut points = Vec::new();

                for (t, spikes) in self.spike_history.iter().enumerate() {
                    for (n, &spike) in spikes.iter().enumerate() {
                        if spike > 0.0 {
                            points.push((t as f64, num_neurons - n as f64));
                        }
                    }
                }

                let datasets = vec![
                    ratatui::widgets::Dataset::default()
                        .marker(ratatui::symbols::Marker::Dot)
                        .graph_type(ratatui::widgets::GraphType::Scatter)
                        .style(Style::default().fg(Color::Yellow))
                        .data(&points)
                ];

                let x_labels = vec![Span::raw("0"), Span::raw(format!("{}", num_timesteps))];
                let y_labels = vec![Span::raw("0"), Span::raw(format!("{}", num_neurons))];

                let chart = ratatui::widgets::Chart::new(datasets)
                    .block(Block::default().title(title).borders(Borders::ALL))
                    .x_axis(ratatui::widgets::Axis::default()
                        .bounds([0.0, num_timesteps.max(1.0)])
                        .labels(x_labels))
                    .y_axis(ratatui::widgets::Axis::default()
                        .bounds([0.0, num_neurons.max(1.0)])
                        .labels(y_labels));

                f.render_widget(chart, area);
                return;
            }
        }
        
        let p = Paragraph::new("Select a layer to view raster plot.")
            .block(Block::default().borders(Borders::ALL).title(title));
        f.render_widget(p, area);
    }

    fn draw_rewards(&self, f: &mut Frame, area: Rect, state: &VisualizationState) {
        let history = &state.epoch_rewards;
        if history.is_empty() {
            let p = Paragraph::new("No reward data yet.")
                .block(Block::default().borders(Borders::ALL).title("Reward History"));
            f.render_widget(p, area);
            return;
        }

        let max_epoch = history.iter().map(|(e, _)| *e).max().unwrap_or(0) as f64;
        let mut min_r = f32::MAX;
        let mut max_r = f32::MIN;
        let mut points = Vec::new();
        for &(e, r) in history.iter() {
            min_r = min_r.min(r);
            max_r = max_r.max(r);
            points.push((e as f64, r as f64));
        }

        let datasets = vec![
            ratatui::widgets::Dataset::default()
                .marker(ratatui::symbols::Marker::Braille)
                .graph_type(ratatui::widgets::GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(&points)
        ];

        let chart = ratatui::widgets::Chart::new(datasets)
            .block(Block::default().title("Reward History").borders(Borders::ALL))
            .x_axis(ratatui::widgets::Axis::default()
                .title("Epoch")
                .bounds([0.0, max_epoch.max(1.0)])
                .labels(vec![Span::raw("0"), Span::raw(format!("{}", max_epoch))]))
            .y_axis(ratatui::widgets::Axis::default()
                .title("Reward")
                .bounds([min_r.min(0.0) as f64, max_r.max(0.1) as f64])
                .labels(vec![Span::raw(format!("{:.1}", min_r)), Span::raw(format!("{:.1}", max_r))]));

        f.render_widget(chart, area);
    }

    fn draw_logs(&mut self, f: &mut Frame, area: Rect) {
        let logs = if let Ok(logs_guard) = super::GLOBAL_LOGS.lock() {
            logs_guard.iter().map(|l| ListItem::new(l.clone())).collect::<Vec<_>>()
        } else { vec![] };
        
        let total_logs = logs.len();
        
        // Auto scroll to latest if user hasn't explicitly scrolled backwards
        if !self.user_scrolled || self.log_state.selected().is_none() {
            if total_logs > 0 {
                self.log_state.select(Some(total_logs - 1));
            }
        } else {
            // Keep bounds in check if they delete themselves
            if let Some(idx) = self.log_state.selected() {
                if idx >= total_logs && total_logs > 0 {
                    self.log_state.select(Some(total_logs - 1));
                }
            }
            // If they reach bottom, reset user_scrolled state to allow auto-scrolling
            if let Some(idx) = self.log_state.selected() {
                if total_logs > 0 && idx == total_logs - 1 {
                    self.user_scrolled = false;
                }
            }
        }
        
        let list = List::new(logs)
            .block(Block::default().title("Execution Logs (Scroll with up/down arrows or mouse)").borders(Borders::ALL))
            .highlight_style(Style::default().bg(Color::DarkGray));
            
        f.render_stateful_widget(list, area, &mut self.log_state);
    }

    fn draw_help(&self, f: &mut Frame, screen_area: Rect) {
        let area = centered_rect(50, 50, screen_area);
        let help_text = vec![
            Line::from(Span::styled("Keyboard Shortcuts", Style::default().add_modifier(Modifier::BOLD))),
            Line::from(""),
            Line::from("  ?       Toggle this help menu"),
            Line::from("  q   Quit application"),
            Line::from("  p       Pause/Resume Training"),
            Line::from("  s       Save Model Checkpoint"),
            Line::from("  l       Load Local Checkpoint"),
            Line::from("  <-/->   Select / Cycle Layer"),
            Line::from("  Up/Down Scroll Execution Logs"),
        ];

        let p = Paragraph::new(help_text)
            .block(Block::default().borders(Borders::ALL).title("Help"))
            .alignment(Alignment::Left);
        
        f.render_widget(ratatui::widgets::Clear, area);
        f.render_widget(p, area);
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
