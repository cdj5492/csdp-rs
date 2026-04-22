use candle_core::{Device, Tensor};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use custom_framework::models::csdp_multi_model::CSDPMultiModel;
use custom_framework::models::ff_multi_model::FFMultiModel;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{BarChart, Block, Borders, Paragraph},
};
use std::error::Error;
use std::io;
use std::time::{Duration, Instant};

fn value_to_class(value: f32, min_ret: f32, max_ret: f32, n_classes: usize) -> usize {
    if n_classes <= 1 {
        return 0;
    }
    let range = max_ret - min_ret;
    if range.abs() < 1e-8 {
        return n_classes / 2;
    }
    let t = ((value - min_ret) / range).clamp(0.0, 1.0);
    (t * (n_classes - 1) as f32).round() as usize
}

fn class_to_value(class_id: usize, min_ret: f32, max_ret: f32, n_classes: usize) -> f32 {
    if n_classes <= 1 || (max_ret - min_ret).abs() < 1e-8 {
        return (min_ret + max_ret) / 2.0;
    }
    min_ret + (class_id as f32 / (n_classes - 1) as f32) * (max_ret - min_ret)
}

fn encode_value(v: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(20);
    for i in 0..20 {
        let mu = (i as f32) * (100.0 / 19.0);
        let sigma = 8.0;
        let p = (-((v - mu).powi(2)) / (2.0 * sigma * sigma)).exp();
        out.push(p);
    }
    // L2 normalize
    let mut sum_sq = 0.0;
    for &val in &out {
        sum_sq += val * val;
    }
    let norm = (sum_sq + 1e-8).sqrt();
    for val in &mut out {
        *val /= norm;
    }
    out
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::cuda_if_available(0)?;

    // Config
    let input_size = 20;
    let num_classes = 50;
    let min_val = 0.0;
    let max_val = 100.0;

    // CSDP setup
    let mut csdp = CSDPMultiModel::new(
        input_size,
        &[1000], // hidden size must be divisible by num_classes (50)
        num_classes,
        &device,
        1.0,
        40,
    )?;

    // FF setup
    let mut ff = FFMultiModel::new(
        &[input_size, 1000],
        num_classes,
        &device,
        5, // num_epochs per train step
    )?;

    let args: Vec<String> = std::env::args().collect();
    let no_tui = args.iter().any(|a| a == "--no-tui");

    if no_tui {
        println!("Training without TUI for 400 batches...");
        for i in 0..400 {
            let batch_size = 64;
            let mut inputs_ff = Vec::with_capacity(batch_size * input_size);
            let mut labels_ff = Vec::with_capacity(batch_size);
            let mut labels_csdp = Vec::with_capacity(batch_size);

            for _ in 0..batch_size {
                let rand_val = rand::random::<f32>() * 100.0;
                let c_id = value_to_class(rand_val, min_val, max_val, num_classes);

                inputs_ff.extend(encode_value(rand_val));
                labels_ff.push(c_id);
                labels_csdp.push(c_id as f32);
            }

            let x_train = Tensor::from_vec(inputs_ff, (batch_size, input_size), &device)?;
            let y_train_csdp = Tensor::from_vec(labels_csdp, (1, batch_size), &device)?;

            let _ = ff.train(&x_train, &labels_ff);
            let _ = csdp.train(&x_train, &y_train_csdp, None);

            if i > 0 && i % 50 == 0 {
                let stats = csdp.synapses[0].synapse.weight_stats().unwrap();
                println!(
                    "Completed {} batches | CSDP Weights: Mean {:.4}, Std {:.4}, Min {:.4}, Max {:.4}",
                    i, stats.mean, stats.std, stats.min, stats.max
                );
            }
        }

        println!("--- Evaluation ---");
        for &target in &[10.0, 50.0, 90.0] {
            let curr_class = value_to_class(target, min_val, max_val, num_classes);
            println!("\nTarget: {} (Class {})", target, curr_class);

            let x_eval = Tensor::from_vec(encode_value(target), (1, input_size), &device)?;

            let tau_ff = 0.15;
            let tau_csdp = 0.002;

            if let Ok(ff_scores) = ff.predict_scores(std::slice::from_ref(&x_eval))
                && let Ok(ff_flat) = ff_scores.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    let max_v = ff_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = ff_flat
                        .iter()
                        .map(|&v| ((v - max_v) / tau_ff).exp())
                        .collect();
                    let sum: f32 = exps.iter().sum();
                    let probs: Vec<f32> = exps.iter().map(|v| v / sum).collect();
                    let best = probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    println!(
                        "FF_Multi2 best prediction: Class {} (Prob: {:.2})",
                        best, probs[best]
                    );
                    if target == 50.0 {
                        println!("FF_Probs: {:.2?}", &probs[15..35]);
                    }
                }
            if let Ok(csdp_scores) = csdp.predict_scores(std::slice::from_ref(&x_eval))
                && let Ok(csdp_flat) = csdp_scores.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    if target == 50.0 {
                        println!("Raw CSDP Goodness: {:.4?}", &csdp_flat[15..35]);
                    }
                    let max_v = csdp_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = csdp_flat
                        .iter()
                        .map(|&v| ((v - max_v) / tau_csdp).exp())
                        .collect();
                    let sum: f32 = exps.iter().sum();
                    let probs: Vec<f32> = exps.iter().map(|v| v / sum).collect();
                    let best = probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    println!(
                        "CSDP best prediction: Class {} (Prob: {:.2})",
                        best, probs[best]
                    );
                    if target == 50.0 {
                        println!("CSDP_Probs: {:.2?}", &probs[15..35]);
                    }
                }
        }
        return Ok(());
    }

    // TUI setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut current_target = 50.0;
    let mut target_dir = 0.5;

    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(50);

    let mut csdp_probs = vec![0.0; num_classes];
    let mut ff_probs = vec![0.0; num_classes];

    let _x_tensor = Tensor::from_vec(vec![1.0], (1, 1), &device)?;

    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(10)])
                .split(f.area());

            let curr_class = value_to_class(current_target, min_val, max_val, num_classes);
            let title = Paragraph::new(format!("Target Continuous Value: {:.1} (Mapped to discrete class: {}) | True Class Value: {:.1} | Press 'q' to quit | Press 'SPACE' to random jump", current_target, curr_class, class_to_value(curr_class, min_val, max_val, num_classes)))
                .block(Block::default().borders(Borders::ALL).title("Continuous Value Encoding Test"));
            f.render_widget(title, chunks[0]);

            let chart_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[1]);

            let curr_class = value_to_class(current_target, min_val, max_val, num_classes);
            
            let best_csdp = csdp_probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
            let best_ff = ff_probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);

            let csdp_data: Vec<(String, u64)> = csdp_probs.iter().enumerate().map(|(i, &p)| {
                let s = if i == curr_class && i == best_csdp { "✓".to_string() }
                        else if i == curr_class { "T".to_string() }
                        else if i == best_csdp { "P".to_string() }
                        else { "·".to_string() };
                (s, f32::min(p * 100.0, 100.0) as u64)
            }).collect();
            
            let csdp_bar_data: Vec<(&str, u64)> = csdp_data.iter().map(|(s, p)| (s.as_str(), *p)).collect();
            
            let avail_width = chart_chunks[0].width.saturating_sub(2);
            let width_per_bar = avail_width.checked_div(num_classes as u16).unwrap_or(1);
            let (bw, gap) = if width_per_bar >= 4 { (3, 1) } else if width_per_bar == 3 { (2, 1) } else if width_per_bar == 2 { (1, 1) } else { (1, 0) };

            let csdp_chart = BarChart::default()
                .block(Block::default().title("CSDP Expected Return Distribution").borders(Borders::ALL))
                .data(&csdp_bar_data)
                .bar_width(bw)
                .bar_gap(gap)
                .max(100)
                .value_style(Style::default().fg(Color::Yellow))
                .bar_style(Style::default().fg(Color::Cyan));
            f.render_widget(csdp_chart, chart_chunks[0]);

            let ff_data: Vec<(String, u64)> = ff_probs.iter().enumerate().map(|(i, &p)| {
                let s = if i == curr_class && i == best_ff { "✓".to_string() }
                        else if i == curr_class { "T".to_string() }
                        else if i == best_ff { "P".to_string() }
                        else { "·".to_string() };
                (s, f32::min(p * 100.0, 100.0) as u64)
            }).collect();
            
            let ff_bar_data: Vec<(&str, u64)> = ff_data.iter().map(|(s, p)| (s.as_str(), *p)).collect();
            let ff_chart = BarChart::default()
                .block(Block::default().title("FF_Multi2 Expected Return Distribution").borders(Borders::ALL))
                .data(&ff_bar_data)
                .bar_width(bw)
                .bar_gap(gap)
                .max(100)
                .value_style(Style::default().fg(Color::Yellow))
                .bar_style(Style::default().fg(Color::Magenta));
            f.render_widget(ff_chart, chart_chunks[1]);
        })?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)?
            && let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char(' ') => {
                        current_target = rand::random::<f32>() * 100.0;
                    }
                    _ => {}
                }
            }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();

            // Move target slowly
            current_target += target_dir;
            if current_target > max_val {
                current_target = max_val;
                target_dir *= -1.0;
            } else if current_target < min_val {
                current_target = min_val;
                target_dir *= -1.0;
            }

            // Train on random values across the continuous space
            let batch_size = 64;
            let mut inputs_ff = Vec::with_capacity(batch_size * input_size);
            let mut labels_ff = Vec::with_capacity(batch_size);
            let mut labels_csdp = Vec::with_capacity(batch_size);

            for _ in 0..batch_size {
                let rand_val = rand::random::<f32>() * 100.0;
                let c_id = value_to_class(rand_val, min_val, max_val, num_classes);

                let v = rand_val / 100.0;
                let norm = (v.powi(2) + 1.0).sqrt();
                inputs_ff.push(v / norm);
                inputs_ff.push(1.0 / norm); // Bias channel to prevent feature erasure on L2 norm
                labels_ff.push(c_id);
                labels_csdp.push(c_id as f32);
            }

            let x_train = Tensor::from_vec(inputs_ff, (batch_size, input_size), &device)?;
            let y_train_csdp = Tensor::from_vec(labels_csdp, (1, batch_size), &device)?;

            let _ = ff.train(&x_train, &labels_ff);
            let _ = csdp.train(&x_train, &y_train_csdp, None);

            // Evaluate on current_target
            let x_eval = Tensor::from_vec(encode_value(current_target), (1, input_size), &device)?;

            if let Ok(ff_scores) = ff.predict_scores(std::slice::from_ref(&x_eval))
                && let Ok(ff_flat) = ff_scores.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    let max_v = ff_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let tau_ff = 0.15; // Makes FF much more narrow and peaky
                    let exps: Vec<f32> = ff_flat
                        .iter()
                        .map(|&v| ((v - max_v) / tau_ff).exp())
                        .collect();
                    let sum: f32 = exps.iter().sum();
                    ff_probs = exps.iter().map(|v| v / sum).collect();
                }

            if let Ok(csdp_scores) = csdp.predict_scores(std::slice::from_ref(&x_eval))
                && let Ok(csdp_flat) = csdp_scores.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    let max_v = csdp_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let tau_csdp = 0.15; // CSDP goodnesses are bounded by 1.0, difference is smaller
                    let exps: Vec<f32> = csdp_flat
                        .iter()
                        .map(|&v| ((v - max_v) / tau_csdp).exp())
                        .collect();
                    let sum: f32 = exps.iter().sum();
                    csdp_probs = exps.iter().map(|v| v / sum).collect();
                }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
