use crate::layer::CellUpdate;
use candle_core::{Tensor, DType, Device, Result as CandleResult};
use std::sync::Arc;

#[derive(Clone)]
pub struct LayerConfig {
    pub n: usize,
}

impl LayerConfig {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

pub struct Layer {
    size: usize,
    state: Tensor,
    // activity is non-binary rate-like activation derived from state (ReLU-thresholded)
    activity: Tensor,
    // boxed update rule
    updater: Box<dyn CellUpdate>,
    device: Device,
    threshold: f32,
}

impl Layer {
    pub fn new(cfg: LayerConfig, device: &Device) -> CandleResult<Self> {
        let state = Tensor::zeros((cfg.n,), DType::F32, device)?;
        let activity = state.zeros_like()?;
        // default leaky integrator updater
        let updater: Box<dyn CellUpdate> = Box::new(LeakyIntegrator { tau: 10.0 });
        Ok(Self {
            size: cfg.n,
            state,
            activity,
            updater,
            device: device.clone(),
            threshold: 0.5,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn step(&mut self, external_input: Option<&Tensor>) -> CandleResult<()> {
        let input = match external_input {
            Some(x) => x.clone(),
            None => Tensor::zeros((self.size,), DType::F32, &self.device)?,
        };
        // dt=1 for simplicity
        let new_state = self.updater.update(&self.state, &input, 1.0)?;
        // update internal state and derive activity = relu(new_state - threshold)
        self.state = new_state;
        let shifted = self.state.affine(1.0, -(self.threshold as f64))?;
        let zeros = shifted.zeros_like()?;
        self.activity = shifted.maximum(&zeros)?;
        Ok(())
    }

    pub fn activity(&self) -> CandleResult<Tensor> {
        self.activity.clone().force_contiguous()
    }
}

// simple leaky integrator
struct LeakyIntegrator {
    tau: f32,
}

impl CellUpdate for LeakyIntegrator {
    fn update(&self, state: &Tensor, input: &Tensor, dt: f32) -> CandleResult<Tensor> {
        // dv = ( -state + input ) * (dt / tau)
        let neg = state.affine(-1.0, 0.0)?;
        let sum = neg.add(input)?;
        let dv = sum.affine((dt / self.tau) as f64, 0.0)?;
        let new_state = state.add(&dv)?;
        Ok(new_state)
    }
}
