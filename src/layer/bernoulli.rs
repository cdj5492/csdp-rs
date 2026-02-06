use crate::layer::{Layer, ModSignal};
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub struct BernoulliLayer {
    mod_signal: ModSignal,
    inputs: Tensor,
    spikes: Tensor,
    size: usize,
}

impl BernoulliLayer {
    pub fn new(size: usize, device: &Device) -> CandleResult<Self> {
        Ok(Self {
            // goodness should not be optimized for on input layers (which is what this will be
            // used for)
            mod_signal: ModSignal::new(size, 0.0, 0.0, 0.0, device)?,
            inputs: Tensor::zeros((size, 1), DType::F32, device)?,
            spikes: Tensor::zeros((size, 1), DType::F32, device)?,
            size,
        })
    }
}

impl Layer for BernoulliLayer {
    fn step(&mut self, _dt: f32) -> CandleResult<()> {
        // clamp input to [0,1]
        let clamped = self.inputs.clamp(0.0, 1.0)?;
        let random_vals = Tensor::rand_like(&clamped, 0.0, 1.1)?;
        self.spikes = clamped.ge(&random_vals)?.to_dtype(DType::F32)?;
        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.inputs)
    }

    fn calc_mod_signal(&mut self, dt: f32) -> CandleResult<Tensor> {
        self.mod_signal
            .calc_mod_signal(&self.spikes, &self.spikes, dt)
    }

    fn output(&self) -> CandleResult<&Tensor> {
        Ok(&self.spikes)
    }

    fn size(&self) -> usize {
        self.size
    }

    fn add_input(&mut self, input: &Tensor) -> CandleResult<()> {
        self.inputs = self.inputs.add(input)?;
        Ok(())
    }

    fn reset_input(&mut self) -> CandleResult<()> {
        self.inputs = self.inputs.zeros_like()?;
        Ok(())
    }

    fn reset(&mut self) -> CandleResult<()> {
        self.inputs = self.inputs.zeros_like()?;
        self.spikes = self.spikes.zeros_like()?;
        Ok(())
    }

    fn set_current_label(&mut self, _label: f32) {}
}
