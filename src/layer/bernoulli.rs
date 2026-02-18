use crate::layer::{Layer, ModSignal};
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub struct BernoulliLayer {
    mod_signal: ModSignal,
    inputs: Tensor,
    spikes: Tensor,
    size: usize,
    current_label: f32,
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
            current_label: 1.0,
        })
    }
}

impl Layer for BernoulliLayer {
    fn step(&mut self, dt: f32) -> CandleResult<()> {
        // clamp input to [0,1]
        let clamped = self.inputs.clamp(0.0, 1.0)?;
        let random_vals = Tensor::rand_like(&clamped, 0.0, 1.1)?;
        self.spikes = clamped.ge(&random_vals)?.to_dtype(DType::F32)?;

        let lab = Tensor::ones((self.size, 1), DType::F32, self.inputs.device())?;
        self.mod_signal.calc_mod_signal(&self.spikes, &lab, dt)?;

        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.inputs)
    }

    fn get_mod_signal(&self) -> &Tensor {
        &self.mod_signal.mod_signal
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

    fn set_positive_sample(&mut self, label: f32) {
        self.current_label = label;
    }
}
