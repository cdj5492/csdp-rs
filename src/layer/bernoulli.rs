use crate::layer::Layer;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub struct BernoulliLayer {
    /// random uniform [0, 1]
    rng_vals: Tensor,
    /// current probabilities
    probs: Tensor,
    /// output spikes
    spikes: Tensor,
    inputs: Tensor,
    size: usize,
    current_label: f32,
    dummy_mod_signal: Tensor,
}

impl BernoulliLayer {
    pub fn new(size: usize, device: &Device) -> CandleResult<Self> {
        let rng_vals = Tensor::rand(0.0f32, 1.0, (size, 1), device)?;
        let probs = Tensor::zeros((size, 1), DType::F32, device)?;
        let spikes = Tensor::zeros((size, 1), DType::F32, device)?;
        let inputs = Tensor::zeros((size, 1), DType::F32, device)?;
        let dummy_mod_signal = Tensor::zeros((size, 1), DType::F32, device)?;

        Ok(Self {
            rng_vals,
            probs,
            spikes,
            inputs,
            size,
            current_label: 1.0,
            dummy_mod_signal,
        })
    }
}

impl Layer for BernoulliLayer {
    fn step(&mut self, _dt: f32) -> CandleResult<()> {
        // Just raw bernoulli distribution of the inputs
        let eps = 1e-4;
        self.probs = self.inputs.clamp(eps, 1.0 - eps)?;

        // reroll rng
        self.rng_vals = Tensor::rand(0.0f32, 1.0, (self.size, 1), self.probs.device())?;

        // 1 if prob > rng
        self.spikes = self
            .probs
            .gt(&self.rng_vals)?
            .to_dtype(candle_core::DType::F32)?;
        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.probs)
    }

    fn get_mod_signal(&self) -> &Tensor {
        &self.dummy_mod_signal
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
        self.inputs = Tensor::zeros((self.size, 1), DType::F32, self.probs.device())?;
        Ok(())
    }

    fn reset(&mut self) -> CandleResult<()> {
        self.inputs = Tensor::zeros((self.size, 1), DType::F32, self.probs.device())?;
        self.probs = Tensor::zeros((self.size, 1), DType::F32, self.probs.device())?;
        self.spikes = Tensor::zeros((self.size, 1), DType::F32, self.probs.device())?;
        Ok(())
    }

    fn set_positive_sample(&mut self, label: f32) {
        self.current_label = label;
    }

    fn set_reward(&mut self, _reward: f32) {}
}
