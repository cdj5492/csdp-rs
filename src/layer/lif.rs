use crate::layer::Layer;
use crate::layer::mod_signal::ModSignalGenerator;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

#[allow(clippy::upper_case_acronyms)]
pub struct LIFLayer {
    mod_signal: Box<dyn ModSignalGenerator>,
    /// input currents
    inputs: Tensor,
    /// membrane potential
    state: Tensor,
    /// output spikes
    spikes: Tensor,
    /// current threshold value
    thresh: f32,
    /// how fast threshold adapts
    thresh_lambda: f32,
    /// membrane time constant
    tau: f32,
    size: usize,
    current_label: f32,
    current_reward: f32,
}

impl LIFLayer {
    pub fn new(
        size: usize,
        tau: f32,
        thresh: f32,
        thresh_lambda: f32,
        mod_signal_generator: Box<dyn ModSignalGenerator>,
        device: &Device,
    ) -> CandleResult<Self> {
        let inputs = Tensor::zeros((size, 1), DType::F32, device)?;
        let state = Tensor::zeros((size, 1), DType::F32, device)?;
        let spikes = Tensor::zeros((size, 1), DType::F32, device)?;
        Ok(Self {
            mod_signal: mod_signal_generator,
            inputs,
            state,
            spikes,
            tau,
            thresh,
            thresh_lambda,
            size,
            current_label: 1.0,
            current_reward: 0.0,
        })
    }
}

impl Layer for LIFLayer {
    fn step(&mut self, dt: f32) -> CandleResult<()> {
        let dv = (((dt / self.tau) as f64) * self.inputs.sub(&self.state)?)?;
        self.state = self.state.add(&dv)?;
        // spikes where state > thresh
        self.spikes = self.state.gt(self.thresh)?.to_dtype(DType::F32)?;
        self.state = self.state.sub(&((self.thresh as f64) * &self.spikes)?)?;

        // adjust threshold adaptively
        self.thresh += dt
            * self.thresh_lambda
            * (self
                .spikes
                .sum_all()?
                .to_device(&Device::Cpu)?
                .to_scalar::<f32>()?
                - (self.size as f32 * 0.02)); // Target 2% firing rate

        if self.thresh < 0.0 {
            self.thresh = 0.0;
        }


        let lab = (self.spikes.ones_like()? * self.current_label as f64)?;
        self.mod_signal
            .calc_mod_signal(&self.spikes, &lab, self.current_reward, dt)?;

        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.state)
    }

    fn get_mod_signal(&self) -> &Tensor {
        self.mod_signal.get_mod_signal()
    }

    fn output(&self) -> CandleResult<&Tensor> {
        Ok(&self.spikes)
    }

    fn size(&self) -> usize {
        self.size
    }

    /// Adds to the input compartment of the layer
    fn add_input(&mut self, input: &Tensor) -> CandleResult<()> {
        self.inputs = self.inputs.add(input)?;
        Ok(())
    }

    /// resets input compartment to zero
    fn reset_input(&mut self) -> CandleResult<()> {
        self.inputs = Tensor::zeros((self.size, 1), DType::F32, self.state.device())?;
        Ok(())
    }

    /// resets internal state fully
    fn reset(&mut self) -> CandleResult<()> {
        self.state = Tensor::zeros((self.size, 1), DType::F32, self.state.device())?;
        self.spikes = Tensor::zeros((self.size, 1), DType::F32, self.state.device())?;
        Ok(())
    }

    fn set_positive_sample(&mut self, label: f32) {
        self.current_label = label;
    }

    fn set_reward(&mut self, reward: f32) {
        self.current_reward = reward;
    }
}
