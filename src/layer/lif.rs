use crate::layer::{Layer, ModSignal};
use candle_core::{DType, Device, Result as CandleResult, Tensor};

#[allow(clippy::upper_case_acronyms)]
pub struct LIFLayer {
    mod_signal: ModSignal,
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
}

impl LIFLayer {
    pub fn new(
        size: usize,
        tau: f32,
        thresh: f32,
        thresh_lambda: f32,
        trace_tau: f32,
        device: &Device,
    ) -> CandleResult<Self> {
        let inputs = Tensor::zeros((size, 1), DType::F32, device)?;
        let state = Tensor::zeros((size, 1), DType::F32, device)?;
        let spikes = Tensor::zeros((size, 1), DType::F32, device)?;
        let max_z = 1.0; // not much of a reason to have it not be 1
        Ok(Self {
            mod_signal: ModSignal::new(
                size,
                trace_tau,
                1.0,
                (size as f32) * max_z * max_z / 2.0,
                device,
            )?,
            inputs,
            state,
            spikes,
            tau,
            thresh,
            thresh_lambda,
            size,
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
                - 1.0);

        println!("thresh: {}", self.thresh);

        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.state)
    }

    fn calc_mod_signal(&mut self, dt: f32) -> CandleResult<Tensor> {
        // TODO: based on positive/negative samples. For now just 1 extended to number of neurons
        // (all positive samples)
        let lab = self.spikes.gt(-1.0)?.to_dtype(DType::F32)?;
        self.mod_signal.calc_mod_signal(&self.spikes, &lab, dt)
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
}
