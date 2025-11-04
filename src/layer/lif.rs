use crate::layer::Layer;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub struct LIFLayer {
    /// input currents
    inputs: Tensor,
    /// membrane potential
    state: Tensor,
    /// output spikes
    spikes: Tensor,
    /// average output spike rates
    avg_rate: Tensor,
    /// averaging constant
    trace_tau: f32,
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
        let avg_rate = Tensor::zeros((size, 1), DType::F32, device)?;
        Ok(Self {
            inputs,
            state,
            spikes,
            avg_rate,
            tau,
            thresh,
            thresh_lambda,
            trace_tau,
            size,
        })
    }
}

impl Layer for LIFLayer {
    fn step(&mut self, dt: f32) -> CandleResult<()> {
        let dv = self.inputs
            .sub(&self.state)?
            .affine((dt / self.tau) as f64, 0.0)?;
        self.state = self.state.add(&dv)?;
        // spikes where state > thresh
        self.spikes = self.state.gt(self.thresh)?.to_dtype(DType::F32)?;
        self.state = self
            .state
            .sub(&self.spikes.affine(self.thresh as f64, 0.0)?)?;

        // adjust threshold adaptively
        self.thresh += dt
            * self.thresh_lambda
            * (self
                .spikes
                .sum_all()?
                .to_device(&Device::Cpu)?
                .to_scalar::<f32>()?
                / (self.size as f32)
                - 1.0);
        
        // update average rate
        let dz = self.spikes.sub(&self.avg_rate)?.affine((dt / self.trace_tau) as f64, 0.0)?;
        self.avg_rate = self.avg_rate.add(&dz)?;

        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.state)
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
        self.avg_rate = Tensor::zeros((self.size, 1), DType::F32, self.state.device())?;
        Ok(())
    }
}
