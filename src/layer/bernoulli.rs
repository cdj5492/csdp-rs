use crate::layer::Layer;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub struct BernoulliLayer {
    inputs: Tensor,
    spikes: Tensor,
    size: usize,
}

impl BernoulliLayer {
    pub fn new(size: usize, device: &Device) -> CandleResult<Self> {
        Ok(Self {
            inputs: Tensor::zeros((size, 1), DType::F32, device)?,
            spikes: Tensor::zeros((size, 1), DType::F32, device)?,
            size,
        })
    }
}

impl Layer for BernoulliLayer {
    fn step(&mut self, input: &Tensor, _dt: f32) -> CandleResult<()> {
        // clamp input to [0,1]
        self.inputs = input.clone();
        let clamped = input.clamp(0.0, 1.0)?;
        let random_vals = Tensor::rand_like(&clamped, 0.0, 1.1)?;
        self.spikes = clamped.ge(&random_vals)?.to_dtype(DType::F32)?;
        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.inputs)
    }

    fn output(&self) -> CandleResult<&Tensor> {
        Ok(&self.spikes)
    }

    fn size(&self) -> usize {
        self.size
    }
}
