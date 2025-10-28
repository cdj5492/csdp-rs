use candle_core::{Device, DType, Tensor, Result as CandleResult};

pub struct XorDataset {
    inputs: Vec<Tensor>,
    labels: Vec<Tensor>,
}

impl XorDataset {
    pub fn new(device: &Device) -> CandleResult<Self> {
        let pairs = vec![
            (vec![0.0f32, 0.0f32], vec![0.0f32]),
            (vec![0.0f32, 1.0f32], vec![1.0f32]),
            (vec![1.0f32, 0.0f32], vec![1.0f32]),
            (vec![1.0f32, 1.0f32], vec![0.0f32]),
        ];
        let mut ins = Vec::with_capacity(4);
        let mut labs = Vec::with_capacity(4);
        for (x, y) in pairs {
            ins.push(Tensor::from_vec(x, (2,), device)?);
            labs.push(Tensor::from_vec(y, (1,), device)?);
        }
        Ok(Self {
            inputs: ins,
            labels: labs,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Tensor, &Tensor)> {
        self.inputs.iter().zip(self.labels.iter())
    }
}
