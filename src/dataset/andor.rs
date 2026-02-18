use candle_core::{Device, Result as CandleResult, Tensor};

pub struct AndOrDataset {
    inputs: Vec<Tensor>,
    labels: Vec<Tensor>,
    is_positive: Vec<f32>,
}

impl AndOrDataset {
    pub fn new(device: &Device) -> CandleResult<Self> {
        let pairs = vec![
            (vec![0.0f32, 0.0f32], vec![0.0f32], 1.0f32),
            (vec![0.0f32, 1.0f32], vec![0.0f32], 1.0f32),
            (vec![1.0f32, 0.0f32], vec![0.0f32], 1.0f32),
            (vec![1.0f32, 1.0f32], vec![1.0f32], 1.0f32),
            (vec![0.0f32, 0.0f32], vec![1.0f32], 0.0f32),
            (vec![0.0f32, 1.0f32], vec![1.0f32], 0.0f32),
            (vec![1.0f32, 0.0f32], vec![1.0f32], 0.0f32),
            (vec![1.0f32, 1.0f32], vec![0.0f32], 0.0f32),
        ];

        let mut ins = Vec::with_capacity(pairs.len());
        let mut labs = Vec::with_capacity(pairs.len());
        let mut pos = Vec::with_capacity(pairs.len());

        for (x, y, p) in pairs {
            ins.push(Tensor::from_vec(x, (2, 1), device)?);
            labs.push(Tensor::from_vec(y, (1, 1), device)?);
            pos.push(p);
        }

        Ok(Self {
            inputs: ins,
            labels: labs,
            is_positive: pos,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Tensor, &Tensor, &f32)> {
        self.inputs
            .iter()
            .zip(self.labels.iter())
            .zip(self.is_positive.iter())
            .map(|((i, l), p)| (i, l, p))
    }
}
