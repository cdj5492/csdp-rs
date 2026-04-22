use super::ModSignalGenerator;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

/// Calculates the modulatory goodness signal used for CSDP synapse adjustment.
/// This uses the Multi-Class Contrastive approach (margin ranking loss).
pub struct MultiClassModSignal {
    /// trace constant
    pub trace_tau: f32,
    /// maximum averaged z value
    pub max_z: f32,
    /// goodness distribution shift (threshold)
    pub threshold: f32,
    /// number of classes the layer is divided into
    pub num_classes: usize,
    /// current z
    pub z: Tensor,
    /// previous loss placeholder
    pub prev_loss: Tensor,
    /// modulatory signal
    pub mod_signal: Tensor,
}

impl MultiClassModSignal {
    pub fn new(
        size: usize,
        trace_tau: f32,
        max_z: f32,
        threshold: f32,
        num_classes: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        assert!(
            size.is_multiple_of(num_classes),
            "Layer size must be divisible by num_classes"
        );
        Ok(MultiClassModSignal {
            trace_tau,
            max_z,
            threshold,
            num_classes,
            z: Tensor::zeros((size, 1), DType::F32, device)?,
            prev_loss: Tensor::zeros((size, 1), DType::F32, device)?,
            mod_signal: Tensor::zeros((size, 1), DType::F32, device)?,
        })
    }
}

impl ModSignalGenerator for MultiClassModSignal {
    fn calc_mod_signal(
        &mut self,
        spikes: &Tensor,
        lab: &Tensor, // This will be (1, batch_size) integer values
        _reward: &Tensor,
        dt: f32,
    ) -> CandleResult<()> {
        let current_batch_size = spikes.dims()[1];
        let size = self.z.dims()[0];

        if self.z.dims()[1] != current_batch_size {
            // Reinitialize z if batch size changes (which means a new sequence boundary anyway)
            self.z = Tensor::zeros((size, current_batch_size), DType::F32, spikes.device())?;
        }

        let dz_dt =
            (((dt / self.trace_tau) as f64) * (((self.max_z as f64) * spikes)?.sub(&self.z)?))?;
        self.z = self.z.add(&dz_dt)?;

        let size = self.z.dims()[0];
        let batch_size = self.z.dims()[1];
        let chunk_size = size / self.num_classes;

        let z_sqr = self.z.sqr()?; // (size, batch_size)
        // chunking z_sqr
        let chunks = z_sqr.chunk(self.num_classes, 0)?;

        let mut mod_signal_chunks = Vec::new();
        // Since lab is a tensor of f32 (Candle normally), let's get the values locally to build mask
        let lab_values = lab.flatten_all()?.to_vec1::<f32>()?;

        for (c, chunk) in chunks.into_iter().enumerate() {
            // chunk is (chunk_size, batch_size)
            let g = chunk.mean_keepdim(0)?; // shape: (1, batch_size)

            let mut y_c_vec = Vec::with_capacity(batch_size);
            for &l in &lab_values {
                if (l as usize) == c {
                    y_c_vec.push(1.0f32);
                } else {
                    y_c_vec.push(0.0f32);
                }
            }
            let y_c = Tensor::from_vec(y_c_vec, (1, batch_size), self.z.device())?;

            // Target chunk wants g -> threshold. Error: threshold - g
            let pos_term = g.affine(-1.0, self.threshold as f64)?;
            // Non-target chunk wants g -> 0.0. Error: g.
            // We use g directly for sig_neg.
            let sig_pos = pos_term; // Pos error (positive means "push weight up")
            let sig_neg = g.clone(); // Neg error (positive means "push weight down")

            // score_c = y_c * sig_pos - (1 - y_c) * alpha * sig_neg
            let alpha = 1.0; // Let's just use 1.0 for aggressive suppression of non-targets
            let not_y_c = y_c.affine(-1.0, 1.0)?;

            let pos_score = sig_pos.mul(&y_c)?;
            let neg_score = sig_neg.mul(&not_y_c)?.affine(-alpha , 0.0)?;
            let score_c = pos_score.add(&neg_score)?; // (1, batch_size)

            // mod_signal_i = score_c * (2.0 / chunk_size) * z_i
            let score_c_expanded = score_c.broadcast_as(chunk.shape())?;
            // Using a subset of original z corresponding to this chunk
            let z_chunk = self.z.narrow(0, c * chunk_size, chunk_size)?;

            let factor = 2.0 / (chunk_size as f64);
            let z_chunk_mod = z_chunk.mul(&score_c_expanded)?.affine(factor, 0.0)?;
            mod_signal_chunks.push(z_chunk_mod);
        }

        self.mod_signal = Tensor::cat(&mod_signal_chunks, 0)?;
        Ok(())
    }

    fn get_mod_signal(&self) -> &Tensor {
        &self.mod_signal
    }
}
