use super::ModSignalGenerator;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::ops::sigmoid;

/// Calculates the modulatory goodness signal used for CSDP synapse adjustment.
/// This uses the standard Cross-Entropy approach and ignores external reward.
pub struct StandardModSignal {
    /// trace constant
    pub trace_tau: f32,
    /// maximum averaged z value
    pub max_z: f32,
    /// goodness distribution shift
    pub omega: f32,
    /// current z
    pub z: Tensor,
    /// previous loss
    pub prev_loss: Tensor,
    /// modulatory signal
    pub mod_signal: Tensor,
}

impl StandardModSignal {
    pub fn new(
        size: usize,
        trace_tau: f32,
        max_z: f32,
        omega: f32,
        device: &Device,
    ) -> CandleResult<Self> {
        Ok(StandardModSignal {
            trace_tau,
            max_z,
            omega,
            z: Tensor::zeros((size, 1), DType::F32, device)?,
            prev_loss: Tensor::zeros((size, 1), DType::F32, device)?,
            mod_signal: Tensor::zeros((size, 1), DType::F32, device)?,
        })
    }
}

/// p[y_type=1; z(t)]
fn calc_goodness(z: &Tensor, thr: f32, maximize: bool) -> CandleResult<f32> {
    let thr = thr as f64;
    let z_sqr = z.mul(z)?;
    let delta = z_sqr.sum_all()?;
    let delta = if maximize {
        ((-thr * thr) + delta)?
    } else {
        ((thr * thr) - delta)?
    };

    let p = sigmoid(&delta)?;
    let eps = 1e-5;
    p.clamp(eps, 1.0 - eps)?.to_scalar::<f32>()
}

/// C[z(t), y_type] (cross-entropy)
fn calc_loss_ce(z: &Tensor, lab: &Tensor, thr: f32) -> CandleResult<Tensor> {
    let logit = calc_goodness(z, thr, true)?;
    let cross_entropy = lab.affine(
        -logit as f64,
        ((1.0 + (-logit.abs()).exp()).log10() + logit.max(0.0)) as f64,
    )?;
    Ok(cross_entropy)
}

impl ModSignalGenerator for StandardModSignal {
    fn calc_mod_signal(
        &mut self,
        spikes: &Tensor,
        lab: &Tensor,
        _reward: f32, // Reward is ignored for the standard case
        dt: f32,
    ) -> CandleResult<()> {
        let dz =
            (((dt / self.trace_tau) as f64) * (((self.max_z as f64) * spikes)?.sub(&self.z)?))?;
        self.z = self.z.add(&dz)?;
        let l = calc_loss_ce(&self.z, lab, self.omega)?;
        let dl = l.sub(&self.prev_loss)?;
        self.prev_loss = l;
        let dz_ep = (&dz + 0.00001)?;

        self.mod_signal = dl.div(&dz_ep)?;
        Ok(())
    }

    fn get_mod_signal(&self) -> &Tensor {
        &self.mod_signal
    }
}
