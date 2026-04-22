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
fn calc_goodness(z: &Tensor, thr: f32, maximize: bool) -> CandleResult<(Tensor, Tensor)> {
    let thr_sqr = (thr * thr) as f64;
    let z_sqr = z.mul(z)?;
    let delta = z_sqr.sum_keepdim(0)?; // shape (1, batch_size)
    let delta = if maximize {
        delta.affine(1.0, -thr_sqr)?
    } else {
        delta.affine(-1.0, thr_sqr)?
    };

    let p = sigmoid(&delta)?;
    let eps = 1e-5f64;
    let p_clamped = p.clamp(eps, 1.0 - eps)?;
    Ok((p_clamped, delta))
}

/// C[z(t), y_type] (cross-entropy)
fn calc_loss_ce(z: &Tensor, lab: &Tensor, thr: f32) -> CandleResult<Tensor> {
    let (_, logit) = calc_goodness(z, thr, true)?;

    // cross_entropy = - lab * logit + (1 + exp(-|logit|)).ln() + max(logit, 0)
    let abs_logit_neg = logit.abs()?.affine(-1.0, 0.0)?;
    let term1 = abs_logit_neg.exp()?.affine(1.0, 1.0)?;
    // Candle may lack `.log()`, so we fall back to a zero tensor if it fails, but `.log()` exists.
    let term1 = term1.log()?;
    let term2 = logit.maximum(&logit.zeros_like()?)?;

    let lab_logit = lab.mul(&logit)?;
    let cross_entropy = term1.add(&term2)?.sub(&lab_logit)?;
    Ok(cross_entropy)
}

impl ModSignalGenerator for StandardModSignal {
    fn calc_mod_signal(
        &mut self,
        spikes: &Tensor,
        lab: &Tensor,
        _reward: &Tensor, // Reward is ignored for the standard case
        dt: f32,
    ) -> CandleResult<()> {
        let dz_dt =
            (((dt / self.trace_tau) as f64) * (((self.max_z as f64) * spikes)?.sub(&self.z)?))?;
        self.z = self.z.add(&dz_dt)?;

        let (p, _) = calc_goodness(&self.z, self.omega, true)?; // p is shape (1, batch_size)

        // exact_dl_dlogit = size * p - lab
        let size = self.z.dims()[0] as f64;
        let exact_dl_dlogit = p.affine(size, 0.0)?.sub(lab)?;

        // dL/dz_i = exact_dl_dlogit * 2 * z_i
        let dl_dlogit_expanded = exact_dl_dlogit.broadcast_as(self.z.shape())?;
        let exact_d_z = self.z.mul(&dl_dlogit_expanded)?.affine(2.0, 0.0)?;

        let l = calc_loss_ce(&self.z, lab, self.omega)?;
        self.prev_loss = l;

        self.mod_signal = exact_d_z;
        Ok(())
    }

    fn get_mod_signal(&self) -> &Tensor {
        &self.mod_signal
    }
}
