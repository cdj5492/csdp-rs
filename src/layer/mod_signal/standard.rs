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
fn calc_goodness(z: &Tensor, thr: f32, maximize: bool) -> CandleResult<(f32, f32)> {
    // returns (p, logit)
    let thr = thr as f64;
    let z_sqr = z.mul(z)?;
    let delta = z_sqr.sum_all()?;
    let delta = if maximize {
        ((-thr * thr) + delta)?
    } else {
        ((thr * thr) - delta)?
    };

    let p = sigmoid(&delta)?;
    let eps = 1e-5f64;
    let p_clamped = p.clamp(eps, 1.0 - eps)?.to_scalar::<f32>()?;
    let logit = delta.to_scalar::<f32>()?;
    Ok((p_clamped, logit))
}

/// C[z(t), y_type] (cross-entropy)
fn calc_loss_ce(z: &Tensor, lab: &Tensor, thr: f32) -> CandleResult<Tensor> {
    let (_, logit) = calc_goodness(z, thr, true)?; // fix 2: extract logit correctly
    let cross_entropy = lab.affine(
        -logit as f64,
        // fix 3: natural log (ln) instead of log10
        ((1.0 + (-logit.abs()).exp()).ln() + logit.max(0.0)) as f64,
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
        let dz_dt =
            (((dt / self.trace_tau) as f64) * (((self.max_z as f64) * spikes)?.sub(&self.z)?))?;
        self.z = self.z.add(&dz_dt)?;

        // fix 1: finite difference approx replaced by exact autodiff analytical derivative of CE loss
        // d_z = dL/dz = sum_j(p - lab_j) * 2z
        let (p, _) = calc_goodness(&self.z, self.omega, true)?;

        let lab_f32 = lab.to_dtype(DType::F32)?;
        let lab_cond = lab_f32.gt(0.0f32)?.to_dtype(DType::F32)?;
        let sum_lab = lab_cond.sum_all()?.to_scalar::<f32>()?;
        let size = self.z.dims()[0] as f32;

        // L = sum_j L_j, where L_j is the CE w.r.t logit. dL/d(logit) = size * p - sum(lab_j > 0)
        let exact_dl_dlogit = size * p - sum_lab;

        // dL/dz_i = dL/d(logit) * d(logit)/dz_i = dL/d(logit) * 2 * z_i
        let exact_d_z = self.z.affine((exact_dl_dlogit * 2.0) as f64, 0.0)?;

        // update tracked formal loss for metrics, but standard mod signal is strictly the exact gradient directly natively
        let l = calc_loss_ce(&self.z, lab, self.omega)?;
        self.prev_loss = l;

        self.mod_signal = exact_d_z;
        Ok(())
    }

    fn get_mod_signal(&self) -> &Tensor {
        &self.mod_signal
    }
}
