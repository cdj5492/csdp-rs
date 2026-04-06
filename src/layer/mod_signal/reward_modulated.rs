use super::ModSignalGenerator;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::ops::sigmoid;

/// Calculates the modulatory goodness signal used for CSDP synapse adjustment.
/// This calculates a cross-entropy loss that is modulated by the reward signal.
/// C(z, y) = - [ y * log(p(y=1|z)) * R + (1-y) * log(p(y=0|z)) ]
pub struct RewardModulatedModSignal {
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

impl RewardModulatedModSignal {
    pub fn new(
        size: usize,
        trace_tau: f32,
        max_z: f32,
        omega: f32,
        device: &Device,
    ) -> CandleResult<Self> {
        Ok(RewardModulatedModSignal {
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

/// C[z(t), y_type] modulated by external reward R(x, a)
fn calc_loss_reward_modulated(
    z: &Tensor,
    lab: &Tensor,
    thr: f32,
    reward: &Tensor,
) -> CandleResult<Tensor> {
    let (_, logit) = calc_goodness(z, thr, true)?;

    let abs_logit_neg = logit.abs()?.affine(-1.0, 0.0)?;
    let term = abs_logit_neg.exp()?.affine(1.0, 1.0)?.log()?;
    
    let log_p = term.add(&logit.maximum(&logit.zeros_like()?)?)?.affine(-1.0, 0.0)?;
    let log_1_minus_p = log_p.sub(&logit)?;

    // Loss = - [ Y * log_p * R + (1 - Y) * log_1_minus_p ]
    let term1 = lab.mul(&log_p)?.mul(reward)?;

    let ones = lab.ones_like()?;
    let one_minus_lab = ones.sub(lab)?;
    let term2 = one_minus_lab.mul(&log_1_minus_p)?;

    let loss = (term1.add(&term2)?).affine(-1.0, 0.0)?;
    Ok(loss)
}

impl ModSignalGenerator for RewardModulatedModSignal {
    fn calc_mod_signal(
        &mut self,
        spikes: &Tensor,
        lab: &Tensor,
        reward: &Tensor,
        dt: f32,
    ) -> CandleResult<()> {
        let dz =
            (((dt / self.trace_tau) as f64) * (((self.max_z as f64) * spikes)?.sub(&self.z)?))?;
        self.z = self.z.add(&dz)?;
        let l = calc_loss_reward_modulated(&self.z, lab, self.omega, reward)?;
        let dl = l.sub(&self.prev_loss)?;
        self.prev_loss = l;
        let dz_ep = (&dz + 0.00001)?;

        let dl_expanded = dl.broadcast_as(dz_ep.shape())?;
        self.mod_signal = dl_expanded.div(&dz_ep)?;
        Ok(())
    }

    fn get_mod_signal(&self) -> &Tensor {
        &self.mod_signal
    }
}
