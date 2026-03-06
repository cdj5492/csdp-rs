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

/// C[z(t), y_type] modulated by external reward R(x, a)
fn calc_loss_reward_modulated(
    z: &Tensor,
    lab: &Tensor,
    thr: f32,
    reward: f32,
) -> CandleResult<Tensor> {
    let logit = calc_goodness(z, thr, true)?;
    // p = sigmoid(logit)
    // log(p) = -log(1 + exp(-logit))
    // log(1-p) = -logit - log(1 + exp(-logit))

    let log_p = -((1.0 + (-logit.abs()).exp()).log10() + logit.max(0.0)) as f64;
    let log_1_minus_p = -logit as f64 + log_p;

    // Loss = - [ Y * log(P) * R + (1 - Y) * log(1 - P) ]
    // Tensor operations:
    // Term 1: lab * log_p * reward
    let term1 = lab.affine(log_p * reward as f64, 0.0)?;

    // Term 2: (1 - lab) * log_1_minus_p
    let ones = lab.ones_like()?;
    let one_minus_lab = ones.sub(lab)?;
    let term2 = one_minus_lab.affine(log_1_minus_p, 0.0)?;

    let loss = (term1.add(&term2)?).affine(-1.0, 0.0)?;
    Ok(loss)
}

impl ModSignalGenerator for RewardModulatedModSignal {
    fn calc_mod_signal(
        &mut self,
        spikes: &Tensor,
        lab: &Tensor,
        reward: f32,
        dt: f32,
    ) -> CandleResult<()> {
        let dz =
            (((dt / self.trace_tau) as f64) * (((self.max_z as f64) * spikes)?.sub(&self.z)?))?;
        self.z = self.z.add(&dz)?;
        let l = calc_loss_reward_modulated(&self.z, lab, self.omega, reward)?;
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
