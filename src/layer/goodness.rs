use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::ops::sigmoid;

use crate::layer::Layer;

/// p[y_type=1; z(t)]
fn calc_goodness(z: &Tensor, thr: f32, maximize: bool) -> CandleResult<f32> {
    let z_sqr = z.mul(z)?;
    let delta = z_sqr.sum_all()?;
    let delta = if maximize {
        delta.affine(1.0 as f64, -(thr * thr) as f64)?
    } else {
        delta.affine(-1.0 as f64, (thr * thr) as f64)?
    };

    let p = sigmoid(&delta)?;
    let eps = 1e-5;
    p.clamp(eps, 1.0 - eps)?.to_scalar::<f32>()
}

/// C[z(t), y_type] (cross-entropy)
fn calc_loss_ce(z: &Tensor, lab: &Tensor, thr: f32) -> CandleResult<Tensor> {
    let lab = lab.gt(0.0)?.to_dtype(DType::F16)?;
    let logit = calc_goodness(z, thr, true)?;
    // TODO: Figure out why this is the way that it is
    let cross_entropy = lab.affine(
        -logit as f64,
        ((1.0 + (-logit.abs()).exp()).log10() + logit.max(0.0)) as f64,
    )?;
    let l = cross_entropy.sum_keepdim(1)?.mean(1)?;
    Ok(l)
}
/// dC/dz
fn calc_mod_signal(z: &Tensor, lab: &Tensor, thr: f32) -> CandleResult<(Tensor, Tensor)> {
    let z = z.detach();
    let l = calc_loss_ce(&z, lab, thr)?;
    // map none to err
    let grads = l.backward()?;
    let grad = grads.get(&l).expect("Couldn't get gradient for loss");
    Ok((l, grad.clone()))
}

pub struct GoodnessLayer {
    pub state: Tensor,
    pub loss: Tensor,
    pub mod_signal: Tensor,
    pub thresh: f32,
    pub size: usize,
}

impl GoodnessLayer {
    pub fn new(size: usize, thresh: f32, device: &candle_core::Device) -> CandleResult<Self> {
        let loss = Tensor::zeros((1, 1), DType::F32, device)?;
        let mod_signal = Tensor::zeros((1, 1), DType::F32, device)?;
        Ok(Self {
            thresh,
            loss,
            mod_signal,
            size,
        })
    }
}

impl Layer for GoodnessLayer {
    fn step(&mut self, input: &Tensor, dt: f32) -> CandleResult<()> {
        // noop
        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.state)
    }

    fn output(&self) -> CandleResult<&Tensor> {
        Ok(&self.state)
    }

    fn size(&self) -> usize {}
}
