pub mod bernoulli;
pub mod lif;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::ops::sigmoid;
// pub mod goodness;

/// p[y_type=1; z(t)]
fn calc_goodness(z: &Tensor, thr: f32, maximize: bool) -> CandleResult<f32> {
    let thr = thr as f64;
    let z_sqr = z.mul(z)?;
    // TODO: I don't think this is right
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
    // TODO: Figure out why this is the way that it is
    let cross_entropy = lab.affine(
        -logit as f64,
        ((1.0 + (-logit.abs()).exp()).log10() + logit.max(0.0)) as f64,
    )?;
    Ok(cross_entropy)
    // let l = cross_entropy.sum_keepdim(0)?;
    // Ok(l)
}

/// used in calculating the modulatory signal dC/dz
struct ModSignal {
    /// trace constant
    trace_tau: f32,
    /// maximum averaged z value
    max_z: f32,
    /// goodness distribution shift
    omega: f32,
    /// current z
    z: Tensor,
    /// previous loss
    prev_loss: Tensor,
    /// modulartory signal
    mod_signal: Tensor,
}

impl ModSignal {
    fn new(
        size: usize,
        trace_tau: f32,
        max_z: f32,
        omega: f32,
        device: &Device,
    ) -> CandleResult<Self> {
        Ok(ModSignal {
            trace_tau,
            max_z,
            omega,
            z: Tensor::zeros((size, 1), DType::F32, device)?,
            prev_loss: Tensor::zeros((size, 1), DType::F32, device)?,
            mod_signal: Tensor::zeros((size, 1), DType::F32, device)?,
        })
    }
    /// dC/dz_i
    fn calc_mod_signal(&mut self, spikes: &Tensor, lab: &Tensor, dt: f32) -> CandleResult<()> {
        let dz =
            (((dt / self.trace_tau) as f64) * (((self.max_z as f64) * spikes)?.sub(&self.z)?))?;
        self.z = self.z.add(&dz)?;
        let l = calc_loss_ce(&self.z, lab, self.omega)?;
        let dl = l.sub(&self.prev_loss)?;
        self.prev_loss = l;
        let dz_ep = (&dz + 0.00001)?;
        // dl.repeat((self.z.dims()[0], 1))?.div(&dz_ep)
        // basic 2 point derivative
        self.mod_signal = dl.div(&dz_ep)?;
        Ok(())
    }
}

pub trait Layer: Send + Sync {
    /// update internal state and calculated output
    fn step(&mut self, dt: f32) -> CandleResult<()>;

    /// internal activity getter
    #[allow(dead_code)]
    fn activity(&self) -> CandleResult<&Tensor>;

    /// calculates the modulatory goodness signal used for CSDP synapse adjustment
    fn get_mod_signal(&self) -> &Tensor;

    /// output getter
    fn output(&self) -> CandleResult<&Tensor>;

    /// how many neurons in this layer
    fn size(&self) -> usize;

    /// Adds to the input compartment of the layer
    fn add_input(&mut self, input: &Tensor) -> CandleResult<()>;

    /// resets input compartment to zero
    fn reset_input(&mut self) -> CandleResult<()>;

    /// resets internal state fully
    fn reset(&mut self) -> CandleResult<()>;

    /// sets the current sample type for the layer
    fn set_positive_sample(&mut self, label: f32);
}

/// Position of a layer in visualization space
#[derive(Debug, Clone, Copy)]
pub struct LayerPosition {
    pub x: f32,
    pub y: f32,
}

/// Metadata about a layer for visualization and configuration
#[derive(Debug, Clone)]
pub struct LayerMetadata {
    pub id: usize,
    pub name: String,
    pub layer_type: String,
    pub size: usize,
    pub position: LayerPosition,
}
