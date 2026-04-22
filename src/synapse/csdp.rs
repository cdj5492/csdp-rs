use crate::layer::Layer;

use super::{SynapseOps, WeightStats};
use candle_core::{Result as CandleResult, Tensor};

#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct CSDP {
    pub weights: Tensor,
    /// biases are kept in a separate tensor
    pub biases: Tensor,
}

impl CSDP {
    pub fn new(
        pre_size: usize,
        post_size: usize,
        device: &candle_core::Device,
    ) -> CandleResult<Self> {
        // Scale weights symmetrically to Uniform(-w_bound, w_bound). Symmetrical weights allow SNN
        // neurons to carve out distinct, orthogonal decision boundaries within dense continuous
        // input spaces rather than grouping uniformly into the all-positive quadrant.
        let w_bound = 2.0f32 / (pre_size as f32).sqrt();
        let weights = Tensor::rand(-w_bound, w_bound, (post_size, pre_size), device)?;
        let biases = Tensor::zeros((post_size, 1), candle_core::DType::F32, device)?;
        Ok(Self { weights, biases })
    }
}

impl SynapseOps for CSDP {
    fn forward(&self, pre: &Tensor) -> CandleResult<Tensor> {
        let w_pre = self.weights.matmul(pre)?;
        let out = w_pre.broadcast_add(&self.biases)?;
        Ok(out)
    }

    fn update_weights(
        &mut self,
        pre_activity: &Tensor,
        post_layer: &mut Box<dyn Layer>,
        _dt: f32,
    ) -> CandleResult<()> {
        let pre = pre_activity;
        let batch_size = pre.dims().get(1).copied().unwrap_or(1);

        let mod_signal = post_layer.get_mod_signal();

        // synaptic decay factor
        // TODO: put at top level
        let lambda_d = 0.00005;

        // outer product (should be same shape as weight matrix)
        let dw = mod_signal.matmul(&pre.t()?)?;
        let dw_avg = dw.affine(1.0 / (batch_size as f64), 0.0)?;

        let delta = dw_avg.sub(&(self.weights.affine(lambda_d, 0.0)?))?;

        self.weights = self.weights.add(&delta)?;

        // biases are treated as connections to a neuron that is always firing every timestep
        let db_avg = mod_signal
            .sum_keepdim(1)?
            .affine(1.0 / (batch_size as f64), 0.0)?;
        self.biases = self.biases.add(&db_avg)?;

        Ok(())
    }

    fn weight_stats(&self) -> CandleResult<WeightStats> {
        let weights_vec = self.weights.flatten_all()?.to_vec1::<f32>()?;
        let num_weights = weights_vec.len();

        if num_weights == 0 {
            return Ok(WeightStats {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                num_weights: 0,
            });
        }

        let sum: f32 = weights_vec.iter().sum();
        let mean = sum / num_weights as f32;

        let variance: f32 =
            weights_vec.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / num_weights as f32;
        let std = variance.sqrt();

        let min = weights_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights_vec
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        Ok(WeightStats {
            mean,
            std,
            min,
            max,
            num_weights,
        })
    }

    fn get_state(&self) -> CandleResult<std::collections::HashMap<String, Tensor>> {
        let mut state = std::collections::HashMap::new();
        state.insert("weights".to_string(), self.weights.clone());
        state.insert("biases".to_string(), self.biases.clone());
        Ok(state)
    }

    fn set_state(&mut self, state: &std::collections::HashMap<String, Tensor>) -> CandleResult<()> {
        if let Some(w) = state.get("weights") {
            self.weights = w.clone();
        } else {
            return Err(candle_core::Error::Msg(
                "weights tensor missing from state".to_string(),
            ));
        }

        if let Some(b) = state.get("biases") {
            self.biases = b.clone();
        } else {
            return Err(candle_core::Error::Msg(
                "biases tensor missing from state".to_string(),
            ));
        }

        Ok(())
    }
}
