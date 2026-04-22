use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{
    Linear, Module, Optimizer, VarBuilder, VarMap, linear,
    optim::{AdamW, ParamsAdamW},
};

pub struct FFLayer {
    pub linear: Linear,
    opt: AdamW,
    threshold: f64,
    num_epochs: usize,
}

impl FFLayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        varmap: &mut VarMap,
        device: &Device,
        num_epochs: usize,
    ) -> CandleResult<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let linear = linear(in_features, out_features, vb.pp("linear"))?;

        let params = ParamsAdamW {
            lr: 0.004,
            ..Default::default()
        };
        let opt = AdamW::new(varmap.all_vars(), params)?;

        Ok(Self {
            linear,
            opt,
            threshold: 2.0,
            num_epochs,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        let x_direction = x.broadcast_div(&(norm + 1e-4)?)?;
        let out = self.linear.forward(&x_direction)?;
        out.gelu()
    }

    pub fn train(
        &mut self,
        x_pos: &Tensor,
        x_neg: &Tensor,
        layer_idx: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        for epoch in 0..self.num_epochs {
            let out_pos = self.forward(x_pos)?;
            let g_pos = out_pos.sqr()?.mean_keepdim(1)?;

            let out_neg = self.forward(x_neg)?;
            let g_neg = out_neg.sqr()?.mean_keepdim(1)?;

            let pos_term = g_pos.affine(-1.0, self.threshold)?;
            let neg_term = g_neg.affine(1.0, -self.threshold)?;

            let concat = Tensor::cat(&[&pos_term, &neg_term], 0)?;
            let concat_safe = concat.clamp(-50.0f32, 50.0f32)?; // prevent loss explosion
            let loss = (concat_safe.exp()? + 1.0)?.log()?.mean_all()?;

            self.opt.backward_step(&loss)?;

            if epoch % 200 == 0 || epoch == self.num_epochs - 1 {
                let current_loss = loss.to_vec0::<f32>()?;
                log::info!(
                    "Layer {} - Epoch {} Loss: {:.4}",
                    layer_idx,
                    epoch,
                    current_loss
                );
            }
        }

        Ok((self.forward(x_pos)?.detach(), self.forward(x_neg)?.detach()))
    }
}

pub struct FFModel {
    pub layers: Vec<FFLayer>,
    pub varmaps: Vec<VarMap>,
}

impl FFModel {
    pub fn new(dims: &[usize], device: &Device, num_epochs: usize) -> CandleResult<Self> {
        let mut layers = Vec::new();
        let mut varmaps = Vec::new();
        for i in 0..dims.len() - 1 {
            let mut varmap = VarMap::new();
            let layer = FFLayer::new(dims[i], dims[i + 1], &mut varmap, device, num_epochs)?;
            layers.push(layer);
            varmaps.push(varmap);
        }
        Ok(Self { layers, varmaps })
    }

    pub fn train(&mut self, x_pos: &Tensor, x_neg: &Tensor) -> CandleResult<()> {
        let mut h_pos = x_pos.clone();
        let mut h_neg = x_neg.clone();

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let (next_pos, next_neg) = layer.train(&h_pos, &h_neg, idx)?;
            h_pos = next_pos;
            h_neg = next_neg;
        }
        Ok(())
    }

    pub fn predict(&self, inputs: &[Tensor], chunk_size: usize) -> CandleResult<Vec<usize>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batched_inputs = Tensor::cat(inputs, 0)?;
        let mut h = batched_inputs;
        let mut total_goodnesses =
            Tensor::zeros((inputs.len(),), candle_core::DType::F32, inputs[0].device())?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
            let goodness = h.sqr()?.mean_keepdim(1)?.squeeze(1)?;
            total_goodnesses = total_goodnesses.broadcast_add(&goodness)?;
        }

        let best_scores: Vec<f32> = total_goodnesses.to_vec1()?;
        // log::info!("Predict scores: {:?}", best_scores); // Muted print for vectorization scale

        let mut best_indices = Vec::new();
        for chunk in best_scores.chunks(chunk_size) {
            let (mut best_idx, mut max_score) = (0, f32::MIN);
            for (i, &score) in chunk.iter().enumerate() {
                if score > max_score {
                    max_score = score;
                    best_idx = i;
                }
            }
            best_indices.push(best_idx);
        }

        Ok(best_indices)
    }

    pub fn predict_scores(&self, inputs: &[Tensor]) -> CandleResult<Vec<f32>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batched_inputs = Tensor::cat(inputs, 0)?;
        let mut h = batched_inputs;
        let mut total_goodnesses =
            Tensor::zeros((inputs.len(),), candle_core::DType::F32, inputs[0].device())?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
            let goodness = h.sqr()?.mean_keepdim(1)?.squeeze(1)?;
            total_goodnesses = total_goodnesses.broadcast_add(&goodness)?;
        }

        total_goodnesses.to_vec1()
    }
}
