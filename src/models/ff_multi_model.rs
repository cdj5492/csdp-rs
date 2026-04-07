use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{
    Linear, Module, Optimizer, VarBuilder, VarMap, linear,
    optim::{AdamW, ParamsAdamW},
};

pub struct FFMultiLayer {
    pub linear: Linear,
    opt: AdamW,
    threshold: f64,
    num_epochs: usize,
    num_classes: usize,
}

impl FFMultiLayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_classes: usize,
        varmap: &mut VarMap,
        device: &Device,
        num_epochs: usize,
    ) -> CandleResult<Self> {
        assert!(
            out_features % num_classes == 0,
            "out_features must be divisible by num_classes"
        );
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
            num_classes,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        let x_direction = x.broadcast_div(&(norm + 1e-4)?)?;
        let out = self.linear.forward(&x_direction)?;
        out.gelu()
    }

    pub fn train(&mut self, x: &Tensor, y: &[usize], layer_idx: usize) -> CandleResult<()> {
        let n_samples = y.len();
        let mini_batch_size = 512.min(n_samples);

        for epoch in 0..self.num_epochs {
            let mut epoch_loss_sum = 0.0f32;
            let mut n_batches = 0;

            let mut offset = 0;
            while offset < n_samples {
                let end = (offset + mini_batch_size).min(n_samples);
                let batch_len = end - offset;

                let x_batch = x.narrow(0, offset, batch_len)?;
                let y_batch = &y[offset..end];

                let out = self.forward(&x_batch)?;
                let chunks = out.chunk(self.num_classes, 1)?;

                let mut total_loss = Tensor::zeros((), DType::F32, x.device())?;

                for (c, chunk) in chunks.into_iter().enumerate() {
                    let g = chunk.sqr()?.mean_keepdim(1)?;

                    let mut pos_mask_vec = Vec::new();
                    let mut neg_mask_vec = Vec::new();
                    for &label in y_batch {
                        if label == c {
                            pos_mask_vec.push(1.0f32);
                            neg_mask_vec.push(0.0f32);
                        } else {
                            pos_mask_vec.push(0.0f32);
                            neg_mask_vec.push(1.0f32);
                        }
                    }
                    let pos_mask = Tensor::from_vec(pos_mask_vec, (batch_len, 1), x.device())?;
                    let neg_mask = Tensor::from_vec(neg_mask_vec, (batch_len, 1), x.device())?;

                    let pos_term = g.affine(-1.0, self.threshold)?;
                    let neg_term = g.affine(1.0, -self.threshold)?;

                    let pos_loss = (pos_term.clamp(-50.0f32, 50.0f32)?.exp()? + 1.0)?.log()?.broadcast_mul(&pos_mask)?;
                    let neg_loss = (neg_term.clamp(-50.0f32, 50.0f32)?.exp()? + 1.0)?.log()?.broadcast_mul(&neg_mask)?;

                    let scale = 1.0 / (self.num_classes as f32 - 1.0);
                    let neg_loss_scaled = (neg_loss * scale as f64)?;

                    total_loss = (total_loss + pos_loss.sum_all()?)?;
                    total_loss = (total_loss + neg_loss_scaled.sum_all()?)?;
                }

                let loss = (total_loss / (batch_len as f64 * 2.0))?;
                self.opt.backward_step(&loss)?;

                epoch_loss_sum += loss.to_vec0::<f32>()?;
                n_batches += 1;
                offset = end;
            }

            if epoch % 10 == 0 || epoch == self.num_epochs - 1 {
                let avg_loss = epoch_loss_sum / n_batches as f32;
                println!("Layer {} - Epoch {} Loss: {:.4}", layer_idx, epoch, avg_loss);
            }
        }

        Ok(())
    }
}

pub struct FFMultiModel {
    pub layers: Vec<FFMultiLayer>,
    pub varmaps: Vec<VarMap>,
    pub num_classes: usize,
}

impl FFMultiModel {
    pub fn new(dims: &[usize], num_classes: usize, device: &Device, num_epochs: usize) -> CandleResult<Self> {
        let mut layers = Vec::new();
        let mut varmaps = Vec::new();
        for i in 0..dims.len() - 1 {
            let mut varmap = VarMap::new();
            let layer = FFMultiLayer::new(dims[i], dims[i + 1], num_classes, &mut varmap, device, num_epochs)?;
            layers.push(layer);
            varmaps.push(varmap);
        }
        Ok(Self { layers, varmaps, num_classes })
    }

    pub fn train(&mut self, x: &Tensor, y: &[usize]) -> CandleResult<()> {
        let mut h = x.clone();
        let n_samples = y.len();
        let infer_batch = 2048.min(n_samples);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            layer.train(&h, y, idx)?;

            // Project all data through this layer in mini-batches for the next layer
            let mut projected_chunks = Vec::new();
            let mut offset = 0;
            while offset < n_samples {
                let end = (offset + infer_batch).min(n_samples);
                let batch_len = end - offset;
                let chunk = h.narrow(0, offset, batch_len)?;
                projected_chunks.push(layer.forward(&chunk)?.detach());
                offset = end;
            }
            h = Tensor::cat(&projected_chunks, 0)?;
        }
        Ok(())
    }

    pub fn predict_scores(&self, inputs: &[Tensor]) -> CandleResult<Tensor> {
        if inputs.is_empty() {
            panic!("predict_scores called with empty inputs");
        }

        let batched_inputs = Tensor::cat(inputs, 0)?;
        let batch_size = batched_inputs.dim(0)?;
        let mut h = batched_inputs;

        let mut total_goodnesses = Tensor::zeros((batch_size, self.num_classes), DType::F32, inputs[0].device())?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
            let chunks = h.chunk(self.num_classes, 1)?;
            let mut layer_goodnesses = Vec::new();
            for chunk in chunks {
                let g = chunk.sqr()?.mean_keepdim(1)?;
                layer_goodnesses.push(g);
            }
            let layer_g_tensor = Tensor::cat(&layer_goodnesses, 1)?;
            total_goodnesses = total_goodnesses.broadcast_add(&layer_g_tensor)?;
        }

        Ok(total_goodnesses)
    }

    pub fn predict(&self, inputs: &[Tensor]) -> CandleResult<Vec<usize>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batched_inputs = Tensor::cat(inputs, 0)?;
        let batch_size = batched_inputs.dim(0)?;
        let mut h = batched_inputs;

        let mut total_goodnesses = Tensor::zeros((batch_size, self.num_classes), DType::F32, inputs[0].device())?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
            let chunks = h.chunk(self.num_classes, 1)?;
            let mut layer_goodnesses = Vec::new();
            for chunk in chunks {
                let g = chunk.sqr()?.mean_keepdim(1)?;
                layer_goodnesses.push(g);
            }
            let layer_g_tensor = Tensor::cat(&layer_goodnesses, 1)?;
            total_goodnesses = total_goodnesses.broadcast_add(&layer_g_tensor)?;
        }

        let best_scores_flat: Vec<f32> = total_goodnesses.flatten_all()?.to_vec1()?;
        let mut best_indices = Vec::new();

        for b in 0..batch_size {
            let start = b * self.num_classes;
            let end = start + self.num_classes;
            let chunk = &best_scores_flat[start..end];
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
}
