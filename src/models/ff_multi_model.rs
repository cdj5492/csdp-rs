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
    /// Entropy regularization coefficient (RLGymPPO-style).
    /// When > 0, a -ent_coef * H(π) term is added to train_binary's loss,
    /// encouraging the action distribution to stay spread out.
    pub ent_coef: f64,
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
            out_features.is_multiple_of(num_classes),
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
            ent_coef: 0.0,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        let x_direction = x.broadcast_div(&(norm + 1e-4)?)?;
        let out = self.linear.forward(&x_direction)?;
        out.gelu()
    }

    /// Reshape `(batch, nc * chunk_size)` → `(batch, nc, chunk_size)` and compute
    /// mean(x²) over the chunk dimension, yielding `(batch, nc)` goodness scores.
    /// Replaces the old `chunk()` + per-class loop — all classes computed in one GPU op.
    fn goodness_all(out: &Tensor, nc: usize) -> CandleResult<Tensor> {
        let batch = out.dim(0)?;
        let chunk_size = out.dim(1)? / nc;
        out.reshape((batch, nc, chunk_size))?.sqr()?.mean(2usize)
    }

    pub fn train(&mut self, x: &Tensor, y: &[usize], layer_idx: usize) -> CandleResult<()> {
        let n_samples = y.len();
        let mini_batch_size = 512.min(n_samples);
        let nc = self.num_classes;
        let neg_scale = 1.0 / (nc as f64 - 1.0);

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
                // All class goodnesses at once: (batch, nc)
                let g = Self::goodness_all(&out, nc)?;

                // Build one-hot pos_mask + complement neg_mask in a single O(batch_len) loop.
                // Old path: O(batch_len × nc) loop + 2×nc GPU tensor uploads per mini-batch.
                // New path: O(batch_len) loop + 2 GPU tensor uploads per mini-batch.
                let mut pos_flat = vec![0.0f32; batch_len * nc];
                let mut neg_flat = vec![1.0f32; batch_len * nc];
                for (i, &label) in y_batch.iter().enumerate() {
                    pos_flat[i * nc + label] = 1.0;
                    neg_flat[i * nc + label] = 0.0;
                }
                let pos_mask = Tensor::from_vec(pos_flat, (batch_len, nc), x.device())?;
                let neg_mask = Tensor::from_vec(neg_flat, (batch_len, nc), x.device())?;

                let pos_term = g.affine(-1.0, self.threshold)?;
                let neg_term = g.affine(1.0, -self.threshold)?;

                let pos_loss = (pos_term.clamp(-50.0f32, 50.0f32)?.exp()? + 1.0)?
                    .log()?.broadcast_mul(&pos_mask)?.sum_all()?;
                let neg_loss = ((neg_term.clamp(-50.0f32, 50.0f32)?.exp()? + 1.0)?
                    .log()?.broadcast_mul(&neg_mask)?.sum_all()? * neg_scale)?;

                let loss = ((pos_loss + neg_loss)? / (batch_len as f64 * 2.0))?;
                self.opt.backward_step(&loss)?;

                epoch_loss_sum += loss.to_vec0::<f32>()?;
                n_batches += 1;
                offset = end;
            }

            if epoch % 10 == 0 || epoch == self.num_epochs - 1 {
                log::info!(
                    "Layer {} - Epoch {} Loss: {:.4}",
                    layer_idx, epoch,
                    epoch_loss_sum / n_batches as f32
                );
            }
        }
        Ok(())
    }

    /// Per-action binary training for policy learning.
    ///
    /// Each action chunk is an independent binary classifier — positive advantage
    /// pushes goodness above threshold, negative pushes it below. No other chunk is
    /// touched for a given sample, eliminating winner-take-all cross-action suppression.
    ///
    /// If `ent_coef > 0`, a Shannon entropy bonus is added (borrowed from RLGymPPO)
    /// to keep the action distribution from collapsing even without recent negative updates.
    pub fn train_binary(
        &mut self,
        x: &Tensor,
        actions: &[usize],
        advantages: &[f32],
        layer_idx: usize,
    ) -> CandleResult<()> {
        let n_samples = actions.len();
        let mini_batch_size = 512.min(n_samples);
        let nc = self.num_classes;

        for epoch in 0..self.num_epochs {
            let mut epoch_loss_sum = 0.0f32;
            let mut n_batches = 0;
            let mut offset = 0;

            while offset < n_samples {
                let end = (offset + mini_batch_size).min(n_samples);
                let batch_len = end - offset;

                let x_batch = x.narrow(0, offset, batch_len)?;
                let acts = &actions[offset..end];
                let advs = &advantages[offset..end];

                let out = self.forward(&x_batch)?;
                // (batch, nc) — all class goodnesses computed in one GPU op
                let g = Self::goodness_all(&out, nc)?;

                // Sparse masks: one nonzero entry per row at most (the taken action).
                // Old: O(batch_len × nc) iterations + 2×nc tensor uploads per mini-batch.
                // New: O(batch_len) iterations + 2 tensor uploads per mini-batch.
                let mut pos_flat = vec![0.0f32; batch_len * nc];
                let mut neg_flat = vec![0.0f32; batch_len * nc];
                let mut has_binary = false;
                for (i, (&act, &adv)) in acts.iter().zip(advs.iter()).enumerate() {
                    if adv > 0.0 {
                        pos_flat[i * nc + act] = 1.0;
                        has_binary = true;
                    } else if adv < 0.0 {
                        neg_flat[i * nc + act] = 1.0;
                        has_binary = true;
                    }
                }

                if !has_binary && self.ent_coef == 0.0 {
                    offset = end;
                    continue;
                }

                let pos_mask = Tensor::from_vec(pos_flat, (batch_len, nc), x.device())?;
                let neg_mask = Tensor::from_vec(neg_flat, (batch_len, nc), x.device())?;

                let pos_term = g.affine(-1.0, self.threshold)?;
                let neg_term = g.affine(1.0, -self.threshold)?;

                let pos_loss = (pos_term.clamp(-50.0f32, 50.0f32)?.exp()? + 1.0)?
                    .log()?.broadcast_mul(&pos_mask)?.sum_all()?;
                let neg_loss = (neg_term.clamp(-50.0f32, 50.0f32)?.exp()? + 1.0)?
                    .log()?.broadcast_mul(&neg_mask)?.sum_all()?;

                let main_loss = ((pos_loss + neg_loss)? / batch_len as f64)?;

                let loss = if self.ent_coef > 0.0 {
                    // Shannon entropy of softmax(g) per sample, averaged over batch.
                    // Subtract from loss so the optimizer maximizes it (RLGymPPO pattern).
                    // Mean-shift g for numerical stability before softmax.
                    let mean_g = g.mean_keepdim(1)?;            // (batch, 1)
                    let g_s = g.broadcast_sub(&mean_g)?;        // (batch, nc)
                    let exp_g = g_s.exp()?;
                    let exp_sum = exp_g.sum_keepdim(1)?;        // (batch, 1)
                    let probs = exp_g.broadcast_div(&exp_sum)?; // (batch, nc)
                    let log_probs = probs.clamp(1e-8f32, 1.0f32)?.log()?;
                    let entropy = (probs.broadcast_mul(&log_probs)?
                        .sum_keepdim(1)?.neg()?.sum_all()? / batch_len as f64)?;
                    (main_loss - (entropy * self.ent_coef)?)?
                } else {
                    main_loss
                };

                self.opt.backward_step(&loss)?;
                epoch_loss_sum += loss.to_vec0::<f32>()?;
                n_batches += 1;
                offset = end;
            }

            if (epoch % 10 == 0 || epoch == self.num_epochs - 1) && n_batches > 0 {
                log::info!(
                    "Layer {} (binary) - Epoch {} Loss: {:.4}",
                    layer_idx, epoch,
                    epoch_loss_sum / n_batches as f32
                );
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
    pub fn new(
        dims: &[usize],
        num_classes: usize,
        device: &Device,
        num_epochs: usize,
    ) -> CandleResult<Self> {
        let mut layers = Vec::new();
        let mut varmaps = Vec::new();
        for i in 0..dims.len() - 1 {
            let mut varmap = VarMap::new();
            let layer = FFMultiLayer::new(
                dims[i],
                dims[i + 1],
                num_classes,
                &mut varmap,
                device,
                num_epochs,
            )?;
            layers.push(layer);
            varmaps.push(varmap);
        }
        Ok(Self {
            layers,
            varmaps,
            num_classes,
        })
    }

    pub fn train(&mut self, x: &Tensor, y: &[usize]) -> CandleResult<()> {
        let mut h = x.clone();
        let n_samples = y.len();
        let infer_batch = 2048.min(n_samples);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            layer.train(&h, y, idx)?;

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

    pub fn train_policy(
        &mut self,
        x: &Tensor,
        actions: &[usize],
        advantages: &[f32],
    ) -> CandleResult<()> {
        let mut h = x.clone();
        let n_samples = actions.len();
        let infer_batch = 2048.min(n_samples);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            layer.train_binary(&h, actions, advantages, idx)?;

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

    /// Inference: returns accumulated goodness scores `(batch, num_classes)`.
    /// Vectorized: replaces per-class chunk() loop with a single reshape + mean op per layer.
    pub fn predict_scores(&self, inputs: &[Tensor]) -> CandleResult<Tensor> {
        if inputs.is_empty() {
            panic!("predict_scores called with empty inputs");
        }

        let batched_inputs = Tensor::cat(inputs, 0)?;
        let batch_size = batched_inputs.dim(0)?;
        let nc = self.num_classes;
        let mut h = batched_inputs;

        let mut total_goodnesses =
            Tensor::zeros((batch_size, nc), DType::F32, inputs[0].device())?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
            // (batch, nc*chunk) → (batch, nc, chunk) → mean² → (batch, nc)
            let chunk_size = h.dim(1)? / nc;
            let layer_g = h.reshape((batch_size, nc, chunk_size))?.sqr()?.mean(2usize)?;
            total_goodnesses = (total_goodnesses + layer_g)?;
        }

        Ok(total_goodnesses)
    }

    pub fn predict(&self, inputs: &[Tensor]) -> CandleResult<Vec<usize>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let scores = self.predict_scores(inputs)?; // (batch, nc)
        let batch_size = scores.dim(0)?;
        let nc = self.num_classes;
        let flat: Vec<f32> = scores.flatten_all()?.to_vec1()?;

        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let start = b * nc;
            let chunk = &flat[start..start + nc];
            let best_idx = chunk
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            result.push(best_idx);
        }
        Ok(result)
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, dir: P) -> CandleResult<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| candle_core::Error::Msg(format!("mkdir: {}", e)))?;
        for (i, vm) in self.varmaps.iter().enumerate() {
            let path = dir.join(format!("layer_{}.safetensors", i));
            vm.save(&path)?;
        }
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(&mut self, dir: P) -> CandleResult<()> {
        let dir = dir.as_ref();
        for (i, vm) in self.varmaps.iter_mut().enumerate() {
            let path = dir.join(format!("layer_{}.safetensors", i));
            if path.exists() {
                vm.load(&path)?;
            } else {
                return Err(candle_core::Error::Msg(format!(
                    "Missing weight file: {:?}",
                    path
                )));
            }
        }
        Ok(())
    }
}
