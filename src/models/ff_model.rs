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
    ) -> CandleResult<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let linear = linear(in_features, out_features, vb.pp("linear"))?;

        let params = ParamsAdamW {
            lr: 0.03,
            ..Default::default()
        };
        let opt = AdamW::new(varmap.all_vars(), params)?;

        Ok(Self {
            linear,
            opt,
            threshold: 2.0,
            num_epochs: 1000,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        let x_direction = x.broadcast_div(&(norm + 1e-4)?)?;
        let out = self.linear.forward(&x_direction)?;
        out.relu()
    }

    pub fn train(&mut self, x_pos: &Tensor, x_neg: &Tensor, layer_idx: usize) -> CandleResult<(Tensor, Tensor)> {
        for epoch in 0..self.num_epochs {
            let out_pos = self.forward(x_pos)?;
            let g_pos = out_pos.sqr()?.mean_keepdim(1)?;
            
            let out_neg = self.forward(x_neg)?;
            let g_neg = out_neg.sqr()?.mean_keepdim(1)?;
            
            let pos_term = g_pos.affine(-1.0, self.threshold)?; 
            let neg_term = g_neg.affine(1.0, -self.threshold)?;
            
            let concat = Tensor::cat(&[&pos_term, &neg_term], 0)?;
            let loss = (concat.exp()? + 1.0)?.log()?.mean_all()?;
            
            self.opt.backward_step(&loss)?;

            if epoch % 200 == 0 || epoch == self.num_epochs - 1 {
                let current_loss = loss.to_vec0::<f32>()?;
                println!("Layer {} - Epoch {} Loss: {:.4}", layer_idx, epoch, current_loss);
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
    pub fn new(dims: &[usize], device: &Device) -> CandleResult<Self> {
        let mut layers = Vec::new();
        let mut varmaps = Vec::new();
        for i in 0..dims.len() - 1 {
            let mut varmap = VarMap::new();
            let layer = FFLayer::new(dims[i], dims[i + 1], &mut varmap, device)?;
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

    pub fn predict(&self, inputs: &[Tensor]) -> CandleResult<Tensor> {
        let mut goodness_per_label = Vec::new();
        for x in inputs {
            let mut h = x.clone();
            let mut total_goodness = Tensor::zeros((x.dim(0)?, 1), DType::F32, x.device())?;

            for layer in self.layers.iter() {
                h = layer.forward(&h)?;
                let goodness = h.sqr()?.mean_keepdim(1)?;
                total_goodness = total_goodness.add(&goodness)?;
            }
            goodness_per_label.push(total_goodness);
        }

        let stacked = Tensor::stack(&goodness_per_label, 1)?.squeeze(2)?;
        stacked.argmax(1)
    }
}
