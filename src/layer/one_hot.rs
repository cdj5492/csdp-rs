use crate::layer::Layer;
use candle_core::{DType, Device, Result as CandleResult, Tensor};

pub struct OneHotLayer {
    /// current probabilities (for compatibility)
    probs: Tensor,
    /// output spikes
    spikes: Tensor,
    /// accumulated inputs
    inputs: Tensor,
    /// total number of neurons (sum of all bounds)
    size: usize,
    /// max value for each variable
    bounds: Vec<usize>,
    /// start index for each variable in the output vector
    offsets: Vec<usize>,
    current_label: f32,
    dummy_mod_signal: Tensor,
}

impl OneHotLayer {
    pub fn new(bounds: Vec<usize>, device: &Device) -> CandleResult<Self> {
        println!("constructing onehot layer with bounds: {:?}", bounds);
        let mut offsets = Vec::with_capacity(bounds.len());
        let mut total_size = 0;
        for &bound in &bounds {
            offsets.push(total_size);
            total_size += bound;
        }

        let probs = Tensor::zeros((total_size, 1), DType::F32, device)?;
        let spikes = Tensor::zeros((total_size, 1), DType::F32, device)?;
        let inputs = Tensor::zeros((total_size, 1), DType::F32, device)?;
        let dummy_mod_signal = Tensor::zeros((total_size, 1), DType::F32, device)?;

        Ok(Self {
            probs,
            spikes,
            inputs,
            size: total_size,
            bounds,
            offsets,
            current_label: 1.0,
            dummy_mod_signal,
        })
    }
}

impl Layer for OneHotLayer {
    fn step(&mut self, _dt: f32) -> CandleResult<()> {
        // One-hot layer is deterministic: inputs are mapped to spikes directly
        self.spikes = self.inputs.clone();
        self.probs = self.inputs.clone();
        Ok(())
    }

    fn activity(&self) -> CandleResult<&Tensor> {
        Ok(&self.probs)
    }

    fn get_mod_signal(&self) -> &Tensor {
        &self.dummy_mod_signal
    }

    fn output(&self) -> CandleResult<&Tensor> {
        Ok(&self.spikes)
    }

    fn size(&self) -> usize {
        self.size
    }

    fn add_input(&mut self, input: &Tensor) -> CandleResult<()> {
        // If input size matches number of variables, perform expansion
        if input.dims().len() >= 1 && input.dims()[0] == self.bounds.len() {
            let data = input.flatten_all()?.to_vec1::<f32>()?;
            let mut expanded = vec![0.0f32; self.size];
            for (i, &val) in data.iter().enumerate() {
                let idx = val.round() as usize;
                if idx < self.bounds[i] {
                    expanded[self.offsets[i] + idx] = 1.0;
                }
            }
            let expanded_tensor = Tensor::from_vec(expanded, (self.size, 1), input.device())?;
            self.inputs = self.inputs.add(&expanded_tensor)?;
        } else {
            // standard addition if already expanded or from synapses
            self.inputs = self.inputs.add(input)?;
        }
        Ok(())
    }

    fn reset_input(&mut self) -> CandleResult<()> {
        self.inputs = Tensor::zeros((self.size, 1), DType::F32, self.probs.device())?;
        Ok(())
    }

    fn reset(&mut self) -> CandleResult<()> {
        self.inputs = Tensor::zeros((self.size, 1), DType::F32, self.probs.device())?;
        self.probs = Tensor::zeros((self.size, 1), DType::F32, self.probs.device())?;
        self.spikes = Tensor::zeros((self.size, 1), DType::F32, self.probs.device())?;
        Ok(())
    }

    fn set_positive_sample(&mut self, label: f32) {
        self.current_label = label;
    }

    fn set_reward(&mut self, _reward: f32) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_one_hot_expansion_counts() -> CandleResult<()> {
        let device = Device::Cpu;
        let bounds = vec![10, 20, 5]; // 3 variables, total size 35
        let mut layer = OneHotLayer::new(bounds, &device)?;

        // Input: 3 values
        let input = Tensor::from_vec(vec![5.0f32, 15.0, 2.0], (3, 1), &device)?;
        layer.add_input(&input)?;
        layer.step(0.1)?;

        let output = layer.output()?.flatten_all()?.to_vec1::<f32>()?;

        // Verify total size
        assert_eq!(output.len(), 35);

        // Verify exactly 3 non-zero outputs
        let non_zero_count = output.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(non_zero_count, 3);

        // Verify specific positions
        // Var 0 (0-9): index 5
        // Var 1 (10-29): index 10 + 15 = 25
        // Var 2 (30-34): index 30 + 2 = 32
        assert_eq!(output[5], 1.0);
        assert_eq!(output[25], 1.0);
        assert_eq!(output[32], 1.0);

        Ok(())
    }

    #[test]
    fn test_one_hot_out_of_bounds() -> CandleResult<()> {
        let device = Device::Cpu;
        let bounds = vec![5, 5]; // 2 variables, total size 10
        let mut layer = OneHotLayer::new(bounds, &device)?;

        // Input: 1 valid, 1 out of bounds
        let input = Tensor::from_vec(vec![2.0f32, 10.0], (2, 1), &device)?;
        layer.add_input(&input)?;
        layer.step(0.1)?;

        let output = layer.output()?.flatten_all()?.to_vec1::<f32>()?;

        // Only 1 non-zero output because index 10 is >= bound 5
        let non_zero_count = output.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(non_zero_count, 1);
        assert_eq!(output[2], 1.0);

        Ok(())
    }

    #[test]
    fn test_one_hot_reset() -> CandleResult<()> {
        let device = Device::Cpu;
        let bounds = vec![5];
        let mut layer = OneHotLayer::new(bounds, &device)?;

        layer.add_input(&Tensor::from_vec(vec![1.0f32], (1, 1), &device)?)?;
        layer.step(0.1)?;
        assert_eq!(
            layer
                .output()?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .sum::<f32>(),
            1.0
        );

        layer.reset_input()?;
        layer.step(0.1)?;
        assert_eq!(
            layer
                .output()?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .sum::<f32>(),
            0.0
        );

        Ok(())
    }
}
