use std::error::Error;
use tqdm::Iter;

mod dataset;
mod layer;
mod model;
mod synapse;
mod utils;
mod robot;

use candle_core::Device;
use dataset::xor::XorDataset;
use model::Model;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    // let device = Device::Cpu;
    let cpu = Device::Cpu;

    let n_epochs = 100;
    // let lif_tau = 20.0;
    // let thresh = 0.5;
    // let thresh_lambda = 1e-2;
    // let trace_gamma = 0.05;
    let dt = 0.1;

    // simple XOR dataset
    let ds = XorDataset::new(&device)?;

    let mut model = Model::new(vec![2, 256, 256, 1], &device, dt).unwrap();

    // training loop: unsupervised Hebbian run for a few epochs:
    for epoch in (0..n_epochs).tqdm() {
        for (input, _label) in ds.iter() {
            let out = model.process(&input, 40, false, &device)?;

            if epoch % 50 == 0 {
                println!(
                    "Epoch {}: Input: {:?}, Output: {:?}",
                    epoch,
                    input.to_device(&cpu),
                    out.final_output.to_device(&cpu)
                );
            }
        }
    }

    println!("Done");
    Ok(())
}
