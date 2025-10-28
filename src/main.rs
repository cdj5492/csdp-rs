use std::error::Error;
use tqdm::Iter;

mod dataset;
mod layer;
mod model;
mod synapse;
mod utils;

use candle_core::{DType, Device, Tensor};
use dataset::xor::XorDataset;
use layer::bernoulli::BernoulliLayer;
use layer::lif::LIFLayer;
use model::{Model, ProcessOutput};
use synapse::hebbian::Hebbian;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    // let device = Device::Cpu;
    let cpu = Device::Cpu;

    let n_epochs = 100;
    let lif_tau = 20.0;
    let thresh = 0.5;
    let thresh_lambda = 1e-2;
    let trace_gamma = 0.05;
    let dt = 0.1;

    // simple XOR dataset
    let ds = XorDataset::new(&device)?;

    let mut model = Model::new(device.clone(), dt);

    model.add_layer(Box::new(BernoulliLayer::new(2, &device)?))?;
    model.add_layer(Box::new(LIFLayer::new(
        256,
        lif_tau,
        thresh,
        thresh_lambda,
        trace_gamma,
        &device,
    )?))?;
    model.add_layer(Box::new(LIFLayer::new(
        256,
        lif_tau,
        thresh,
        thresh_lambda,
        trace_gamma,
        &device,
    )?))?;
    model.add_layer(Box::new(LIFLayer::new(
        2,
        lif_tau,
        thresh,
        thresh_lambda,
        trace_gamma,
        &device,
    )?))?;

    let heb = Hebbian::new(1e-2_f32); // learning rate
    model.add_synapse(0, 1, Box::new(heb.clone()))?;
    model.add_synapse(1, 2, Box::new(heb.clone()))?;
    model.add_synapse(2, 3, Box::new(heb))?;

    // training loop: unsupervised Hebbian run for a few epochs:
    for epoch in (0..n_epochs).tqdm() {
        for (input, _label) in ds.iter() {
            let out = model.process(&input, None, 20, false)?;

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
