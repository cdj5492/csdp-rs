use std::error::Error;
use tqdm::Iter;

mod dataset;
mod layer;
mod model;
mod synapse;
mod utils;

use candle_core::{Device, DType, Tensor};
use dataset::xor::XorDataset;
use model::{Model, ProcessOutput};
use synapse::hebbian::Hebbian;
use layer::basic::{LayerConfig, Layer};

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    // let device = Device::Cpu;

    let n_epochs = 100;

    // simple XOR dataset
    let ds = XorDataset::new(&device)?;

    let mut model = Model::new(device.clone());

    let l0 = Layer::new(LayerConfig::new(2), &device)?;
    let l1 = Layer::new(LayerConfig::new(256), &device)?;
    let l2 = Layer::new(LayerConfig::new(256), &device)?;
    let l3 = Layer::new(LayerConfig::new(2), &device)?;
    model.add_layer(l0);
    model.add_layer(l1);
    model.add_layer(l2);
    model.add_layer(l3);

    let heb = Hebbian::new(1e-2_f32); // learning rate
    model.add_synapse(0, 1, Box::new(heb.clone()))?;
    model.add_synapse(1, 2, Box::new(heb.clone()))?;
    model.add_synapse(2, 3, Box::new(heb))?;

    // training loop (toy): unsupervised Hebbian run for a few epochs:
    for epoch in (0..n_epochs).tqdm() {
        for (input, _label) in ds.iter() {
            // process for some timesteps, clamping input to first layer's external_input
            let out = model.process(Some(&input), None, 20)?;
            // optionally dump internals at a low frequency
            if epoch % 10 == 0 && out.timestep_activity.len() > 0 {
                // write first timestep activity of layer0 to CSV (example)
                utils::save_tensor_flat_csv("activity_epoch.csv", &out.timestep_activity[0])?;
            }
        }
    }

    println!("Done");
    Ok(())
}
