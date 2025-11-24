use std::error::Error;
use tqdm::Iter;

mod dataset;
mod layer;
mod model;
mod robot;
mod synapse;
mod utils;

use candle_core::Device;
use dataset::xor::XorDataset;
use model::Model;

use crate::robot::real_lerobot::LeRobot;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    // let device = Device::Cpu;
    let cpu = Device::Cpu;

    // robot stuff
    let mut _follower = LeRobot::new(
        "/dev/ttyACM0",
        [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
        [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
        [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )
    .ok();
    let mut _leader = LeRobot::new(
        "/dev/ttyACM1",
        [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
        [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
        [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )
    .ok();

    let n_epochs = 2000;
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
