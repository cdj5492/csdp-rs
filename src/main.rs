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
    let mut follower = LeRobot::new(
        "/dev/ttyACM0",
        [-0.0276, -1.6, 1.29, 1.1, 0.254, 0.0],
        [-1.3, -1.6, -1.94, -2.0, -1.5, -0.0122],
        [1.0, 1.7, 1.29, 1.2, 1.5, 1.1],
    )
    .ok();
    let mut leader = LeRobot::new(
        "/dev/ttyACM1",
        [
            0.05982525072754008,
            -0.32366994624387013,
            0.08743690490948142,
            -0.018407769454627854,
            1.6659031356438065,
            -1.0676506283684062,
        ],
        [-1.77, -0.32, -3.0, -3.0, -3.0, -1.07],
        [2.22, 3.0, 0.085, -0.069, 3.0, 0.65],
    )
    .ok();

    if follower.is_none() {
        println!("No physical follower robot connected");
    }
    if leader.is_none() {
        println!("No physical leader robot connected");
    }

    let n_epochs = 1000;
    // let lif_tau = 20.0;
    // let thresh = 0.5;
    // let thresh_lambda = 1e-2;
    // let trace_gamma = 0.05;
    let dt = 0.1;

    // simple XOR dataset
    let ds = XorDataset::new(&device)?;

    let mut model = Model::new(vec![2, 256, 256, 1], &device, dt).unwrap();

    // training loop: unsupervised Hebbian run for a few epochs:
    for epoch in (1..=n_epochs).tqdm() {
        for (input, _label) in ds.iter() {
            let out = model.process(&input, 40, false, &device)?;

            if epoch % 50 == 0 {
                println!(
                    "Epoch {}: Input: {:?}, Output: {}",
                    epoch,
                    input.to_device(&cpu),
                    out.final_output.to_device(&cpu)?
                );
            }
        }
    }

    println!("Done");
    Ok(())
}
