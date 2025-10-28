use candle_core::{Result as CandleResult, Tensor};
use std::fs::File;
use std::io::Write;

pub fn save_tensor_flat_csv(path: &str, t: &Tensor) -> CandleResult<()> {
    // flatten and write a single-column CSV of floats
    let flat = t.flatten_all()?;
    // to_vec1::<f32>() expects f32; fallback to f64 if needed.
    let v = flat.to_vec1::<f32>()?;
    let mut w = File::create(path)?;
    for x in v {
        writeln!(w, "{}", x)?;
    }
    Ok(())
}
