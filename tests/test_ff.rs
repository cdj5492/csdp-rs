use candle_core::{Device, Tensor};
use custom_framework::models::ff_model::FFModel;

fn overlay_y_on_x(x: &Tensor, y: &[usize], num_classes: usize) -> candle_core::Result<Tensor> {
    let (batch, features) = x.dims2()?;
    let mut x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let max_val = *x_vec
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&1.0);

    for b in 0..batch {
        for c in 0..num_classes {
            x_vec[b * features + c] = 0.0;
        }
        x_vec[b * features + y[b]] = max_val;
    }

    Tensor::from_vec(x_vec, (batch, features), x.device())
}

#[test]
fn test_ff_logic_learning() {
    let device = Device::Cpu;

    // For XOR logic: 2 inputs. Let's make an input vector of 4 dimensions (2 for one-hot label, 2 for input).
    let input_size = 4;
    let hidden_sizes = vec![64, 32];
    let mut dims = vec![input_size];
    dims.extend(hidden_sizes);
    
    // Explicitly configure 1000 epochs for the singular logic testing batch
    let mut model = FFModel::new(&dims, &device, 1000).unwrap();

    let x_data = vec![
        0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    ]; // shape [4, 4]

    let y_data = vec![0, 1, 1, 0];

    let x = Tensor::from_vec(x_data.clone(), (4, 4), &device).unwrap();

    let x_pos = overlay_y_on_x(&x, &y_data, 2).unwrap();
    let neg_y_data: Vec<usize> = y_data.iter().map(|&y| 1 - y).collect();
    let x_neg = overlay_y_on_x(&x, &neg_y_data, 2).unwrap();

    // Train the model
    model.train(&x_pos, &x_neg).unwrap();

    // Test the model
    let input_c0 = overlay_y_on_x(&x, &vec![0; 4], 2).unwrap();
    let input_c1 = overlay_y_on_x(&x, &vec![1; 4], 2).unwrap();

    // Let's create combinations: one pos, one neg
    let inputs = vec![input_c0.clone(), input_c1.clone()];
    
    // Test inference via chunk_size
    let best_classes = model.predict(&inputs, 2).unwrap();
    
    println!("Best class selected per batch: {:?}", best_classes);
}
