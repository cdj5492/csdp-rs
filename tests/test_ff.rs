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
    let mut model = FFModel::new(&[4, 64, 32], &device).unwrap();

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

    let predictions = model.predict(&[input_c0, input_c1]).unwrap();
    let pred_vec = predictions.to_vec1::<u32>().unwrap();

    println!("Predictions: {:?}", pred_vec);
    assert_eq!(pred_vec, vec![0, 1, 1, 0]);
}
