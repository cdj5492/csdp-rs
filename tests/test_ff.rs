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
    let input_size = 4;
    let hidden_sizes = vec![64, 32];
    let mut dims = vec![input_size];
    dims.extend(hidden_sizes);

    let mut model = FFModel::new(&dims, &device, 1000).unwrap();

    let x_data = vec![
        0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    ];
    let y_data = vec![0, 1, 1, 0];
    let x = Tensor::from_vec(x_data, (4, 4), &device).unwrap();

    let x_pos = overlay_y_on_x(&x, &y_data, 2).unwrap();
    let neg_y_data: Vec<usize> = y_data.iter().map(|&y| 1 - y).collect();
    let x_neg = overlay_y_on_x(&x, &neg_y_data, 2).unwrap();

    model.train(&x_pos, &x_neg).unwrap();

    let mut prediction_inputs = Vec::new();
    for i in 0..4 {
        let sample = x.narrow(0, i, 1).unwrap();
        for label in 0..2 {
            prediction_inputs.push(overlay_y_on_x(&sample, &[label], 2).unwrap());
        }
    }

    let best_classes = model.predict(&prediction_inputs, 2).unwrap();

    println!("Predicted: {:?}", best_classes);
    assert_eq!(best_classes, y_data);
}
