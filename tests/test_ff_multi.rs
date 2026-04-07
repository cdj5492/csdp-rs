use candle_core::{Device, Tensor};
use custom_framework::models::ff_multi_model::FFMultiModel;

#[test]
fn test_ff_multi_logic_learning() {
    let device = Device::Cpu;
    
    // Test mapping 4 classes (0 to 3) from 2D input
    let input_size = 2;
    // Ensure hidden sizes are multiples of 4 (number of classes)
    let hidden_sizes = vec![64, 32];
    let num_classes = 4;
    
    let mut dims = vec![input_size];
    dims.extend(hidden_sizes);
    
    let mut model = FFMultiModel::new(&dims, num_classes, &device, 1000).unwrap();

    // Inputs: [0,0], [0,1], [1,0], [1,1]
    let x_data = vec![
        0.0f32, 0.0, 
        0.0, 1.0, 
        1.0, 0.0, 
        1.0, 1.0,
    ]; 
    let y_data = vec![0, 1, 2, 3];
    let x = Tensor::from_vec(x_data, (4, 2), &device).unwrap();

    println!("Starting training for multi-class forward-forward...");
    model.train(&x, &y_data).unwrap();

    println!("Evaluating predictions...");
    let best_classes = model.predict(&[x]).unwrap();
    
    println!("Expected: {:?}", y_data);
    println!("Predicted: {:?}", best_classes);
    
    // Check if the accuracy is 100%
    assert_eq!(best_classes, y_data);
}
