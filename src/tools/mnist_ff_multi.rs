use candle_core::{DType, Device, Tensor};
use custom_framework::models::ff_multi_model::FFMultiModel;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

const MNIST_BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const DATA_DIR: &str = "data/mnist";

fn download_if_needed(filename: &str) -> std::io::Result<()> {
    let dir = Path::new(DATA_DIR);
    std::fs::create_dir_all(dir)?;
    let path = dir.join(filename);
    if path.exists() {
        return Ok(());
    }
    let url = format!("{}{}", MNIST_BASE_URL, filename);
    log::info!("Downloading {}...", url);
    let resp = std::process::Command::new("curl")
        .args(["-sL", "-o", path.to_str().unwrap(), &url])
        .status()?;
    if !resp.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to download {}", url),
        ));
    }
    Ok(())
}

fn read_mnist_images(filename: &str) -> Vec<Vec<f32>> {
    let path = Path::new(DATA_DIR).join(filename);
    let file = File::open(&path).expect("Failed to open image file");
    let mut decoder = GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).expect("Failed to decompress");

    // MNIST IDX format: magic(4) | count(4) | rows(4) | cols(4) | pixels...
    let count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let pixel_size = rows * cols;

    log::info!(
        "  Loaded {} images of size {}x{} from {}",
        count,
        rows,
        cols,
        filename
    );

    let mut images = Vec::with_capacity(count);
    for i in 0..count {
        let start = 16 + i * pixel_size;
        let pixels: Vec<f32> = buf[start..start + pixel_size]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        images.push(pixels);
    }
    images
}

fn read_mnist_labels(filename: &str) -> Vec<usize> {
    let path = Path::new(DATA_DIR).join(filename);
    let file = File::open(&path).expect("Failed to open label file");
    let mut decoder = GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).expect("Failed to decompress");

    let count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    log::info!("  Loaded {} labels from {}", count, filename);

    buf[8..8 + count].iter().map(|&b| b as usize).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    log::info!("=== MNIST Multi-Class Forward-Forward Benchmark ===\n");

    // Download MNIST data files
    download_if_needed("train-images-idx3-ubyte.gz")?;
    download_if_needed("train-labels-idx1-ubyte.gz")?;
    download_if_needed("t10k-images-idx3-ubyte.gz")?;
    download_if_needed("t10k-labels-idx1-ubyte.gz")?;

    log::info!("\nLoading dataset...");
    let train_images = read_mnist_images("train-images-idx3-ubyte.gz");
    let train_labels = read_mnist_labels("train-labels-idx1-ubyte.gz");
    let test_images = read_mnist_images("t10k-images-idx3-ubyte.gz");
    let test_labels = read_mnist_labels("t10k-labels-idx1-ubyte.gz");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    log::info!("\nUsing device: {:?}", device);

    // Flatten all training images into a single [N, 784] tensor
    let n_train = train_images.len();
    let flat_train: Vec<f32> = train_images.into_iter().flatten().collect();
    let train_tensor = Tensor::from_vec(flat_train, (n_train, 784), &device)?;

    let n_test = test_images.len();
    let flat_test: Vec<f32> = test_images.into_iter().flatten().collect();
    let test_tensor = Tensor::from_vec(flat_test, (n_test, 784), &device)?;

    let input_size = 784;
    let num_classes = 10;
    // 2000 neurons per layer, 200 per class subset
    let hidden_sizes = vec![2000, 2000];
    let epochs_per_layer = 200;

    let mut dims = vec![input_size];
    dims.extend(&hidden_sizes);

    log::info!("\n--- Model Config ---");
    log::info!("  Input:   {}", input_size);
    log::info!("  Hidden:  {:?}", hidden_sizes);
    log::info!("  Classes: {}", num_classes);
    log::info!("  Epochs:  {} per layer", epochs_per_layer);
    log::info!("  Train N: {}", n_train);
    log::info!("  Test N:  {}", n_test);

    let mut model = FFMultiModel::new(&dims, num_classes, &device, epochs_per_layer)?;

    log::info!("\n--- Training ---");
    let start = Instant::now();
    model.train(&train_tensor, &train_labels)?;
    let train_time = start.elapsed();
    log::info!("Training complete in {:.2}s\n", train_time.as_secs_f64());

    log::info!("--- Evaluation ---");
    let eval_start = Instant::now();
    let predictions = model.predict(&[test_tensor])?;
    let eval_time = eval_start.elapsed();

    let mut correct = 0;
    for (pred, &label) in predictions.iter().zip(test_labels.iter()) {
        if *pred == label {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / n_test as f64 * 100.0;

    log::info!("Evaluation complete in {:.3}s", eval_time.as_secs_f64());
    log::info!(
        "\n  Test Accuracy: {} / {} ({:.2}%)\n",
        correct,
        n_test,
        accuracy
    );

    // Per-class breakdown
    let mut class_correct = vec![0usize; num_classes];
    let mut class_total = vec![0usize; num_classes];
    for (pred, &label) in predictions.iter().zip(test_labels.iter()) {
        class_total[label] += 1;
        if *pred == label {
            class_correct[label] += 1;
        }
    }
    log::info!("  Per-class accuracy:");
    for c in 0..num_classes {
        let acc = if class_total[c] > 0 {
            class_correct[c] as f64 / class_total[c] as f64 * 100.0
        } else {
            0.0
        };
        log::info!(
            "    Digit {}: {:4} / {:4} ({:.1}%)",
            c,
            class_correct[c],
            class_total[c],
            acc
        );
    }

    Ok(())
}
