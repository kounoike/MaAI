use anyhow::{Context, Result};
use clap::Parser;
use hound::WavReader;
use ndarray::Array3;
use ort::session::Session;
use ort::value::Value;
use std::path::PathBuf;

/// VAP-BC (Backchannel) inference using ONNX Runtime
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the ONNX model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to the first WAV file (user/speaker 1)
    #[arg(short = '1', long)]
    wav1: PathBuf,

    /// Path to the second WAV file (system/speaker 2)
    #[arg(short = '2', long)]
    wav2: PathBuf,

    /// Show detailed frame-by-frame output
    #[arg(short, long)]
    verbose: bool,
}

fn load_wav_as_f32(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();

    // Ensure 16kHz mono (VAP-BC expects 16kHz)
    if spec.sample_rate != 16000 {
        eprintln!(
            "Warning: Expected 16kHz but got {}Hz for {}. Results may be inaccurate.",
            spec.sample_rate,
            path.display()
        );
    }

    if spec.channels != 1 {
        eprintln!(
            "Warning: Expected mono (1 channel) but got {} channels for {}. Using first channel only.",
            spec.channels,
            path.display()
        );
    }

    // Read samples and convert to f32 normalized to [-1.0, 1.0]
    let samples: Result<Vec<f32>> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i32 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .enumerate()
                .filter_map(|(i, s)| {
                    // Take only first channel if multi-channel
                    if i % spec.channels as usize == 0 {
                        Some(s.map(|v| v as f32 / max_val))
                    } else {
                        None
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("Failed to read sample: {}", e))
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .enumerate()
            .filter_map(|(i, s)| {
                if i % spec.channels as usize == 0 {
                    Some(s)
                } else {
                    None
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Failed to read sample: {}", e)),
    };

    samples
}

fn run_inference(session: &mut Session, wav1: Vec<f32>, wav2: Vec<f32>) -> Result<Array3<f32>> {
    // Ensure both inputs have the same length (pad with zeros if needed)
    // Model expects fixed length (determined at export time)
    // For 10Hz frame rate and 20s context: 16000 * 1.2 = 19200 samples
    let expected_len = 19200;

    let max_len = wav1.len().max(wav2.len());

    // Truncate or pad to expected length
    let target_len = if max_len > expected_len {
        eprintln!("Warning: Input audio is longer than model expects ({} > {}). Truncating to first {:.2}s",
                 max_len, expected_len, expected_len as f32 / 16000.0);
        expected_len
    } else {
        max_len
    };

    let mut wav1_padded = wav1;
    let mut wav2_padded = wav2;

    // Truncate if too long
    wav1_padded.truncate(target_len);
    wav2_padded.truncate(target_len);

    // Pad if too short
    if wav1_padded.len() < target_len {
        wav1_padded.resize(target_len, 0.0);
    }
    if wav2_padded.len() < target_len {
        wav2_padded.resize(target_len, 0.0);
    }

    // Create input tensors with shape [1, 1, target_len]
    // B=1 (batch), C=1 (channel), T=samples
    let input1 = Array3::from_shape_vec((1, 1, target_len), wav1_padded)
        .context("Failed to create input1 tensor")?;
    let input2 = Array3::from_shape_vec((1, 1, target_len), wav2_padded)
        .context("Failed to create input2 tensor")?;

    let input1_value = Value::from_array(input1)?;
    let input2_value = Value::from_array(input2)?;

    // Run inference
    let outputs = session
        .run(ort::inputs![input1_value, input2_value])
        .context("Failed to run ONNX inference")?;

    // Extract output tensor (p_bc_seq)
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor")?;

    // Shape should be [1, F, 1] where F is the number of frames
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    if dims.len() != 3 {
        anyhow::bail!("Expected 3D output, got shape: {:?}", dims);
    }

    let output = Array3::from_shape_vec((dims[0], dims[1], dims[2]), data.to_vec())
        .context("Failed to reshape output tensor")?;

    Ok(output)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading ONNX model from: {}", args.model.display());
    let mut session = Session::builder()?
        .with_intra_threads(4)?
        .commit_from_file(&args.model)
        .context("Failed to load ONNX model")?;

    println!("Loading WAV file 1 (user): {}", args.wav1.display());
    let wav1 = load_wav_as_f32(&args.wav1)?;
    println!("  Loaded {} samples", wav1.len());

    println!("Loading WAV file 2 (system): {}", args.wav2.display());
    let wav2 = load_wav_as_f32(&args.wav2)?;
    println!("  Loaded {} samples", wav2.len());

    println!("\nRunning inference...");
    let output = run_inference(&mut session, wav1, wav2)?;

    // Output shape: [B, F, 1] where F is the number of frames
    let shape = output.shape();
    println!("Output shape: {:?}", shape);
    println!("Number of frames: {}", shape[1]);

    // Extract probabilities (shape[1] is the number of frames)
    let probabilities = output.slice(ndarray::s![0, .., 0]).to_vec();

    if args.verbose {
        println!("\n=== Frame-by-frame backchannel probabilities ===");
        for (i, &prob) in probabilities.iter().enumerate() {
            println!("Frame {:4}: {:.6}", i, prob);
        }
    }

    // Show statistics
    let max_prob = probabilities
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_prob = probabilities.iter().copied().fold(f32::INFINITY, f32::min);
    let avg_prob = probabilities.iter().sum::<f32>() / probabilities.len() as f32;

    println!("\n=== Summary Statistics ===");
    println!("Max probability: {:.6}", max_prob);
    println!("Min probability: {:.6}", min_prob);
    println!("Avg probability: {:.6}", avg_prob);

    // Find top-5 frames with highest backchannel probability
    let mut indexed_probs: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n=== Top 5 backchannel timing candidates ===");
    for (rank, (frame_idx, prob)) in indexed_probs.iter().take(5).enumerate() {
        // Assuming 10 Hz frame rate (100ms per frame) as per the export command
        let time_sec = *frame_idx as f32 * 0.1;
        println!(
            "Rank {}: Frame {} (t={:.2}s) - probability: {:.6}",
            rank + 1,
            frame_idx,
            time_sec,
            prob
        );
    }

    println!("\n✓ Inference completed successfully");

    Ok(())
}
