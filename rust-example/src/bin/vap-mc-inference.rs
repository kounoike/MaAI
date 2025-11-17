use anyhow::{Context, Result};
use clap::Parser;
use hound::WavReader;
use ndarray::Array3;
use ort::session::Session;
use ort::value::Value;
use std::path::PathBuf;

/// VAP-MC (Multi-channel, noise-robust VAP with VAD) inference using ONNX Runtime
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

    /// VAD threshold for detecting voice activity (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    vad_threshold: f32,
}

fn load_wav_as_f32(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();

    // Ensure 16kHz mono (VAP-MC expects 16kHz)
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

#[derive(Debug)]
struct VapMcOutput {
    p_now: Array3<f32>,    // [B, F, 2] - turn-taking probability "now"
    p_future: Array3<f32>, // [B, F, 2] - turn-taking probability "future"
    vad: Array3<f32>,      // [B, F, 2] - voice activity detection [ch1, ch2]
}

fn run_inference(session: &mut Session, wav1: Vec<f32>, wav2: Vec<f32>) -> Result<VapMcOutput> {
    // Model expects fixed length (determined at export time)
    // For 20Hz frame rate and 20s context: 16000 * 20 = 320000 samples
    let expected_len = 320000;

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

    // Extract output tensors: p_now, p_future, vad
    if outputs.len() != 3 {
        anyhow::bail!(
            "Expected 3 outputs (p_now, p_future, vad), got {}",
            outputs.len()
        );
    }

    // Extract p_now [B, F, 2]
    let (shape_p_now, data_p_now) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract p_now tensor")?;
    let dims_p_now: Vec<usize> = shape_p_now.iter().map(|&d| d as usize).collect();
    let p_now = Array3::from_shape_vec(
        (dims_p_now[0], dims_p_now[1], dims_p_now[2]),
        data_p_now.to_vec(),
    )
    .context("Failed to reshape p_now tensor")?;

    // Extract p_future [B, F, 2]
    let (shape_p_future, data_p_future) = outputs[1]
        .try_extract_tensor::<f32>()
        .context("Failed to extract p_future tensor")?;
    let dims_p_future: Vec<usize> = shape_p_future.iter().map(|&d| d as usize).collect();
    let p_future = Array3::from_shape_vec(
        (dims_p_future[0], dims_p_future[1], dims_p_future[2]),
        data_p_future.to_vec(),
    )
    .context("Failed to reshape p_future tensor")?;

    // Extract vad [B, F, 2]
    let (shape_vad, data_vad) = outputs[2]
        .try_extract_tensor::<f32>()
        .context("Failed to extract vad tensor")?;
    let dims_vad: Vec<usize> = shape_vad.iter().map(|&d| d as usize).collect();
    let vad = Array3::from_shape_vec((dims_vad[0], dims_vad[1], dims_vad[2]), data_vad.to_vec())
        .context("Failed to reshape vad tensor")?;

    Ok(VapMcOutput {
        p_now,
        p_future,
        vad,
    })
}

/// Detect speech segments based on VAD output
fn detect_speech_segments(vad: &[f32], threshold: f32, frame_rate_hz: f32) -> Vec<(f32, f32)> {
    let mut segments = Vec::new();
    let mut in_segment = false;
    let mut segment_start = 0.0;

    for (i, &vad_val) in vad.iter().enumerate() {
        let time = i as f32 / frame_rate_hz;
        let is_active = vad_val >= threshold;

        if is_active && !in_segment {
            // Start new segment
            segment_start = time;
            in_segment = true;
        } else if !is_active && in_segment {
            // End current segment
            segments.push((segment_start, time));
            in_segment = false;
        }
    }

    // Close last segment if still open
    if in_segment {
        let end_time = vad.len() as f32 / frame_rate_hz;
        segments.push((segment_start, end_time));
    }

    segments
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
    println!(
        "  Loaded {} samples ({:.2}s)",
        wav1.len(),
        wav1.len() as f32 / 16000.0
    );

    println!("Loading WAV file 2 (system): {}", args.wav2.display());
    let wav2 = load_wav_as_f32(&args.wav2)?;
    println!(
        "  Loaded {} samples ({:.2}s)",
        wav2.len(),
        wav2.len() as f32 / 16000.0
    );

    println!("\nRunning VAP-MC inference...");
    let output = run_inference(&mut session, wav1, wav2)?;

    let num_frames = output.vad.shape()[1];
    println!("Number of frames: {}", num_frames);
    let frame_rate = 20.0; // 20Hz (50ms per frame)

    // Extract VAD for each channel
    let vad_ch1: Vec<f32> = (0..num_frames).map(|i| output.vad[[0, i, 0]]).collect();
    let vad_ch2: Vec<f32> = (0..num_frames).map(|i| output.vad[[0, i, 1]]).collect();

    if args.verbose {
        println!("\n=== Frame-by-frame output ===");
        for i in 0..num_frames {
            let time = i as f32 * 0.05; // 50ms per frame
            println!(
                "Frame {:4} (t={:5.2}s): VAD[ch1={:.3}, ch2={:.3}], P_now[s1={:.3}, s2={:.3}], P_future[s1={:.3}, s2={:.3}]",
                i, time,
                output.vad[[0, i, 0]], output.vad[[0, i, 1]],
                output.p_now[[0, i, 0]], output.p_now[[0, i, 1]],
                output.p_future[[0, i, 0]], output.p_future[[0, i, 1]],
            );
        }
    }

    // VAD statistics
    println!("\n=== VAD Statistics ===");
    println!("VAD threshold: {:.2}", args.vad_threshold);

    let vad_ch1_max = vad_ch1.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let vad_ch1_avg = vad_ch1.iter().sum::<f32>() / vad_ch1.len() as f32;
    println!(
        "Channel 1 (user):   max={:.3}, avg={:.3}",
        vad_ch1_max, vad_ch1_avg
    );

    let vad_ch2_max = vad_ch2.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let vad_ch2_avg = vad_ch2.iter().sum::<f32>() / vad_ch2.len() as f32;
    println!(
        "Channel 2 (system): max={:.3}, avg={:.3}",
        vad_ch2_max, vad_ch2_avg
    );

    // Detect speech segments
    let segments_ch1 = detect_speech_segments(&vad_ch1, args.vad_threshold, frame_rate);
    let segments_ch2 = detect_speech_segments(&vad_ch2, args.vad_threshold, frame_rate);

    println!("\n=== Speech Segments (Channel 1 - User) ===");
    if segments_ch1.is_empty() {
        println!(
            "No speech detected above threshold {:.2}",
            args.vad_threshold
        );
    } else {
        for (i, (start, end)) in segments_ch1.iter().enumerate() {
            println!(
                "  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s)",
                i + 1,
                start,
                end,
                end - start
            );
        }
    }

    println!("\n=== Speech Segments (Channel 2 - System) ===");
    if segments_ch2.is_empty() {
        println!(
            "No speech detected above threshold {:.2}",
            args.vad_threshold
        );
    } else {
        for (i, (start, end)) in segments_ch2.iter().enumerate() {
            println!(
                "  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s)",
                i + 1,
                start,
                end,
                end - start
            );
        }
    }

    // Turn-taking analysis
    println!("\n=== Turn-Taking Analysis ===");
    let p_now_ch1: Vec<f32> = (0..num_frames).map(|i| output.p_now[[0, i, 0]]).collect();
    let p_now_ch2: Vec<f32> = (0..num_frames).map(|i| output.p_now[[0, i, 1]]).collect();

    // Find top-5 turn-taking moments for each channel
    let mut indexed_p_now_ch1: Vec<(usize, f32)> = p_now_ch1.iter().copied().enumerate().collect();
    indexed_p_now_ch1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 turn-taking opportunities (Channel 1 → Channel 2):");
    for (rank, (frame_idx, prob)) in indexed_p_now_ch1.iter().take(5).enumerate() {
        let time_sec = *frame_idx as f32 * 0.05; // 50ms per frame (20Hz)
        println!(
            "  Rank {}: Frame {} (t={:.2}s) - probability: {:.3}",
            rank + 1,
            frame_idx,
            time_sec,
            prob
        );
    }

    println!("\n✓ VAP-MC inference completed successfully");

    Ok(())
}
