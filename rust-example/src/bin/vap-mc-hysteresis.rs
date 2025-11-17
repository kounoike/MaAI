use anyhow::{Context, Result};
use clap::Parser;
use hound::WavReader;
use ndarray::Array3;
use ort::session::Session;
use ort::value::Value;
use std::path::PathBuf;
use std::time::Instant;

/// VAP-MC inference with hysteresis and debouncing for stable turn-taking detection
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

    /// Turn-taking upper threshold (trigger when crossing up)
    #[arg(long, default_value = "0.7")]
    tt_high_threshold: f32,

    /// Turn-taking lower threshold (release when crossing down)
    #[arg(long, default_value = "0.5")]
    tt_low_threshold: f32,

    /// Minimum consecutive frames to trigger turn-taking
    #[arg(long, default_value = "3")]
    tt_min_frames: usize,

    /// Minimum consecutive frames to release turn-taking
    #[arg(long, default_value = "5")]
    tt_release_frames: usize,

    /// Optional: dump per-frame outputs (p_now, p_future, vad) as CSV to this path
    #[arg(long)]
    dump_csv: Option<PathBuf>,

    /// Model context length in seconds (e.g., 2.5, 20.0)
    #[arg(long, default_value = "20.0")]
    context_sec: f32,

    /// Number of times to run inference for benchmarking (default: 1)
    #[arg(long, default_value = "1")]
    benchmark_runs: usize,
}

fn load_wav_as_f32(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();

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

    let samples: Result<Vec<f32>> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i32 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .enumerate()
                .filter_map(|(i, s)| {
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
    p_now: Array3<f32>,
    p_future: Array3<f32>,
    vad: Array3<f32>,
}

fn run_inference(
    session: &mut Session,
    wav1: Vec<f32>,
    wav2: Vec<f32>,
    context_sec: f32,
) -> Result<VapMcOutput> {
    let expected_len = (16000.0 * context_sec) as usize; // e.g., 40000 for 2.5s, 320000 for 20s
    let max_len = wav1.len().max(wav2.len());

    // Always use expected_len (pad short audio, truncate long audio)
    let target_len = expected_len;

    if max_len > expected_len {
        eprintln!(
            "Warning: Input audio is longer than model expects ({} > {}). Truncating to first {:.2}s",
            max_len, expected_len, context_sec
        );
    } else if max_len < expected_len {
        eprintln!(
            "Info: Input audio is shorter than model expects ({} < {}). Padding with silence to {:.2}s",
            max_len, expected_len, context_sec
        );
    }

    let mut wav1_padded = wav1;
    let mut wav2_padded = wav2;

    // Truncate if longer than expected
    wav1_padded.truncate(target_len);
    wav2_padded.truncate(target_len);

    // Pad with silence (0.0) if shorter than expected
    if wav1_padded.len() < target_len {
        wav1_padded.resize(target_len, 0.0);
    }
    if wav2_padded.len() < target_len {
        wav2_padded.resize(target_len, 0.0);
    }

    let input1 = Array3::from_shape_vec((1, 1, target_len), wav1_padded)
        .context("Failed to create input1 tensor")?;
    let input2 = Array3::from_shape_vec((1, 1, target_len), wav2_padded)
        .context("Failed to create input2 tensor")?;

    let input1_value = Value::from_array(input1)?;
    let input2_value = Value::from_array(input2)?;

    let outputs = session
        .run(ort::inputs![input1_value, input2_value])
        .context("Failed to run ONNX inference")?;

    if outputs.len() != 3 {
        anyhow::bail!(
            "Expected 3 outputs (p_now, p_future, vad), got {}",
            outputs.len()
        );
    }

    let (shape_p_now, data_p_now) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract p_now tensor")?;
    let dims_p_now: Vec<usize> = shape_p_now.iter().map(|&d| d as usize).collect();
    let p_now = Array3::from_shape_vec(
        (dims_p_now[0], dims_p_now[1], dims_p_now[2]),
        data_p_now.to_vec(),
    )
    .context("Failed to reshape p_now tensor")?;

    let (shape_p_future, data_p_future) = outputs[1]
        .try_extract_tensor::<f32>()
        .context("Failed to extract p_future tensor")?;
    let dims_p_future: Vec<usize> = shape_p_future.iter().map(|&d| d as usize).collect();
    let p_future = Array3::from_shape_vec(
        (dims_p_future[0], dims_p_future[1], dims_p_future[2]),
        data_p_future.to_vec(),
    )
    .context("Failed to reshape p_future tensor")?;

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

/// Hysteresis state machine for turn-taking detection
#[derive(Debug, Clone)]
struct TurnTakingDetector {
    is_active: bool,
    high_threshold: f32,
    low_threshold: f32,
    min_trigger_frames: usize,
    min_release_frames: usize,
    high_counter: usize,
    low_counter: usize,
}

impl TurnTakingDetector {
    fn new(
        high_threshold: f32,
        low_threshold: f32,
        min_trigger_frames: usize,
        min_release_frames: usize,
    ) -> Self {
        Self {
            is_active: false,
            high_threshold,
            low_threshold,
            min_trigger_frames,
            min_release_frames,
            high_counter: 0,
            low_counter: 0,
        }
    }

    fn update(&mut self, value: f32) -> bool {
        if self.is_active {
            // Currently active - check for release condition
            if value < self.low_threshold {
                self.low_counter += 1;
                self.high_counter = 0;

                if self.low_counter >= self.min_release_frames {
                    self.is_active = false;
                    self.low_counter = 0;
                }
            } else {
                // Still above low threshold, reset release counter
                self.low_counter = 0;
            }
        } else {
            // Currently inactive - check for trigger condition
            if value > self.high_threshold {
                self.high_counter += 1;
                self.low_counter = 0;

                if self.high_counter >= self.min_trigger_frames {
                    self.is_active = true;
                    self.high_counter = 0;
                }
            } else {
                // Below high threshold, reset trigger counter
                self.high_counter = 0;
            }
        }

        self.is_active
    }

    fn is_active(&self) -> bool {
        self.is_active
    }
}

/// Detect speech segments with hysteresis
fn detect_speech_segments_hysteresis(
    vad: &[f32],
    high_threshold: f32,
    low_threshold: f32,
    min_frames: usize,
    frame_rate_hz: f32,
) -> Vec<(f32, f32)> {
    let mut segments = Vec::new();
    let mut detector =
        TurnTakingDetector::new(high_threshold, low_threshold, min_frames, min_frames * 2);
    let mut segment_start: Option<f32> = None;

    for (i, &vad_val) in vad.iter().enumerate() {
        let time = i as f32 / frame_rate_hz;
        let is_active = detector.update(vad_val);

        if is_active && segment_start.is_none() {
            segment_start = Some(time);
        } else if !is_active && segment_start.is_some() {
            segments.push((segment_start.unwrap(), time));
            segment_start = None;
        }
    }

    if let Some(start) = segment_start {
        let end_time = vad.len() as f32 / frame_rate_hz;
        segments.push((start, end_time));
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

    // Run inference multiple times for benchmarking
    println!(
        "\nRunning VAP-MC inference ({} runs)...",
        args.benchmark_runs
    );
    let mut inference_times = Vec::with_capacity(args.benchmark_runs);
    let mut output = None;

    for run in 0..args.benchmark_runs {
        let inference_start = Instant::now();
        let result = run_inference(&mut session, wav1.clone(), wav2.clone(), args.context_sec)?;
        let inference_time = inference_start.elapsed();
        inference_times.push(inference_time);

        // Show detailed timing for small number of runs or first/last runs
        if args.benchmark_runs <= 20 {
            println!(
                "  Run {}: {:.2}ms",
                run + 1,
                inference_time.as_secs_f64() * 1000.0
            );
        } else if run == 0 {
            println!(
                "  Run 1 (cold): {:.2}ms",
                inference_time.as_secs_f64() * 1000.0
            );
        } else if run == args.benchmark_runs - 1 {
            println!(
                "  Run {} (warm): {:.2}ms",
                run + 1,
                inference_time.as_secs_f64() * 1000.0
            );
        }

        // Keep the last output for analysis
        output = Some(result);
    }

    let output = output.unwrap();
    let num_frames = output.vad.shape()[1];

    // Calculate statistics
    if args.benchmark_runs > 1 {
        let total_ms: f64 = inference_times
            .iter()
            .map(|t| t.as_secs_f64() * 1000.0)
            .sum();
        let mean_ms = total_ms / inference_times.len() as f64;
        let first_ms = inference_times[0].as_secs_f64() * 1000.0;
        let min_ms = inference_times
            .iter()
            .map(|t| t.as_secs_f64() * 1000.0)
            .fold(f64::INFINITY, f64::min);
        let max_ms = inference_times
            .iter()
            .map(|t| t.as_secs_f64() * 1000.0)
            .fold(f64::NEG_INFINITY, f64::max);

        // Calculate warm runs average (excluding first run)
        let warm_avg_ms = if inference_times.len() > 1 {
            let warm_total: f64 = inference_times[1..]
                .iter()
                .map(|t| t.as_secs_f64() * 1000.0)
                .sum();
            warm_total / (inference_times.len() - 1) as f64
        } else {
            mean_ms
        };

        println!("\nInference Statistics:");
        println!("  Runs: {}", args.benchmark_runs);
        println!("  First (cold): {:.2}ms", first_ms);
        println!(
            "  Warm average: {:.2}ms ({:.2}ms per frame)",
            warm_avg_ms,
            warm_avg_ms / num_frames as f64
        );
        println!("  Min: {:.2}ms", min_ms);
        println!("  Max: {:.2}ms", max_ms);
        println!("  Mean: {:.2}ms", mean_ms);
    } else {
        let inference_time = inference_times[0];
        println!("Number of frames: {}", num_frames);
        println!(
            "Inference time: {:.2}ms ({:.2}ms per frame)",
            inference_time.as_secs_f64() * 1000.0,
            inference_time.as_secs_f64() * 1000.0 / num_frames as f64
        );
    }

    let frame_rate = 20.0; // 20Hz (50ms per frame)

    // Extract probabilities
    let p_now_ch1: Vec<f32> = (0..num_frames).map(|i| output.p_now[[0, i, 0]]).collect();
    let p_now_ch2: Vec<f32> = (0..num_frames).map(|i| output.p_now[[0, i, 1]]).collect();
    let vad_ch1: Vec<f32> = (0..num_frames).map(|i| output.vad[[0, i, 0]]).collect();
    let vad_ch2: Vec<f32> = (0..num_frames).map(|i| output.vad[[0, i, 1]]).collect();
    let p_future_ch1: Vec<f32> = (0..num_frames)
        .map(|i| output.p_future[[0, i, 0]])
        .collect();
    let p_future_ch2: Vec<f32> = (0..num_frames)
        .map(|i| output.p_future[[0, i, 1]])
        .collect();

    // Optional: dump per-frame outputs as CSV for cross-runtime comparison
    if let Some(csv_path) = &args.dump_csv {
        use std::io::Write;
        if let Some(parent) = csv_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut file = std::fs::File::create(csv_path)
            .with_context(|| format!("Failed to create dump CSV: {}", csv_path.display()))?;
        // header
        writeln!(
            file,
            "frame,time_sec,p_now_ch1,p_now_ch2,p_future_ch1,p_future_ch2,vad_ch1,vad_ch2"
        )?;
        for i in 0..num_frames {
            let t = i as f32 * 0.05;
            writeln!(
                file,
                "{},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
                i,
                t,
                p_now_ch1[i],
                p_now_ch2[i],
                p_future_ch1[i],
                p_future_ch2[i],
                vad_ch1[i],
                vad_ch2[i]
            )?;
        }
        println!(
            "Dumped per-frame outputs to CSV: {} ({} frames)",
            csv_path.display(),
            num_frames
        );
    }

    // Hysteresis-based turn-taking detection
    println!("\n=== Turn-Taking Detection (with Hysteresis) ===");
    println!("Configuration:");
    println!("  High threshold: {:.2}", args.tt_high_threshold);
    println!("  Low threshold: {:.2}", args.tt_low_threshold);
    println!("  Min trigger frames: {}", args.tt_min_frames);
    println!("  Min release frames: {}", args.tt_release_frames);

    let mut tt_detector_ch1 = TurnTakingDetector::new(
        args.tt_high_threshold,
        args.tt_low_threshold,
        args.tt_min_frames,
        args.tt_release_frames,
    );

    let mut tt_detector_ch2 = TurnTakingDetector::new(
        args.tt_high_threshold,
        args.tt_low_threshold,
        args.tt_min_frames,
        args.tt_release_frames,
    );

    let mut tt_events_ch1 = Vec::new();
    let mut tt_events_ch2 = Vec::new();

    for i in 0..num_frames {
        let time = i as f32 * 0.05; // 50ms per frame
        let tt_ch1 = tt_detector_ch1.update(p_now_ch1[i]);
        let tt_ch2 = tt_detector_ch2.update(p_now_ch2[i]);

        // Record state changes
        if i > 0 {
            let prev_tt_ch1 = tt_detector_ch1.is_active();
            let prev_tt_ch2 = tt_detector_ch2.is_active();

            if tt_ch1 && i == 0 || (tt_ch1 && !prev_tt_ch1) {
                tt_events_ch1.push(("START", time, p_now_ch1[i]));
            } else if !tt_ch1 && prev_tt_ch1 {
                tt_events_ch1.push(("END", time, p_now_ch1[i]));
            }

            if tt_ch2 && i == 0 || (tt_ch2 && !prev_tt_ch2) {
                tt_events_ch2.push(("START", time, p_now_ch2[i]));
            } else if !tt_ch2 && prev_tt_ch2 {
                tt_events_ch2.push(("END", time, p_now_ch2[i]));
            }
        }

        if args.verbose {
            println!(
                "Frame {:4} (t={:5.2}s): p_now[ch1={:.3}{}, ch2={:.3}{}], VAD[ch1={:.3}, ch2={:.3}]",
                i,
                time,
                p_now_ch1[i],
                if tt_ch1 { " ✓" } else { "" },
                p_now_ch2[i],
                if tt_ch2 { " ✓" } else { "" },
                vad_ch1[i],
                vad_ch2[i]
            );
        }
    }

    // Report turn-taking events
    println!("\n=== Turn-Taking Events (Channel 1 - User) ===");
    if tt_events_ch1.is_empty() {
        println!("No turn-taking events detected");
    } else {
        let mut in_event = false;
        let mut event_start = 0.0;
        for (event_type, time, prob) in &tt_events_ch1 {
            if *event_type == "START" {
                event_start = *time;
                in_event = true;
                println!("  Turn-taking opportunity at {:.2}s (p={:.3})", time, prob);
            } else if *event_type == "END" && in_event {
                println!(
                    "    → Released at {:.2}s (duration: {:.2}s, p={:.3})",
                    time,
                    time - event_start,
                    prob
                );
                in_event = false;
            }
        }
    }

    println!("\n=== Turn-Taking Events (Channel 2 - System) ===");
    if tt_events_ch2.is_empty() {
        println!("No turn-taking events detected");
    } else {
        let mut in_event = false;
        let mut event_start = 0.0;
        for (event_type, time, prob) in &tt_events_ch2 {
            if *event_type == "START" {
                event_start = *time;
                in_event = true;
                println!("  Turn-taking opportunity at {:.2}s (p={:.3})", time, prob);
            } else if *event_type == "END" && in_event {
                println!(
                    "    → Released at {:.2}s (duration: {:.2}s, p={:.3})",
                    time,
                    time - event_start,
                    prob
                );
                in_event = false;
            }
        }
    }

    // VAD with hysteresis
    println!("\n=== Voice Activity Detection (with Hysteresis) ===");
    let vad_high = args.vad_threshold;
    let vad_low = args.vad_threshold * 0.7; // 30% lower for hysteresis

    let segments_ch1 =
        detect_speech_segments_hysteresis(&vad_ch1, vad_high, vad_low, 2, frame_rate);
    let segments_ch2 =
        detect_speech_segments_hysteresis(&vad_ch2, vad_high, vad_low, 2, frame_rate);

    println!("Hysteresis: high={:.2}, low={:.2}", vad_high, vad_low);

    println!("\nChannel 1 (User) - {} segments:", segments_ch1.len());
    for (i, (start, end)) in segments_ch1.iter().enumerate() {
        println!(
            "  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s)",
            i + 1,
            start,
            end,
            end - start
        );
    }

    println!("\nChannel 2 (System) - {} segments:", segments_ch2.len());
    for (i, (start, end)) in segments_ch2.iter().enumerate() {
        println!(
            "  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s)",
            i + 1,
            start,
            end,
            end - start
        );
    }

    println!("\n✓ VAP-MC inference with hysteresis completed successfully");

    Ok(())
}
