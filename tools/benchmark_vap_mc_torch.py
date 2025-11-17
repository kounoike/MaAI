#!/usr/bin/env python3
"""
Benchmark VAP-MC PyTorch inference performance.

Usage:
    python tools/benchmark_vap_mc_torch.py \
      --wav1 example/wav_sample/jpn_inoue_16k.wav \
      --wav2 example/wav_sample/jpn_sumida_16k.wav \
      --language jp_kyoto \
      --frame-rate 5 \
      --context-sec 3.0 \
      --runs 50
"""

import argparse
import time
from pathlib import Path
import sys

# Add src to path
_repo_root = Path(__file__).resolve().parents[1]
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import torch
import numpy as np
from scipy.io import wavfile

# Import from export script (in same directory)
sys.path.insert(0, str(Path(__file__).parent))
from export_vap_mc_onnx import build_model


def load_wav_as_tensor(path: str, target_len: int) -> torch.Tensor:
    """Load WAV file and convert to tensor with padding/truncation."""
    sample_rate, audio = wavfile.read(path)
    
    if sample_rate != 16000:
        print(f"Warning: Expected 16kHz but got {sample_rate}Hz")
    
    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    
    # Handle stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Pad or truncate
    if len(audio) < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded
    elif len(audio) > target_len:
        audio = audio[:target_len]
    
    # Convert to tensor: (1, 1, T)
    return torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Benchmark VAP-MC PyTorch inference")
    parser.add_argument("--wav1", required=True, help="Path to first WAV file")
    parser.add_argument("--wav2", required=True, help="Path to second WAV file")
    parser.add_argument("--language", default="jp_kyoto", help="Language variant")
    parser.add_argument("--frame-rate", type=int, default=5, help="Frame rate (Hz)")
    parser.add_argument("--context-sec", type=float, default=3.0, help="Context length (seconds)")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--threads", type=int, default=None, help="Number of PyTorch threads (default: auto)")
    
    args = parser.parse_args()
    
    # Set thread count if specified
    if args.threads:
        torch.set_num_threads(args.threads)
        print(f"PyTorch using {args.threads} threads")
    else:
        print(f"PyTorch using default threads ({torch.get_num_threads()})")
    
    device = args.device
    
    # Build model
    print(f"Loading VAP-MC model ({args.frame_rate}Hz, {args.context_sec}s, {args.language})...")
    wrapper, _ = build_model(
        device=device,
        cpc_model_path=Path.home() / ".cache" / "cpc" / "60k_epoch4-d0f474de.pt",
        frame_rate=args.frame_rate,
        context_len_sec=args.context_sec,
        language=args.language,
        cache_dir=None,
        force_download=False,
        local_weights=None,
    )
    
    # Load audio
    target_len = int(16000 * args.context_sec)
    print(f"\nLoading WAV file 1: {args.wav1}")
    wav1 = load_wav_as_tensor(args.wav1, target_len).to(device)
    print(f"  Shape: {wav1.shape} ({wav1.shape[2] / 16000:.2f}s)")
    
    print(f"Loading WAV file 2: {args.wav2}")
    wav2 = load_wav_as_tensor(args.wav2, target_len).to(device)
    print(f"  Shape: {wav2.shape} ({wav2.shape[2] / 16000:.2f}s)")
    
    # Warm-up run
    print("\nWarm-up run...")
    with torch.no_grad():
        _ = wrapper(wav1, wav2)
    
    # Benchmark
    print(f"\nRunning {args.runs} inference iterations...")
    times = []
    
    for i in range(args.runs):
        start = time.perf_counter()
        with torch.no_grad():
            p_now, p_future, vad = wrapper(wav1, wav2)
        
        # Sync for CUDA
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
        
        if args.runs <= 20:
            print(f"  Run {i+1}: {elapsed * 1000:.2f}ms")
        elif i == 0:
            print(f"  Run 1 (cold): {elapsed * 1000:.2f}ms")
        elif i == args.runs - 1:
            print(f"  Run {args.runs} (warm): {elapsed * 1000:.2f}ms")
    
    # Get output info
    num_frames = p_now.shape[1]
    frame_interval_ms = 1000.0 / args.frame_rate
    
    # Statistics
    times = np.array(times)
    first_time = times[0]
    warm_times = times[1:] if len(times) > 1 else times
    
    print("\nInference Statistics:")
    print(f"  Runs: {args.runs}")
    print(f"  Output frames: {num_frames}")
    print(f"  Frame interval: {frame_interval_ms:.1f}ms ({args.frame_rate}Hz)")
    print(f"  First (cold): {first_time:.2f}ms")
    
    if len(warm_times) > 0:
        warm_avg = warm_times.mean()
        print(f"  Warm average: {warm_avg:.2f}ms ({warm_avg / num_frames:.2f}ms per frame)")
        print(f"  Min: {times.min():.2f}ms")
        print(f"  Max: {times.max():.2f}ms")
        print(f"  Mean: {times.mean():.2f}ms")
        print(f"  Std: {times.std():.2f}ms")
        
        # Real-time assessment
        print("\nReal-time Assessment:")
        if warm_avg < frame_interval_ms:
            margin = frame_interval_ms - warm_avg
            print(f"  ✅ CAN run real-time (margin: {margin:.2f}ms, {frame_interval_ms / warm_avg:.2f}x)")
        else:
            deficit = warm_avg - frame_interval_ms
            print(f"  ❌ CANNOT run real-time (deficit: {deficit:.2f}ms, {warm_avg / frame_interval_ms:.2f}x slower)")


if __name__ == "__main__":
    main()
