#!/usr/bin/env python3
"""
Benchmark VAP-MC PyTorch inference in streaming mode (frame-by-frame with KV cache).
This mimics the actual Maai class behavior.

Usage:
    python tools/benchmark_vap_mc_torch_streaming.py \
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

_repo_root = Path(__file__).resolve().parents[1]
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import torch
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))
from export_vap_mc_onnx import build_model


def load_wav(path: str) -> np.ndarray:
    """Load WAV file as float32 numpy array."""
    sample_rate, audio = wavfile.read(path)
    
    if sample_rate != 16000:
        print(f"Warning: Expected 16kHz but got {sample_rate}Hz")
    
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    return audio


def main():
    parser = argparse.ArgumentParser(description="Benchmark VAP-MC PyTorch (streaming mode)")
    parser.add_argument("--wav1", required=True)
    parser.add_argument("--wav2", required=True)
    parser.add_argument("--language", default="jp_kyoto")
    parser.add_argument("--frame-rate", type=int, default=5)
    parser.add_argument("--context-sec", type=float, default=3.0)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--use-kv-cache", action="store_true", help="Use KV cache (like Maai class)")
    
    args = parser.parse_args()
    
    if args.threads:
        torch.set_num_threads(args.threads)
        print(f"PyTorch using {args.threads} threads")
    else:
        print(f"PyTorch using default threads ({torch.get_num_threads()})")
    
    device = args.device
    
    # Build model (get the underlying VapGPT, not the wrapper)
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
    
    # Get the underlying VapGPT model from the wrapper
    vap_model = wrapper.core  # VapMCExportWrapper has .core attribute
    
    # Calculate frame parameters (matching Maai class)
    sampling_rate = 16000
    frame_context_padding = 320
    audio_frame_size = sampling_rate // args.frame_rate + frame_context_padding
    audio_context_len = int(args.context_sec * args.frame_rate)
    
    print(f"\nFrame parameters:")
    print(f"  Frame size: {audio_frame_size} samples")
    print(f"  Context padding: {frame_context_padding} samples")
    print(f"  Context length: {audio_context_len} frames")
    
    # Load full audio
    print(f"\nLoading WAV files...")
    wav1_full = load_wav(args.wav1)
    wav2_full = load_wav(args.wav2)
    print(f"  Audio length: {len(wav1_full)} samples ({len(wav1_full) / 16000:.2f}s)")
    
    # Calculate how many frames we can process
    max_frames = min(len(wav1_full), len(wav2_full)) // (sampling_rate // args.frame_rate)
    print(f"  Max processable frames: {max_frames}")
    
    # Warm-up
    print("\nWarm-up run...")
    current_x1 = np.zeros(frame_context_padding, dtype=np.float32)
    current_x2 = np.zeros(frame_context_padding, dtype=np.float32)
    cache = None
    
    for frame_idx in range(min(10, max_frames)):
        start_idx = frame_idx * (sampling_rate // args.frame_rate)
        end_idx = start_idx + (sampling_rate // args.frame_rate)
        
        x1_chunk = wav1_full[start_idx:end_idx]
        x2_chunk = wav2_full[start_idx:end_idx]
        
        current_x1 = np.concatenate([current_x1, x1_chunk])
        current_x2 = np.concatenate([current_x2, x2_chunk])
        
        x1_ = torch.from_numpy(current_x1).float().unsqueeze(0).unsqueeze(0).to(device)
        x2_ = torch.from_numpy(current_x2).float().unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            e1, e2 = vap_model.encode_audio(x1_, x2_)
            if args.use_kv_cache:
                out, cache = vap_model.forward(e1, e2, cache=cache)
                # Trim cache (like Maai class)
                if cache is not None:
                    new_cache = {}
                    for key, (k_list, v_list) in cache.items():
                        new_k_list = [t[..., -(audio_context_len - 1):, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in k_list]
                        new_v_list = [t[..., -(audio_context_len - 1):, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in v_list]
                        new_cache[key] = (new_k_list, new_v_list)
                    cache = new_cache
            else:
                out, _ = vap_model.forward(e1, e2, cache=None)
        
        current_x1 = current_x1[-frame_context_padding:]
        current_x2 = current_x2[-frame_context_padding:]
    
    # Benchmark
    print(f"\nRunning {args.runs} streaming inference runs ({max_frames} frames each)...")
    print(f"KV Cache: {'Enabled' if args.use_kv_cache else 'Disabled'}")
    
    times = []
    for run in range(args.runs):
        current_x1 = np.zeros(frame_context_padding, dtype=np.float32)
        current_x2 = np.zeros(frame_context_padding, dtype=np.float32)
        cache = None
        
        start = time.perf_counter()
        
        for frame_idx in range(max_frames):
            start_idx = frame_idx * (sampling_rate // args.frame_rate)
            end_idx = start_idx + (sampling_rate // args.frame_rate)
            
            x1_chunk = wav1_full[start_idx:end_idx]
            x2_chunk = wav2_full[start_idx:end_idx]
            
            current_x1 = np.concatenate([current_x1, x1_chunk])
            current_x2 = np.concatenate([current_x2, x2_chunk])
            
            x1_ = torch.from_numpy(current_x1).float().unsqueeze(0).unsqueeze(0).to(device)
            x2_ = torch.from_numpy(current_x2).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                e1, e2 = vap_model.encode_audio(x1_, x2_)
                if args.use_kv_cache:
                    out, cache = vap_model.forward(e1, e2, cache=cache)
                    if cache is not None:
                        new_cache = {}
                        for key, (k_list, v_list) in cache.items():
                            new_k_list = [t[..., -(audio_context_len - 1):, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in k_list]
                            new_v_list = [t[..., -(audio_context_len - 1):, :] if isinstance(t, torch.Tensor) and t.dim() >= 3 else t for t in v_list]
                            new_cache[key] = (new_k_list, new_v_list)
                        cache = new_cache
                else:
                    out, _ = vap_model.forward(e1, e2, cache=None)
            
            current_x1 = current_x1[-frame_context_padding:]
            current_x2 = current_x2[-frame_context_padding:]
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
        
        if args.runs <= 20:
            print(f"  Run {run+1}: {elapsed * 1000:.2f}ms ({elapsed * 1000 / max_frames:.2f}ms per frame)")
        elif run == 0:
            print(f"  Run 1 (cold): {elapsed * 1000:.2f}ms ({elapsed * 1000 / max_frames:.2f}ms per frame)")
        elif run == args.runs - 1:
            print(f"  Run {args.runs} (warm): {elapsed * 1000:.2f}ms ({elapsed * 1000 / max_frames:.2f}ms per frame)")
    
    # Statistics
    times = np.array(times)
    first_time = times[0]
    warm_times = times[1:] if len(times) > 1 else times
    frame_interval_ms = 1000.0 / args.frame_rate
    
    print("\nInference Statistics (total time for all frames):")
    print(f"  Runs: {args.runs}")
    print(f"  Frames per run: {max_frames}")
    print(f"  Frame interval: {frame_interval_ms:.1f}ms ({args.frame_rate}Hz)")
    print(f"  First (cold): {first_time:.2f}ms total ({first_time / max_frames:.2f}ms per frame)")
    
    if len(warm_times) > 0:
        warm_avg = warm_times.mean()
        warm_per_frame = warm_avg / max_frames
        print(f"  Warm average: {warm_avg:.2f}ms total ({warm_per_frame:.2f}ms per frame)")
        print(f"  Min: {times.min():.2f}ms ({times.min() / max_frames:.2f}ms per frame)")
        print(f"  Max: {times.max():.2f}ms ({times.max() / max_frames:.2f}ms per frame)")
        print(f"  Mean: {times.mean():.2f}ms ({times.mean() / max_frames:.2f}ms per frame)")
        print(f"  Std: {times.std():.2f}ms")
        
        print("\nReal-time Assessment (per frame):")
        if warm_per_frame < frame_interval_ms:
            margin = frame_interval_ms - warm_per_frame
            print(f"  ✅ CAN run real-time (margin: {margin:.2f}ms per frame, {frame_interval_ms / warm_per_frame:.2f}x)")
        else:
            deficit = warm_per_frame - frame_interval_ms
            print(f"  ❌ CANNOT run real-time (deficit: {deficit:.2f}ms per frame, {warm_per_frame / frame_interval_ms:.2f}x slower)")


if __name__ == "__main__":
    main()
