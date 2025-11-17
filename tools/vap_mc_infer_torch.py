#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Ensure repo src/tools on path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
TOOLS_PATH = REPO_ROOT / "tools"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))

import torch

from export_vap_mc_onnx import build_model  # Reuse model builder/wrapper


def read_wav_mono_f32(path: Path) -> Tuple[np.ndarray, int]:
    import wave
    try:
        with wave.open(str(path), "rb") as wf:
            nch = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            data = wf.readframes(nframes)
        if sampwidth == 2:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            if nch > 1:
                samples = samples.reshape(-1, nch)[:, 0]
            samples = samples / 32768.0
            return samples, sr
        else:
            try:
                import soundfile as sf
                x, sr = sf.read(str(path), always_2d=True, dtype="float32")
                x = x[:, 0]
                return x, sr
            except Exception as fe:
                raise RuntimeError(
                    f"Unsupported WAV format (sampwidth={sampwidth}); install soundfile to handle it."
                ) from fe
    except wave.Error as we:
        try:
            import soundfile as sf
            x, sr = sf.read(str(path), always_2d=True, dtype="float32")
            x = x[:, 0]
            return x, sr
        except Exception as fe:
            raise RuntimeError(f"Failed to read WAV: {path}") from fe


def save_csv(csv_path: Path, p_now: np.ndarray, p_future: np.ndarray, vad: np.ndarray, frame_rate: float = 20.0):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,time_sec,p_now_ch1,p_now_ch2,p_future_ch1,p_future_ch2,vad_ch1,vad_ch2\n")
        for i in range(p_now.shape[0]):
            t = i / frame_rate
            f.write(
                f"{i},{t:.6f},{p_now[i,0]:.9f},{p_now[i,1]:.9f},{p_future[i,0]:.9f},{p_future[i,1]:.9f},{vad[i,0]:.9f},{vad[i,1]:.9f}\n"
            )


def compare_csv(csv_a: Path, csv_b: Path, atol: float = 1e-6, rtol: float = 1e-6) -> dict:
    def load(csv: Path) -> np.ndarray:
        arr = np.loadtxt(csv, delimiter=",", skiprows=1)
        return arr
    a = load(csv_a)
    b = load(csv_b)
    if a.shape != b.shape:
        return {"equal": False, "reason": f"Shape mismatch: {a.shape} vs {b.shape}"}
    diff = np.abs(a[:, 2:] - b[:, 2:])
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    ok = np.allclose(a[:, 2:], b[:, 2:], atol=atol, rtol=rtol)
    return {"equal": bool(ok), "max_abs": max_abs, "mean_abs": mean_abs, "frames": a.shape[0]}


def main():
    ap = argparse.ArgumentParser(description="VAP-MC PyTorch inference (20Hz/20s) and comparison utility")
    ap.add_argument("--wav1", required=True, type=Path)
    ap.add_argument("--wav2", required=True, type=Path)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--cpc-model", default=os.path.expanduser("~/.cache/cpc/60k_epoch4-d0f474de.pt"))
    ap.add_argument("--frame-rate", type=int, default=20)
    ap.add_argument("--context-sec", type=float, default=20.0)
    ap.add_argument("--language", default="jp_kyoto")
    ap.add_argument("--local-weights", type=Path, help="Optional local .pt state_dict path")
    ap.add_argument("--dump-csv", type=Path)
    ap.add_argument("--compare-csv", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    # Load audio
    x1, sr1 = read_wav_mono_f32(args.wav1)
    x2, sr2 = read_wav_mono_f32(args.wav2)
    if sr1 != 16000:
        print(f"Warning: Expected 16kHz but got {sr1}Hz for {args.wav1}")
    if sr2 != 16000:
        print(f"Warning: Expected 16kHz but got {sr2}Hz for {args.wav2}")

    # Time alignment: cut to context length or max
    expected_len = int(16000 * args.context_sec)
    max_len = max(len(x1), len(x2))
    target_len = expected_len if max_len > expected_len else max_len
    x1 = x1[:target_len]
    x2 = x2[:target_len]
    if len(x1) < target_len:
        x1 = np.pad(x1, (0, target_len - len(x1)))
    if len(x2) < target_len:
        x2 = np.pad(x2, (0, target_len - len(x2)))

    # Build model (wrapper forward returns p_now, p_future, vad)
    wrapper, _ = build_model(
        device=args.device,
        cpc_model_path=str(args.cpc_model),
        frame_rate=args.frame_rate,
        context_len_sec=args.context_sec,
        language=args.language,
        cache_dir=None,
        force_download=False,
        local_weights=str(args.local_weights) if args.local_weights else None,
    )

    with torch.no_grad():
        x1_t = torch.from_numpy(x1.astype(np.float32))[None, None, :].to(args.device)
        x2_t = torch.from_numpy(x2.astype(np.float32))[None, None, :].to(args.device)
        p_now_t, p_future_t, vad_t = wrapper(x1_t, x2_t)
        # Move to CPU numpy and drop batch dim
        p_now = p_now_t.detach().cpu().numpy()[0]
        p_future = p_future_t.detach().cpu().numpy()[0]
        vad = vad_t.detach().cpu().numpy()[0]

    print(f"Frames: {p_now.shape[0]}")

    if args.dump_csv:
        save_csv(args.dump_csv, p_now, p_future, vad)
        print(f"Dumped CSV: {args.dump_csv}")

    if args.compare_csv:
        res = compare_csv(args.dump_csv if args.dump_csv else None or Path("/tmp/pytorch_tmp.csv"), args.compare_csv, atol=args.atol, rtol=args.rtol)
        if not args.dump_csv:
            # ensure tmp saved
            save_csv(Path("/tmp/pytorch_tmp.csv"), p_now, p_future, vad)
            res = compare_csv(Path("/tmp/pytorch_tmp.csv"), args.compare_csv, atol=args.atol, rtol=args.rtol)
        print("\nComparison (PyTorch vs Other CSV):")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
