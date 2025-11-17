#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    raise SystemExit("onnxruntime is required. Please install with pip install onnxruntime.")


def read_wav_mono_f32(path: Path) -> Tuple[np.ndarray, int]:
    """
    Read WAV as float32 mono [-1, 1].
    - Prefer stdlib wave for 16-bit PCM.
    - Fallback to soundfile if available for other formats.
    Returns (audio, sample_rate)
    """
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            nch = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            data = wf.readframes(nframes)

        if sampwidth == 2:  # 16-bit PCM
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            if nch > 1:
                samples = samples.reshape(-1, nch)[:, 0]
            samples = samples / 32768.0
            return samples, sr
        else:
            # Fallback to soundfile if available
            try:
                import soundfile as sf

                x, sr = sf.read(str(path), always_2d=True, dtype="float32")
                if x.shape[1] > 1:
                    x = x[:, 0]
                else:
                    x = x[:, 0]
                return x, sr
            except Exception as fe:
                raise RuntimeError(
                    f"Unsupported WAV format (sampwidth={sampwidth}); install soundfile to handle it."
                ) from fe
    except wave.Error as we:
        # Not a PCM WAV; try soundfile
        try:
            import soundfile as sf

            x, sr = sf.read(str(path), always_2d=True, dtype="float32")
            if x.shape[1] > 1:
                x = x[:, 0]
            else:
                x = x[:, 0]
            return x, sr
        except Exception as fe:
            raise RuntimeError(f"Failed to read WAV: {path}") from fe


class TurnTakingDetector:
    def __init__(self, high_threshold: float, low_threshold: float, min_trigger_frames: int, min_release_frames: int):
        self.is_active_flag = False
        self.high = high_threshold
        self.low = low_threshold
        self.min_trig = min_trigger_frames
        self.min_rel = min_release_frames
        self.high_cnt = 0
        self.low_cnt = 0

    def update(self, value: float) -> bool:
        if self.is_active_flag:
            if value < self.low:
                self.low_cnt += 1
                self.high_cnt = 0
                if self.low_cnt >= self.min_rel:
                    self.is_active_flag = False
                    self.low_cnt = 0
            else:
                self.low_cnt = 0
        else:
            if value > self.high:
                self.high_cnt += 1
                self.low_cnt = 0
                if self.high_cnt >= self.min_trig:
                    self.is_active_flag = True
                    self.high_cnt = 0
            else:
                self.high_cnt = 0
        return self.is_active_flag

    def is_active(self) -> bool:
        return self.is_active_flag


def detect_speech_segments_hysteresis(vad: np.ndarray, high: float, low: float, min_frames: int, frame_rate_hz: float) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    det = TurnTakingDetector(high, low, min_frames, min_frames * 2)
    start: Optional[float] = None
    for i, v in enumerate(vad):
        t = i / frame_rate_hz
        active = det.update(float(v))
        if active and start is None:
            start = t
        elif not active and start is not None:
            segs.append((start, t))
            start = None
    if start is not None:
        segs.append((start, len(vad) / frame_rate_hz))
    return segs


def run_onnx(model: Path, wav1: Path, wav2: Path, expected_len: int = 320000):
    # Load audio
    x1, sr1 = read_wav_mono_f32(wav1)
    x2, sr2 = read_wav_mono_f32(wav2)
    if sr1 != 16000:
        print(f"Warning: Expected 16kHz but got {sr1}Hz for {wav1}")
    if sr2 != 16000:
        print(f"Warning: Expected 16kHz but got {sr2}Hz for {wav2}")

    max_len = max(len(x1), len(x2))
    target_len = expected_len if max_len > expected_len else max_len
    x1 = x1[:target_len]
    x2 = x2[:target_len]
    if len(x1) < target_len:
        x1 = np.pad(x1, (0, target_len - len(x1)))
    if len(x2) < target_len:
        x2 = np.pad(x2, (0, target_len - len(x2)))

    # Shape to (1,1,T)
    x1 = x1.astype(np.float32)[None, None, :]
    x2 = x2.astype(np.float32)[None, None, :]

    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model), sess_options=sess_opt, providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    # Assume two inputs in the order they were exported
    feed = {
        inputs[0].name: x1,
        inputs[1].name: x2,
    }
    outs: List[np.ndarray] = sess.run(None, feed)  # [p_now, p_future, vad]

    p_now = outs[0]
    p_future = outs[1]
    vad = outs[2]
    assert p_now.shape[0] == 1 and p_future.shape[0] == 1 and vad.shape[0] == 1
    return p_now[0], p_future[0], vad[0], target_len


def save_csv(csv_path: Path, p_now: np.ndarray, p_future: np.ndarray, vad: np.ndarray, frame_rate: float = 20.0):
    # p_* shape: (frames, 2)
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
        # columns: frame,time,p_now_ch1,p_now_ch2,p_future_ch1,p_future_ch2,vad_ch1,vad_ch2
        return arr

    a = load(csv_a)
    b = load(csv_b)
    if a.shape != b.shape:
        return {"equal": False, "reason": f"Shape mismatch: {a.shape} vs {b.shape}"}

    # Compare only the numeric probability columns (exclude frame/time)
    diff = np.abs(a[:, 2:] - b[:, 2:])
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    ok = np.allclose(a[:, 2:], b[:, 2:], atol=atol, rtol=rtol)
    return {"equal": bool(ok), "max_abs": max_abs, "mean_abs": mean_abs, "frames": a.shape[0]}


def main():
    ap = argparse.ArgumentParser(description="VAP-MC ONNX inference in Python (20Hz/20s) and comparison utility")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--wav1", required=True, type=Path, help="User channel (speaker 1)")
    ap.add_argument("--wav2", required=True, type=Path, help="System channel (speaker 2)")
    ap.add_argument("--dump-csv", type=Path, help="Dump per-frame outputs to CSV here")
    ap.add_argument("--compare-csv", type=Path, help="Compare with another CSV (e.g., dumped by Rust)")
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--tt-high-threshold", type=float, default=0.7)
    ap.add_argument("--tt-low-threshold", type=float, default=0.5)
    ap.add_argument("--tt-min-frames", type=int, default=3)
    ap.add_argument("--tt-release-frames", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"Loading ONNX: {args.model}")
    print(f"Loading WAV1: {args.wav1}")
    print(f"Loading WAV2: {args.wav2}")
    p_now, p_future, vad, target_len = run_onnx(args.model, args.wav1, args.wav2)

    frames = p_now.shape[0]
    print(f"Frames: {frames}")

    if args.dump_csv:
        save_csv(args.dump_csv, p_now, p_future, vad)
        print(f"Dumped CSV: {args.dump_csv}")

    # Turn-taking detection (same hysteresis logic)
    frame_rate = 20.0
    tt1 = TurnTakingDetector(args.tt_high_threshold, args.tt_low_threshold, args.tt_min_frames, args.tt_release_frames)
    tt2 = TurnTakingDetector(args.tt_high_threshold, args.tt_low_threshold, args.tt_min_frames, args.tt_release_frames)

    if args.verbose:
        for i in range(frames):
            t = i / frame_rate
            s1 = tt1.update(float(p_now[i,0]))
            s2 = tt2.update(float(p_now[i,1]))
            print(
                f"Frame {i:4d} t={t:5.2f}s: p_now[{p_now[i,0]:.3f},{p_now[i,1]:.3f}] VAD[{vad[i,0]:.3f},{vad[i,1]:.3f}] {'*' if s1 else ''}{'*' if s2 else ''}"
            )

    # VAD segments with hysteresis
    vad_high = args.vad_threshold
    vad_low = args.vad_threshold * 0.7
    seg1 = detect_speech_segments_hysteresis(vad[:,0], vad_high, vad_low, 2, frame_rate)
    seg2 = detect_speech_segments_hysteresis(vad[:,1], vad_high, vad_low, 2, frame_rate)

    print("\nVAD segments (User):", len(seg1))
    for i,(s,e) in enumerate(seg1,1):
        print(f"  {i}: {s:.2f}s - {e:.2f}s (dur {e-s:.2f}s)")
    print("VAD segments (System):", len(seg2))
    for i,(s,e) in enumerate(seg2,1):
        print(f"  {i}: {s:.2f}s - {e:.2f}s (dur {e-s:.2f}s)")

    # Optional comparison to another CSV (e.g., Rust dump)
    if args.compare_csv and args.dump_csv:
        res = compare_csv(args.dump_csv, args.compare_csv)
        print("\nComparison (Python vs Other CSV):")
        print(json.dumps(res, indent=2))
    elif args.compare_csv:
        # If only compare target given, compute a temp csv to compare
        tmp = Path("/tmp/vap_mc_python_dump.csv")
        save_csv(tmp, p_now, p_future, vad)
        res = compare_csv(tmp, args.compare_csv)
        print("\nComparison (Python vs Other CSV):")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
