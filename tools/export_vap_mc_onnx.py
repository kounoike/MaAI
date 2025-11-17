#!/usr/bin/env python3
"""
Export VAP-MC (Voice Activity Projection - Multi-Condition) model to ONNX for Rust inference.

This script loads the MaAI VapGPT model with CPC encoder, downloads the specified
state_dict from Hugging Face if needed, and exports an ONNX model that takes two
raw audio waveforms (B x 1 x T) and outputs:
  - p_now: short-term turn-taking prediction (B x F x 2)
  - p_future: long-term turn-taking prediction (B x F x 2)
  - vad: voice activity detection (B x F x 2)

Where F is the number of frames after encoder downsampling.

Weights example:
  repo: https://huggingface.co/maai-kyoto/vap_mc_jp_kyoto
  file: vap-mc_state_dict_jp_kyoto_10hz_20000msec.pt

Usage:
  python tools/export_vap_mc_onnx.py --out vap_mc_jp_10hz_20s.onnx

Notes:
- This exports the full model, including the CPC encoder.
- Dynamic time dimension is enabled for input waveforms and output frames.
- Opset 18 is used by default (legacy exporter compatible).
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

# Ensure local 'src' is on sys.path when running from repo without install
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
_src_path = _repo_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import torch
import torch.nn as nn
import torch._dynamo as dynamo

# Import MaAI modules
from maai.models.config import VapConfig
from maai.models.vap import VapGPT
from maai.util import load_vap_model


class VapMCExportWrapper(nn.Module):
    """
    Thin wrapper to expose an ONNX-friendly forward that:
      - Accepts raw waveforms for ch1/ch2: (B, 1, T)
      - Encodes them via CPC encoders
      - Runs AR stacks (no KV cache for simplicity)
      - Returns p_now, p_future, and vad sequences (B, F, 2)
    """

    def __init__(self, core: VapGPT):
        super().__init__()
        self.core = core
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1_wav: torch.Tensor, x2_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x1_wav, x2_wav: (B, 1, T)
        # Encode audio to embeddings
        e1, e2 = self.core.encode_audio(x1_wav, x2_wav)  # (B, F, D)

        # Run AR channel layers directly to avoid .tolist() in VapGPT.forward()
        o1 = self.core.ar_channel(e1, past_kv=None)
        o2 = self.core.ar_channel(e2, past_kv=None)
        
        # Run cross-channel AR
        out = self.core.ar(
            o1["x"],
            o2["x"],
            past_kv1=None,
            past_kv2=None,
            past_kv1_c=None,
            past_kv2_c=None,
        )

        # Compute VAD for each channel
        vad1 = self.core.va_classifier(o1["x"])  # (B, F, 1)
        vad2 = self.core.va_classifier(o2["x"])  # (B, F, 1)
        vad = torch.cat([vad1, vad2], dim=-1)    # (B, F, 2)
        vad = self.sigmoid(vad)                  # Apply sigmoid

        # Compute VAP logits and probabilities
        logits = self.core.vap_head(out["x"])    # (B, F, n_classes)
        probs = logits.softmax(dim=-1)

        # Aggregate probabilities for p_now and p_future
        p_now = self.core.objective.probs_next_speaker_aggregate(
            probs,
            from_bin=self.core.BINS_P_NOW[0],
            to_bin=self.core.BINS_P_NOW[-1],
        )  # (B, F, 2)

        p_future = self.core.objective.probs_next_speaker_aggregate(
            probs,
            from_bin=self.core.BINS_PFUTURE[0],
            to_bin=self.core.BINS_PFUTURE[1],
        )  # (B, F, 2)

        return p_now, p_future, vad


def _apply_encoder_downsample_weights(vap: VapGPT, sd: dict) -> None:
    """Load downsample weights that aren't covered by strict=False load_state_dict.

    This mirrors the logic used in maai.model.Maai.__init__.
    """
    vap.encoder1.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
    vap.encoder1.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
    vap.encoder1.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
    vap.encoder1.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])

    vap.encoder2.downsample[1].weight = nn.Parameter(sd['encoder.downsample.1.weight'])
    vap.encoder2.downsample[1].bias = nn.Parameter(sd['encoder.downsample.1.bias'])
    vap.encoder2.downsample[2].ln.weight = nn.Parameter(sd['encoder.downsample.2.ln.weight'])
    vap.encoder2.downsample[2].ln.bias = nn.Parameter(sd['encoder.downsample.2.ln.bias'])


def build_model(device: str, cpc_model_path: str, frame_rate: int, context_len_sec: float,
                language: str = "jp", cache_dir: str | None = None,
                force_download: bool = False, local_weights: str | None = None) -> Tuple[VapMCExportWrapper, dict]:
    """Construct VapGPT, load CPC + MaAI weights, wrap for ONNX export.

    Returns the export wrapper and the loaded state_dict (for diagnostics).
    """
    conf = VapConfig()
    model = VapGPT(conf)

    # Load weights
    if local_weights is None:
        sd = load_vap_model("vap_mc", frame_rate, context_len_sec, language, device, cache_dir, force_download)
    else:
        sd = torch.load(local_weights, map_location=torch.device(device))

    # Load CPC encoder weights
    model.load_encoder(cpc_model=cpc_model_path)

    # Core MaAI weights
    model.load_state_dict(sd, strict=False)
    _apply_encoder_downsample_weights(model, sd)

    # Eval mode
    model.to(device)
    model.eval()

    wrapper = VapMCExportWrapper(model)
    wrapper.to(device)
    wrapper.eval()

    return wrapper, sd


def main():
    parser = argparse.ArgumentParser(description="Export VAP-MC model to ONNX")
    parser.add_argument("--out", required=True, help="Output ONNX file path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for export")
    parser.add_argument("--cpc-model", default=os.path.expanduser("~/.cache/cpc/60k_epoch4-d0f474de.pt"),
                        help="Path to CPC checkpoint; will be downloaded if missing")
    parser.add_argument("--frame-rate", type=int, default=10, help="Frame rate Hz used by the weights")
    parser.add_argument("--context-sec", type=float, default=20.0, help="Context length seconds used by the weights")
    parser.add_argument("--language", default="jp", help="Language tag for weights (default: jp)")
    parser.add_argument("--local-weights", default=None, help="Optional local .pt state_dict path to load instead of HF")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")

    args = parser.parse_args()

    device = args.device

    # Build model + load weights
    wrapper, _ = build_model(
        device=device,
        cpc_model_path=args.cpc_model,
        frame_rate=args.frame_rate,
        context_len_sec=args.context_sec,
        language=args.language,
        cache_dir=None,
        force_download=False,
        local_weights=args.local_weights,
    )

    # Dummy input: full context length for export
    sample_rate = 16000
    T = int(sample_rate * args.context_sec)  # Full context (e.g., 16000 * 20 = 320000 for 20s)

    x1 = torch.zeros(1, 1, T, dtype=torch.float32, device=device)
    x2 = torch.zeros(1, 1, T, dtype=torch.float32, device=device)

    # Export
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    input_names = ["x1", "x2"]
    output_names = ["p_now", "p_future", "vad"]

    with torch.no_grad():
        # Force static shape export to avoid symbolic shape solver issues
        dynamo.config.dynamic_shapes = False
        torch.onnx.export(
            wrapper,
            (x1, x2),
            args.out,
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Exported ONNX to: {args.out}")


if __name__ == "__main__":
    main()
