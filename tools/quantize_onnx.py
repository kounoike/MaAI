#!/usr/bin/env python3
"""
Quantize ONNX models for faster CPU inference.

This script performs dynamic quantization on ONNX models to reduce their size
and improve inference speed on CPU.

Usage:
    python tools/quantize_onnx.py --input model.onnx --output model_int8.onnx --mode int8
    python tools/quantize_onnx.py --input model.onnx --output model_fp16.onnx --mode fp16
"""

import argparse
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnx import version_converter


def quantize_int8(input_model: str, output_model: str):
    """
    Apply INT8 dynamic quantization to the model.
    This converts weights to INT8 and performs dynamic quantization at runtime.
    """
    print(f"Quantizing {input_model} to INT8...")
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QInt8,
    )
    print(f"✓ INT8 quantized model saved to: {output_model}")


def convert_fp16(input_model: str, output_model: str):
    """
    Convert model to FP16 (half precision).
    This reduces model size by half and can improve inference speed.
    """
    print(f"Converting {input_model} to FP16...")
    
    # Load the model
    model = onnx.load(input_model)
    
    # Convert to FP16
    from onnxconverter_common import float16
    model_fp16 = float16.convert_float_to_float16(model)
    
    # Save the converted model
    onnx.save(model_fp16, output_model)
    print(f"✓ FP16 model saved to: {output_model}")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output ONNX model path")
    parser.add_argument(
        "--mode",
        choices=["int8", "fp16"],
        default="int8",
        help="Quantization mode: int8 (dynamic quantization) or fp16 (half precision)",
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Perform quantization
    try:
        if args.mode == "int8":
            quantize_int8(args.input, args.output)
        elif args.mode == "fp16":
            convert_fp16(args.input, args.output)
        
        # Report file sizes
        input_size = Path(args.input).stat().st_size / (1024 * 1024)
        output_size = Path(args.output).stat().st_size / (1024 * 1024)
        reduction = (1 - output_size / input_size) * 100
        
        print(f"\nFile size comparison:")
        print(f"  Original: {input_size:.2f} MB")
        print(f"  Quantized: {output_size:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
