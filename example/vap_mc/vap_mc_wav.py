"""
This script is an example of using WAV files with the VapGPT model.
Processes two WAV files (for two speakers) and displays real-time turn-taking predictions.
"""

import sys
import os
import argparse
import time

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# This allows you to import modules from the src directory without pip installing the package.
# Uncomment the line below if you need to run this script directly without installing the package.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput


def benchmark_mode(args):
    """Run in benchmark mode: measure inference performance without visualization."""
    
    print(f"Loading WAV files...")
    print(f"  Speaker 1: {args.wav1}")
    print(f"  Speaker 2: {args.wav2}")
    
    wav1 = MaaiInput.Wav(args.wav1)
    wav2 = MaaiInput.Wav(args.wav2)
    
    print(f"\nInitializing VAP-MC model...")
    print(f"  Language: {args.language}")
    print(f"  Frame rate: {args.frame_rate}Hz")
    print(f"  Context: {args.context_sec}s")
    print(f"  Device: {args.device}")
    print(f"  KV Cache: {'Enabled' if args.use_kv_cache else 'Disabled'}")
    
    maai = Maai(
        mode="vap_mc",
        lang=args.language,
        frame_rate=args.frame_rate,
        context_len_sec=args.context_sec,
        audio_ch1=wav1,
        audio_ch2=wav2,
        device=args.device,
        use_kv_cache=args.use_kv_cache,
    )
    
    print("\nStarting inference...")
    maai.start()
    
    frame_times = []
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            result = maai.get_result()
            frame_time = time.time() - frame_start
            
            frame_times.append(frame_time * 1000)  # Convert to ms
            frame_count += 1
            
            # Print progress every 50 frames
            if frame_count % 50 == 0:
                avg_time = sum(frame_times[-50:]) / min(50, len(frame_times))
                print(f"  Processed {frame_count} frames, avg: {avg_time:.2f}ms/frame")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        # WAV file ended
        pass
    finally:
        maai.stop()
    
    total_time = time.time() - start_time
    
    # Print statistics
    if frame_times:
        import numpy as np
        frame_times = np.array(frame_times)
        frame_interval_ms = 1000.0 / args.frame_rate
        
        print("\n" + "="*60)
        print("Benchmark Results (Maai class - Real streaming)")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Frame rate: {args.frame_rate}Hz (interval: {frame_interval_ms:.1f}ms)")
        print(f"\nPer-frame inference time:")
        print(f"  First frame: {frame_times[0]:.2f}ms")
        print(f"  Average: {frame_times.mean():.2f}ms")
        print(f"  Min: {frame_times.min():.2f}ms")
        print(f"  Max: {frame_times.max():.2f}ms")
        print(f"  Std: {frame_times.std():.2f}ms")
        
        print(f"\nReal-time capability:")
        avg_time = frame_times.mean()
        if avg_time < frame_interval_ms:
            margin = frame_interval_ms - avg_time
            print(f"  ✅ CAN run real-time (margin: {margin:.2f}ms, {frame_interval_ms / avg_time:.2f}x)")
        else:
            deficit = avg_time - frame_interval_ms
            print(f"  ❌ CANNOT run real-time (deficit: {deficit:.2f}ms, {avg_time / frame_interval_ms:.2f}x slower)")
        print("="*60)


def visualization_mode(args):
    """Run in visualization mode: display real-time predictions."""
    
    print(f"Loading WAV files...")
    print(f"  Speaker 1: {args.wav1}")
    print(f"  Speaker 2: {args.wav2}")
    
    wav1 = MaaiInput.Wav(args.wav1)
    wav2 = MaaiInput.Wav(args.wav2)
    
    # Choose output visualization
    if args.output == "console":
        output = MaaiOutput.ConsoleBar()
    elif args.output == "gui_bar":
        output = MaaiOutput.GuiBar()
    elif args.output == "gui_plot":
        output = MaaiOutput.GuiPlot()
    else:
        raise ValueError(f"Unknown output mode: {args.output}")
    
    print(f"\nInitializing VAP-MC model...")
    print(f"  Language: {args.language}")
    print(f"  Frame rate: {args.frame_rate}Hz")
    print(f"  Context: {args.context_sec}s")
    print(f"  Device: {args.device}")
    
    maai = Maai(
        mode="vap_mc",
        lang=args.language,
        frame_rate=args.frame_rate,
        context_len_sec=args.context_sec,
        audio_ch1=wav1,
        audio_ch2=wav2,
        device=args.device,
    )
    
    print("\nStarting inference (press Ctrl+C to stop)...")
    maai.start()
    
    try:
        while True:
            result = maai.get_result()
            output.update(result)
    except KeyboardInterrupt:
        print("\nEnding the script.")
    finally:
        maai.stop()


def main():
    parser = argparse.ArgumentParser(
        description="VAP-MC inference from WAV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark mode (measure performance)
  python vap_mc_wav.py --wav1 speaker1.wav --wav2 speaker2.wav --benchmark

  # Visualization mode (console bar)
  python vap_mc_wav.py --wav1 speaker1.wav --wav2 speaker2.wav --output console

  # Custom settings
  python vap_mc_wav.py --wav1 speaker1.wav --wav2 speaker2.wav \\
    --language jp_kyoto --frame-rate 5 --context-sec 3.0 --benchmark
        """
    )
    
    # Input files
    parser.add_argument("--wav1", required=True, help="Path to first WAV file (speaker 1)")
    parser.add_argument("--wav2", required=True, help="Path to second WAV file (speaker 2)")
    
    # Model parameters
    parser.add_argument("--language", default="jp", choices=["jp", "jp_kyoto", "en", "en_kyoto", "ch", "ch_kyoto", "tri", "tri_kyoto"],
                        help="Language variant (default: jp)")
    parser.add_argument("--frame-rate", type=int, default=10, choices=[5, 10, 20],
                        help="Frame rate in Hz (default: 10)")
    parser.add_argument("--context-sec", type=float, default=20.0,
                        help="Context length in seconds (default: 20.0)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    parser.add_argument("--use-kv-cache", action="store_true",
                        help="Use KV cache for faster inference (default: True)")
    parser.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false",
                        help="Disable KV cache")
    parser.set_defaults(use_kv_cache=True)
    
    # Mode selection
    parser.add_argument("--benchmark", action="store_true",
                        help="Run in benchmark mode (no visualization)")
    parser.add_argument("--output", default="console", choices=["console", "gui_bar", "gui_plot"],
                        help="Output visualization mode (default: console)")
    
    args = parser.parse_args()
    
    # Check if WAV files exist
    if not os.path.exists(args.wav1):
        print(f"Error: WAV file not found: {args.wav1}")
        sys.exit(1)
    if not os.path.exists(args.wav2):
        print(f"Error: WAV file not found: {args.wav2}")
        sys.exit(1)
    
    # Run in appropriate mode
    if args.benchmark:
        benchmark_mode(args)
    else:
        visualization_mode(args)


if __name__ == "__main__":
    main()
