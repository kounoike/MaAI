# MaAI ONNX Inference Example (Rust)

Rustで書かれたVAP-BC（相槌予測）モデルのONNX推論サンプルです。

## 必要なもの

- Rust 1.70以上（`rustc --version`で確認）
- エクスポート済みのONNXモデル（例: `vap_bc_jp_10hz_20s.onnx`）
- 16kHz モノラル WAVファイル2本（ユーザとシステムの音声）

## セットアップ

### 1. 依存関係のビルド

```bash
cd rust-example
cargo build --release
```

初回ビルドには数分かかります（ONNX Runtime等の大きなクレートをコンパイル）。

### 2. ONNXモデルの準備

プロジェクトルートで以下を実行してモデルをエクスポート：

```bash
# .venv-export環境を使用
source .venv-export/bin/activate
PYTHONPATH=src python tools/export_vap_bc_onnx.py \
  --out ./vap_bc_jp_10hz_20s.onnx \
  --device cpu --frame-rate 10 --context-sec 20 --opset 17
deactivate
```

## 使い方

### 基本的な実行

```bash
cargo run --release -- \
  --model ../vap_bc_jp_10hz_20s.onnx \
  --wav1 path/to/user_audio.wav \
  --wav2 path/to/system_audio.wav
```

### 詳細出力（全フレームの確率を表示）

```bash
cargo run --release -- \
  --model ../vap_bc_jp_10hz_20s.onnx \
  --wav1 user.wav \
  --wav2 system.wav \
  --verbose
```

### ビルド済みバイナリで実行

```bash
./target/release/vap-bc-inference \
  --model ../vap_bc_jp_10hz_20s.onnx \
  --wav1 user.wav \
  --wav2 system.wav
```

## 出力例

```
Loading ONNX model from: ../vap_bc_jp_10hz_20s.onnx
Loading WAV file 1 (user): user.wav
  Loaded 160000 samples
Loading WAV file 2 (system): system.wav
  Loaded 160000 samples

Running inference...
Output shape: [1, 100, 1]
Number of frames: 100

=== Summary Statistics ===
Max probability: 0.923451
Min probability: 0.012345
Avg probability: 0.234567

=== Top 5 backchannel timing candidates ===
Rank 1: Frame 45 (t=4.50s) - probability: 0.923451
Rank 2: Frame 67 (t=6.70s) - probability: 0.891234
Rank 3: Frame 23 (t=2.30s) - probability: 0.765432
Rank 4: Frame 89 (t=8.90s) - probability: 0.654321
Rank 5: Frame 12 (t=1.20s) - probability: 0.543210

✓ Inference completed successfully
```

## コマンドライン引数

- `--model, -m`: ONNXモデルファイルのパス（必須）
- `--wav1, -1`: 1番目のWAVファイル（ユーザ/話者1）のパス（必須）
- `--wav2, -2`: 2番目のWAVファイル（システム/話者2）のパス（必須）
- `--verbose, -v`: 全フレームの確率を表示（オプション）

## 注意事項

### 音声フォーマット

- **サンプリングレート**: 16kHz推奨（他のレートでも動作しますが警告が出ます）
- **チャンネル数**: モノラル（1チャンネル）推奨（ステレオの場合は左チャンネルのみ使用）
- **ビット深度**: 16bit整数または32bit floatに対応

### フレームレート

このサンプルは10Hzのフレームレート（100msごと）を想定しています。
異なるフレームレートでエクスポートした場合は、出力時刻計算を調整してください：

```rust
// 例: 20Hz（50ms）の場合
let time_sec = *frame_idx as f32 * 0.05;
```

### パフォーマンス

- 初回実行時はモデルの初期化に数秒かかります
- 推論速度は音声の長さに依存しますが、通常はリアルタイムの数十〜数百倍速です
- CPUのみで動作します（CUDA版が必要な場合は`ort`の機能フラグを変更）

## トラブルシューティング

### ビルドエラー

```bash
# キャッシュをクリアして再ビルド
cargo clean
cargo build --release
```

### 実行時エラー: "Failed to load ONNX model"

- ONNXファイルのパスが正しいか確認
- ファイルが破損していないか確認（サイズは約23MB）

### 実行時エラー: "Failed to open WAV file"

- WAVファイルのパスが正しいか確認
- ファイル形式が標準的なWAV（RIFF）か確認

### 警告: "Expected 16kHz but got ..."

モデルは16kHzで学習されているため、異なるサンプリングレートでは精度が落ちます。
`ffmpeg`で変換できます：

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 output_16k.wav
```

## ライセンス

このサンプルコードはMITライセンスです。
使用するONNXモデルのライセンスは各モデルのREADMEを参照してください。
