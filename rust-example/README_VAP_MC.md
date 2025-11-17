# VAP-MC (Voice Activity Projection - Multi-channel) ONNX Inference

VAP-MCモデルをONNX形式でエクスポートし、Rustで推論を実行するための手順とツールです。

## VAP-MCとは

VAP-MCは、ノイズに強い音声活動投影(Voice Activity Projection)モデルで、以下の機能を提供します:

- **VAD (Voice Activity Detection)**: 各チャンネルの音声活動を検出
- **ターンテイキング予測**: 話者交代のタイミングを予測
- **ノイズロバスト性**: 雑音環境でも安定した性能

### 出力

モデルは3つのテンソルを出力します:

1. **p_now** `[B, F, 2]`: 現在のターンテイキング確率 (話者1→話者2, 話者2→話者1)
2. **p_future** `[B, F, 2]`: 未来のターンテイキング確率
3. **vad** `[B, F, 2]`: 各チャンネルの音声活動検出スコア (0.0-1.0)

- `B`: バッチサイズ (通常1)
- `F`: フレーム数 (10Hz = 100msごと)
- 次元2: [チャンネル1, チャンネル2]

## ONNXモデルのエクスポート

### 1. エクスポート環境のセットアップ

```bash
# PyTorch 2.2.1でエクスポート環境を作成 (legacy ONNX exporterを使用)
python -m venv .venv-export
source .venv-export/bin/activate
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install onnx
```

### 2. VAP-MCモデルのエクスポート

```bash
source .venv-export/bin/activate
PYTHONPATH=src python tools/export_vap_mc_onnx.py \
  --out ./vap_mc_jp_kyoto_20hz_20s.onnx \
  --device cpu \
  --frame-rate 20 \
  --context-sec 20 \
  --language jp \
  --opset 17
```

**パラメータ説明:**

- `--out`: 出力ONNXファイルのパス
- `--device`: `cpu` または `cuda`
- `--frame-rate`: フレームレート (20Hz推奨、10Hzも可能)
- `--context-sec`: コンテキスト長 (秒)。20秒推奨
- `--language`: `jp` (日本語) または `en` (英語)
- `--opset`: ONNX opsetバージョン。17を推奨 (18はReduceMeanで問題あり)

### 3. エクスポート結果

成功すると以下のようなファイルが生成されます:

```bash
$ ls -lh vap_mc_jp_kyoto_20hz_20s.onnx
-rw-r--r-- 1 user user 23M Nov 17 16:58 vap_mc_jp_kyoto_20hz_20s.onnx
```

モデルサイズは約23MBです。

## Rust推論の実行

### 1. ビルド

```bash
cd rust-example
cargo build --release --bin vap-mc-inference
```

### 2. 実行

基本的な使い方:

```bash
./target/release/vap-mc-inference \
  --model ../vap_mc_jp_kyoto_20hz_20s.onnx \
  --wav1 ../example/wav_sample/jpn_inoue_16k.wav \
  --wav2 ../example/wav_sample/jpn_sumida_16k.wav
```

詳細出力 (フレームごとの値を表示):

```bash
./target/release/vap-mc-inference \
  --model ../vap_mc_jp_kyoto_20hz_20s.onnx \
  --wav1 ../example/wav_sample/jpn_inoue_16k.wav \
  --wav2 ../example/wav_sample/jpn_sumida_16k.wav \
  --verbose
```

VADしきい値を変更:

```bash
./target/release/vap-mc-inference \
  --model ../vap_mc_jp_kyoto_20hz_20s.onnx \
  --wav1 ../example/wav_sample/jpn_inoue_16k.wav \
  --wav2 ../example/wav_sample/jpn_sumida_16k.wav \
  --vad-threshold 0.3
```

### 3. 出力例

```
Loading ONNX model from: ./vap_mc_jp_kyoto_20hz_20s.onnx
Loading WAV file 1 (user): ./example/wav_sample/jpn_inoue_16k.wav
  Loaded 4250624 samples (265.66s)
Loading WAV file 2 (system): ./example/wav_sample/jpn_sumida_16k.wav
  Loaded 4250624 samples (265.66s)

Running VAP-MC inference...
Warning: Input audio is longer than model expects (4250624 > 320000). Truncating to first 20.00s
Number of frames: 400

=== VAD Statistics ===
VAD threshold: 0.50
Channel 1 (user):   max=0.909, avg=0.446
Channel 2 (system): max=0.280, avg=0.218

=== Speech Segments (Channel 1 - User) ===
  Segment 1: 0.05s - 0.85s (duration: 0.80s)
  Segment 2: 1.00s - 1.20s (duration: 0.20s)
  Segment 3: 1.30s - 2.45s (duration: 1.15s)

=== Speech Segments (Channel 2 - System) ===
No speech detected above threshold 0.50

=== Turn-Taking Analysis ===

Top 5 turn-taking opportunities (Channel 1 → Channel 2):
  Rank 1: Frame 6 (t=0.30s) - probability: 0.964
  Rank 2: Frame 4 (t=0.20s) - probability: 0.961
  Rank 3: Frame 8 (t=0.40s) - probability: 0.958
  Rank 4: Frame 42 (t=2.10s) - probability: 0.948
  Rank 5: Frame 40 (t=2.00s) - probability: 0.940

✓ VAP-MC inference completed successfully
```

## リアルタイム音声セグメンテーションへの応用

VAP-MCモデルは、リアルタイムストリーミング音声からVoice Activityのある部分を切り出すのに適しています:

### 実装例の考え方

1. **チャンク処理**: 50ms単位で音声をチャンク化 (20Hz)
2. **スライディングウィンドウ**: 過去20秒のコンテキストを保持
3. **VAD判定**: `vad` 出力を使って音声活動を検出
4. **セグメント抽出**: VADスコアがしきい値を超えている区間を抽出

```rust
// 擬似コード
let vad_threshold = 0.5;
let mut segments = Vec::new();
let mut in_segment = false;
let mut segment_start = 0.0;

for (frame_idx, vad_score) in vad_output.iter().enumerate() {
    let time = frame_idx as f32 / 20.0; // 20Hz (50ms per frame)
    
    if vad_score >= vad_threshold && !in_segment {
        // 音声開始
        segment_start = time;
        in_segment = true;
    } else if vad_score < vad_threshold && in_segment {
        // 音声終了
        segments.push((segment_start, time));
        in_segment = false;
    }
}
```

### 推奨設定

- **フレームレート**: 20Hz (50msごと) - 高精度でリアルタイム処理も可能
- **コンテキスト長**: 20秒 (320000サンプル@16kHz) - 十分な会話コンテキスト
- **VADしきい値**: 0.3-0.5 - 用途に応じて調整

## トラブルシューティング

### エクスポート時のエラー

**エラー**: `Unrecognized attribute: axes for operator ReduceMean`

**解決策**: `--opset 17` を指定してください。opset 18ではReduceMean演算子の互換性に問題があります。

```bash
python tools/export_vap_mc_onnx.py ... --opset 17
```

**エラー**: `.tolist()` に関するTracerWarning

**解決策**: これは想定内です。`export_vap_mc_onnx.py`の`VapMCExportWrapper`が`.tolist()`を回避するように実装されています。

### 推論時のエラー

**エラー**: 音声ファイルが長すぎる警告

**解決策**: これは正常です。モデルは固定長(20秒)入力を期待し、長い音声は自動的に切り詰められます。リアルタイム処理では、スライディングウィンドウで処理してください。

**エラー**: VADが検出されない

**解決策**: 
- `--vad-threshold` を下げてみてください (例: 0.3)
- `--verbose` で各フレームのVAD値を確認してください
- 入力音声が16kHz モノラルであることを確認してください

## 参照

- [MaAI GitHub](https://github.com/maai-kyoto/MaAI)
- [Hugging Face: maai-kyoto/vap_mc_jp_kyoto](https://huggingface.co/maai-kyoto/vap_mc_jp_kyoto)
- [ONNX Runtime for Rust](https://github.com/pykeio/ort)
