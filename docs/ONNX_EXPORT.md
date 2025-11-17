# ONNX Export Guide

このドキュメントでは、MaAIモデルをONNX形式にエクスポートする正確な手順を説明します。エクスポートの一貫性を保ち、推論結果の再現性を確保するため、必ずこの手順に従ってください。

## 目次

- [前提条件](#前提条件)
- [環境構築](#環境構築)
- [VAP-MC モデルのエクスポート](#vap-mc-モデルのエクスポート)
- [VAP-BC モデルのエクスポート](#vap-bc-モデルのエクスポート)
- [エクスポート結果の検証](#エクスポート結果の検証)
- [トラブルシューティング](#トラブルシューティング)

---

## 前提条件

- Python 3.12以上
- Git
- 十分なディスク容量（モデルとキャッシュで約1GB）

---

## 環境構築

### 1. リポジトリのクローン

```bash
git clone https://github.com/kounoike/MaAI.git
cd MaAI
```

### 2. エクスポート専用の仮想環境を作成

**重要**: エクスポートには PyTorch 2.2.1 の legacy exporter が必要です。メイン環境とは分離してください。

```bash
# エクスポート専用の仮想環境を作成
python3 -m venv .venv-export

# 仮想環境を有効化
source .venv-export/bin/activate  # Linux/macOS
# または
.venv-export\Scripts\activate  # Windows
```

### 3. 必要なパッケージをインストール

```bash
# PyTorch 2.2.1 (legacy exporter用)
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# その他の依存関係
pip install -e .
pip install huggingface_hub>=0.23
pip install numpy>=1.26,<2.0
```

### 4. 環境の確認

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
# 出力: PyTorch version: 2.2.1+cpu (または 2.2.1)

python -c "from maai.models.vap import VapGPT; print('MaAI modules: OK')"
# 出力: MaAI modules: OK
```

---

## VAP-MC モデルのエクスポート

VAP-MC（Voice Activity Projection - Multi-Condition）モデルは、ノイズに強いターンテイキング予測とVADを提供します。

### 基本的なエクスポート（20Hz、20秒コンテキスト、日本語 Kyotoデータセット）

```bash
python tools/export_vap_mc_onnx.py \
  --out vap_mc_jp_kyoto_20hz_20s.onnx \
  --frame-rate 20 \
  --context-sec 20 \
  --language jp_kyoto \
  --opset 17
```

### パラメータの説明

- `--out`: 出力ONNXファイルのパス
- `--frame-rate`: フレームレート（Hz）。10 または 20 を推奨
  - 10Hz: 100ms/フレーム、低レイテンシ用
  - 20Hz: 50ms/フレーム、高精度用
- `--context-sec`: コンテキスト長（秒）。通常は 20.0
- `--language`: 学習済みモデルの言語
  - `jp_kyoto`: 日本語 Kyoto Corpus
  - `en_kyoto`: 英語 Kyoto Corpus
  - `ch_kyoto`: 中国語 Kyoto Corpus
  - `tri_kyoto`: 多言語（英・中・日）Kyoto Corpus
  - `jp`: 日本語（旧データセット）
  - `en`: 英語（旧データセット）
  - `ch`: 中国語（旧データセット）
  - `tri`: 多言語（旧データセット）
- `--opset`: ONNX opset バージョン。**必ず 17 を指定**
  - opset 18 では ReduceMean op のエラーが発生します
- `--device`: エクスポートに使うデバイス（cpu または cuda）。デフォルトは cpu
- `--cpc-model`: CPCエンコーダのチェックポイントパス
  - デフォルト: `~/.cache/cpc/60k_epoch4-d0f474de.pt`
  - 初回実行時に自動ダウンロードされます

### 他のバリエーション例

```bash
# 10Hz、20秒、日本語
python tools/export_vap_mc_onnx.py \
  --out vap_mc_jp_kyoto_10hz_20s.onnx \
  --frame-rate 10 \
  --context-sec 20 \
  --language jp_kyoto \
  --opset 17

# 20Hz、20秒、英語
python tools/export_vap_mc_onnx.py \
  --out vap_mc_en_kyoto_20hz_20s.onnx \
  --frame-rate 20 \
  --context-sec 20 \
  --language en_kyoto \
  --opset 17

# ローカルの学習済み重みを使用
python tools/export_vap_mc_onnx.py \
  --out vap_mc_custom.onnx \
  --frame-rate 20 \
  --context-sec 20 \
  --language jp_kyoto \
  --local-weights /path/to/state_dict.pt \
  --opset 17
```

### エクスポート時の出力例

```
########################################
Load pretrained CPC
########################################
Froze EncoderCPC!
########################################
Load pretrained CPC
########################################
Froze EncoderCPC!
freeze encoder
Froze EncoderCPC!
Froze EncoderCPC!
Exported ONNX to: vap_mc_jp_kyoto_20hz_20s.onnx
```

エクスポートが成功すると、約 23MB の ONNX ファイルが生成されます。

---

## VAP-BC モデルのエクスポート

VAP-BC（Voice Activity Projection - Backchannel）モデルは、バックチャネル（相槌）の予測に特化しています。

```bash
python tools/export_vap_bc_onnx.py \
  --out vap_bc_jp_10hz_20s.onnx \
  --frame-rate 10 \
  --context-sec 20 \
  --language jp \
  --opset 17
```

パラメータは VAP-MC と同様です。

---

## エクスポート結果の検証

エクスポートしたONNXモデルが正しく動作することを確認します。

### 1. Python での ONNX 推論テスト

```bash
python tools/vap_mc_infer_onnx.py \
  --model vap_mc_jp_kyoto_20hz_20s.onnx \
  --wav1 example/wav_sample/jpn_inoue_16k.wav \
  --wav2 example/wav_sample/jpn_sumida_16k.wav \
  --dump-csv tmp/onnx_test.csv
```

正常に動作すれば、VAD区間とフレーム数が表示されます。

### 2. PyTorch モデルとの比較検証

```bash
# PyTorch モデルで推論（比較用のベースライン）
python tools/vap_mc_infer_torch.py \
  --wav1 example/wav_sample/jpn_inoue_16k.wav \
  --wav2 example/wav_sample/jpn_sumida_16k.wav \
  --language jp_kyoto \
  --frame-rate 20 \
  --context-sec 20 \
  --dump-csv tmp/pytorch_baseline.csv

# ONNX モデルで推論
python tools/vap_mc_infer_onnx.py \
  --model vap_mc_jp_kyoto_20hz_20s.onnx \
  --wav1 example/wav_sample/jpn_inoue_16k.wav \
  --wav2 example/wav_sample/jpn_sumida_16k.wav \
  --dump-csv tmp/onnx_test.csv \
  --compare-csv tmp/pytorch_baseline.csv
```

**期待される結果**:
- `max_abs` < 3e-05（最大絶対誤差）
- `mean_abs` < 1e-06（平均絶対誤差）
- VAD区間の開始・終了時刻が ±1フレーム以内で一致

これらの基準を満たせば、エクスポートは成功です。

### 3. Rust での推論テスト（オプション）

Rustの推論環境がある場合：

```bash
cd rust-example
cargo build --release

./target/release/vap-mc-hysteresis \
  --model ../vap_mc_jp_kyoto_20hz_20s.onnx \
  --wav1 ../example/wav_sample/jpn_inoue_16k.wav \
  --wav2 ../example/wav_sample/jpn_sumida_16k.wav \
  --dump-csv ../tmp/rust_test.csv

# PyTorchと比較
cd ..
python -c "
import numpy as np
from pathlib import Path
a = np.loadtxt('tmp/pytorch_baseline.csv', delimiter=',', skiprows=1)
b = np.loadtxt('tmp/rust_test.csv', delimiter=',', skiprows=1)
diff = np.abs(a[:, 2:] - b[:, 2:])
print(f'Max diff: {diff.max():.6e}')
print(f'Mean diff: {diff.mean():.6e}')
print(f'Match (tol=1e-4): {np.allclose(a[:, 2:], b[:, 2:], atol=1e-4, rtol=1e-4)}')
"
```

---

## トラブルシューティング

### エラー: ReduceMean operator のエラー

```
RuntimeError: Unsupported: ONNX export of operator ReduceMean with axes attribute
```

**原因**: opset 18 以降で ReduceMean の仕様が変更されました。

**解決策**: `--opset 17` を必ず指定してください。

### エラー: .tolist() が ONNX トレースを妨げる

```
TracerWarning: Converting a tensor to a Python list might cause the trace to be incorrect
```

**原因**: `VapGPT.forward()` 内で `.tolist()` が呼ばれています。

**解決策**: エクスポートスクリプトは既に `VapMCExportWrapper` を使用して回避しています。スクリプトをそのまま使用してください。

### エラー: CPC モデルのダウンロード失敗

```
FileNotFoundError: CPC checkpoint not found
```

**解決策**: 
1. インターネット接続を確認
2. 手動でダウンロード:
   ```bash
   mkdir -p ~/.cache/cpc
   wget https://github.com/facebookresearch/CPC_audio/raw/master/models/60k_epoch4-d0f474de.pt \
     -O ~/.cache/cpc/60k_epoch4-d0f474de.pt
   ```

### エラー: PyTorch と ONNX の出力が大きく異なる

**症状**: `max_abs` が 0.01 以上、または VAD 区間が完全に異なる

**原因**:
1. 異なる PyTorch バージョンでエクスポートした
2. CPCモデルのチェックポイントが異なる
3. コードの変更が反映されていない

**解決策**:
1. PyTorch 2.2.1 を使用していることを確認
2. この手順書に従って最初から環境を構築し直す
3. `--cpc-model` パスが正しいことを確認
4. エクスポート後、必ず検証手順を実行

### 警告: Batch size に関する警告

```
UserWarning: Exporting a model to ONNX with a batch_size other than 1
```

**影響**: この警告は無視して構いません。推論時は常に batch_size=1 を使用するためです。

---

## ベストプラクティス

1. **専用環境の使用**
   - エクスポート専用の仮想環境（`.venv-export`）を使用
   - PyTorch 2.2.1 を固定

2. **バージョン管理**
   - エクスポートしたONNXファイルには日付とコミットハッシュをタグ付け
   - 例: `vap_mc_jp_kyoto_20hz_20s_20251117_abc1234.onnx`

3. **必ず検証**
   - エクスポート後、必ず PyTorch モデルとの比較を実行
   - CI/CD パイプラインに検証を組み込むことを推奨

4. **ドキュメント化**
   - エクスポート時のパラメータと環境をログに記録
   - 例:
     ```bash
     python tools/export_vap_mc_onnx.py \
       --out vap_mc_jp_kyoto_20hz_20s.onnx \
       --frame-rate 20 \
       --context-sec 20 \
       --language jp_kyoto \
       --opset 17 \
       2>&1 | tee export_log_$(date +%Y%m%d_%H%M%S).txt
     ```

5. **依存関係の固定**
   - `requirements-export.txt` を作成して依存関係を固定:
     ```bash
     pip freeze > requirements-export.txt
     ```

---

## 環境の削除

エクスポート作業が完了したら、専用環境を削除できます：

```bash
# 仮想環境を無効化
deactivate

# 仮想環境を削除
rm -rf .venv-export
```

---

## 参考情報

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [PyTorch ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)
- [MaAI GitHub Repository](https://github.com/kounoike/MaAI)

---

## 変更履歴

- 2025-11-17: 初版作成。PyTorch 2.2.1 + opset 17 での確実なエクスポート手順を確立
