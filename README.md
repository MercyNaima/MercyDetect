# 🎥 ConvLSTMベースの動画異常検知システム

本プロジェクトは、軽量かつ実用性の高い動画異常検知システムを構築するために、DeepEYEをベースとして改良されたものです。3D-CNNとConvLSTMを組み合わせた自己符号化モデルにより、時空間の動きパターンを学習し、正常な動作との再構成誤差から異常を判定します。

---

## 📁 ディレクトリ構成

```
DeepEYE_test/
├── data/
│   ├── training_videos/      ← 正常行動の動画
│   └── testing_videos/       ← 異常または未知の行動を含む動画
│
├── dataset/                  
│   ├── training.npy
│   └── test.npy
│
├── models/
│   └── conv_lstm_model.h5
│
├── results/
│   ├── anomaly_results.json
│   └── anomaly_score_plot.png
│
├── vid2array.py              ← 動画→フレーム列への変換
├── train.py                  ← ConvLSTMモデルの学習
├── test.py                   ← 異常判定（推論）
├── visualize.py              ← スコアの可視化
└── README.md                 ← 説明文書（本ファイル）
```

---

## ⚙️ 実行手順

### 1. ライブラリのインストール

```bash
pip install -r requirements.txt
```

または個別にインストール：

```bash
pip install tensorflow scikit-image matplotlib opencv-python
```

---

### 2. データの準備

- 正常動画：`data/training_videos/`
- テスト動画：`data/testing_videos/`

---

### 3. フレーム列の生成

```bash
python vid2array.py
```

---

### 4. モデルの学習

```bash
python train.py
```

---

### 5. 異常の推論とスコア出力

```bash
python test.py
```

---

### 6. 異常スコアの可視化

```bash
python visualize.py
```

出力：`results/anomaly_score_plot.png`

---

## 🔍 技術のポイント

- ConvLSTM自己符号化モデルによる時空間特徴の抽出
- SSIMとMSEを組み合わせた異常スコア評価
- 教師なしで正常行動のみ学習、異常を再構成困難として検出
- スクリプト分離型設計（前処理・学習・推論・可視化）

---

## ✍️ 開発者・出典

本研究は、卒業研究の一環として開発したものであり、[DeepEYE](https://github.com/MathGeniusMan/DeepEYE)プロジェクトをベースに構築されています。

---

## 📜 ライセンス

本ソフトウェアは教育・研究目的に限り使用可能です。商用利用を禁じ、利用時には出典の明記をお願いします。
