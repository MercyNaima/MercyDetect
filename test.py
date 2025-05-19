import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from skimage.metrics import structural_similarity as ssim
import os
import json

# === パラメータ ===
SEQUENCE_LEN = 10
HEIGHT, WIDTH, CHANNELS = 64, 64, 1
THRESHOLD = 0.3  # しきい値（調整可）

# === モデル読み込み ===
print("Loading model...")
model = load_model('models/conv_lstm_model.h5', custom_objects={'mse': MeanSquaredError()})

# === テストデータ読み込み ===
print("Loading test data...")
test_data = np.load('dataset/test.npy')  # (N, 10, 64, 64, 1)
print(f"Test samples: {test_data.shape}")

# === 異常スコア計算（MSE + SSIM）===
def compute_anomaly_score(original, reconstructed):
    mse_total, ssim_total = 0, 0
    for i in range(SEQUENCE_LEN):
        orig = original[i].squeeze()
        recon = reconstructed[i].squeeze()
        mse = np.mean((orig - recon) ** 2)
        ssim_score = ssim(orig, recon, data_range=1.0)
        mse_total += mse
        ssim_total += ssim_score
    mse_avg = mse_total / SEQUENCE_LEN
    ssim_avg = ssim_total / SEQUENCE_LEN
    score = 0.7 * mse_avg + 0.3 * (1 - ssim_avg)  # 重み付きスコア
    return score

# === 推論開始 ===
print("Starting inference...")
results = []

for idx, sequence in enumerate(test_data):
    input_seq = np.expand_dims(sequence, axis=0)  # shape: (1, 10, 64, 64, 1)
    reconstructed = model.predict(input_seq, verbose=0)
    score = compute_anomaly_score(sequence, reconstructed[0])
    label = "Anomaly" if score > THRESHOLD else "Normal"
    print(f"[{idx:03d}] Score = {score:.4f} → {label}")
    results.append({
        "index": int(idx),
        "score": float(score),
        "label": label
    })

# === 結果保存 ===
os.makedirs('results', exist_ok=True)
with open('results/anomaly_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("All done. Results saved to results/anomaly_results.json")
