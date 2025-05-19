import matplotlib.pyplot as plt
import json
import os

# === ファイルパス ===
RESULT_PATH = "results/anomaly_results.json"
OUTPUT_IMG = "results/anomaly_score_plot.png"
THRESHOLD = 0.3  # test.py 中的判定基準と合わせる

# === JSON 読み込み ===
if not os.path.exists(RESULT_PATH):
    raise FileNotFoundError(f"{RESULT_PATH} が見つかりません。先に test.py を実行してください。")

with open(RESULT_PATH, 'r') as f:
    data = json.load(f)

indices = [d["index"] for d in data]
scores = [d["score"] for d in data]
labels = [d["label"] for d in data]

# === 可視化 ===
plt.figure(figsize=(12, 5))
plt.plot(indices, scores, label="Anomaly Score", color="blue")
plt.axhline(y=THRESHOLD, color="red", linestyle="--", label=f"Threshold = {THRESHOLD}")
plt.xlabel("Sequence Index")
plt.ylabel("Score")
plt.title("Anomaly Score per Test Sequence")
plt.grid(True)
plt.legend()

# === 保存と表示 ===
os.makedirs("results", exist_ok=True)
plt.savefig(OUTPUT_IMG)
plt.show()

print(f"✅ 保存済み: {OUTPUT_IMG}")
