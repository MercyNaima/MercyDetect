import cv2
import os
import numpy as np

# === パラメータ設定 ===
sequence_len = 10
resize_dim = (64, 64)
channels = 1

def video_to_sequences(video_path, seq_len=sequence_len):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, resize_dim)
        frames.append(resized)
    cap.release()

    # シーケンス化
    sequences = []
    for i in range(len(frames) - seq_len + 1):
        seq = frames[i:i + seq_len]
        seq = np.array(seq).astype('float32') / 255.0  # 正規化
        seq = np.expand_dims(seq, axis=-1)  # (10, 64, 64, 1)
        sequences.append(seq)
    return sequences

def process_folder(folder_path, save_name):
    all_sequences = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            full_path = os.path.join(folder_path, filename)
            print(f"Processing: {full_path}")
            seqs = video_to_sequences(full_path)
            all_sequences.extend(seqs)

    all_sequences = np.array(all_sequences)
    print(f"Total sequences: {len(all_sequences)}, Shape: {all_sequences.shape}")
    np.save(f'dataset/{save_name}.npy', all_sequences)
    print(f"Saved to dataset/{save_name}.npy")

if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    process_folder('data/train_videos', 'training')
    process_folder('data/test_videos', 'test')
