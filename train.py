import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, ConvLSTM2D, TimeDistributed, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# === パラメータ設定 ===
SEQUENCE_LEN = 10
HEIGHT = 64
WIDTH = 64
CHANNELS = 1
EPOCHS = 50
BATCH_SIZE = 4

# === データ読み込み ===
print("Loading training data...")
X_train = np.load('dataset/training.npy')  # shape: (N, 10, 64, 64, 1)
print(f"Training samples: {X_train.shape}")

# === モデル構築 ===
def build_model():
    inputs = Input(shape=(SEQUENCE_LEN, HEIGHT, WIDTH, CHANNELS))

    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)

    x = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
    x = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)

    x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))(x)
    outputs = TimeDistributed(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# === モデル訓練 ===
model = build_model()
model.summary()

# モデル保存用ディレクトリ
os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint('models/conv_lstm_model.h5', monitor='loss', save_best_only=True, verbose=1)

print("Training started...")
model.fit(X_train, X_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          callbacks=[checkpoint])

print("Training complete. Model saved to models/conv_lstm_model.h5")
