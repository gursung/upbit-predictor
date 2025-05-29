import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 랜덤 입력 데이터 (100개 샘플, 10개 시퀀스, 1개 특징)
X = np.random.rand(100, 10, 1)
y = np.random.rand(100, 1)

# 모델 생성
model = Sequential()
model.add(LSTM(32, input_shape=(10, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 모델 학습
model.fit(X, y, epochs=5)

# 모델 저장
model.save("model.h5")

print("✅ model.h5 파일이 생성되었습니다.")
