import streamlit as st
import pyupbit
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.title("업비트 코인 예측기 (랜덤포레스트 기반)")

coin = st.selectbox("코인 선택", pyupbit.get_tickers(fiat="KRW"))
df = pyupbit.get_ohlcv(coin, interval="day", count=200)

if df is None or df.empty:
    st.error("데이터를 불러오지 못했습니다.")
    st.stop()

df['label'] = df['close'].shift(-1)
df = df.dropna()

X = df[['open', 'high', 'low', 'volume']]
y = df['label']

model = RandomForestRegressor()
model.fit(X, y)
predicted = model.predict(X)

st.subheader("📈 실제 vs 예측 종가")
plt.plot(y.values, label="실제")
plt.plot(predicted, label="예측")
plt.legend()
st.pyplot(plt)

last_price = df['close'].iloc[-1]
predict_price = model.predict([X.iloc[-1]])[0]

if predict_price > last_price * 1.02:
    st.success(f"📢 매수 추천! 내일 예측가: {predict_price:.0f}원")
elif predict_price < last_price * 0.98:
    st.warning(f"📉 매도 추천! 내일 예측가: {predict_price:.0f}원")
else:
    st.info(f"🔍 관망 추천. 내일 예측가: {predict_price:.0f}원")
