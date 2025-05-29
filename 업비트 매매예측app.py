import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# -----------------------------
# 📌 업비트 시세 데이터 가져오기
# -----------------------------
def get_coin_data(ticker="KRW-BTC", count=200):
    url = f"https://api.upbit.com/v1/candles/days"
    headers = {"Accept": "application/json"}
    params = {"market": ticker, "count": count}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df = df[["candle_date_time_kst", "trade_price"]]
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# -----------------------------
# 📌 모델 학습 함수
# -----------------------------
def train_model(prices):
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

    X, y = [], []
    for i in range(20, len(prices_scaled)):
        X.append(prices_scaled[i-20:i])
        y.append(prices_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    return model, scaler, prices_scaled

# -----------------------------
# 📌 Streamlit 앱 구성
# -----------------------------
st.set_page_config(page_title="업비트 코인 예측 앱", layout="centered")
st.title("📈 업비트 코인 예측 앱")
st.markdown("원하는 코인을 선택하고 다음 시점 가격을 예측하세요!")

coin_options = {
    "비트코인 (BTC)": "KRW-BTC",
    "이더리움 (ETH)": "KRW-ETH",
    "리플 (XRP)": "KRW-XRP",
    "솔라나 (SOL)": "KRW-SOL"
}
selected_coin = st.selectbox("코인 선택", list(coin_options.keys()))
ticker = coin_options[selected_coin]

# 데이터 불러오기 및 모델 학습
with st.spinner("데이터 불러오는 중..."):
    df = get_coin_data(ticker)
    prices = df["close"].values
    model, scaler, prices_scaled = train_model(prices)

# 차트 출력
st.line_chart(df.set_index("date")["close"])

# 예측 버튼 클릭 시
if st.button("다음 시점 가격 예측"):
    latest_data = prices_scaled[-20:].reshape(1, 20, 1)
    predicted = model.predict(latest_data)
    predicted_price = scaler.inverse_transform(predicted)
    current_price = prices[-1]

    st.success(f"📌 예측 가격: {predicted_price[0][0]:,.0f} 원")
    st.info(f"현재 가격: {current_price:,.0f} 원")

    if predicted_price[0][0] > current_price:
        st.warning("📈 상승 예측 → 매수 고려!")
    else:
        st.error("📉 하락 예측 → 관망 또는 매도 고려!")

