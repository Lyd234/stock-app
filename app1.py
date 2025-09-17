import streamlit as st
from datetime import datetime
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Default date range constants
START = "2015-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

# Streamlit page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

def inject_css():
    css = """
    :root{
      --bg:#0f1724;
      --card:#0b1220;
      --accent:#00b4d8;
      --muted:#94a3b8;
      --glass: rgba(255,255,255,0.03);
    }
    .stApp {
      background: linear-gradient(180deg, #071126 0%, #0b1220 100%);
      color: #e6eef8;
      font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .block-container{
      padding: 1.5rem 2rem;
      max-width: 1300px;
    }
    .stSidebar .block-container { padding: 1rem 1rem; }
    div.stButton > button {
      background: linear-gradient(90deg, #06b6d4, #3b82f6);
      color: white; border-radius: 8px;
    }
    .stDataFrame, .stPlotlyChart, .stImage, .stChart {
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));
      padding: 12px;
    }
    footer { visibility: hidden; }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_css()

# App title and sidebar controls
st.title("Stock Price Prediction")
st.sidebar.header("Controls")
user_input = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start date", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.today())
run_button = st.sidebar.button("Load & Predict")

@st.cache_data
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if not df.empty:
        df = df.reset_index()
    return df

if not run_button:
    st.info("Set options in the sidebar and click 'Load & Predict'.")
    st.stop()

with st.spinner("Downloading data..."):
    df = load_data(user_input, start_date.isoformat(), end_date.isoformat())

if df.empty:
    st.error("No data returned for that ticker / date range.")
    st.stop()

df['Date'] = pd.to_datetime(df['Date'])

st.subheader(f"Data for {user_input} — {start_date} to {end_date}")
st.write(df.describe())

if 'Close' not in df.columns:
    st.error("Downloaded data missing 'Close' column.")
    st.stop()

# Plot 1: Closing Price chart 
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(10, 5), facecolor='white')
ax = fig.add_subplot()
ax.plot(df['Date'], df['Close'], color='#1f77b4', linewidth=1.5)
ax.set_title(f"{user_input} Closing Price", color='black')   
ax.set_xlabel("Date", color='black')
ax.set_ylabel("Price", color='black')
ax.tick_params(colors='black')
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig)

# Plot 2: Moving averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig2 = plt.figure(figsize=(10, 5), facecolor='white')
ax2 = fig2.add_subplot()
ax2.plot(df['Date'], df['Close'], label='Close', color='#1f77b4')
ax2.plot(df['Date'], ma100, label='MA100', color='#ff7f0e')
ax2.plot(df['Date'], ma200, label='MA200', color='#2ca02c')
ax2.legend()
ax2.set_title(f"{user_input} Close with MAs", color='black')
ax2.tick_params(colors='black')
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig2)

# Prepare train/test and model
train_size = int(len(df) * 0.70)
data_train = pd.DataFrame(df['Close'].iloc[:train_size]).reset_index(drop=True)
data_test = pd.DataFrame(df['Close'].iloc[train_size:]).reset_index(drop=True)

if len(data_train) < 101:
    st.error("Not enough training data (need >100 rows).")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_array = scaler.fit_transform(data_train)

model_path = 'keras_model.keras'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please upload it to your repo.")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model '{model_path}': {e}")
    st.stop()

past_100_days = data_train.tail(100).reset_index(drop=True)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])   
    y_test.append(input_data[i, 0])

if len(x_test) == 0:
    st.warning("Not enough data to build test windows (need >100 rows after combining).")
    st.stop()

x_test = np.array(x_test)
y_test = np.array(y_test)   

y_predicted = model.predict(x_test, verbose=0)

if y_predicted.ndim == 3:
    y_predicted = y_predicted[:, -1, 0]
else:
    y_predicted = y_predicted.reshape(-1)

y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1)).reshape(-1)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

dates_for_final = pd.concat([
    df['Date'].iloc[train_size-100:train_size].reset_index(drop=True),
    df['Date'].iloc[train_size:].reset_index(drop=True)
], ignore_index=True)

pred_dates = pd.to_datetime(dates_for_final.iloc[100:].reset_index(drop=True))

st.subheader("Predictions vs Original")
fig3 = plt.figure(figsize=(12, 6), facecolor='white')
ax3 = fig3.add_subplot()
ax3.plot(pred_dates, y_test, 'b', label='Original Price')
ax3.plot(pred_dates, y_predicted, 'r', label='Predicted Price')
ax3.set_title("Predictions vs Original", color='black')
ax3.set_xlabel('Date', color='black')
ax3.set_ylabel('Price', color='black')
ax3.legend()
ax3.tick_params(colors='black')
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig3)

pred_df = pd.DataFrame({"date": pred_dates, "true": y_test, "predicted": y_predicted})
st.subheader("Prediction samples")
st.dataframe(pred_df.head(50))

csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button("Download predictions (csv)", csv, file_name=f"{user_input}_predictions.csv", mime="text/csv")
