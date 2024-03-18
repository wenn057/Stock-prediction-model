# Stock-prediction-model

import yfinance as yf

df = yf.Ticker("2330.tw").history(period="10y")
df

df = df.filter(["Close"])
df = df.rename(columns={"Close" : "GT"})
df

# 日k圖
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(df["GT"], linewidth=1)
plt.show()

# 把數值縮小到(0,1)之間的數值
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(df.values)
scaled_prices

# 以收集歷史股價60天為基準，預測第61天股價
import numpy as np

MOVING_MIN_SIZE = 60

all_x, all_y = [], []
for i in range(len(scaled_prices) - MOVING_MIN_SIZE):
  x = scaled_prices[i:i+MOVING_MIN_SIZE]
  y = scaled_prices[i+MOVING_MIN_SIZE]
  all_x.append(x)
  all_y.append(y)


all_x, all_y = np.array(all_x), np.array(all_y)

all_x.shape
all_y.shape

train_x.shape
train_y.shape
test_x.shape
test_y.shape

# construct AI model

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.summary()

model.compile(optimizer="adam", loss="mean_squared_error")

# 設定early stopping以避免模型會overfitting
# 劃分20%當作測試數據集
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(train_x, train_y,
     validation_split=0.2,
     callbacks=[callback],
     epochs=1000)

preds = model.predict(test_x)
preds

preds = scaler.inverse_transform(preds)
preds

train_df = df[:train_ds_size+MOVING_MIN_SIZE]
test_df = df[train_ds_size+MOVING_MIN_SIZE:]
test_df = test_df.assign(Predict=preds)

# 列印圖表結果，發現很接近模型是複製前一天股價
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(train_df["GT"],linewidth=2)
plt.plot(test_df["GT"],linewidth=2)
plt.plot(test_df["Predict"],linewidth=1)
plt.legend(["Train","GT","Predict"])
plt.show()

plt.plot(train_df["GT"][-20:],linewidth=2)
plt.plot(test_df["GT"][:30],linewidth=2)
plt.plot(test_df["Predict"][:30],linewidth=1)
plt.legend(["Train","GT","Predict"])
plt.show()

# 檢驗模型與實際複製前一天股價哪個比較合理
# 結論是模型趨近於複製前一天股價
# 股票預測模型實際利用在現實的功效不大
test_df = test_df.assign(Shifted=test_df["GT"].shift(1))
test_df.iat[0, -1] = train_df.iat[-1,-1]
test_df

from sklearn.metrics import mean_squared_error

predict_rmse = mean_squared_error(test_df["GT"], test_df["Predict"], squared=False)
predict_cvrmse = predict_rmse / test_df["GT"].mean()*100
predict_cvrmse

shifted_rmse = mean_squared_error(test_df["GT"], test_df["Shifted"], squared=False)
shifted_cvrmse = shifted_rmse / test_df["GT"].mean()*100
shifted_cvrmse
