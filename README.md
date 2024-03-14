# Stock-prediction-model

import yfinance as yf

df = yf.Ticker("2330.tw").history(period="10y")
df

df = df.filter(["Close"])
df = df.rename(columns={"Close" : "GT"})
df

import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(df["GT"], linewidth=1)
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(df.values)
scaled_prices

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
