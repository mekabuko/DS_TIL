# サマリ
- LSTM の使い方と、そのための前処理などをざっと。

# メモ
## 目的
- LSTM を使って、シャンパンの打ち上げを予想するモデルを作成する

## データの読み込み
- LSTM, 今回は float 形で読み込む

## データの分割
- 時系列だと、データの順序がとても重要になる＝ランダムに分けられない。
```
# `dataset`の67%をトレーニングデータ、33%をテストデータに分割
train_size = int(len(dataset) * 0.67)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
```

## データのスケーリング
- 正規化：最低値と最大値を決めて、そのスケールに変換
- 標準化：平均０、分散１となるように変換
- 注意点：スケーリングの基準は、トレーニングデータのみ。テストデータは使わない
    - トレーニングデータの中に、テストデータの除法が含まれる形になってしまうため

```
from sklearn.preprocessing import MinMaxScaler

# データのスケーリング（正規化）を行います
# 最小値が0, 最大値が1となるようにスケーリング方法を定義します
scaler = MinMaxScaler(feature_range=(0, 1))
# `train`のデータを基準にスケーリングするようパラメータを定義します
scaler_train = scaler.fit(train)
# パラメータを用いて`train`データをスケーリングします
train = scaler_train.transform(train)
# パラメータを用いて`test`データをスケーリングします
test = scaler_train.transform(test)
```

## 正解ラベルの作成
- 時系列モデルでは `Yt = f(Yt-1, Yt-2, ..., Yt-k)` というモデルを構成する
- その為、X = [Yt-1, ..., Yt-k], Y = [Yt] となるような変換を行う
```
# 入力データ・正解ラベルを作成する関数を定義
def create_dataset(dataset, look_back):
    data_X, data_Y = [], []
    for i in range(look_back, len(dataset)):
        data_X.append(dataset[i-look_back:i, 0])
        data_Y.append(dataset[i, 0])
    return np.array(data_X), np.array(data_Y)
```

## データ整形
- LSTM で分析するためには、行数×変数数×カラム数の3次元の行列に変換する必要がある
    - 行数：データの総数
    - 変数数：1セットのデータに含まれる要素数
    - カラム数：扱うデータの種別数
```
# 3次元のnumpy.ndarrayに変換
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
```

## モデルの構築
- keras で簡単に使える
```
# LSTMモデルを作成
model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(look_back, 1), return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(1)) # 全結合層
# モデルをコンパイル
model.compile(loss='mean_squared_error', optimizer='adam')
# 訓練
model.fit(train_X, train_Y, epochs=15, batch_size=1, verbose=15)
```

## コードサンプル全体
```
# 何も変更せずに実行してください
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 入力データ・正解ラベルを作成する関数を定義
def create_dataset(dataset, look_back):
    data_X, data_Y = [], []
    for i in range(look_back, len(dataset)):
        data_X.append(dataset[i-look_back:i, 0])
        data_Y.append(dataset[i, 0])
    return np.array(data_X), np.array(data_Y)


# データセットの読み込み
dataframe = pd.read_csv('./6160_sales_forecast_data/monthly-champagne-sales-in-1000s.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# `dataset`をトレーニングデータ、テストデータに分割
train_size = int(len(dataset) * 0.67)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# データのスケーリング（正規化）
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_train = scaler.fit(train)
train = scaler_train.transform(train)
test = scaler_train.transform(test)

# 入力データと正解ラベルを作成
look_back = 3
train_X, train_Y = create_dataset(train, look_back)
test_X, test_Y = create_dataset(test, look_back)

# データの整形
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

# LSTMモデルの作成と訓練
model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(look_back, 1), return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_Y, epochs=15, batch_size=1, verbose=2)

# 予測データの作成
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# スケールしたデータを元に戻す
train_predict = scaler_train.inverse_transform(train_predict)
train_Y = scaler_train.inverse_transform([train_Y])
test_predict = scaler_train.inverse_transform(test_predict)
test_Y = scaler_train.inverse_transform([test_Y])

# 予測精度の計算
train_score = math.sqrt(mean_squared_error(train_Y[0], train_predict[:, 0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = math.sqrt(mean_squared_error(test_Y[0], test_predict[:, 0]))
print('Test  Score: %.2f RMSE' % (test_score))

# プロットのためのデータ整形
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(look_back*2):len(dataset), :] = test_predict

# データのプロット
plt.title("monthly-champagne-sales-in-1000s")
plt.xlabel("time(month)")
plt.ylabel("Earnings")
# 読み込んだままのデータをプロットします
plt.plot(dataset, label='dataset')
# トレーニングデータから予測した値をプロットします
plt.plot(train_predict_plot, label='train_predict')
# テストデータから予測した値をプロットします
plt.plot(test_predict_plot, label='test_predict')

plt.legend(loc='lower right')
plt.show()
```