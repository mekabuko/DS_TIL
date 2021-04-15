#　サマリ
- tensorflowに組み込まれたKerasを用いれば、非常に簡単にDNN(３層以上のニューラルネットワーク)を実装できる
- ハイパーパラメータのチューニングは重要。すぐ発散したり、収束しなくなったりする。過学習も起こしやすい。

# メモ
## モデル生成コードサンプル(引用)
- `model = Sequential()`: インスタンス作成
- `model.add(Dense(256, input_dim=784))`: 層の追加
- `model.add(Activation("sigmoid"))`: 活性化関数の定義
- `model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])`: 学習処理を設定して完成
```
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# (◯, 28, 28)のデータを(◯, 784)に次元削減します。(簡単のためデータ数を減らします)
shapes = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], shapes)[:6000]
X_test = X_test.reshape(X_test.shape[0], shapes)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

model = Sequential()
# 入力ユニット数は784, 1つ目の全結合層の出力ユニット数は256
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))

# 2つ目の全結合層の出力ユニット数は128。活性化関数はrelu。
model.add(Dense(128))
model.add(Activation("relu"))

# 3つ目の全結合層（出力層）の出力ユニット数は10
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# モデル構造の出力
plot_model(model, show_layer_names=True, dpi=150)

# モデル構造の可視化
image = plt.imread("model125.png")
plt.figure(dpi=150)
plt.imshow(image)
plt.axis('off')
plt.show()
```
## モデル生成コードサンプル(引用)
- `model.fit(X_train, y_train, verbose=1, epochs=3)`
    - `verbose`: １ を指定した場合は学習の進捗度合いを出力し、 0 の場合は出力しない
    - `epochs`:同じデータセットで行う学習の回数
```
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784)[:6000]
X_test = X_test.reshape(X_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

model = Sequential()
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, verbose=1, epochs=3)

#acc, val_accのプロット
plt.plot(history.history["accuracy"], label="accuracy", ls="-", marker="o")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()
```

## モデルの評価(引用)
- `score = model.evaluate(X_test, y_test, verbose=1)`
```
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784)[:6000]
X_test = X_test.reshape(X_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

model = Sequential()
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, verbose=1)

score = model.evaluate(X_test, y_test, verbose=1)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
```

## モデルによる予測(引用)
-  `model.predict(x, batch_size=None, verbose=0, steps=None)`
    - `x`: 入力データ。Numpy配列の形式。
    - `batch_size`: 整数。デフォルトは32。
    - `verbose`: 進行状況メッセージ出力モード。0または1。
    - `steps`: 評価ラウンド終了を宣言するまでの総ステップ数（サンプルのバッチ）。None（デフォルト値）の場合は無視。
- 出力は、分類の場合、n次元の配列になっているので、最大の値となるインデックスが予測分類クラスになる
    - `np.argmax(x, axis=1)`
```
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784)[:6000]
X_test = X_test.reshape(X_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

model = Sequential()
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))

# テストデータの最初の10枚を表示します
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i].reshape((28,28)), "gray")
plt.show()

# X_testの最初の10枚の予測されたラベルを表示しましょう
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print("予測値 :" + str(pred))
```

## 深層学習のチューニング
### ドロップアウト
- `model.add(Dropout(rate=0.3))` でドロップアウトを追加できる
    - 指定した割合のニューロンをランダムに削除し、過学習を防ぐ

### 活性化関数の利用
- 活性化関数を使うことで、モデルに非線形性を持たせて、線形分離不可能なデータにも対応できる様にする
#### シグモイド関数
- 0-1 を取るので、極端な出力が少ない
#### ReLU(ランプ関数)
- 0以上の値で、極端な出力も可能

### 損失関数
- 出力データと教師データの差を評価する関数
#### 平均二乗誤差
- 主に回帰モデルに向く
- 最小値の付近ではゆっくりと更新される＝収束しやすい
#### クロスエントロピー誤差
- 主に分類に特化
- 正解ラベルと、予測値の誤差の対数の総和

### 最適化関数
- 誤差関数を重みで微分した値を使って、各重みの更新量を決定するが、その際に使用するのが最適化関数
    - 学習率、エポック数、過去の重みの更新量などを使って、更新量を決定スル

### 学習率
- 各層の重みを一度にどの程度更新するかを決めるパラメータ
- 適切な値を設定しないと、収束しなかったり、学習効率が悪くなる

### ミニバッチ学習
- １データで重みを更新すると、偏ったデータの影響を大きく受けてしまう可能性がある
- なので、ある程度まとめたデータで重みの更新を行う
    - そのひとまとまりのサイズを「バッチサイズ」と呼ぶ
    - バッチサイズ＝１: オンライン学習（確率的勾配降下法）
    - バッチサイズ＝データ全て: バッチ学習（最急降下法）
    - バッチサイズ＝2〜全部未満：ミニバッチ学習

### 反復学習
- 同じデータを何回か使用して学習を繰り返す
- 何回同じデータを使うのかを「エポック数」と呼ぶ
    