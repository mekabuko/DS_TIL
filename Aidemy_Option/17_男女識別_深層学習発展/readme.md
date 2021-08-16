# サマリ
- CNNの実際の使用イメージを確認
- 正直、16.CNNを用いた...の劣化版のような。。
- 簡潔に要点のみまとまっているといえばそうかも。。

# 1. データ収集〜データクレンジング
## 1.1 データ収集
### 1.1.1 データ収集
- 学習に使えるデータセットが世には溢れている。
    - keras 標準の MNIST, Cifar10
    - lfs(http://vis-www.cs.umass.edu/lfw/): 男女識別のための画像データセット
    - imdb(https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
## 1.2 データクレンジング
### 1.2.1 画像の読み込みと表示
- サンプル（引用）
```
# OpenCVを用いるためにCV2とNumPyをインポート。画像表示用にMatplotlibを使用。
import cv2
import matplotlib.pyplot as plt
import numpy as np

# OpenCVでの読み込み。引数にはパスを指定。
img = cv2.imread('./6100_gender_recognition_data/male/Aaron_Eckhart_0001.jpg')

# OpenCVで読み込んだデータをmatplotlibで表示できるように加工。b,g,r順となっているのでr,g,bに変形（cv2.COLOR_BGR2RGB)
my_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# matplotlibを用いて画像を出力
plt.imshow(my_img)
plt.show()
```
### 1.2.2 画像の縮小
- あまり大きいと学習効率が悪い
- また、同じサイズでないと学習しづらい場合がある
- そのため、統一したサイズにリサイズしてから使用する
- サンプル（引用）
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

#読み込み
img = cv2.imread('./6100_gender_recognition_data/male/Aaron_Eckhart_0001.jpg')

# cv2でリサイズ、BGR2RGB調整。
my_img = cv2.resize(img, (50, 50))
my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)

# 確認
plt.imshow(my_img)
plt.show()

# 保存するなら、imwriteを使って画像を保存
# cv2.imwrite('resize.jpg', my_img)
```
### 1.2.3 色空間の変換
- 機械学習ではモノクロ変換もよく使用される。
- サンプル（引用）
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./6100_gender_recognition_data/male/Aaron_Eckhart_0001.jpg')

# cvtColorの第二引数をRGB2GRAYにして、モノクロに。
my_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

plt.imshow(my_img)
plt.gray()
plt.show()
```

# 2.CNN
## 2.1 CNN
### 2.1.1 モデルの定義
#### 初期化
```
model = Sequential()
```
#### レイヤ追加
```
model.add(XXX)
```
#### 全結合層
- units: 出力の次元数
```
Dense(units, input_dim=784)
```
#### 畳み込み層
```
Conv2D(filters = 32, kernel_size=(2, 2), strides=(1, 1), padding="same", input_shape=(28, 28, 3))
```
#### プーリング層
```
MaxPooling2D(pool_size=(2, 2), strides=None)
```
#### 平坦化層
```
Flatten()
```
#### 活性化関数
- sigmoid, relu, softmax などが使用可能
```
Activation('sigmoid')
```
#### 構造出力
```
model.summary()
```
#### サンプル（引用）
```
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(2, 2), strides=(1, 1), padding="same", input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
model.add(Conv2D(filters = 32, kernel_size=(2, 2), strides=(1, 1), padding="same", input_shape=(16,16,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
model.add(Flatten())
model.add(Dense(256, input_dim=2048))
model.add(Activation('sigmoid'))
model.add(Dense(128,input_dim=256))
model.add(Activation('sigmoid'))
model.add(Dense(10, input_dim = 128))
model.add(Activation('sigmoid'))
model.summary()
```
### 2.1.2 モデルの運用
#### パラメータ指定とコンパイル
- optimizer: 学習率の更新方法
- loss: 損失関数
- metrics: 精度指標
```
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
```
#### 学習の実行
```
model.fit(X_train, y_train, batch_size=32, epochs=10)
```
#### 予測
```
pred = np.argmax(model.predict(data[0]))
```
#### サンプル（引用）
```
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train[:1000]
X_test = X_test[:1000]
y_train = to_categorical(y_train)[:1000]
y_test = to_categorical(y_test)[:1000]

model = Sequential()
model.add(Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=2) # 実行時間を短くするためにエポックを小さくした。

for i in range(1):
    x = X_test[i]
    plt.imshow(x)
    plt.show()
    pred = np.argmax(model.predict(x.reshape(1,32,32,3)))
    print(pred)
```
### 2.1.3 転移学習
#### 転移元（VGG16)サンプル
```
from tensorflow.keras.applications.vgg16 import VGG16

input_tensor = Input(shape=(32, 32, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
```
#### 転移先サンプル
```
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(10, activation='softmax'))
#入力はvgg.input, 出力は, top_modelにvgg16の出力を入れたもの
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
```
#### 学習時に、転移元の重みは固定する
```
# modelの19層目までがvggのモデルなので、19層目までを固定する
for layer in model.layers[:19]:
    layer.trainable = False
```
#### コンパイル
- 転移学習の最適化は、SGD(`optimizers.SGD`)が良い
```
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
```
#### 全体サンプル
```
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train[:1000]
X_test = X_test[:1000]
y_train = to_categorical(y_train)[:1000]
y_test = to_categorical(y_test)[:1000]

# vgg16のインスタンスの生成
#---------------------------
input_tensor = Input(shape=(32, 32, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
#---------------------------

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(10, activation='softmax'))

# モデルの連結
#---------------------------
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
#---------------------------

# vgg16の重みの固定
#---------------------------
for layer in model.layers[:19]:
    layer.trainable = False
#---------------------------

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=1)

for i in range(1):
    x = X_test[i]
    plt.imshow(x)
    plt.show()
    pred = np.argmax(model.predict(x.reshape(1,32,32,3)))
    print(pred)
```

### 2.1.4 精度向上のための手法
#### ドロップアウト層の導入
- 過学習を防ぐための手法として、ドロップアウト層を追加することができる
    - 入力データをランダムに除去する（0で上書きする）手法
    - より汎用的な(学習データのみに依存しない)特徴を学習するようになることが期待できる
- `rate`: ドロップアウトする割合
```
model.add(Dropout(rate=0.5))
```
#### 分析精度の表示
```
score = model.evaluate(X_test, y_test)
```