# メモ
## 1 CNNを用いた画像認識の基礎
### 1.1 深層学習画像認識
#### 1.1.1 画像認識
- 画像認識：画像や映像に映る文字や顔などといった 「モノ」 や 「特徴」 を検出する技術
    - 画像の分類
    - モノの位置の推定
    - 画像の領域分割
### 1.2 CNN
#### 1.2.1 CNNの概要
- CNN: 人間の脳の視覚野と似た構造を持つ 「畳み込み層」 という層を使って特徴抽出を行うニューラルネットワーク
- 畳み込み層
    - 全結合層と同じように特徴の抽出を行う層
    - 全結合層とは違い2次元のままの画像データを処理し特徴の抽出が行える
    - 線や角といった 2次元的な特徴 を抽出するのに優れる
- プーリング層
    - 畳み込み層 から得た情報を縮約し、最終的に画像の分類などを行う
#### 1.2.2 畳み込み層
- 入力データの一部分に注目しその部分画像の特徴を調べる層
- どのような特徴に注目すれば良いかは、自動的に学習される
- (例)顔認識をするCNNの場合
    - 入力層に近い畳み込み層では線や点といった低次元な概念の特徴に注目
    - 出力層に近い層では目や鼻といった高次元な概念の特徴に注目
- 特徴は、プログラム内部では フィルター(カーネル) と呼ばれる重み行列として扱われ、各特徴につき一つのフィルターが用いられる
#### 1.2.3 プーリング層
- 畳み込み層の出力を縮約しデータの量を削減する層
    - 特徴マップの部分区間の最大値を取ったり（ Maxプーリング ）、あるいは平均を取ったり（ Averageプーリング ）する
- データの無駄を削減し、情報の損失を抑えながらデータを圧縮することが可能
- 特徴が元画像の平行移動などでも影響を受けないようなロバスト性を与える
#### 1.2.4 CNNの実装
- TensorFlow による実装例
##### 1. インスタンス作成
```
model = Sequential()
```
##### 2. 畳み込み層の追加
```
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
```
##### 3. プーリング層の追加
```
model.add(MaxPooling2D(pool_size=(2, 2)))
```
##### 4. 平滑化: 多次元のデータを全結合層に入力する場合には、データを一次元に平滑化する必要がある
```
mmodel.add(Flatten())
```
##### 5. 全結合層
```
model.add(Dense(256))
```
##### 6. コンパイル
```
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
```
##### 7. モデル構造の出力
```
model.summary()
```
#### 1.2.5 CNNを用いた分類（MNIST）
- コード例（引用）
```
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
import matplotlib.pyplot as plt

# データのロード
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 今回は全データのうち、学習には300、テストには100個のデータを使用します。
# Convレイヤーは4次元配列を受け取ります。（バッチサイズx縦x横xチャンネル数）
# MNISTのデータはRGB画像ではなくもともと3次元のデータとなっているので予め4次元に変換します。
X_train = X_train[:300].reshape(-1, 28, 28, 1)
X_test = X_test[:100].reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train)[:300]
y_test = to_categorical(y_test)[:100]

# モデルの定義
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, y_train,
            batch_size=128,
            epochs=1,
            verbose=1,
            validation_data=(X_test, y_test))

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# データの可視化（テストデータの先頭の10枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape((28,28)), 'gray')
plt.suptitle("10 images of test data",fontsize=20)
plt.show()

# 予測（テストデータの先頭の10枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)

model.summary()
```
#### 1.2.6 CNNを用いた分類（CIFAR-10）
- CIFAR-10: 10種類のオブジェクトが映った画像のデータセット
    - 0: 飛行機
    - 1: 自動車
    - 2: 鳥
    - 3: 猫
    - 4: 鹿
    - 5: 犬
    - 6: 蛙
    - 7: 馬
    - 8: 船
    - 9: トラック
- 実装例(引用)
```
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# データのロード
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 今回は全データのうち、学習には300、テストには100個のデータを使用します
X_train = X_train[:300]
X_test = X_test[:100]
y_train = to_categorical(y_train)[:300]
y_test = to_categorical(y_test)[:100]


# モデルの定義
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# コンパイル
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# 重みデータ param_cifar10.hdf5 を読み込みます
# model.load_weights('./param_cifar10.hdf5')

# 学習
model.fit(X_train, y_train, batch_size=32, epochs=1)

# 学習によって得た重みを保存する場合は、 save_weights() メソッドを使います。本環境では実行できません。
# model.save_weights('param_cifar10.hdf5')

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# データの可視化（テストデータの先頭の10枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i])
plt.suptitle("10 images of test data",fontsize=20)
plt.show()

# 予測（テストデータの先頭の10枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)

model.summary()
```
### 1.3 ハイパーパラメータ
- Conv: 畳込み層
- Pool: プーリング層
#### 1.3.1 filters （Conv層）
- 生成する特徴マップの数 つまり 抽出する特徴の種類 を指定
- 小さすぎると学習できず、大きすぎると同じようなマップが増えて過学習につながる
#### 1.3.2 kernel_size （Conv層）
- カーネル(畳み込みに使用する重み行列)の大きさ 
- 程よい大きさでないと、特徴検出の精度が落ちる
#### 1.3.3 strides （Conv層）
- 特徴を抽出する間隔 、つまり カーネルを動かす距離
- 基本は小さくて良い（１）。
#### 1.3.4 padding （Conv層）
- 畳み込んだ時の画像の縮小を抑えるため、入力画像の周囲にピクセルを追加すること
- 一般的には、ゼロパディング(入力画像の周辺を画素値が0のピクセルで埋める)する
- 端のデータの特徴もよく考慮されるようになる
- 畳み込み演算の回数が増える
- 各層の入出力ユニット数の調整が行える
#### 1.3.5 pool_size （Pool層）
- 一度にプーリングを適用する領域のサイズ（プーリングの粗さ）
- 基本的に pool_size は2×2にすれば良い
    - pool_size を大きくすることで、位置に対するロバスト性が上がる（画像の中でオブジェクトが映る位置が多少変化しても出力が変化しないこと）とされる
#### 1.3.6 strides （Pool層）
- 特徴マップをプーリングする間隔
- kerasのPoolingレイヤーではstridesはデフォルトでpool_sizeと一致させるようになっている
#### 1.3.7 padding （Pool層）
- パディングの仕方を指定
- keras の MaxPooling2D 層では、 padding=valid, padding=same などのようにしてパディングの仕方を指定
    - valid: paddingなし
    - same 出力される特徴マップが入力のサイズと一致するように、入力にパディング実行
## 2 CNNを用いた画像認識の応用
### 2.1 データのかさ増し
#### 2.1.1 ImageDataGenerator
- 画像認識では、画像データとそのラベル（教師データ）の組み合わせが大量に必要
    - そのため、データの水増しが有効
    - 画像を反転したり、ずらしたりして新たなデータを作ることができる
- TensorFlow の ImageDataGenerator
```
datagen = ImageDataGenerator(rotation_range=0.,
                            width_shift_range=0.,
                            height_shift_range=0.,
                            shear_range=0.,
                            zoom_range=0.,
                            channel_shift_range=0,
                            horizontal_flip=False,
                            vertical_flip=False)
```
- パラメータ
    - rotation_range: ランダムに回転する回転範囲（単位degree）
    - width_shift_range: ランダムに水平方向に平行移動する、画像の横幅に対する割合
    - height_shift_range: ランダムに垂直方向に平行移動する、画像の縦幅に対する割合
    - shear_range: せん断の度合い。大きくするとより斜め方向に押しつぶされたり伸びたりしたような画像になる（単位degree）
    - zoom_range: ランダムに画像を圧縮、拡大させる割合。最小で 1-zoom_range まで圧縮され、最大で 1+zoom_rangeまで拡大される。
    - channel_shift_range: 入力がRGB3チャンネルの画像の場合、R,G,Bそれぞれにランダムな値を足したり引いたりする。(0~255)
    - horizontal_flip: Trueを指定すると、ランダムに水平方向に反転。
    - vertical_flip: Trueを指定すると、ランダムに垂直方向に反転。
- flow: numpyデータとラベルの配列を受け取り，拡張/正規化したデータのバッチを生成する関数
```
flow(x, y=None, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
```
- パラメータ
    - x: データ,4次元データである必要がある。グレースケールデータではチャネルを1、RGBデータではチャネルを3に。
    - y: ラベル
    - batch_size: 整数（デフォルト: 32）。データのバッチのサイズを指定する。
    - shuffle: 真理値（デフォルト: True）。データをシャッフルするかどうか。
    - save_to_dir: Noneまたは文字列（デフォルト: None)。生成された拡張画像を保存するディレクトリを指定できる（行ったことの可視化に有用です）
    - save_prefix: 文字列（デフォルト'')。画像を保存する際にファイル名に付けるプリフィックス（save_to_dirに引数が与えられた時のみ有効）
    - save_format: "png"または"jpeg"（save_to_dirに引数が与えられた時のみ有効)。デフォルトは"png"

### 2.2 正規化
#### 2.2.1 様々な正規化手法
- 正規化：データにある決まりに従って処理を行い、使いやすくすること
    - 明るさや色味の調整など
- 近年、深いニューラルネットワークモデルにおいて正規化はあまり必要ないとされることもある
- 正規化にはさまざまな手法がある
    - バッチ正規化(BN)
    - 主成分分析（PCA）
    - 特異値分解（SVD）
    - ゼロ位相成分分析（ZCA）
    - 局所的応答正規化（LRN）
    - 大域コントラスト正規化（GCN）
    - 局所コントラスト正規化（LCN）
#### 2.2.2 標準化
- 標準化: 個々の特徴を平均0、分散1にすること
- ImageDataGeneratorでの実装サンプル
    - samplewise_center=True : 平均0
    - samplewise_std_normalization=True : 分散1
```
# generator
datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
# 標準化
g = datagen.flow(X_train, y_train, shuffle=False)
```
#### 2.2.3 白色化
- 白色化: データの特徴間の相関を無くす処理
- 情報量の多いエッジ等を強調することができる
- ImageDataGeneratorでの実装サンプル
    - sfeaturewise_center=True, zca_whitening=True: ゼロ位相成分分析
```
# ジェネレーターの生成
datagen = ImageDataGenerator(featurewise_center=True, zca_whitening=True)
# 白色化
datagen.fit(X_train)
g = datagen.flow(X_train, y_train, shuffle=False)
```
#### 2.2.4 バッチ正規化
- バッチ正規化: ミニバッチ学習の際にバッチごとに標準化を行うこと
- TensorFlowでは以下でモデルに組みことができる
```
model.add(BatchNormalization())
```
- データの前処理としてだけではなく、中間層の出力に適用することもできる
- 特に、活性化関数ReLUなど、出力値の範囲が限定されてない関数の出力に対してバッチ正規化を使うと、学習がスムーズに進みやすくなり大きな効果を発揮する
### 2.3 転移学習
#### 2.3.1 VGG
- TensorFlowでは、ImageNet（120万枚，1000クラスからなる巨大な画像のデータセット）で学習した画像分類モデルとその重みをダウンロードし、転移学習に利用できる
- 一例として、VGGモデルは、2014年のILSVRCで2位になったネットワークモデル
    - 1000クラスの分類モデルなので出力ユニットは1000個あるが、最後の全結合層は使わずに途中までの層を特徴抽出のための層として使うことで、転移学習に用いることができる
    - 入力画像のサイズも気にしないでよい。これは、VGG16モデルは、畳み込み層のカーネルサイズは 3x3 と小さく、また padding='same' とされており、極端に入力画像が小さくない限り13層を経て抽出される特徴の数が一定数確保されるため。
#### 2.3.2 CNNを用いた分類（cifar）

- 実装例（引用）
```
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train[:300]
X_test = X_test[:100]
y_train = to_categorical(y_train)[:300]
y_test = to_categorical(y_test)[:100]

#input_tensorを定義
input_tensor = Input(shape=(32, 32, 3))
#vgg16を定義
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
#自作モデルを定義
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='sigmoid'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))
#vgg16とtop_modelを連結（inputに転移元を使う）
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
#転移元モデルの重みは更新しないように固定する。ここでは19層目まで。
for layer in model.layers[:19]:
    layer.trainable = False
#コンパイル。転移学習では、最適化関数はSDGが良いとされる
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


model.load_weights('./5100_cnn_data/param_vgg.hdf5')

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=1)

# 以下の式でモデルの重みを保存することが可能
#model.save_weights('./5100_cnn_data/param_vgg.hdf5')

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# データの可視化（テストデータの先頭の10枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i])
plt.suptitle("10 images of test data",fontsize=16)
plt.show()

# 予測（テストデータの先頭の10枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)

model.summary()
```