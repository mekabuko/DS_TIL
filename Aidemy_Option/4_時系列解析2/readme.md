# サマリ
- RNN/LSTMを使用すれば、時系列的に影響関係にあるデータの学習ができる
    - RNN には長期データを扱えない問題点があり、LSTMはそれを解決
- 自然言語処理も、前後の単語などを意識するという点で、時系列と同じ考えができる
    - 同じようにRNNなどが適用可能


# メモ
## ニューラルネットワークの設定値
- 中間層の数: 中間層の次元（ユニット数）の増加が大きくなりそうな時、１層追加する。
- 中間層のユニット数：(入力層 + 出力層) * (2/3) から始める。２層以上の中間層がある時には、１層目から最適化、２層目以降はそれに準じる。
- 出力層のユニット数：分類であれば分類したいカテゴリ数

## 過学習を防ぐ手法
- DropOut
```
# Kerasを用いたDrop Outの実装
from keras.layers.core import Dropout

# 出力のニューロン数32、活性化関数reluの層を定義
model.add(Dense(32, activation='relu'))

# 各層ごとにニューロン数、活性化関数を定義した後にDrop Outを指定
# p=Drop Outする確率、0.5が一般的な設定
model.add(Dropout(p))
```
- EarlyStopping
```
# Kerasを用いたEarly Stoppingの実装
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
                      monitor=‘val_loss’, # monitorは監視値（今回は誤差）
                      patience=10, # patienceは”過去どれだけの誤差を見るか”
                      verbose=1, # verboseは”学習進捗の表示”
                      mode=‘auto’) # modeは収束判定の規定（上限か下限か）、を決めるパラメータ
```

## RNN
- 過去の中間層のデータを、次の時点の入力として用いる
- 中間層から見て、過去データを入力の一つとして使用できるので、通常のインプットと同様に学習できる
- ただし、そのままだと勾配消失や勾配爆発をを起こすなどして、長時間データを処理できない問題がある
    - ざっくり、勾配とは「パラメータの更新量の指標」
        - 大きすぎれば学習が発散し、小さすぎれば学習に無影響になってしまう

## 勾配クリッピング
- 勾配爆発を解決するために、勾配が大きくなりすぎないように、勾配がある閾値を超えたら修正する、という処置を行う
    - 勾配消失には対応できない

## LSTM(Long Short Term Memory)
- RNN の問題を解決する、長期的な記憶を保持するための方法
- 中間層のRNNのセルを、LSTMブロックに置き換える形で実装
- 構成要素
    - CEC: 過去データを保持するユニット
    - 入力ゲート:前のユニットの入力の重みを調整するゲート
    - 出力ゲート:前のユニットの出力の重みを調整するゲート
    - 忘却ゲート:過去の情報が入っているCECの中身をどの程度残すか調整するゲート

## 実装サンプル
### 概要
- 作成するモデル:「重力未分離加速度情報と角速度情報から被験者の行動ラベルを予測する」モデル
- データセット
    - 被験者：ボランティア 30人
    - 年齢：19-48歳
    - データ収集デバイス：smartphone
    - デバイス上のセンサ：3軸加速度、3軸角速度
    - ラベリング手法：同時に撮影した映像からラベルを手動生成
    - データセット：学習データ:テストデータ = 7:3（被験者単位で分割）
    - データソース：https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- 特徴量データ
    - 重力未分離加速度情報
    - 角速度情報
- 予測対象のデータ: 予測対象としたい行動にラベルを振り分ける
    - WALKING
    - WALKING_UPSTAIRS
    - WALKING_DOWNSTAIRS
    - SITTING
    - STANDING
    - LAYING

### RNN実装例（引用）
```
# 分類を行うRNNモデルです
# -------------------- モジュールインポート部分 --------------------
# 必要なモジュールをインポートする
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

plt.style.use('ggplot')


# -------------------- データ読み込み部分 --------------------
# 特徴量に使用するデータの読み込みを指定する
# 正解ラベルのマスタを読み込む
activity_labels = pd.read_csv("./5130_rnn_lstm_data/activity_labels.txt",
                              sep=" ", header=None, names=["activity_id", "activity"])

# 通常は使用するデータからトレーニングデータとテストデータを分ける必要があるが、今回は準備してあるのでそのファイルを読み込む
# 読み込む行数を指定
nrows = 300
# トレーニングデータ
subject_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/subject_train.txt", sep=" ", header=None, nrows=nrows)
y_train = pd.read_csv("./5130_rnn_lstm_data/train/y_train.txt",
                      sep=" ", header=None, nrows=nrows)
total_acc_x_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/total_acc_x_train.txt", sep="\s+", header=None, nrows=nrows)
total_acc_y_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/total_acc_y_train.txt", sep="\s+", header=None, nrows=nrows)
total_acc_z_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/total_acc_z_train.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_x_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/body_gyro_x_train.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_y_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/body_gyro_y_train.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_z_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/body_gyro_z_train.txt", sep="\s+", header=None, nrows=nrows)

# テストデータ
subject_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/subject_test.txt", sep=" ", header=None, nrows=nrows)
y_test = pd.read_csv("./5130_rnn_lstm_data/test/y_test.txt",
                     sep=" ", header=None, nrows=nrows)
total_acc_x_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/total_acc_x_test.txt", sep="\s+", header=None, nrows=nrows)
total_acc_y_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/total_acc_y_test.txt", sep="\s+", header=None, nrows=nrows)
total_acc_z_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/total_acc_z_test.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_x_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/body_gyro_x_test.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_y_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/body_gyro_y_test.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_z_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/body_gyro_z_test.txt", sep="\s+", header=None, nrows=nrows)


# -------------------- 入力データの処理 --------------------
def convert_threeDarray_for_nn(df_list):

    # ニューラルネットに適した、3次元のnumpy.ndarrayにする
    # arrayを(サンプル数, シーケンス長, 次元数)の形に変換する
    array_list = []
    for df in df_list:
        ndarray = np.array(df)
        array_list.append(np.reshape(
            ndarray, (ndarray.shape[0], ndarray.shape[1], 1)))

    return np.concatenate(array_list, axis=2)


# トレーニングデータに含めたい特徴量データをリスト化する
train_df_list = [
    total_acc_x_train,
    total_acc_y_train,
    total_acc_z_train,
    body_gyro_x_train,
    body_gyro_y_train,
    body_gyro_z_train
]
# テストデータに含めたい特徴量データをリスト化する
test_df_list = [
    total_acc_x_test,
    total_acc_y_test,
    total_acc_z_test,
    body_gyro_x_test,
    body_gyro_y_test,
    body_gyro_z_test
]

# 作成した関数にリスト化したトレーニングデータとテストデータを渡す
X_train = convert_threeDarray_for_nn(train_df_list)
X_test = convert_threeDarray_for_nn(test_df_list)


# -------------------- 正解ラベルの処理 --------------------
# カテゴリ数を`n_classes`へ格納
n_classes = len(activity_labels.activity_id.unique())

# numpyのarrayに変換し、1 ~ 6の数値を0 ~ 5に直す
Y_train = np.array(y_train)[:, -1] - 1

# カテゴリ予測なので、正解ラベルを0, 1のバイナリデータの配列に変換する
Y_train = to_categorical(Y_train, n_classes)

# テストデータにも同様の処理を行う
Y_test = np.array(y_test)[:, -1] - 1
Y_test = to_categorical(Y_test, n_classes)

# 学習する順番によってデータが偏ってしまうので、シャッフルする
shuffle_index = [i for i in range(len(y_train))]
random.shuffle(shuffle_index)

# データの順番をシャッフルする
X_train_shuffle = X_train[shuffle_index]
Y_train_shuffle = Y_train[shuffle_index]


# -------------------- RNNモデルの構築 --------------------
# 入力するデータサイズを指定する
input_size = [X_train_shuffle.shape[1], X_train_shuffle.shape[2]]

# レイヤーを定義してください
rnn_model = keras.Sequential()

rnn_model.add(
    layers.SimpleRNN(
        input_shape=(input_size[0], input_size[1]),
        units=60,
        return_sequences=False # さらにRNNレイヤーを重ねるのであればTrueにする
    )
)
rnn_model.add(layers.Dense(units=n_classes, activation='softmax'))
rnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

# 訓練スタート
history = rnn_model.fit(
    X_train_shuffle, Y_train_shuffle,
    batch_size=100, epochs=10, validation_split=0.3,
    verbose=0
    # verbose=0：学習過程のログを出力しない、1：プログレスバーでログを出力する、2：エポックごとにログを出力する
)

# 精度の推移図を出力
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 損失関数の推移図を出力
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = rnn_model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

### LSTM実装例(パラメータチューニングを含む)
```
# 分類を行うLSTMモデルです
# -------------------- モジュールインポート部分 --------------------
# 必要なモジュールをインポートする
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


plt.style.use('ggplot')


# -------------------- データ読み込み部分 --------------------
# 特徴量に使用するデータの読み込みを指定する
# 正解ラベルのマスタを読み込む
activity_labels = pd.read_csv("./5130_rnn_lstm_data/activity_labels.txt",
                              sep=" ", header=None, names=["activity_id", "activity"])

# 通常は使用するデータからトレーニングデータとテストデータを分ける必要があるが、今回は準備してあるのでそのファイルを読み込む
# 読み込む行数を指定
nrows = 300
# トレーニングデータ
subject_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/subject_train.txt", sep=" ", header=None, nrows=nrows)
y_train = pd.read_csv("./5130_rnn_lstm_data/train/y_train.txt",
                      sep=" ", header=None, nrows=nrows)
total_acc_x_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/total_acc_x_train.txt", sep="\s+", header=None, nrows=nrows)
total_acc_y_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/total_acc_y_train.txt", sep="\s+", header=None, nrows=nrows)
total_acc_z_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/total_acc_z_train.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_x_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/body_gyro_x_train.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_y_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/body_gyro_y_train.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_z_train = pd.read_csv(
    "./5130_rnn_lstm_data/train/body_gyro_z_train.txt", sep="\s+", header=None, nrows=nrows)

# テストデータ
subject_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/subject_test.txt", sep=" ", header=None, nrows=nrows)
y_test = pd.read_csv("./5130_rnn_lstm_data/test/y_test.txt",
                     sep=" ", header=None, nrows=nrows)
total_acc_x_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/total_acc_x_test.txt", sep="\s+", header=None, nrows=nrows)
total_acc_y_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/total_acc_y_test.txt", sep="\s+", header=None, nrows=nrows)
total_acc_z_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/total_acc_z_test.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_x_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/body_gyro_x_test.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_y_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/body_gyro_y_test.txt", sep="\s+", header=None, nrows=nrows)
body_gyro_z_test = pd.read_csv(
    "./5130_rnn_lstm_data/test/body_gyro_z_test.txt", sep="\s+", header=None, nrows=nrows)


# -------------------- 入力データの処理 --------------------
def convert_threeDarray_for_nn(df_list):

    # ニューラルネットに適した、３次元の numpy.ndarray にする
    # array を (サンプル数, シーケンス長, 次元数) の形に変換する
    array_list = []
    for df in df_list:
        ndarray = np.array(df)
        array_list.append(np.reshape(
            ndarray, (ndarray.shape[0], ndarray.shape[1], 1)))

    return np.concatenate(array_list, axis=2)


# トレーニングデータに含めたい特徴量データをリスト化する
train_df_list = [
    total_acc_x_train,
    total_acc_y_train,
    total_acc_z_train,
    body_gyro_x_train,
    body_gyro_y_train,
    body_gyro_z_train
]
# テストデータに含めたい特徴量データをリスト化する
test_df_list = [
    total_acc_x_test,
    total_acc_y_test,
    total_acc_z_test,
    body_gyro_x_test,
    body_gyro_y_test,
    body_gyro_z_test
]

# 上で作成した関数にリスト化したトレーニングデータとテストデータを渡す
X_train = convert_threeDarray_for_nn(train_df_list)
X_test = convert_threeDarray_for_nn(test_df_list)


# -------------------- 正解ラベルの処理 --------------------
# カテゴリ数を n_classes へ格納
n_classes = len(activity_labels.activity_id.unique())

# numpyのarrayに変換し、 1 ~ 6 の数値を 0 ~ 5 に直す
Y_train = np.array(y_train)[:, -1] - 1

# カテゴリ予測なので、正解ラベルを0,1のバイナリデータの配列に変換する
Y_train = to_categorical(Y_train, n_classes)

# テストデータにも同様の処理を行う
Y_test = np.array(y_test)[:, -1] - 1
Y_test = to_categorical(Y_test, n_classes)

# 学習する順番によってデータが偏ってしまうので、シャッフルする
shuffle_index = [i for i in range(len(y_train))]
random.shuffle(shuffle_index)

# データの順番をシャッフルする
X_train_shuffle = X_train[shuffle_index]
Y_train_shuffle = Y_train[shuffle_index]


# -------------------- LSTMモデルの構築 --------------------
# 入力するデータサイズを指定する
input_size = [X_train_shuffle.shape[1], X_train_shuffle.shape[2]]

# ハイパーパラメータや学習方法を変更しながら実行したいため、ここではモデルを関数化します
def pred_activity_lstm(input_dim,
                       activate_method='softmax',  # 活性化関数
                       loss_method='categorical_crossentropy',  # 損失関数
                       optimizer_method='adam',  # パラメータの更新方法
                       kernel_init_method='glorot_normal',  # 重みの初期化方法
                       batch_normalization=False,  # バッチ正規化
                       dropout_rate=None  # ドロップアウト率
                       ):
    # レイヤーを定義
    model = keras.Sequential()
    # 入力層を定義
    model.add(
        layers.LSTM(
            input_shape=(input_dim[0], input_dim[1]),
            units=60,
            kernel_initializer=kernel_init_method,
            return_sequences=True  # この後にLSTMレイヤーを重ねるのでTrueにする
        ))
# １つ目の層のパラメータを変更する場合、ここのコードを変更する
    # バッチごとに正規化を行う
    # pred_activity_lstm(batch_normalization=True)にする場合
    if batch_normalization:
        model.add(layers.BatchNormalization())

    # ドロップアウトにより、ニューロンをランダムに削除する
    # pred_activity_lstm(dropout_rate=0.5)にする場合
    if dropout_rate:
        model.add(layers.Dropout(dropout_rate))
# ここまで

    # 中間層を定義
    model.add(
        layers.LSTM(
            units=30,
            kernel_initializer=kernel_init_method,
            return_sequences=False  # この後にLSTMレイヤーはないのでFalseにする
        ))
# ２つ目の層のパラメータを変更する場合、ここのコードを変更する
    # バッチごとに正規化を行う
    if batch_normalization:
        model.add(layers.BatchNormalization())

    # ドロップアウトにより、ニューロンをランダムに削除する
    if dropout_rate:
        model.add(layers.Dropout(dropout_rate))
# ここまで

    # 出力層
    model.add(layers.Dense(units=n_classes, activation=activate_method))
    model.compile(loss=loss_method, optimizer=optimizer_method,
                  metrics=['accuracy'])

    return model


# モデルの作成
input_size = [X_train_shuffle.shape[1], X_train_shuffle.shape[2]]
turned_model = pred_activity_lstm(
    input_dim=input_size,
    activate_method='softmax',
    loss_method='categorical_crossentropy',
    optimizer_method='adam',
    kernel_init_method='glorot_normal',
    batch_normalization=True
)

# early_stoppingの定義
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# 学習スタート
history = turned_model.fit(
    X_train_shuffle,
    Y_train_shuffle,
    batch_size=100,
    epochs=5,
    validation_split=0.3,
    callbacks=[early_stopping],
    verbose=2
)
```

## Attention
- 過去データの重要な点に着目して学習を進める手法。
- 自然言語処理で広く使用されている。

## CNN と RNN の組み合わせ
- RNNはふたつの対になった時系列データを扱う問題（機械翻訳、自動要約、質疑応答、メールの自動返信など）に利用できる
- CNN と RNN を組み合わせて、「画像」から「文章」を生成することもできる
    - Encoder-Decoder Network（深層学習のモデル構築手法）を用いる
        1. EncoderをCNNにすることで、3次元の特徴マップを出力します。入力から抽出できる特徴は一通りとは限らず、複数の特徴を抽出できます。
        2. 1.で出力された特徴マップをDecoderであるLSTMにより重みを計算します。
        3. 2.で計算された重みと、LSTMの状態を入力として、LSTMで単語を出力します。
        4. 文末記号または設定した文字数を超えるまで2.〜3. を繰り返し、1語ずつ出力して文章を生成します。
    
## データの正規化
- 学習の効率を上げるために、データを正規化する。
- `sklearn.preprocessing.StandardScaler` が使用できる
- `fit_transform` で、平均０、分散１に正規化
- ただし、データ数が少ない時には正規化がうまくできない時がある：テストデータの場合など
    - その場合には、fitは訓練データで行い、transformのみを適用することができる
```
from sklearn.preprocessing import StandardScaler

data1 = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
data2 = [[-1,2]]

# ---------- データをスケールします ----------
# StandardScalerのインスタンスを作ります。
scaler = StandardScaler()

# fit_transformは、データの平均・分散を求めるfitと、データを変換するtransformをまとめて処理します
data1= scaler.fit_transform(data1)
data2 = scaler.transform(data2)

print(data1)
print(data2)
```

## 自然言語処理の種別
- N-gram
    - 文を文字単位の記号列として捉えてN単語ごとに文を切り分け、出現頻度を集計
    - 辞書や文甫的な解釈は不要
- 形態素解析
    - 文を形態素（言語で意味を持つ最小単位）に区切る
    - 日本語に対応するツールの例には以下のようなものがある
        - MeCab
        - ChaSen
        - JUMAN
- 構文解析
    - 形態素に切り分けられた単語の位置関係を明確にする技術
    - 形態素の品詞ではなく、文節同士の係受けを決定する
    - 日本語に対応するツールの例
        - CaboCha
        - KNP(Kurohashi-Nagao Parser)

## 自然言語の前処理
1. クレンジング処理
    - テキスト内のノイズ（JavaScriptのコードやHTMLタグなど）を除去する処理
2. 形態素解析
    - 入力文章を形態素（単語）に分割する処理
3. 単語の正規化
    - 単語の文字種の統一、つづりや表記揺れなどを統一する処理
    - ex）全角の「ネコ」、半角の「ﾈｺ」、ひらがなの「ねこ」
4. ストップワード除去
    - 頻出する割に役に立たない単語を除去する処理
    - ex）助詞や助動詞などの機能語（「が」「は」「です」「ます」）など
    - 機械学習と異なり、深層学習では行わなくても良い場合もあります
5. ベクトル化
    - 単語をベクトルに変換（数値化）する処理

## 自然言語の処理例
```
# 分かち書きを行い、頻出単語を確認します
from collections import Counter
import pandas as pd
import numpy as np
import re
import MeCab
from pathlib import Path


# 青空文庫から取得しておいたデータを読み込む
p = Path(".")
txt_data = (p / "./5130_rnn_lstm_data/shayo.txt").open("r", encoding="utf-8").read()

# 分かち書きのインスタンスを作成してください
mt = MeCab.Tagger("-Owakati")
txt_data = mt.parse(txt_data)
word_list = txt_data.split(" ")

# 単語の一意なリストを取得（辞書順）
vocab = sorted(set(word_list))

# 単語->id(index)のエンコーダー
vocab_to_int = {c: i for i, c in enumerate(vocab)}

# id->単語のデコーダー
int_to_vocab = dict(enumerate(vocab))

# 数字の羅列をリスト化する
int_list = [vocab_to_int[x] for x in word_list]

# DataFrameを作成し、出現単語の数をカウントして書き出す
df = pd.DataFrame \
    .from_dict(dict(Counter(word_list)), orient="index") \
    .reset_index() \
    .rename(columns={"index": "word", 0: "freq"})

df.sort_values(by="freq", ascending=False)
```

## Embedding
- 単語を特徴ベクトルで表現する
    - 「王様」‐「男」＋「女」＝「女王」や、「ワイン」‐「フランス」＋「ドイツ」＝「ビール」などが可能になる
- KerasではEmbeddingレイヤーが利用可能
    - `model.add(Embedding(input_dim=, output_dim=,　input_length=))`
        - `input_dim`：語彙数（入力データの最大インデックス + 1を正の整数で指定）
        - `output_dim`：密なembedding層の次元数（0以上の整数で指定）
        - `input_length`：入力データの系列の長さ
    - 基本的にはインプットを変換するものなので、入力層で使用される

```
from keras.models import Sequential
from keras.layers.embeddings import Embedding

# ----- ここから前出（分かち書き）のコードを引用 -----
import numpy as np
import re
import MeCab
from pathlib import Path

# 青空文庫から取得しておいたデータを読み込む
p = Path(".")
txt_data = (p / "./5130_rnn_lstm_data/shayo.txt").open("r", encoding="utf-8").read()

# 分かち書きのインスタンスを作成する
mt = MeCab.Tagger("-Owakati")
txt_data = mt.parse(txt_data)
word_list = txt_data.split(" ")

# 単語の一意なリストを取得（辞書順）
vocab = sorted(set(word_list))

# 単語->id(index)のエンコーダー
vocab_to_int = {c: i for i, c in enumerate(vocab)}

# id->単語のデコーダー
int_to_vocab = dict(enumerate(vocab))

# 数字の羅列をリスト化する
int_list = [vocab_to_int[x] for x in word_list]
# ----- ここまで前出（分かち書き）のコードを引用 -----

## Embedding層の定義
# バッチサイズを指定する
batch_size = -1
# 扱う語彙数を入力する
vocab_size = len(vocab)
# 単語ベクトルの次元数を指定する
embedding_dim = 100
# 文の長さを指定する
seq_length = 30

# `word_list`をidの羅列に変換したデータの一部を入力データとする
# 例として batch_size と seq_length を使用
residue = len(int_list) % seq_length
input_data = np.array(int_list)[:-residue].reshape(batch_size, seq_length)

# modelの入力層にEmbeddingを追加する
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                    input_length=seq_length))
model.compile('rmsprop', 'mse')

output = model.predict(input_data)
print(output.shape)

(2035, 30, 100)
```

## LSTMによる自然言語処理の実装例
```
# -------------------- パッケージの読み込み --------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------- モデルの構築 --------------------
# モデルの定義
model = keras.Sequential()

# 1層目
# return_sequencesに適切な値を入力してください
model.add(layers.LSTM(input_dim=3, input_length=3, units=30, 
               return_sequences=True))
# 2層目
# return_sequencesに適切な値を入力してください
model.add(layers.LSTM(units=30, return_sequences=False))

# 出力のクラス数と活性化関数を指定する
model.add(layers.Dense(3 ,activation='linear'))
# 損失関数と最適化アルゴリズムを指定してコンパイルする
model.compile(loss="mse", optimizer="rmsprop")

# 値の変化が停止した時に訓練を終了することを指定する
earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)

# ========== ここから先の処理は前処理が必要なため動作しません ==========
# モデルをトレーニングデータで訓練する
# hist = model.fit(X_Train, y_train, batch_size=batchSize, nb_epoch=epochD, verbose=0, shuffle = False,
#                  validation_split=validRate, callbacks = [earlyStopping])
# print(hist.history)

# -------------------- モデルを使って予測 --------------------
# テストデータで予測を行う
# predictTrain = model.predict(y_train)  
# predict_df = pd.DataFrame(predictTrain)
# y_train_df = pd.DataFrame(y_train)
```