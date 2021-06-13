# メモ
## 1. 基礎編：自然言語処理における深層学習
### 1.1 自然言語処理における深層学習
#### 1.1.1 自然言語処理における深層学習
- 古典的自然言語処理の手法
    - one hot vector
    - TFIDF
- 上記のような古典的なベクトルによる表現には以下の問題点が存在
    - スパース性：ベクトル次元が大きすぎて、メモリを大量に必要とする
    - 意味を扱えない: すべての単語が出現頻度という特商でしか差別化されず、個性は失われる
- ニューラルネットワークのモデルを使うときの利点
    - 誤差逆伝播法によって単語のベクトルをがky集でくるので、各単語にわずか数百次元のベクトル割り当て（Embedding）で良い
    - 文脈を考慮できるので、単語の意味を扱うことができる
### 1.2 Embedding
#### 1.2.1 Embedding
- Embedding: 埋め込み。単語という記号をd次元ベクトル（dは100〜300程度）に埋め込む。
- tensorflow.keras での実装例: `Embedding(input_dim, output_dim, input_length)`
    - input_dim: 語彙数、単語の種類の数
    - output_dim: 単語ベクトルの次元
    - input_length: 各文の長さ
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

vocab_size = 1000 # 扱う語彙の数
embedding_dim = 100 # 単語ベクトルの次元
seq_length = 20 # 文の長さ

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length))
```
- Embedding Matrix: 全単語の単語ベクトルを結合したもの
    - 前提として各単語に特有のIDを割り振り、そのIDがEmbedding Matrixの何行目になるのかを指す
    - 各セルの値はtensorflow.kerasが自動でランダムな値を格納してくれる
- 0行目にはunk、すなわちUnknown（未知語）を割り当てる
    - unkを使う理由は、扱う語彙を限定してその他のマイナーな単語は全てUnknownとすることでメモリを節約するため
    - 語彙の制限の仕方は、扱うコーパス（文書）に出現する頻度が一定以上のものだけを扱うなどが一般的
- 文の長さは全てのデータで統一する必要がある(Padding)
    - 長さD以下の文は長さがDになるよう0を追加する
    - 長さD以上の文は長さがDになるよう末尾から単語を削る
### 1.3 RNN
#### 1.3.1 RNN
- RNN: Recurrent Neural Networkの略称で、日本語では「再帰ニューラルネット」
    - 可変長系列、すなわち任意の長さの入力列を扱うことに優れており、自然言語処理において頻繁に使われる重要な機構    
    - 言語以外でも、時系列データであれば使えるので、音声認識など活用の幅は広い
### 1.4 LSTM
#### 1.4.1 LSTMとは
- LSTM: RNNの一種で、 LSTMはRNNに特有な欠点を補う機構を持っている
    - 長期記憶が苦手な点：「ゲート」を導入することで、長期記憶と短期記憶の両方を可能にしている
#### 1.4.2 LSTMの実装
- 実装例: `STM(units, return_sequences=True)`
    - units: LSTMの隠れ状態ベクトルの次元数。大抵100から300程度。
    - return_sequences: LSTMの出力の形式をどのようにするかを決めるための引数
```
from tensorflow.keras.layers import LSTM
units = 200
lstm = LSTM(units, return_sequences=True)
```
#### 1.4.3 BiLSTM
- 通常、CNNは過去の情報を未来に入力するような形を取る
    - それを逆にしたり、あるいは組み合わせるようなことも可能である
- 双方向再帰ニューラルネット: 各時刻において先頭から伝播してきた情報と後ろから伝播してきた情報、すなわち入力系列全体の情報を考慮できる
- BiLSTM: 2方向のLSTMを繋げたもの
- 実装例
    - merge_mode: 2方向のLSTMをどうつなげるか
        - sum: 要素和
        - mul: 要素積
        - concat: 結合
        - ave: 平均
        - None: 結合せずにlistを返す
```
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
bilstm = Bidirectional(LSTM(units, return_sequences=True), merge_mode='sum')
```
### 1.5 Softmax関数
#### 1.5.1 Softmax関数
- Softmax関数: 活性化関数の一種。クラス分類を行う際にニューラルネットの一番出力に近い層で使われる。
- 実装例
    - バッチごとに softmax関数を適用して使う
```
from tensorflow.keras.layers import Activation
# xのサイズ: [バッチサイズ、クラス数]
y = Activation('softmax')(x)
# sum(y[0]) = 1, sum(y[1]) = 1, ...
```
### 1.6 Attention
#### 1.6.1 Attentionとは
- 仮にsを質問文とし、tをそれに対する回答文の候補とする
- この時、機械に自動でtがsに対する回答文として妥当かどうか判断させるにはどのようにしたら良いか
- こういった際に、Attention Mechanism（注意機構）が使用できる
    - 参照元の系列sのどこに”注意”するかを対象の系列tの各時刻において計算することで、臨機応変に参照元の系列の情報を考慮しながら対象の系列の情報を抽出することが可能になる
    - 双方向や、参照元と対象の最大時刻（隠れベクトルの総数）が異なる場合でも適用可能
#### 1.6.2 Attentionの実装
- tensorflow.kerasでAttentionを実装するためには、Mergeレイヤーを使う必要がある
    - 但し、tensorflow.kerasのバージョン2.0以降では前の章まで使っていたSequential ModelをMergeすることができない
    - そのため、Functional API を使用する
- 実装例
```
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import dot, concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model

batch_size = 32 # バッチサイズ
vocab_size = 1000 # 扱う語彙の数
embedding_dim = 100 # 単語ベクトルの次元
seq_length1 = 20 # 文1の長さ
seq_length2 = 30 # 文2の長さ
lstm_units = 200 # LSTMの隠れ状態ベクトルの次元数
hidden_dim = 200 # 最終出力のベクトルの次元数

input1 = Input(shape=(seq_length1,))
embed1 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length1)(input1)
bilstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed1)

input2 = Input(shape=(seq_length2,))
embed2 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length2)(input2)
bilstm2 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed2)

# 要素ごとの積を計算する
product = dot([bilstm2, bilstm1], axes=2) # productのサイズ：[バッチサイズ、文2の長さ、文1の長さ]

a = Activation('softmax')(product)
c = dot([a, bilstm1], axes=[2, 1])
c_bilstm2 = concatenate([c, bilstm2], axis=2)
h = Dense(hidden_dim, activation='tanh')(c_bilstm2)

model = Model(inputs=[input1, input2], outputs=h)
```
### 1.7 Dropout
#### 1.7.1 Dropout
- Dropout: 訓練時に変数の一部をランダムに0に設定することによって、汎化性能を上げ、過学習を防ぐための手法
- 実装例
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
model = Sequential()
...
model.add(Dropout(0.3))
```
- 実装例(Functional API)
```
from tensorflow.keras.layers import Dropout
y = Dropout(0.3)(x)
```
## 2. 実践編：回答文選択システムの実装
### 2.1 回答文選択システム
#### 2.1.1 回答文選択システム
### 2.2 データの前処理
#### 2.2.1 正規化・分かち書き
#### 2.2.2 単語のID化
#### 2.2.3 Padding
### 2.3 Attention-based QA-LSTM
#### 2.3.1 全体像
#### 2.3.2 質問と回答のBiLSTM
#### 2.3.3 質問から回答へのAttention
#### 2.3.4 出力層、コンパイル
### 2.4 訓練
#### 2.4.1 訓練
### 2.5 テスト
#### 2.5.1 テスト
### 2.6 Attentionの可視化
#### 2.6.1 Attentionの可視化