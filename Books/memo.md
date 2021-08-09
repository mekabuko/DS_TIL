#　メモ
## 1章 自然言語処理の基礎
### 自然言語処理のタスク
1. 形態素解析
- テキストを形態素に分割して品詞を付与する処理
- ツール
    - MeCab: 高性能かつ高速
    - janome: MeCabより遅いが使いやすい
2. 構文解析
- 文の構造（係り受けなど）を解析する
- ツール
    - CaboCha
    - KNP
3. 意味解析
- テキストの意味を解析
- 同時に単語の意味の曖昧性解消を含む
    - bank が「銀行」なのか「土手」なのか、のようなもの
4. 文脈解析
- 複数の文のつながりに関する解析を行う
### 自然言語処理の応用
- 機械翻訳システム
- 質問応答システム
- 情報抽出
- 対話システム
## ２章 機械学習
- 特にメモなし
## 3章 コーパス
### コーパスとは
- 自然言語のデータ
    - ラベル付き、ラベルなしに大分できる
- フリーの公開コーパスも多い
    - テキスト分類用: 20 Newsgroups, IMDb
    - 固有表現認識用: CoNLL-2003
    - 質問応答用: SQuAD, MS MACRO
### コーパス読み込みの実装例
ファイルを読み書きするなら`with`を使うと、open/close が不要
```
with open('example.txt', 'r', encoding='utf-8') as f:
    f.read()
    f.readline()
    for line in f:
        print(line)
```
CSVを読み込むには pandas が便利
```
import pandas as pd
df = pd.read_csv('example.csv', encoding='utf-8')
```
JSONにも
```
pd.read_json('example.json', encoding='utf-8')
```
### コーパスの作成（クローリング/スクレイピング）
- ぐるなびのAPIなど、情報を取得できるAPIを使用することもできるし、クローリングなどの手法もある
    - ぐるなびAPI: 2021/06で終了していた。。
- ライブラリには、 requests や Scrapy などがある
### コーパスの作成（アノテーションツール）
アノテーションツール: データにラベルをつけるためのツール
- OSSの例
    - brat: 系列ラベリングと関係認識用ラベリングが可能
    - doccano: テキスト分類、系列ラベリング、系列変換が可能
- サービスの例(有料のものもあり)
    - prodigy
    - tagtog
    - LightTag
## 4章　テキストの前処理
[コード(Colab)](https://colab.research.google.com/drive/1c0NBJQBKQ_YkVqQWzNUD69J5VEPyV9BR?usp=sharing)
[実践サンプル](https://colab.research.google.com/drive/18OM5FoF6Ep4QxSML7pqaf3cd3KPt0wUr?usp=sharing)
### クリーニング
HTMLタグやjavascriptコードなどのノイズを除去する処理。
BeautifulSoup や Python標準の re パッケージが活用可能。
- 正規表現を試すことのできるオンラインエディタ: [Regex101](https://regex101.com/)
### 単語分割
テキストの分割や、品詞の抽出を行う
- 分割: janome.tokenizer
- フィルタ: janome.analyzer & janome.tokenfilter
場合によっては、辞書の修正も有効な方法になる
### 単語の正規化
表記揺れなどを統一する
- 文字種の統一
    - 大文字、小文字に統一など
- 数字の置き換え
    - 数字は邪魔になるので、全て置き換えてしまった方が良い
- 辞書を用いた単語の統一
    - 辞書を用いた表記揺れの修正は、地道な手作業となることが多い
    - 例: ソニー、Sony　→ Sony に統一、など
### スワップワードの除去
一般的で役に立たないものなど、処理対象外となる単語（ストップワード）を除去する
- 辞書方式:
    - [SlothLibの日本語データ](http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt)
- 出現頻度方式
    - カウントして、上位 n 個の文字をストップワード認定する、という方法
### 単語のID化
単語にIDを割り振って、無駄な情報を落とす
### パディング
データの系列長を合わせる操作: 機械学習フレームワークが、一定長を求める場合が多いから。
## 5章 特徴エンジニアリング
[実践サンプル](https://colab.research.google.com/drive/1YygZfK9-WRuUwGu3dW9qNlBI7nl_nK1l?usp=sharing)
### 質的変数の処理
順序特徴量は、変換用のマップを使って変換する
```
# size を順序変数に変換するためのマップ
size2intMap = {'S': 0, 'M': 1, 'L': 2}
df['size'] = df['size'].map(size2intMap)
```
名義特徴量は、`LabelEncoder`で整数値にしてから、`pd.get_dummies`で one-hot encoding する
```
from sklearn.processing import LabelEncoder

encoder = LabelEncoder()
df['name'] = encoder.fit_transform('df['name'])
pd.get_dummies(df)
```
### 量的変数の処理
二値化
```
df['Fare'] = (df['Fare'] > 10).astype(int) # 0 or 1 で二値化
```
丸め
```
df['Fare'] = df['Fare'].round().astype(int) # 丸め込み
```
### テキストのベクトル表現
テキストをベクトル化するには２ステップ。
1. 単語分割
2. ベクトル化
ベクトル化にはいくつか手法が存在。

#### N-gram ベクトル
- n-gram = 連続する n 個のトークン
    - n = 1 を uni-gram
    - n = 2 を bi-gram
- ベクトル化の手順
1. 各 n-gram に数値を割り当て（これを「語彙」と呼ぶ）
2. テキストを Bag-of-Ngrams(BoW) でベクトル表現に変換
    - One-hot エンコーディング
        - ある単語がテキストに存在するかどうかでベクトルを作成
        ```
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer()
        docs = ['the cat is out of the bag', 'dogs are']

        bow = vectorizer.fit_transform(docs)
        bow.toarray() # 出力されたベクトル
        vectorizer.vocabulary_ # 単語とインデックスの対応
        ```
    - Count エンコーディング
        - ある単語がテキストに、何回存在するかでベクトルを作成
        ```
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(binary=False)
        docs = ['the cat is out of the bag', 'dogs are']

        bow = vectorizer.fit_transform(docs)
        bow.toarray() # 出力されたベクトル
        vectorizer.vocabulary_ # 単語とインデックスの対応
        ```
    - tf-idf
        - Countエンコーディングの弱点＝the, is などの頻出単語に影響を受けやすくなってしまう
        - その欠点を軽減する。単語の出現頻度をそのまま使うのではなく、その単語が出現する文書数の逆数（逆文書頻度）をかけて用いる
            - 珍しい単語に、大きな重みを持たせる＝特徴的な単語、として考える。
        ```
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer()
        docs = ['the cat is out of the bag', 'dogs are']

        bow = vectorizer.fit_transform(docs)

        bow.toarray() # 出力されたベクトル
        vectorizer.get_feature_names() # 単語とインデックスの対応
        ```
        - 日本語を使う場合には以下のように少し操作が必要
        ```
        from sklearn.feature_extraction.text import TfidfVectorizer
        from janome.tokenizer import Tokenizer

        t = Tokenizer(wakati=True)

        vectorizer = TfidfVectorizer(tokenizer=t.tokenize)
        docs = ['猫の子子猫', '犬の子子犬']

        bow = vectorizer.fit_transform(docs)

        bow.toarray() # 出力されたベクトル
        vectorizer.get_feature_names() # 単語とインデックスの対応
        ```
### 特徴量のスケーリング
#### 正規化
データをある範囲に収まるようにスケーリング
- Min-Max スケーリング
```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(data)
scaler.transform(data)
```
#### 標準化
平均０、分散１にスケーリング
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)
```
### 特徴選択
モデル構築に用いる特徴を選択するプロセス
- フィルター法
    - 各特徴と目的変数間の関係を考慮して、重要度でランク付けする方法
    - 相関係数など
- ラッパー法
    - 特徴の部分集合を使って機械学習モデルを学習、その性能を使って特徴に重要度を与えてランクづけ
    - 良い性能を出せる特徴量を使っていく、という方法
- 組み込み法
    - 機械学習アルゴリズム自体に特徴選択の機能が組み込まれているもの
## 6章 機械学習アルゴリズム
[コード](https://colab.research.google.com/drive/1smMOripIegLGiGgLXxCXnp3gU2IlBaYq?usp=sharing)
[実践例](https://colab.research.google.com/drive/1qoURsDec8aY886kk9HwqCz9STZmFz5kU?usp=sharing)
### ロジスティック回帰
回帰という名前だが、分類のためのモデル。
シグモイド関数を使って、[0, 1]の確率を出力できる。

### 損失関数（コスト関数)
モデルの評価を行うかんすう。
- 平均二乗誤差: 予測と正解との誤差の二乗の総和。
- クロスエントロピー
    - 間違えた時の予想確率の大きさがペナルティになる（自信があったのに間違えた、が大きなペナルティ）
### オプティマイザ
損失関数の値に応じてモデルの重みを更新するアルゴリズム。
- 勾配降下法: 損失関数が小さくなる方向に向かって進む（降下する）ことで、モデルを更新していく方法。
- 学習率: 勾配降下法で言えば、どれくらい降下するか。
### k分割交差検証(クロスバリデーション/CV)
データをk個に分割して、k-1個を学習に、残った1個を検証に使う、を繰り返す方法。
過学習の抑制効果が期待できる。
### 汎化性能
未知のデータに対する予測性能
### 学習曲線
横軸にデータ数、縦軸に性能をプロットした図。
学習時、検証時でそれぞれプロットする。
- 未学習の時: 両方低くなる
- 過学習の時: 学習時は高いが、検証時は低い
### 正則化
損失関数に重みによるペナルティを加える＝モデルの複雑さにペナルティを与えることで、過学習を防ぐ。
- L1正則化: 重みの係数の絶対値に比例した値をペナルティとして加える
- L2正則化: 重みの係数の2条に比例する値をペナルティとして加える
    - scikit-learnのロジスティック回帰はデフォルトでL2正則化が使われている
### ハイパーパラメータチューニング
モデル学習時に与えるパラメータを設定すること。手動でもいいが、自動でサーチする方法もあル。
- グリッドサーチ: 与えられたすべての候補値の組み合わせを試す
- ランダムサーチ: パラメータの分布を指定し、そこからサンプリングしてチューニングする
    - 少数のハイパーパラメータが大きく影響を与える場合に効果的
## ７章 ニューラルネットワーク
[実践例](https://colab.research.google.com/drive/1TJhoOye7yXlbuRDjwEzPVPfb2KjXLkQi?usp=sharing)
## 8章 単語分散表現
[実践例](https://colab.research.google.com/drive/15HQJQe3PfWjP41fOZpYDaXyHXgNK7XJc?usp=sharing)
### 単語分散表現とは
1単語を 数百次元程度のベクトルで表す方法
### one-hot エンコーディング
1単語に1次元を割り当て、その単語であるかどうか、を示すベクトルに変換する方法
- 次元数 = ボキャブラリ数(単語の種類の数)
    - 単語を追加するごとに、データの次元数が変わるので、再学習が必要になってしまう
- ベクトル間の演算（内積など）で意味のある結果を得られない = 単語同士の類似度などを測る方法はない
### 分散表現
単語を低次元(50-300程度)の実数値ベクトルで表す方法
- ベクトル演算に意味が出る
    - 単語間の類似度を、ベクトルの内積によって測る
    - 加減算で単語の意味を表現できる
        - King - Man + Woman = Queen など
- ボキャブラリ数と次元数は別なので、単語の追加で次元数を増やす必要はない
### 分布仮説
単語の意味は、文の中の周辺の単語(単語の分布)によって決定される、という考え方
### Word2Vec
単語分散表現を学習するためのソフトウェア。
内部で用いられているのは「Skip-gram」「CBOW」というモデル。
### Skip-gram
単語の分散表現を計算するモデル。２層のニューラルネットワークで構成される。
- あるタスクに対してNNを学習し、入力層と隠れ層の間の重みを得る
    - この重みが、単語の分散表現
- ある単語を入力した時、その単語の周辺に出現しやすい単語を予測する
### 学習済み単語分散表現
フリーで使えるものも多数存在
- word2vec
- GloVe
- facebook製の、日本語Wikipediaから学習したもの: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
### 単語分散表現の評価
- 内省的評価: 単語分散表現の性能を直接評価する方法
    - 人間の判断との相関: 単語のペアに対して、人間がつけた相関のスコアを用いて評価
    - アナロジータスク: 推論による問題の正解率を評価
    - 日本語の評価用データセットとしては、[JapaneseWordSimilarityDataset](https://github.com/tmu-nlp/JapaneseWordSimilarityDataset)がある
- 外省的評価
    - 実際のタスクに単語分散表現を使って、性能が高くなる単語分散表現を良いものと判断する方法
    - 基本的には外省的評価を使った方が、実際のタスクとしては良い結果が得られやすい
### 文脈を考慮した分散的表現
- word2Vecなどは、文脈を考慮することはできない
- 文脈を考慮したものとしては ELMo, それを改善した BEAT など。
## 9章 テキスト分類
[実践例](https://colab.research.google.com/drive/17ZLUAT4Zst_QwXcXf1imTar0tsDsX7lS?usp=sharing)
### テキスト分類
テキストを事前に決められたクラス（カテゴリ）に分類すること
#### テキスト分類の例
- メールがスパムなのか否か
- レビューがポジティブかネガティブか
- 文章が何語で記述されているか
#### テキスト分類の種別
- 手動
    - 性能が高い
    - 遅く、高コストで、スケールしない
- ルールベース
    - ルールが正しければ性能は高い
    - ルールを作るコストは高く、メンテナンスコストも高い
- 機械学習ベース
    - 性能が高くスケールが可能
    - 学習データが必要
### RNN(リカレントネットワーク)
NNでは、可変調の入力を扱いづらい

RNNでは
- 可変調の入力を扱うことができる
- 入力系列の要素感に存在する依存性を扱うことができる
    - 時系列データのほか、自然言語などにも適用される
- 隠れ状態(hidden state)に過去の入力の値を圧縮して持っておき、入力の一部としてつかう
- tanh という構造を内部に持ち、それで隠れ状態を表現

### テキスト分類の評価
- 混合行列
    - 予想、実際のそれぞれがP/Nのどちらなのかで、TP, TN, FP, FN の４区分に分ける方法
- 正解率
    - 全体のうち、正しく予想した割合
- 適合率
    - Positive と予想したうち、どれだけPositive か
- 再現率
    - Positive のうち、どれだけPositiveと予測できたか
    - 適合率と再現率の間にはトレードオフの関係がある
- F1
    - 敵同率と再現率の両方を考慮（適合率と再現率の調和平均

### 多クラス分類の評価
各クラスに対して、TP, TN, FP, FN を考えることができ、それによって正解率、適合率、再現率、F1などを求められる
- マイクロ平均
    - 適合率: 各クラスの真陽性の合計/(各クラスにおける真陽性+偽陽性)
- マクロ平均
    - 適合率: 各クラスに対する適合率Pkの平均

### LSTM
RNN よりも長期の依存性を考慮できるモデル
- RNNが１つのtanhだったのに対して、4つのそうで表現
- 簡単にいうと「まず重要でないことを忘れて、重要なことを記憶する」という仕組み

### CNN
畳み込みニューラルネットワーク。画像処理などの例も多いが、自然言語に対しても適用可能。
高い性能が出る可能性があるが、職人芸でアーキテクチャやハイパーパラメータの設定が必要。

### 学習済み単語分散表現の使い方
学習済み単語分散表現を Embedding 層の初期値として使うことが可能。


# Links
- [amazon](https://www.amazon.co.jp/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%BB%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92%E3%81%AB%E3%82%88%E3%82%8B%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E5%85%A5%E9%96%80-Compass-Books%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E4%B8%AD%E5%B1%B1-%E5%85%89%E6%A8%B9-ebook/dp/B084WPRT44)
- [公式サポート](https://book.mynavi.jp/supportsite/detail/9784839966607.html)
- [Google Colabのファイルリンク](https://gist.github.com/Hironsan/1f1cc629613cbd7de042a7ce269b91d6)
