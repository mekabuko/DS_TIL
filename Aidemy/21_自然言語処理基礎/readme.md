# サマリ

# メモ
## 単語集（引用）
- 自然言語(NL, Natural Language): 日本語や英語のような 自然発生的に生まれた言語
- 自然言語処理(NLP, Natural Language Processing): 人間が日常的に使っている 自然言語をコンピュータに処理させる技術。文書分類・機械翻訳・文書要約・質疑応答・対話などの活用がある。
- トークン：自然言語を解析する際、文章の最小単位として扱われる文字や文字列のこと。
- タイプ：単語の種類を表す用語。
- 文章：まとまった内容を表す文のこと。自然言語処理では一文を指すことが多い。
- 文書：複数の文章から成るデータ一件分を指すことが多い。
- コーパス：文書または音声データにある種の情報を与えたデータ。
- シソーラス：単語の上位/下位関係、部分/全体関係、同義関係、類義関係などによって単語を分類し、体系づけた類語辞典・辞書。
- 形態素：意味を持つ最小の単位。「食べた」という単語は、2つの形態素「食べ」と「た」に分解できる。
- 単語：単一または複数の形態素から構成される小さな単位。
- 表層：原文の記述のこと。
- 原形：活用する前の記述のこと。
- 特徴：文章や文書から抽出された情報のこと。
- 辞書：自然言語処理では、単語のリストを指す。
- 単語分割：文章を単語に分割すること
- 品詞タグ付け：単語を品詞に分類して、タグ付けをする処理のこと
- 形態素解析：形態素への分割と品詞タグ付けの作業をまとめたもの
- 表記揺れ：同じ文書の中で、同音・同義で使われるべき語句が異なって表記されていること
- 正規化：表記揺れを防ぐためルールベースで文字や数字を変換すること

## 自然言語処理における「日本語」
- 単語の区切りは難しい
- 単語ごとの解釈は少ないので、品詞や基本形を求めることは容易

## MeCab
- 形態素解析のツール
- `MeCab.Tagger("-Owakati").parse(str)`: わかちがき
- `MeCab.Tagger("-Ochasen").parse(str)`: 形態素解析

## janome
- 形態素解析のツール
- `Tokenizer(wakati=True).tokenize(str)`: わかちがき
- `Tokenizer().tokenize(str)`: 形態素解析
    - `Token.surface`： 表層形
    - `Token.part_of_speech`: 品詞部分
```
from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()
tokens = tokenizer.tokenize("明日は晴れるだろうか。")
for token in tokens:
    print(token)
```

## Ngram
- N文字ごとに単語を切り分ける 、または N単語ごとに文章を切り分ける 解析手法
- 単語のNgramを求めたい場合: 引数に単語と切り出したい数を入れる
- 文章のNgramを求めたい場合: janomeのtokenize関数を用いて分かち書きのリストを作成し、その分かち書きのリストと切り出したい数を引数に入れる
```
from janome.tokenizer import Tokenizer
t = Tokenizer()
tokens = t.tokenize("太郎はこの本を二郎を見た女性に渡した。", wakati=True)

def gen_Ngram(words,N):
    # Ngramを生成してください
    ngram = []
    for i in range(len(words)-N+1):
        cw = "".join(words[i:i+N])
        ngram.append(cw)
    return ngram

print(gen_Ngram(tokens, 2))
print(gen_Ngram(tokens, 3))
```

## 正規化
- 全角を半角に統一や大文字を小文字に統一等、ルールベースで文字を変換すること
- `neologdn.normalize("正規化したい文字列")`
- 単純なものなら`str.lower()`, `str.upper()`なども可能

### 数字の置き換え
- 数字は多様だが出現頻度が高い＝自然言語処理のノイズになりがち
- そのため、記号などに置き換えてしまいことが多い
- `re.sub('\d', "!", text)`

## 文書のベクトル表現
- 文書中に単語がどのように分布しているかをベクトルとして表現すること
- Bag of Words：構造や語順の情報は失われた状態
### 方法
- カウント表現：文書中の各単語の出現数に着目する方法
- バイナリ表現：出現頻度を気にせず、文章中に各単語が出現したかどうかのみに着目する方法
- tf-idf表現：tf-idfという手法で計算された、文章中の各単語の重み情報を扱う方法
    - 単語の出現頻度であるtf(Term frequency)と、単語がどれだけ珍しいかを示す逆文書頻度 idf(Inverse Document Frequency)の積で求められる
    - 結果、どの文章にも出てくるような単語の重要度は下げられて、との文書特有の単語が特徴として強調されることになる。

## カウント表現の実装
- `corpora.Dictionary(分かち書きされた文章)` で作成
```
from gensim import corpora
from janome.tokenizer import Tokenizer

text1 = "すもももももももものうち"
text2 = "料理も景色もすばらしい"
text3 = "私の趣味は写真撮影です"

t = Tokenizer()
tokens1 = t.tokenize(text1, wakati=True)
tokens2 = t.tokenize(text2, wakati=True)
tokens3 = t.tokenize(text3, wakati=True)
documents = [tokens1, tokens2, tokens3]

# 単語辞書
dictionary = corpora.Dictionary(documents)

# 各単語のid
print(dictionary.token2id)

# BOW
bow_corpus = [dictionary.doc2bow(d) for d in documents]
print(bow_corpus)
```

## tf-idf表現の実装
- `vectorizer = TfidfVectorizer(use_idf=True, token_pattern="(?u)\\b\\w+\\b")`
    - `use_idf=False`: tfのみの重み付け
    - `token_pattern="(?u)\\b\\w+\\b"`: 標準だと1文字の単語は除外されてしまうので、除外しないうように対象を正規表現で「１文字以上の任意の文字列：とする
- `vectorizer.fit_transform(分かち書きされた文章)`
```
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

np.set_printoptions(precision=2)
docs = np.array([
    "リンゴ リンゴ", "リンゴ ゴリラ", "ゴリラ ラッパ"
])

# ベクトル表現に変換
vectorizer = TfidfVectorizer(use_idf=True, token_pattern="(?u)\\b\\w+\\b")
vecs = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names())
print(vecs.toarray())
```

## cos類似度
- ベクトル同士の類似度として、三角関数cosignを用いたcos類似度を使える
```
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

docs = np.array([
    "リンゴ リンゴ", "リンゴ ゴリラ", "ゴリラ ラッパ"
])
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u"(?u)\\b\\w+\\b")
vecs = vectorizer.fit_transform(docs)
vecs = vecs.toarray()

# cos類似度を求める関数
def cosine_similarity(v1, v2):
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    return cos_sim

# 類似度を比較
print("{:1.3F}".format(cosine_similarity(vecs[0], vecs[1])))
```

## Word2Vec
- 単語をベクトル化するツール
- `model = word2vec.Word2Vec(リスト, size=, min_count=, window=)`
    - size ：ベクトルの次元数。
    - window ：この数の前後の単語を、関連性のある単語と見なして学習を行う。
    - min_count ：n回未満登場する単語を破棄。
- `model.most_similar(positive=["単語"])` で、類似度の高いものを出力できる
- サンプル（引用）
```
import glob
from janome.tokenizer import Tokenizer
from gensim.models import word2vec

# livedoor newsの読み込みと分類
def load_livedoor_news_corpus():
    category = {
        "dokujo-tsushin": 1,
        "it-life-hack":2,
        "kaden-channel": 3,
        "livedoor-homme": 4,
        "movie-enter": 5,
        "peachy": 6,
        "smax": 7,
        "sports-watch": 8,
        "topic-news":9
    }
    docs  = []
    labels = []

    for c_name, c_id in category.items():
        files = glob.glob("./5050_nlp_data/{c_name}/{c_name}*.txt".format(c_name=c_name))

        text = ""
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines() 

                #1,2行目に書いたあるURLと時間は関係ないので取り除きます。
                url = lines[0]  
                datetime = lines[1]  
                subject = lines[2]
                body = "".join(lines[3:])
                text = subject + body

            docs.append(text)
            labels.append(c_id)

    return docs, labels

# 品詞を取り出し「名詞、動詞、形容詞、形容動詞」のリスト作成
def tokenize(text):
    tokens = t.tokenize(",".join(text))
    word = []
    for token in tokens:
        part_of_speech = token.part_of_speech.split(",")[0]
 
        if part_of_speech in ["名詞", "動詞", "形容詞", "形容動詞"]:
            word.append(token.surface)            
    return word

# ラベルと文章に分類
docs, labels = load_livedoor_news_corpus()
t = Tokenizer() # 最初にTokenizerインスタンスを作成する
sentences = tokenize(docs[0:100])  # データ量が多いため制限している

# word2vec
model = word2vec.Word2Vec(sentences, size=100, min_count=20, window=15)
print(model.most_similar(positive=["男"]))
```
## Doc2Vec
- Word2Vecを応用した 文章をベクトル化する技術
- BOWと違い、 文の語順 も特徴として考慮に入れることができる点
    - 結果、単語の語順情報がない、単語の意味の表現が苦手、というBOWの欠点を克服
- `Doc2Vec(documents=training_docs, min_count=1)` : モデル生成
- `model.docvecs.most_similar(document)`: 類似度の表示
```
import glob
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer

# livedoor newsの読み込みと分類
def load_livedoor_news_corpus():
    category = {
        "dokujo-tsushin": 1,
        "it-life-hack":2,
        "kaden-channel": 3,
        "livedoor-homme": 4,
        "movie-enter": 5,
        "peachy": 6,
        "smax": 7,
        "sports-watch": 8,
        "topic-news":9
    }
    docs  = []
    labels = []

    for c_name, c_id in category.items():
        files = glob.glob("./5050_nlp_data/{c_name}/{c_name}*.txt".format(c_name=c_name))

        text = ""
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines() 

                #1,2行目に書いたあるURLと時間は関係ないので取り除きます。
                url = lines[0]  
                datetime = lines[1]  
                subject = lines[2]
                body = "".join(lines[3:])
                text = subject + body

            docs.append(text)
            labels.append(c_id)

    return docs, labels
docs, labels = load_livedoor_news_corpus()

# Doc2Vecの処理
token = [] # 各docsの分かち書きした結果を格納するリストです
training_docs = [] # TaggedDocumentを格納するリストです
t = Tokenizer() # 最初にTokenizerインスタンスを作成する
for i in range(4):
    
    # docs[i] を分かち書きして、tokenに格納します
    token.append(t.tokenize(docs[i], wakati=True))
    
    # TaggedDocument クラスのインスタンスを作成して、結果をtraining_docsに格納します
    # タグは "d番号"とします
    training_docs.append(TaggedDocument(words=token[i], tags=["d" + str(i)]))

# Doc2Vec モデル化
model = Doc2Vec(documents=training_docs, min_count=1)

for i in range(4):
    print(model.docvecs.most_similar("d"+str(i)))
```

## コーパスカテゴリの分類（ランダムフォレスト） サンプルコード（引用）
```
import glob
import random
from sklearn.model_selection import train_test_split
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def load_livedoor_news_corpus():
# livedoor newsの読み込みと分類(自然言語処理基礎2.2.4)
    category = {
        "dokujo-tsushin": 1,
        "it-life-hack":2,
        "kaden-channel": 3,
        "livedoor-homme": 4,
        "movie-enter": 5,
        "peachy": 6,
        "smax": 7,
        "sports-watch": 8,
        "topic-news":9
    }
    docs  = []
    labels = []

    for c_name, c_id in category.items():
        files = glob.glob("./5050_nlp_data/{c_name}/{c_name}*.txt".format(c_name=c_name))

        text = ""
        for file in files[:15]:#実行時間の関係上、読み込むファイルを制限しています。
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines() 
                url = lines[0]  
                datetime = lines[1]  
                subject = lines[2]
                body = "".join(lines[3:])
                text = subject + body

            docs.append(text)
            labels.append(c_id)

    return docs, labels

docs, labels = load_livedoor_news_corpus()

# データをトレイニングデータとテストデータに分割(機械学習概論2.2.2)
train_data, test_data, train_labels, test_labels = train_test_split(docs, labels, test_size=0.2, random_state=0)

# 文章を分割する関数
t=Tokenizer()
tokenize = lambda doc : t.tokenize(doc, wakati=True)


# tf-idfでトレイニングデータとテストデータをベクトル化(自然言語処理基礎2.1.4, 2.4.2)
# 以下に回答を作成してください
vectorizer = TfidfVectorizer(use_idf=True, token_pattern="(?u)\\b\\w+\\b")
train_matrix = vectorizer.fit_transform(train_data)
test_matrix = vectorizer.transform(test_data)

# ランダムフォレストで学習
clf = RandomForestClassifier(n_estimators=2)
clf.fit(train_matrix, train_labels)

# 精度の出力
print(clf.score(train_matrix, train_labels))
print(clf.score(test_matrix, test_labels))
```

## scikit-learn における fit関数
- `fit()` ：渡されたデータの統計（最大値、最小値、平均、など）を取得して、メモリに保存。
- `transform()` ：fit()で取得した情報を用いてデータを書き換える。
    - テストデータなら、トレーニングデータの fit()結果を使うので、こちらを使う
- `fit_transform()` ：fit()の後にtransform()を実施する。
    - トレーニングデータの場合にはこちらを使えば良い


## ファイル操作などに使えるモジュール群
### glob
- ファイルやディレクトリ操作のためのモジュール。
    - 正規表現なども利用可能なので、柔軟な絞り込みなどができる。
- `glob.glob("test/*.txt")`

### with文
- `open(), read(), close()` でファイル操作はできるが、close忘れなどの問題がある
- そういった問題を起こさないために `with`が使用できる
```
with open("text/sports-watch/LICENSE.txt", "r", encoding="utf-8") as f:
    print(f.read())
```

