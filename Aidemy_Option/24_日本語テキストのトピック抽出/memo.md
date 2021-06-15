# メモ
## 日本語コーパスの例
- [青空文庫](https://www.aozora.gr.jp/)
- [現代日本語書き言葉均衡コーパス(BCCWJ)](http://pj.ninjal.ac.jp/corpus_center/bccwj/)
- [雑談対話コーパス](https://sites.google.com/site/dialoguebreakdowndetection/chat-dialogue-corpus)
- [名大会話コーパス(日本語自然会話書き起こしコーパス)](https://mmsrv.ninjal.ac.jp/nucc/)
- [日本語話し言葉コーパス(CSJ)](http://pj.ninjal.ac.jp/corpus_center/csj/)
- [livedoorニュースコーパス](https://www.rondhuit.com/download/ldcc-20140209.tar.gz)
## 形態素解析ツールの例
- ChaSen ：奈良先端科学技術大学院大学 松本研究室が開発・提供。
- JUMAN ：京都大学 黒橋・河原研究室が開発・提供。
- MeCab ：工藤拓氏が開発・オープンソースとして提供。
    - 実装例
    ```
    import MeCab

    # 形態素解析をしてください
    m = MeCab.Tagger()
    print(m.parse('すもももももももものうち'))

    # 分かち書きをしてください
    w = MeCab.Tagger('-Owakati')
    print(w.parse('すもももももももものうち'))
    ```
- Janome ：打田智子氏が開発・Pythonライブラリとして提供。
    - 実装例
    ```
    from janome.tokenizer import Tokenizer
    from janome.tokenfilter import POSKeepFilter
    from janome.analyzer import Analyzer

    # 形態素解析オブジェクトの生成
    t = Tokenizer()

    # 名詞のみ抽出するフィルターを生成
    token_filters = [POSKeepFilter(['名詞'])]

    # フィルターを搭載した解析フレームワークの生成
    analyzer = Analyzer([], t, token_filters)

    for token in analyzer.analyze('すもももももももものうち'):
    print(token)
    ```
- Rosette Base Linguistics ：ベイシステクノロジー社が開発・提供（有償）。

## 単語文書行列（term-document matrix）
- 文書に出現する単語の頻度 を表形式で表したもの
    - 各文書に含まれる単語データは形態素解析によって得ることができ、そこから各単語の出現数をカウントして数値データに変換して得られる
- ライブラリ
    - Python標準: collections.Counter()
    - scikit-learn: CountVectorizer()
        - テキストを単語に分割し、単語の出現回数を数える
        - デフォルトでは 1文字の単語はカウントされない
        - １文字の単語をカウントするオプション：`CountVectorizer(token_pattern='(?u)\\b\\w+\\b')`
- 実装例(scikit-learn)
```
from sklearn.feature_extraction.text import CountVectorizer

# `CountVectorizer()`を用いた変換器を生成します
CV = CountVectorizer()
corpus = ['This is a pen.',
          'That is a bot.',]

# `fit_transform()`で`corpus`の学習と、単語の出現回数を配列に変換します
X = CV.fit_transform(corpus)
print(X)

>>> 出力結果
  (0, 2)	1
  (0, 1)	1
  (0, 4)	1
  (1, 0)	1
  (1, 3)	1
  (1, 1)	1

# `get_feature_names()`で学習した単語が入っているリストを確認します
print(CV.get_feature_names())

>>> 出力結果
['bot', 'is', 'pen', 'that', 'this']

# カウントした出現回数を`toarray()`でベクトルに変換して表示します
print(X.toarray())

>>> 出力結果
# 行：`corpus`で与えた文章の順
# 列：`get_feature_names()`で確認した単語の順
[[0 1 1 0 1]
 [1 1 0 1 0]]
```
### 重みあり単語文書行列
- 単語文書行列では、どの文書においても普遍的に出現する単語（例えば「私」「です」など）の出現頻度が高くなる
    - その結果、特定の文書にのみ出現する単語の出現頻度が低くなり、単語から各文書を特徴付けることが難しくなる
- そこで用いられるのが、TF-IDF値: TF（Term Frequency）に逆文書頻度 IDF（Inverse Document Frequency）を掛けた値
    - IDF: log（総文書数／ある単語が出現する文書数）＋ 1 
- 実装例
    1. vectorizer = TfidfVectorizer()で、ベクトル表現化（単語を数値化すること）を行う変換器を生成
        - use_idf=Falseにすると、tfのみの重み付け になる
    2. vectorizer.fit_transform()で、文書をベクトルに変換。引数には、空白文字によって分割された（分かち書きされた）配列を与える。
    3. toarray()によって出力をNumPyのndarray配列に変換。

```
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 小数点以下を有効数字２桁で表示する
np.set_printoptions(precision=2)
docs = np.array([
    "白　黒　赤", "白　白　黒", "赤　黒"
])

# `TfidfVectorizer()`を用いた変換器を生成します
vectorizer = TfidfVectorizer(use_idf=True, token_pattern="(?u)\\b\\w+\\b")

# `fit_transform()`で`docs`の学習と、重み付けされた単語の出現回数を配列に変換します
vecs = vectorizer.fit_transform(docs)
print(vecs.toarray())
```

## 特徴量
- 単語文書行列における特徴量
    - CountVectorizer()で作成した単語文書行列では単語の出現回数
    - TfidfVectorizer()で作成した単語文書行列では単語のTF-IDF値
### 類似度
- ２つの単語の出現の仕方がどの程度似ているか、といった特徴量
    - 相関係数
    - コサイン類似度: ベクトル同士の類似度を測る
    - Jaccard係数: 集合同士の類似度を測る 
- `corr = DataFrame.corr()`
    - 'pearson'：ピアソンの積率相関係数（デフォルト）
    - 'kendall'：ケンドールの順位相関係数
    - 'spearman'：スピアマンの順位相関係数
- 実装例
```
import os
import json
import pandas as pd
import numpy as np
import re
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


# init100ディレクトリを指定
file_path = './6110_nlp_preprocessing_data/init100/'
file_dir = os.listdir(file_path)

# フラグと発話内容を格納する空のリストを作成
label_text = []

# JSONファイルを1ファイルずつ10ファイル分処理
for file in file_dir[:10]:
    r = open(file_path + file, 'r', encoding='utf-8')
    json_data = json.load(r)

    # 発話データ配列`turns`から発話内容とフラグを抽出
    for turn in json_data['turns']:
        turn_index = turn['turn-index']
        speaker = turn['speaker']
        utterance = turn['utterance']
        if turn_index != 0:
            if speaker == 'U':
                u_text = utterance
            else:
                for annotate in turn['annotations']:
                    a = annotate['breakdown']
                    tmp1 = str(a) + '\t' + u_text
                    tmp2 = tmp1.split('\t')
                    label_text.append(tmp2)

# リスト`label_text`をDataFrameに変換し、重複を削除
df_label_text = pd.DataFrame(label_text)
df_label_text = df_label_text.drop_duplicates()
# 破綻ではない発話のみを抽出
df_label_text_O = df_label_text[df_label_text[0] == 'O']

t = Tokenizer()

# 空の破綻ではない発話データセットを作成
morpO = []

# 数字とアルファベットの大文字・小文字を除去
for row in df_label_text_O.values.tolist():
    reg_row = re.sub('[0-9a-zA-Z]+', '', row[1])
    reg_row = reg_row.replace('\n', '')
    # Janomeで形態素解析
    tmp1 = []
    tmp2 = ''
    for token in t.tokenize(reg_row):
        tmp1.append(token.surface)
        tmp2 = ' '.join(tmp1)
    morpO.append(tmp2)

# TF-IDF値による重みありの単語文書行列を作成
morpO_array = np.array(morpO)
tfidf_vecO = TfidfVectorizer(use_idf=True)
morpO_tfidf_vecs = tfidf_vecO.fit_transform(morpO_array)
morpO_tfidf_array = morpO_tfidf_vecs.toarray()

# 単語の出現回数をDataFrame形式に変換
dtmO = pd.DataFrame(morpO_tfidf_array, columns=tfidf_vecO.get_feature_names(), 
             index=morpO)

# 相関行列を作成してください
corr_matrixO = dtmO.corr()

# 相関行列の表示
corr_matrixO.head(20)
```
## N-gram
- テキストを連続したN個の文字で分割する方法
- 単語の連続性（どの順番で出現するか）に着目
- トピックの抽出を行う場合には、N-gramモデルを作成する
    - 分類なら、単語文書行列を作成する
- 実装例
```
import os
import json
import pandas as pd
import re
from janome.tokenizer import Tokenizer
from collections import Counter
import itertools
import networkx as nx
import matplotlib.pyplot as plt


# 破綻ではない発話データセットの作成
file_path = './6110_nlp_preprocessing_data/init100/'
file_dir = os.listdir(file_path)

label_text = []
for file in file_dir[:100]:
    r = open(file_path + file, 'r', encoding='utf-8')
    json_data = json.load(r)
    for turn in json_data['turns']:
        turn_index = turn['turn-index']
        speaker = turn['speaker']
        utterance = turn['utterance']
        if turn_index != 0:
            if speaker == 'U':
                u_text = utterance
            else:
                for annotate in turn['annotations']:
                    a = annotate['breakdown']
                    tmp1 = str(a) + '\t' + u_text
                    tmp2 = tmp1.split('\t')
                    label_text.append(tmp2)

df_label_text = pd.DataFrame(label_text)
df_label_text = df_label_text.drop_duplicates()
df_label_text_O = df_label_text[df_label_text[0] == 'O']

# 分かち書きし、正規表現で不要な文字列を除去
t = Tokenizer()
wakatiO = []
for row in df_label_text_O.values.tolist():
    reg_row = re.sub('[0-9a-zA-Z]+', '', row[1])
    reg_row = reg_row.replace('\n', '')
    tmp1 = t.tokenize(reg_row, wakati=True)
    wakatiO.append(tmp1)

# 単語の出現数をカウント、並べ替え、dicに追加
word_freq = Counter(itertools.chain(*wakatiO))
dic = []
for word_uniq in word_freq.most_common():
    dic.append(word_uniq[0])

# 単語にIDを付与し辞書を作成
dic_inv = {}
for i, word_uniq in enumerate(dic, start=1):
    dic_inv.update({word_uniq: i})

# 単語をIDへ変換
wakatiO_n = [[dic_inv[word] for word in waka] for waka in wakatiO]

# 2-gramリストを作成
bigramO = []

for i in range(0, len(wakatiO_n)):
    row = wakatiO_n[i]
    # 2-gramの作成
    tmp = []
    for j in range(len(row)-1):
        tmp.append([row[j], row[j+1]])
    bigramO.extend(tmp)

# 配列`bigramO`をDataFrameに変換しcolumnを設定
df_bigramO = pd.DataFrame(bigramO)
df_bigramO = df_bigramO.rename(columns={0: 'node1', 1: 'node2'})

# `weight`列を追加し、値を1で統一する
df_bigramO['weight'] = 1

# 2-gram個数をカウント
df_bigramO = df_bigramO.groupby(['node1', 'node2'], as_index=False).sum()

# 出現数が1を超えるリストを抽出
df_bigramO = df_bigramO[df_bigramO['weight'] > 1]

# 有向グラフの作成
G_bigramO = nx.from_pandas_edgelist(
    df_bigramO, 'node1', 'node2', ['weight'], nx.DiGraph)

# 破綻ではない発話ネットワーク
# 入次数の度数を求めてください
indegree = sorted([d for n, d in G_bigramO.in_degree(weight='weight')], reverse=True)
indegreeCount = Counter(indegree)
indeg, cnt = zip(*indegreeCount.items())

# 出次数の度数を求めてください
outdegree = sorted([d for n, d in G_bigramO.out_degree(weight='weight')], reverse=True)
outdegreeCount = Counter(outdegree)
outdeg, cnt = zip(*outdegreeCount.items())

# 次数分布の作成
plt.subplot(1, 2, 1)
plt.bar(indeg, cnt, color='r')
plt.title('in_degree')
plt.subplot(1, 2, 2)
plt.bar(outdeg, cnt, color='b')
plt.title('out_degree')
plt.show()
```

## 類似度ネットワーク
- ネットワーク: 対象と対象の関係を表現する方法 の一つ
    - 対象はノードで、関係はエッジで表現される
    - エッジ(関係)は重みを持つ
- 無向グラフ: エッジに方向の概念がないもの 
- 有向グラフ: エッジに方向の概念があるもの 
- 実装例
    - df：グラフの元となるPandasのDataFrame名
    - source：ソースノードの列名
        - str（文字列型）またはint（整数型）で指定する
    - target：対象ノードの列名
        - trまたはintで指定する
    - edge_attr：それぞれのデータのエッジ（重み）      
        - strまたはint、iterable、Trueで指定する
    - create_using：グラフのタイプ（オプション）
        - 無向グラフ：nx.Graph（デフォルト）
        - 有向グラフ：nx.DiGraph
```
# ライブラリ`NetworkX`をimport
import networkx as nx

# 無向グラフの作成
network = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=None, create_using=None)
```
- 可視化の実装例
```
# ライブラリ`Matplotlib`から`pyplot`をimport
from matplotlib import pyplot

# 各ノードの最適な表示位置を計算
pos = nx.spring_layout(graph)

# グラフを描画
nx.draw_networkx(graph, pos)

# Matplolibを用いてグラフを表示
plt.show()
```
## ネットワークの特徴
- 次数：ノードが持つエッジの本数を表します。
- 次数分布：ある次数を持つノード数のヒストグラムを表します。
- クラスタ係数：ノード間がどの程度密に繋がっているかを表します。
- 経路長：あるノードから他のノードへ至るまでの距離です。
- 中心性：あるノードがネットワークにおいて中心的な役割を果たす度合いを表します。
### 類似度ネットワークにおける特徴
- クラスタ係数：高いほど、単語間のつながりの密度が高い
    - 実装例: `nx.average_clustering(G, weight=None)`
- 媒介中心性：高いほど、単語がハブ（色々な単語をつなぐ）になっている
    - 実装例: `nx.betweenness_centrality(G, weight=None)`
- 入次数: 
    - 実装例: `G_bigramO.in_degree(weight='weight')`
- 出次数:
    - 実装例: `G_bigramO.out_degree(weight='weight')`

## ネットワークにおけるコミュニティ
- ネットワークは、複数の部分ネットワークに分けられる＝コミュニティ
- モジュラリティ: コミュニティ内のノードのつながりの大きさ
    - 実装例: `greedy_modularity_communities(G, weight=None)`

