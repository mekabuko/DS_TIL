# 1.ネットワーク分析のための基礎知識を身につけよう
## 1.1 グラフのイメージをつかもう
### 1.1.1 ネットワーク分析の応用例
- ネットワーク分析: グラフを用いて様々な事象と事象の関係について調べることを目的とした、分析手法
    - ネットワーク: 点と線からなる関係構造
    - グラフ: ネットワークを可視化したもの
- 適用領域
    - ソーシャルネットワーク
    - 企業間ネットワーク
    - 交通ネットワーク
    - コンピューターネットワーク
    - 論文のネットワーク(萌芽領域の特定)
        - 萌芽領域とは近いうちに伸びると予測される領域
        - 実装として、例えば論文をノードにその引用関係でネットワーク構造を作り、クラスタリングをしたものを時系列順に並べることでそれぞれの領域の拡大推移を見ることができ、萌芽領域を特定することができる
### 1.1.2 グラフを描画してみよう
- 実装例
```
import networkx as nx
import matplotlib.pyplot as plt

# 一繋ぎの5個の点を持つグラフに初期化
G = nx.path_graph(5)

# 点5, 6を結ぶ線を追加
G.add_edge(5, 6)

nx.draw_networkx(G)
plt.show()
```
### 1.1.3 グラフに点や辺を加えてみよう
## 1.2 グラフ理論の基本事項を学ぼう
### 1.2.1 グラフ理論の基本用語を学ぼう①
- グラフ理論: グラフを数学的に扱う分野
- 基礎用語
    - 頂点(vertex) : グラフを構成する点(ノードと呼ばれることもある)
    - 辺(edge) : 頂点を結ぶ線(エッジと呼ばれることもある)
    - 有向グラフ(directed path) : 一方通行であるような辺が存在するグラフ
        - 実装例: `G = nx.DiGraph()`
    - 無向グラフ(undirected path) : すべての辺が双方向であるようなグラフ
        - 実装例: `G = nx.path_graph()`
    - 道(path) : 頂点と辺が代わる代わる連続したもので同じ頂点が二度以上現れないもの
### 1.2.2 グラフ理論の基本用語を学ぼう②
- 重み付きグラフ
    - グラフの各辺に数字が書いてあるもの
    - その数字を「重み」と呼ぶ
- 最短経路
    - ある頂点から頂点への経路の内、通過する辺の重みの和が最小になるような道
- 閉路
    - 閉じた道。辿っていくと、同じところに必ず戻ってくる。
- 木
    - 閉路を持たないようなグラフ(有向/無向ともに)
        - (辺の数) = (頂点の数) - 1 である
    - 根：木の一番上の頂点
    - 葉：一番下の頂点
    - 自分より上の頂点を親、下の頂点を子、と呼ぶ
- 二部グラフ
    - 頂点を2つのグループに分けて、各グループ内の頂点同士を結ぶような辺がなく、別々のグループ同士の頂点を結ぶ辺しか存在しないようなもの
### 1.2.3 グラフ理論の基本用語を学ぼう③
- 強連結
    - 任意の2頂点間に道が存在する有向グラフ
    - どの頂点と頂点の間にも、有向な道が存在
- 弱連結
    - 各辺の方向を無視した時に任意の2頂点間に道が存在する有向グラフ
    - 有向だとたどることのできない２頂点が存在
- 隣接行列
    - 隣接すれば1, しなければ0 として、ネットワークを表現した「行列」
    - 行列計算する際の計算が扱いやすい
- 隣接リスト
    - 隣接すれば1, しなければ0 として、ネットワークを表現した「リスト」
    - 消費するメモリが少ない
    - ある頂点から辺を一本通って行ける頂点を調べたい時には特に有効
- 次数
    - ある頂点につながっている辺の数
    - 入次数: 自身に入ってくるような辺の数
    - 出次数: 自身から出て行くような辺の数
## 1.3 networkxを学ぼう
### 1.3.1 自分でグラフを描画してみよう
### 1.3.2 色々なグラフを描画してみよう
```
import networkx as nx
import matplotlib.pyplot as plt

# 初期化
G = nx.DiGraph()

# エッジ（辺）をまとめて追加する
G.add_edges_from([(0, 1), (1, 0), (2, 3), (2, 1), (3, 4)])

# ノード（頂点）を描画する位置を決める
pos = nx.spring_layout(G)

# グラフの描画
# draw_networkx_nodes()でノード(頂点)の描画
# draw_networkx_edges()でエッジ(辺)の描画

# ---引数の説明---
# G: graph nodes
# pos: ポジション(必須)
# nodelist: 描画したいノード(頂点)のリスト
# node_color: ノードの色
# node_size: ノードの大きさ
# alpha: アルファ値(透過度) 0から1の範囲の値
# width: ノードの太さ

nx.draw_networkx_nodes(
    G, pos, nodelist=[1, 2], node_color="blue", node_size=1000, alpha=0.5)
nx.draw_networkx_nodes(
    G, pos, nodelist=[0], node_color="yellow", node_size=500, alpha=1)
nx.draw_networkx_edges(G, pos, edgelist=[(
    0, 1), (1, 2), (2, 3), (3, 0), (3, 1)], , alpha=0.5, edge_color="black")
plt.show()
```
### 1.3.3 networkxの様々なメソッドを使ってみよう①
- `nx.info(G)`: graph nodes Gの基本情報を表示
    - Type:　グラフの種類
    - Number of nodes:　ノード(頂点)の総数
    - Number of edges:　エッジ　(道)の総数
    - Average degree:　全てのノードから出ているエッジの平均数
- `G.nodes()`: ノードをlist型で出力
- `G.degree()`: 全ての点における次数(ある点が隣接する他の点の数)
- `G.edges()`: Gのエッジの両端のノードをリスト型に出力
- `nx.adjacency_matrix(G)`: Gの隣接行列を入手
    - 例: (4,6)間のエッジ : `nx.adjacency_matrix(G)[(4, 6)]`
### 1.3.4 networkxの様々なメソッドを使ってみよう②
- `connected_component_subgraphs()`: グラフを連結成分で分割
- `all_pairs_dijkstra_path()`: ノード間の最短経路を求める
# 2.karate clubネットワークの分析をしてみよう
## 2.1 簡単なネットワークの分析をしてみよう
### 2.1.1 karate clubに触れよう
### 2.1.2 中心人物を見つけよう①
- 中心性: 各頂点がどの程度、ネットワーク構造上有利な状態にあるのか、という指標を数値化したもの
- 固有ベクトル中心性:
    - より多くの人と繋がりを持ったほうが有利である
    - 多くの人と繋がりを持っている人と、繋がりを持っている方が有利である
    - `nx.eigenvector_centrality_numpy(G)`
### 2.1.3 中心人物を見つけよう②
- 媒介中心性:
    - 情報伝達の役割を担っている頂点を"有利な状態"と考えて定義
    - `nx.communicability_betweenness_centrality(G)`
### 2.1.4 一番中心性が高い頂点を求めてみよう
- 固有ベクトル中心性の最も高い頂点を求める実装例
    - `output = max(list(eignv_cent.keys()), key=lambda val : eignv_cent[val])`
### 2.1.5 中心性の大きさに応じて頂点の大きさを変えよう
- 媒介中心性の大きさを描画のノードの大きさとする実装例
    - `node_size = [10000 * size for size in list(between_cent.values())]`
## 2.2 グラフを分割しよう
### 2.2.1 連結成分
- サブグループ: グラフを分割したときのそれぞれのグループ
- 連結成分: グラフがいくつかの繋がっている部分に分割されるときの、それぞれのサブグループ
    - 連結成分に分割する例: `G_sub_graph = list(nx.connected_component_subgraphs(G))`
### 2.2.2 クリーク
- グラフが全て繋がっているとき(全結合グラフ)には連結成分への分割はできない
- その場合には、グラフの密度に着目した分割を行う
    - 密度： n頂点m辺の場合
        - 有向グラフ: m/(n(n-1))
        - 無向グラフ: 2m/(n(n-1))
- クリーク: 頂点が三つ以上あるサブグラフで、密度が１となるようなサブグラフ
    - サブグラフに含まれる頂点間の距離が全て1である
### 2.2.3 n-クリーク
- クリークになることはほとんどない
- なので、クリークの条件を緩くしたものが n-クリーク
    - サブグラフに含まれる頂点間の距離の最大値がn以下である
### 2.2.4 
- ネットワーク分析におけるクラスタリング = グラフ理論を用いた分類手法
- クラスタリング: 連結なグラフをいくつかのサブグラフに分割すること
    - `community.best_partition(G)`
- コミュニティ: 分割された各サブグラフ
### 2.2.5 コミュニティごとに色を分けて描画しよう
### 2.2.6 モジュラリティ指標
- モジュラリティ指標:
    - 各コミュニティ内の結びつきが強く、コミュニティとコミュニティの結びつきが弱いような分割を「良い」とする指標
    - `community.modularity(partition, G)`
# 3.実際のデータを用いてネットワーク分析をしよう
## 3.1 JSONファイルを扱う
### 3.1.1 JSONファイルを知ろう
### 3.1.2 JSONファイルを読み込もう
## 3.2 実際のデータを用いてネットワークを分析してみよう
### 3.2.1 データの読み込みと整理
### 3.2.2 整理したデータの保存
### 3.2.3 グラフの描画
```
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np

G = nx.Graph()
f = open('./6150_network_analysis_data/my_sample_data.json')
data = json.load(f)
f.close()

# dataをint型に変更しています
data_int = {}
for key in data.keys():
    sub_data = {}
    for sub_key in data[key].keys():
        sub_data[int(sub_key)] = data[key][sub_key]
        data_int[int(key)] = sub_data

# 11~17行目は以下のように1行で書くこともできます
# data_int = {int(key): {int(sub_key): data[key][sub_key] for sub_key in data[key].keys()} for key in data.keys()}
for key in data_int.keys():
    for sub_key in data_int[key].keys():
        if(key < sub_key):
            # add_edge()内を埋めて、keyとsub_keyのノードをweightを指定して繋げてください
            G.add_edge(key, sub_key, weight = data_int[key][sub_key])

# ノード間の重みを辺の太さにしたグラフを作っています
plt.figure(figsize=(20, 20))
pos=nx.spring_layout(G, k=4)
nx.draw_networkx_nodes(G, pos, node_size=400)
nx.draw_networkx_labels(G, pos, font_size=15)
width = np.array([d['weight'] for (u, v, d) in G.edges(data=True)])
width_std = 5 * (width - min(width)) / (max(width) - min(width)) + 0.1
nx.draw_networkx_edges(G, pos, width=width_std)

plt.show()
```
### 3.2.4 グラフのビジュアルを整える
```
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json


G = nx.Graph()

f = open('./6150_network_analysis_data/my_sample_data.json')
data = json.load(f)
f.close()

# データをint型に直しています
data = {int(key) : {int(sub_key) : data[key][sub_key] for sub_key in data[key].keys()} for key in data.keys()}

# ２つのノードとその重みをまとめてリストにしています
edge_list = [(key, sub_key, data[key][sub_key]) for key in data.keys() for sub_key in data[key].keys() if sub_key > key]

G.add_weighted_edges_from(edge_list)

plt.figure(figsize=(20, 20))

# 固有ベクトル中心性
eigv_cent = nx.eigenvector_centrality_numpy(G)

# node_sizeを指定してください
node_size = np.array(list(eigv_cent.values())) * 10000

# 中心性をノードの大きさにしてグラフを描画しています
pos=nx.spring_layout(G, k=4)
nx.draw_networkx_nodes(G, pos, node_size=node_size)
nx.draw_networkx_labels(G, pos, fontsize=15)
width = np.array([d['weight'] for (u, v, d) in G.edges(data=True)])
width_std = 5 * (width - min(width)) / (max(width) - min(width)) + 0.1
nx.draw_networkx_edges(G, pos, width=width_std)

plt.show()
```
### 3.2.5 クラスタリングをしよう
```
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import community


G = nx.Graph()

f = open('./6150_network_analysis_data/my_sample_data.json')
data = json.load(f)
f.close()

data = {int(key) : {int(sub_key) : data[key][sub_key] for sub_key in data[key].keys()} for key in data.keys()}

edge_list = [(key, sub_key, data[key][sub_key]) for key in data.keys() for sub_key in data[key].keys() if sub_key > key]

G.add_weighted_edges_from(edge_list)

plt.figure(figsize=(20, 20))            

# 固有ベクトル中心性
eigv_cent = nx.eigenvector_centrality_numpy(G, weight='weight')

# コミュニティを辞書型で格納しています
partition = community.best_partition(G, weight='weight')

# nodeのサイズを指定しています
node_size = np.array(list(eigv_cent.values())) * 10000

# nodeの色を指定して、リスト型にしてください
node_color = [partition[i] for i in G.nodes()]

pos=nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_color, cmap=plt.cm.RdYlBu)
nx.draw_networkx_labels(G, pos, fontsize=15)
nx.draw_networkx_edges(G, pos, width=0.2)

plt.show()
```
### 3.2.6 中心性が高い人物を抽出しよう
```
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import community


G = nx.Graph()
f = open('./6150_network_analysis_data/my_sample_data.json')
data = json.load(f)
f.close()

data = {int(key): {int(sub_key): data[key][sub_key]
                   for sub_key in data[key].keys()} for key in data.keys()}
edge_list = [(key, sub_key, data[key][sub_key]) for key in data.keys()
             for sub_key in data[key].keys() if sub_key > key]

G.add_weighted_edges_from(edge_list)

# クラスタリングをしてください
partition = community.best_partition(G)

# コミュニティごとに頂点を分けてリストにいます
part_com = [[] for _ in set(list(partition.values()))]
for key in partition.keys():
    part_com[partition[key]].append(key)

# 各コミュニティごとに媒介中心性の最大値およびその頂点を求めています
for part in part_com:
    # グラフの初期化
    G_part = nx.Graph()
    for edge in edge_list:
        if edge[0] in part and edge[1] in part:
            G_part.add_weighted_edges_from([edge])
    print(max(G_part.nodes(), key=lambda val: nx.betweenness_centrality(
        G_part, weight='weight')[val]))
```
### 3.2.7 コミュニティーよるモジュラリティー指標
```
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import community

G = nx.Graph()

f = open('./6150_network_analysis_data/my_sample_data.json')
data = json.load(f)
f.close()

data = {int(key): {int(sub_key): data[key][sub_key]
                   for sub_key in data[key].keys()} for key in data.keys()}
edge_list = [(key, sub_key, data[key][sub_key]) for key in data.keys()
             for sub_key in data[key].keys() if sub_key > key]

G.add_weighted_edges_from(edge_list)

eigv_cent = nx.eigenvector_centrality_numpy(G, weight='weight')

partition1 = community.best_partition(G, weight='weight')
partition2 = community.best_partition(G, partition={i : 0 if i < 10 else 1 for i in G.nodes()}, weight='weight')

# partition1, partition2のそれぞれのモジュラリティー指標を出力してください
print(community.modularity(partition1, G))
print(community.modularity(partition2, G))
```