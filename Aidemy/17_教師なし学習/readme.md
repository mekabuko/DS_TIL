# サマリ

# メモ
## ユークリッド距離
- `((x1 - y1)**2 + (x2 - y2)**2 + ... + (xn - yn)**2)**(1/2)` で求められる距離
- **ノルム**とも
- numpy では、`np.linalg.norm(series1 - series2)`で計算可能

## コサイン類似度
- ベクトルの内角θを用いて、より内角が小さい＝類似している、と考える
- 内角θが小さくなるほど大きくなる`cosθ`を用いて、コサイン類似度という指標として使える
- numpyでは `np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))` で計算可能

## 階層的クラスタリング
- 順番に似ている組み合わせを探していく
- 途中経過で階層構造となる
- この、データがどう纏まって行ったのかを示す樹形図をデンドログラムと呼ぶ

## 非階層的クラスタリング
- 最初からいくつのクラスタに分割するかを決めておく必要がある

## sklearn.datasets.make_blobs によるデータ生成
- 指定したクラスタ数のデータを生成できる`make_blobs`
- サンプル(引用)
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Xには1つのプロットの(x,y)が、Yにはそのプロットの所属するクラスター番号が入る
X, Y = make_blobs(n_samples=150,       # サンプル点の総数
                  n_features=2,          # 特徴量（次元数）の指定 
                  centers=2,             # クラスタの個数
                  cluster_std=0.5,       # クラスタ内の標準偏差
                  shuffle=True,          # サンプルをシャッフル
                  random_state=0)        # 乱数生成器の状態を指定

plt.scatter(X[:, 0], X[:, 1], c="black", marker="*", s=50)
plt.grid()
plt.show()
```

## k-means法
- 非階層的クラスタリングの代表的な手法
- データを分散の等しいn個のクラスターに分けることができる
- データの重心＝セントロイドを適当に動かしながら、SSE=データ点とセントロイドの距離、を等しく、かつ最小化するようにデータを分割していく。
    - 出来るだけクラスターの中心にデータが集まるようにクラスタリングする
    - 必然的にクラスターは円形(球状)に近い形を取る
    - クラスターの大きさ・形に偏りがないときは効果を発揮
- クラスターの大きさ・形に偏りがあるデータの場合は良いクラスタリングができない傾向にある

### エルボー法
- クラスタ数を大きくして行ったときに、SSEが大きく変化した点を最適と見なす方法
- ハイパーパラメータであるクラスタ数を求めるために使用する
- サンプル(引用)
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# サンプルデータの生成
X, Y = make_blobs(n_samples=150, n_features=2, centers=3,
                  cluster_std=0.5, shuffle=True, random_state=0)

distortions = []
for i in range(1, 11):                # クラスター数1~10を一気に計算
    km = KMeans(n_clusters=i,
                init="k-means++",     # k-means++法によりクラスタ中心を選択
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)                         # クラスタリングのを実行
    distortions.append(km.inertia_)   # km.fitするとkm.inertia_が得られる

# グラフのプロット
plt.plot(range(1, 11), distortions, marker="o")
plt.xticks(np.arange(1, 11, 1))
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.show()
```

## DMSCAN
- 非階層クラスタリングの１手法
- クラスターの高密度(データが凝集している)の場所を低密度の場所から分離して捉える
- クラスターサイズ・形に偏りがある際に真価を発揮
- min_samplesとepsの2つの指標を用いて、次の3種類にデータ点を分類
    1. あるデータの半径 eps 内に min_sample 数だけのデータがある場合 = コア点
        - コア点でクラスタを形成
    2. コア点ではないが、コア点から半径 eps 内に入っているデータ = ボーダー点
        - ボーダーテンは一番近いコア点のクラスタに入る
    3. どちらにも満たさないデータ点 = ノイズ点
        - ノイズは除去
- サンプル（引用）
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# 月型のデータを生成
X, Y = make_moons(n_samples=200, noise=0.05, random_state=0)

# グラフと2つの軸を定義 左のax1はk-means法用、右のax2はDBSCAN用
f, (ax2) = plt.subplots(1, 1, figsize=(8, 3))

# DBSCANでクラスタリング # コードを完成してください
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric="euclidean")
Y_db = db.fit_predict(X)

ax2.scatter(X[Y_db == 0, 0], X[Y_db == 0, 1], c="lightblue",
            marker="o", s=40, label="cluster 1")
ax2.scatter(X[Y_db == 1, 0], X[Y_db == 1, 1], c="red",
            marker="s", s=40, label="cluster 2")
ax2.set_title("DBSCAN clustering")
ax2.legend()
plt.show()
```

## 主成分分析
- scikit-learnにおけるサンプル(引用)
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

df_wine = pd.read_csv("./5030_unsupervised_learning_data/wine.csv", header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X = (X - X.mean(axis=0)) / X.std(axis=0)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可視化
color = ["r", "b", "g"]
marker = ["s", "x", "o"]
for label, color, marker in zip(np.unique(y), color, marker):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                c=color, marker=marker, label=label)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.show()
```

## カーネル主成分分析
- 線形分離不可能を、カーネルトリックにより写像変換し、線形分離可能にする
- サンプル（引用）
```
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

# 月形データを取得
X, y = make_moons(n_samples=100, random_state=123)

# KernelPCAクラスをインスタンス化
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
# データXをKernelPCAを用いて変換
X_kpca =kpca.fit_transform(X)

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.scatter(X[y == 0, 0], X[y == 0, 1], c="r")
ax1.scatter(X[y == 1, 0], X[y == 1, 1], c="b")
ax1.set_title("moon_data")
ax2.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], c="r")
ax2.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], c="b")
ax2.set_title("kernel_PCA")
plt.show()
```