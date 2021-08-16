# 1. バイオインフォマティクス(オミックス)データと機械学習の概要
## 1.1 バイオインフォマティクスデータの概要
### 1.1.1 医学/生物学系のデータと機械学習
- 医学/生物学系のデータ例
    - X線写真やCT、MRIなどの画像データ -> 画像データ
    - 心電図や脳波などの波形データ -> 時系列データ
    - 血液成分が数値で表現される検査データ -> 多次元数値データ
### 1.1.2 バイオインフォマティクス(オミックス)データの概要
- バイオインフォマティクス: 生命情報科学のことで、生命の持つ情報を情報科学や統計学を用いて分析する学問
- オミックスデータ: 以下のようなデータ
    - 遺伝子配列データ：ゲノム(Genome)データ
    - 遺伝子発現データ: トランスクリプトーム(Transcriptome)データ
    - タンパク発現データ：プロテオーム(Proteome)データ
    - 代謝産物の網羅的データ：メタボローム(Metabolome)データ
### 1.1.3 遺伝子発現データの概要
- 遺伝子発現データ: 細胞内で各種遺伝子がどのくらい発現しているかを定量的に測定したデータ
- PCR産物の増加量をリアルタイムで調べることによって発現量を算出する定量的PCR(quantitative polymerase chain reaction (qPCR))という手法が手軽で一般的
- また既知(アノテーションされている)遺伝子に関して、細胞の中にどのくらい発現しているかを主にRNAの発現に注目して網羅的に測定する方法が最近開発されている
    - cDNAマイクロアレイ(microarry)
    - RNAシークエンス(RNA-seq)
### 1.1.4 遺伝子発現データ各論：RNA-seqの概要
- RNA シークエンス: 細胞内で発現しているRNAを回収し、そのままもしくは逆転写反応(cDNAの合成)を使ってcDNAを作らせた後、断片化したRNAもしくはcDNAの配列を、シークエンサー(塩基配列やアミノ酸配列などを解析する装置)を用いて読みとる方法
- 読み取られた情報はそれだけだと意味をなさない情報
    - マッピング(ゲノム配列情報と突き合わせること)により、染色体のどの部分の遺伝子がどれだけ細胞内で発現しているかがわかる
- RNAシークエンスデータはfastqと呼ばれる特殊なファイル形式で出力されるため、前処理が必要
    1. データの質の評価(quality control (QC)) : fastqcなどのソフトで行う。
    2. マッピング：Bowtie, STAR, Kallistoなどのソフトで行う。
    3. 正規化：edgeR, Cufflinksなどのソフトで行う。
        - CPM (counts per million)
        - FPKM (fragments per kilobase of exon per million mapped fragments)
        - RPKM (reads per kilobase of exon per million mapped reads)
        - TPM (transcripts per million)
## 1.2 機械学習の概要
### 1.2.1 機械学習概論
# 2. 遺伝子発現データを使って教師なし学習をさせてみよう
## 2.1 遺伝子データをつかって主成分分析させてみよう
### 2.1.1 次元削減とデータの可視化
### 2.1.2 主成分分析(PCA)の概要
- 主成分分析: 多次元のデータに対し、散布図において分散を最大にするような互いに直行する元の次元よりも少ない次元数の主成分軸を引く手法
    - ばらつきが最大のものを第一主成分軸、ばらつきが２番目に大きいのものを第二主成分軸と言う
## 2.2 遺伝子発現データに触れてみよう
### 2.2.1 遺伝子発現データファイル
### 2.2.2 メタデータファイル
## 2.3 遺伝子発現データを主成分分析してみよう
### 2.3.1 主成分分析実装 1
### 2.3.2 主成分分析実装 2
## 2.4 可視化・解析手法
### 2.4.1 MDS (multi-dimesional scaling)
- MDS: 遺伝子発現データなどのバイオインフォマティクスデータの可視化や解析に用いられる手法の一つ
    - データに含まれるサンプル間の類似度に基づき、別の次元でサンプルを配置する手法
    - 性質が類似するサンプルは近くに、性質が異なるサンプルは遠くに配置
### 2.4.2 t-SNE (t-distributed Stochastic Neighbor Embedding)
- t-SNE: 主成分分析以外に遺伝子発現データなどのバイオインフォマティクスデータの可視化や解析に用いられる手法の一つ
    - サンプル間の近さを確率分布の形で表現するのが特徴
    - 次元削減前の基準点からそれぞれのサンプルまでの距離の確率分布と、次元削減後(2次元)の同様の確率分布ができるだけ近くなるように機械学習させる手法
## 2.5 遺伝子データをつかってクラスタリングしてみよう
### 2.5.1 クラスタリングとは
### 2.5.2 k-means法について
### 2.5.3 k-means法による乳がん遺伝子発現データのクラスタリング
### 2.5.4 シルエットスコアを使用してクラスタリングの妥当性を評価してみよう
- シルエット係数: クラスタリングの妥当性の評価指標の一つ
    - クラスター内のサンプルがどの程度凝集しているか、一つのクラスターが隣接するクラスターとどのくらい離れているかを合わせて評価した指標
    - 1から-1の値をとり、0に近い値は、重複するクラスターを示す
    - 負の値は、異なるクラスターがより類似しているため、サンプルが誤ったクラスターに割り当てられていることを示す
# 3. 遺伝子発現データをつかって、がんの診断してみよう
## 3.1 教師あり学習(分類)によるがんの診断
### 3.1.1 ロジスティック回帰
```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# seed値の固定
np.random.seed(10)

# データの読み込み
X = pd.read_csv("./5150_gene_data/Breast_500.csv", sep=",", header=0, index_col=0, nrows=300)

# データ分析しやすいようにデータを転置
X = X.T 

# メタデータファイル(Sample_data.csv)の読み込み
sp = pd.read_csv("./5150_gene_data/Sample_data.csv", sep=",", header=0, nrows=300)

# 正常組織とがん組織を区別するラベルのはいったカラム"Type 2"を配列yに格納
y = np.array(sp["Type 2"].tolist())

# 正常組織とがん組織を区別するラベルのはいったカラム"Type 2"を配列pre_cluster_lebelsに格納
pre_cluster_labels = np.array(sp["Type 2"].tolist())

# ラベルに応じて色を割り振って色分け表(カラーマップ)の作成
cmap = plt.cm.get_cmap('gnuplot', len(pre_cluster_labels))

# 以下のコードを完成させて、TSNEモデルを構築してください
TS = TSNE()

# 以下のコードを完成させて、データから変換モデルを学習し、変換
TS_coords = TS.fit_transform(X[:300])

X1 = TS_coords

train_X, test_X, train_y, test_y = train_test_split(X1, y, random_state=42)

# ロジスティック回帰のモデルを構築してください
model = LogisticRegression()

# train_Xとtrain_yを使ってモデルに学習
model.fit(train_X, train_y)

# test_Xに対するモデルの予測をしてください
pred_y = model.predict(test_X)

# F値の算出
print("LogisticRegression: {:.4f}".format(f1_score(pre_cluster_labels[:75], pred_y)))

# 生成したデータをプロット
plt.scatter(TS_coords[:,0], TS_coords[:,1], c=y, marker=".",cmap=get_cmap(name="bwr"), alpha=0.7)

# 学習して導出した識別境界線のプロット
Xi = np.linspace(-60000, 60000)
Y = -model.coef_[0][0] / model.coef_[0][1] * \
    Xi - model.intercept_ / model.coef_[0][1]
plt.plot(Xi, Y)

# グラフのスケールを調整
plt.xlim(min(TS_coords[:,0]) - 35, max(TS_coords[:,0]) + 35)
plt.ylim(min(TS_coords[:,0]) - 35, max(TS_coords[:,0]) + 35)
plt.axes().set_aspect("equal", "datalim")

# グラフにタイトルの設定
plt.title("classification data using LogisticRegression")

# x軸、y軸それぞれに名前を設定
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()
```
### 3.1.2 ロジスティック回帰によるがん診断の実装
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# seed値の固定
np.random.seed(42)

# データの読み込み
X = pd.read_csv("./5150_gene_data/Breast_500.csv", sep=",", header=0, index_col=0, nrows=300)

# 数値データだけを取り出し、X１に代入
X1 = X.iloc[:,:].values

# メタデータファイル(Sample_data.csv)の読み込み
sp = pd.read_csv("./5150_gene_data/Sample_data.csv", sep=",", header=0, nrows=300)

# 正常組織とがん組織を区別するラベルのはいったカラム"Type 2"を配列yに格納
y = np.array(sp["Type 2"].tolist())

# トレーニングデータとテストデータに分割
train_X, test_X, train_y, test_y = train_test_split(X1, y, test_size=0.3)

# 以下にコードを記述してください
# モデルを構築してください
model = LogisticRegression()

# train_Xとtrain_yを使ってモデルに学習させてください
model.fit(train_X,train_y)

# test_Xに対するモデルの分類予測をしてください
pred_y = model.predict(test_X)

print("予測結果: ", pred_y)
print("正解: ", test_y)

# F値の算出
print("F値: {:.4f}".format(f1_score(test_y, pred_y)))
```
### 3.1.3 線形SVMによるがん診断の実装
```
# モデルを構築してください
model = LinearSVC()

# train_Xとtrain_yを使ってモデルに学習させてください
model.fit(train_X,train_y)

# test_Xに対するモデルの分類予測をしてください
pred_y = model.predict(test_X)
```
### 3.1.4 非線形SVMによるがん診断の実装
```# モデルを構築してください
model = SVC()

# train_Xとtrain_yを使ってモデルに学習させてください
model.fit(train_X,train_y)

# test_Xに対するモデルの分類予測をしてください
pred_y = model.predict(test_X)
```
### 3.1.5 決定木によるがん診断の実装
```
# モデルを構築してください
model = DecisionTreeClassifier()

# train_Xとtrain_yを使ってモデルに学習させてください
model.fit(train_X,train_y)

# test_Xに対するモデルの分類予測をしてください
pred_y = model.predict(test_X)
```
### 3.1.6 ランダムフォレストによるがん診断の実装
```
# モデルの構築をしてください
model = RandomForestClassifier()

# train_Xとtrain_yを使ってモデルに学習させてください
model.fit(train_X,train_y)

# test_Xに対するモデルの分類予測をしてください
pred_y = model.predict(test_X)
```
### 3.1.7 k-Nearest Neighbors (K-NN)によるがん診断の実装
```
# モデルの構築をしてください
model = KNeighborsClassifier(n_neighbors=5)

# train_Xとtrain_yを使ってモデルに学習させてください
model.fit(train_X, train_y)

# 予測結果
pred_y = model.predict(test_X)
```
### 3.1.8 各種分類法によるがん診断の比較
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# seed値の固定
np.random.seed(42)

# データの読み込み
X = pd.read_csv("./5150_gene_data/Breast_500.csv", sep=",", header=0, index_col=0, nrows=300)

# 数値データだけを取り出し、X１に代入
X1 = X.iloc[:, :].values

# メタデータファイル(Sample_data.csv)の読み込み
sp = pd.read_csv("./5150_gene_data/Sample_data.csv", sep=",", header=0, nrows=300)

# 正常組織とがん組織を区別するラベルのはいったカラム"Type 2"を配列yに格納
y = np.array(sp["Type 2"].tolist())

# トレーニングデータとテストデータに分割
train_X, test_X, train_y, test_y = train_test_split(X1, y, test_size=0.3)

# モデルの構築
model_param = {LogisticRegression(): "ロジスティック回帰: {}", LinearSVC(): "線形SVM: {}",
               SVC(): "非線形SVM: {}", DecisionTreeClassifier(): "決定木: {}",
               RandomForestClassifier(): "ランダムフォレスト: {}", KNeighborsClassifier(): "K-NN: {}"}

for model, output_format in model_param.items():
    # train_Xとtrain_yを使ってモデルに学習
    model.fit(train_X, train_y)
    # test_Xに対するモデルの予測結果
    pred_y = model.predict(test_X)
    # F値を出力してください
    print(output_format.format(f1_score(test_y, pred_y)))
```