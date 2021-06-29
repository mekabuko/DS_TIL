# 1. 異常検知基礎
## 1.1 基礎
### 1.1.1 異常パターンの分類
- 異常検知の分野において「異常」を客観的に捉えるために異常パターンを分類する必要がある
- 外れ値: 値が他と比べて大きく離れている値
    - 外れ値を見つける問題を「外れ値検出」と呼ぶ
- 変化点: 値が他と異なった振る舞いをする、振る舞いが変化した時点
    - 変化点を見つける問題を「変化(点)検知」または「異常部位検知」と呼ぶ
### 1.1.2 異常度・閾値・誤報率
- 異常度: どれだけ異常かを定量的に示す数値
- 閾値: どれだけ異常度が高ければ異常と見做すか
- 誤報率: 正常なものを、異常だと判断してしまった割合
    - `1 - (正しく正常と判断した数)/(正常の数)`
## 1.2 ホテリング法
### 1.2.1 ホテリング法とは
- ホテリング法: データが単一の正規分布から発生しているという仮定のもとで、異常度と閾値を定義
- 制約: 
    - データが単一の正規分布から発生している
    - データ中に異常な値を含まないか、含んでいたとしてもごくわずか
- 手法:
    1. 自分で設定した誤報率に基づき、閾値を求める
    2. 正常と信用できる値の平均値および共分散行列を計算する
    3. テストデータに対してマハラノビス距離(異常度)を計算し、閾値を超えていたら異常と判定する

### 1.2.2 マハラノビス距離
- マハラノビス距離: あるデータがデータ全体の平均値からどれくらい離れているか
- ユークリッド距離の問題点: 
    - 分散の大きい変数の寄与が大きく、分散の小さい変数の寄与が小さい(スケールに左右される)
    - 変数同士の相関を反映できない
- マハラノビス距離の特性:
    - ユークリッド距離の問題点を、共分散行列の逆行列である逆共分散行列(精度行列)を噛ませて正規化することによって解決
        - スケール不変性
        - 特徴量間の相関を補正
- 実装例
```
from scipy.spatial import distance

# データ集合の平均値
# axis=0 と指定することで列ごとの平均を求めていく
mean = np.mean(data, axis=0)
# データ集合の共分散行列(分散を集めたもの)
cov = np.cov(data.T)
# データx, 平均値mean, 共分散行列の逆行列np.linalg.pinv(cov) から距離を計算
distance.mahalanobis(x, mean, np.linalg.pinv(cov))
```
### 1.2.3 T二乗法（閾値の決定）
- ホテリングのT二乗法: ホテリング法を用いた外れ値検出のこと
- 手順1. 自分で設定した誤報率に基づき、閾値を求める
    - 必要なパラメータである 誤報率は自身で設定
    - 誤報率を設定し、データ量が十分ある状態であれば、近似的にχ(カイ)二乗検定を用いて閾値を設定できる
- 実装例
```
from scipy import stats as st

# 異常度a, 次元数df, 誤報率f1
a = 10
df = 4
f1 = 0.05

# 閾値を計算
threshold = st.chi2.ppf(1 - f1, df)

# 閾値を越えているかによって分岐
if a <= threshold:
    print("normal")
else:
    print("abnormal")
```
### 1.2.4 T二乗法（標本値の計算）
- 手順2: 正常と信用できる値の平均値および共分散行列を計算する
    - データから平均値と共分散行列を計算
- 実装例
```
import numpy as np

# 平均値
mean = np.mean(data, axis=0)
# 共分散行列
cov = np.cov(data.T)
```
### 1.2.5 T二乗法（異常度の計算）
- 手順3. テストデータに対してマハラノビス距離を計算し、閾値を超えていたら異常と判定する
- 実装例
```
# xのマハラノビス距離
mahalanobis = distance.mahalanobis(x, mean, np.linalg.pinv(cov))
```
### 1.2.6 ホテリング法実践
- 実装例
```
import numpy as np
from scipy.spatial import distance
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns

# X_testの読み込み
X_test = np.loadtxt('anomaly_detection_data/hoteling_test_data.csv', dtype='float64', delimiter=',')

# 閾値を設定して下さい
thr = st.chi2.ppf(1 - 0.15, 2)

# X_testの標本値を取得して下さい
mean = np.mean(X_test, axis=0)
cov = np.cov(X_test.T)

# X_testの各データの異常度mahを計算して下さい
mah = [distance.mahalanobis(x, mean, np.linalg.pinv(cov)) for x in X_test]

# X_err, X_normを分類して下さい
X_err = X_test[mah > thr]
X_nom = X_test[thr >= mah]

# プロットしています
plt.plot(X_err[:, 0], X_err[:, 1], "o", color="r")
plt.plot(X_nom[:, 0], X_nom[:, 1], "o", color="b")
plt.title("T二乗法によるX_testについての異常値検知")
plt.show()
```
## 1.3 単純ベイズ法
### 1.3.1 単純ベイズ法とは
- 異常検知の問題において変数が多くなると(次元が増えると)計算量が増え、手に負えなくなるという問題点が存在
- そこで、単純ベイズ法を用いて多変数の問題を1変数の問題に帰結させることで問題を単純にする、と言う手法がある
- 単純ベイズ法の主な考え方
    - 変数同士の相関が無い、つまり 各変数が独立である
    - 変数が独立であると仮定すれば、データが異常であるか正常であるかの確率は各変数の異常である確率に重み付けしたものの積で求められる
- 単純ベイズ法
    - 異常度はデータと重みの内積(データの各変数に各重みを掛けたものの和)で表す
        - これによりデータの次元を1にすることができる。
    - 閾値はホテリング法とは違って理論的に求めることは難しいので、検証用データを用いて最適化する
- 流れ
    1. 訓練データを用いたベイズ法による重みの計算
    2. 検証データを用いた閾値の最適化
    3. 異常度の計算、閾値と比較
### 1.3.2 ベクトルの重み
- データ処理における考え方:
    - データをベクトルとして考えた際に、適切なベクトルとの内積を取ることでそのデータの特徴を表す数値が得られる
- 重みベクトル: データとの内積を取るためのベクトル
- 内積の実装: `np.dot(a,b)`
### 1.3.3 単純ベイズ分類（重みの計算）
- 実装例
```
import numpy as np
import numpy.random as rd

# 各変数が意味する単語
words = ["hoge", "foo", "bar", "po", "do",.....] # リストの長さ=単語数
# 単語袋詰表現のデータ
X_train = np.array([[2, 5, 1, 9, 0, 0,...], [1, 1, 0, 2, 7, 3,...], [5, 2, 7, 1, 0, 0,...],...]) # 単語数×文章データ数
y_train = np.array([1, 1, 0, 1, 0, 0,...]) # リストの長さ=文章データ数

# 重みを0にしないためのゲタ(あとで対数をとるため)
alpha = 1

# 異常データの集合X1, 正常データの集合X0に分類
X1 = X_train[y_train == 1]
X0 = X_train[y_train == 0]

# ベイズ法による計算過程
w1 = (np.sum(X1, axis=0) + alpha) / (np.sum(X1) + X1.shape[1] * alpha)
w0 = (np.sum(X0, axis=0) + alpha) / (np.sum(X0) + X0.shape[1] * alpha)

# ベイズ法による重みの計算
weight = np.log(w1 / w0)
```
### 1.3.4 単純ベイズ分類（閾値の最適化）
- 実装例
```
from sklearn import metrics

# 評価用データの異常度を計算
valid_score = np.dot(X_valid, weight)

# 偽陽性率、真陽性率、閾値候補の配列をそれぞれ返す
fpr, tpr, thr_arr = metrics.roc_curve(y_valid, valid_score)

# roc曲線において最も適した閾値を求める
threshold = thr_arr[(tpr - fpr).argmax()]
```
### 1.3.5 単純ベイズ分類（異常度の計算）
- 実装例
```
import numpy as np
import numpy.random as rd
from sklearn import metrics

rd.seed(0)
# 正規分布に従う正常データX0
mean_n = rd.randint(0, 7, 100)
cov_n = rd.randint(0, 5, 100) * np.identity(100)
X0 = rd.multivariate_normal(mean_n, cov_n, 200).astype('int64')
X0[X0 < 0] = 0
# 正規分布に従わない異常データX1
X1 = rd.randint(0, 10, (50, 100))
# ゲタalpha
alpha = 1
# 重みweightの計算
w0 = (np.sum(X0, axis=0) + alpha) / (np.sum(X0) + X0.shape[1] * alpha)
w1 = (np.sum(X1, axis=0) + alpha) / (np.sum(X1) + X1.shape[1] * alpha)

weight = np.log(w1 / w0)
# 閾値thresholdの設定
threshold = 123

# 異常判定したいデータx
x = rd.randint(0, 10, 100)

# xの異常度を計算して下さい
score = np.dot(x, weight)

print("score:" + str(score))

# 閾値を越えているかによって分岐し、結果を出力して下さい
if score > threshold: 
    print("abnormal")
else:
    print("normal")
```
# 2. 外れ値検出
## 2.1 k近傍法
### 2.1.1 k近傍法とは
- 基本的な考え方: 判別したいデータとの距離が近いデータk個の中で異常データの割合から異常度を計算し、閾値を超えたら異常と判定する。
- k近傍法のk: 距離の近いデータを何個とるかを意味し、k=1ならば最も近いデータのみを見るため最近傍法と呼ぶ。
- 長所:
    - データの分布に関する前提条件がいらないため、実用が簡単
    - 正常なデータが複数箇所のまとまりから構成されていても使用可能
    - 異常度の式が簡単なため、実装が難しくない
- 短所:
    - データの分布に関する前提条件がないため、閾値が数式で定まらない
    - パラメータ k の厳密なチューニングには複雑な数式が必要
    - 怠惰学習(事前にモデルを構築しない学習)のため、新しいデータを分類するための計算量が減らない
### 2.1.2 異常度
- 判別したいデータを x として `ln((π0N1(x))/(π1N0(x)))` で定義
    - π0: 正常なデータの割合
    - π1: 異常なデータの割合
    - N0(x): xのk近傍中の正常なデータの割合
    - N1(x): xのk近傍中の異常なデータの割合
- 実装例
```
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 標本データXとラベルy, 近傍k
X = np.array([
    [0, 7],
    [5, 2],
    [4, 4],
    [1, 5],
    [5, 3],
    [3, 2]
])
y = np.array([0, 0, 1, 1, 1, 0])
k = 3

# Xのk近傍のラベルを取得してください
clf = KNeighborsClassifier(n_neighbors=k+1)
clf.fit(X, y)

dist, ind = clf.kneighbors(X)
neighbors =  y[ind][:, 1:]

# pi0, pi1の計算
pi0 = y[y == 0].size / y.size
pi1 = 1 - pi0

# N0, N1を計算してください
N1 = neighbors.sum(axis=1) / k
N0 = 1 - N1

# 異常度を計算し、出力してください
abnorm = np.log((pi0 * N1) / (pi1 * N0))
print(abnorm)
```
### 2.1.3 一つ抜き交差確認法
- 一つ抜き交差確認法: N個の標本データのうちN-1個をモデルにして残り一個のデータについて予測し、当たり外れを見るという方法
- 実装例
```
# 最近傍(自分自身)は除外するため、kを一つ増やす
clf = KNeighborsClassifier(n_neighbors=k+1)
clf.fit(X, y)
# 標本データの近傍を取得
dist, ind = clf.kneighbors(X)
# [:, 1:]で配列の一列目を除外したラベル
neighbors = y[ind][:, 1:]
```
### 2.1.4 閾値の設定
- k近傍法は標本データの分布を仮定しないので、閾値を計算で求めることができない
- kの値も自分で設定する必要がある
- 閾値とkの値の組み合わせを試して、最も良い精度のものを使用する
- 実装例
```
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 標本データX, ラベルyの読み込み
X = np.load("anomaly_detection_data/knnsample.npz")["X"]
y = np.load("anomaly_detection_data/knnsample.npz")["y"].reshape(40)
# k, 閾値の候補
param_k = np.arange(1, 6)
param_ath = np.arange(-10, 11) / 10

# pi0の計算
pi0 = y[y == 0].size / y.size
# pi1の計算
pi1 = 1 - pi0
k_opt = 0
ath_opt = 0
score_max = 0
# 1. kと閾値の候補の配列を使ってループ
for k in param_k:
    for ath in param_ath:
        # 2. ループ内で各標本データの異常度を計算
        # 一つ抜き交差確認法により異常度abnormを計算
        # KNeighborsClassifierの用意
        clf = KNeighborsClassifier(n_neighbors=k+1)
        clf.fit(X, y)
        dist, ind = clf.kneighbors(X)
        neighbors = y[ind][:, 1:]
        # N1の計算
        N1 = neighbors.sum(axis=1) / k
        # N0の計算
        N0 = 1 - N1
        abnorm = np.log((N1 * pi0) / (N0 * pi1))
        
        # 3. 閾値を超えたものを異常と判定、ラベルを作成
        y_valid = abnorm > ath
        # 4. 精度を計算
        score = f1_score(y, y_valid)
        # 5. 精度が良くなったらk,athを更新
        if score  > score_max :
            score_max = score
            k_opt = k
            ath_opt = ath
            
# 出力
print("k : {:0}\nath  : {:1}\nscore: {:2}".format(k_opt, ath_opt, score_max))
```
### 2.1.5 k近傍法（異常判定）
- 実装例
```
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# 標本データX, ラベルyの読み込み
X = np.load("anomaly_detection_data/knnsample.npz")["X"]
y = np.load("anomaly_detection_data/knnsample.npz")["y"].reshape(40)
# k, 閾値ath
k = 3
ath = 0.5
# データx
x = np.array([
    [ 1.52,  3.60],
    [-2.50,  0   ],
    [ 5.32, -1.89]
])

# x の異常度を計算しています
# KNeighborsClassifierを用意し、x近傍のラベルを取得しています
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)
dist, ind = clf.kneighbors(x)
neighbors = y[ind]

# pi0, pi1, N0, N1 を計算しています
pi0 = y[y == 0].size / y.size
pi1 = 1 - pi0
N1 = neighbors.sum(axis=1) / k
N0 = 1 - N1

# 異常度の計算
abnorm = np.log((N1 * pi0) / (N0 * pi1))

# 異常判定し、y_predを作成してください
y_pred = np.asarray(abnorm > ath, dtype="int")

# 結果の出力
print("xの異常判定結果:" + str(y_pred))
x0 = x[y_pred==0]
x1 = x[y_pred==1]
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "o", color="skyblue")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "o", color="pink")
plt.plot(x0[:, 0], x0[:, 1], "o", color="b")
plt.plot(x1[:, 0], x1[:, 1], "o", color="r")
plt.title("データxの異常判定結果")
plt.show()
```
## 2.2 1クラスSVM
### 2.2.1 1クラスSVMとは
- 手続き
    1. SVMの識別器を用意して、各データの異常度を計算
    2. 標本データの内何割が異常かをあらかじめ決定して閾値を設定
    3. 閾値と異常度によって異常判定
- 長所:
    - 教師なしの異常検知  
    - 機械学習の実用例が多い(= scikit-learn で簡単に実装できる)
    - データが複数箇所のまとまりから構成されていても使用可能  
    - データの次元数が大きくなっても精度を保てる  
- 短所:
    - 数式のパラメータの変化によって精度が大きく上下する  
    - データが増えると計算量が急上昇する  
### 2.2.2 異常度の計算
- 実装例
```
# 識別器(sklearnのOneClassSVM)を用意
clf = OneClassSVM(kernel="rbf", gamma="auto")
# 標本データで学習
clf.fit(data)
# 異常度の計算、.ravel()で配列を整頓
y_score = clf.decision_function(data).ravel()
```
- kernel: どのように空間を歪めるかを意味し、基本は孤立した点とまとまった点を分けるように歪める"rbf"を用いる
- gamma: 厳密には"rbf"のパラメータで、sklearnのモジュールを使う際には"auto"と指定すると最適化される
### 2.2.3 閾値の設定
- サポートベクトルデータ記述法ではラベル無し標本データを用いる上にホテリング法のように標本の分布を仮定しないので、標本データには一定の割合(a%)で異常データが入っているものとみなして閾値を設定する必要がある
- 実装例
```
# 記述統計のモジュールをimport
import scipy.stats as st

# thrより小さいscoreの割合がa%になるようなthrを求める
thr = st.scoreatpercentile(score, a)
```
### 2.2.4 1クラスSVM（異常判定）
- 実装例
```
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.svm import OneClassSVM

# データの読み込み
data = np.load("./anomaly_detection_data/OCSVM_sample.npy")

# 異常度の計算
clf = OneClassSVM(kernel="rbf", gamma="auto")
clf.fit(data)
y_score = clf.decision_function(data).ravel()

# 閾値は10%分位点
threshold = st.scoreatpercentile(y_score, 10)

# y_scoreとthresholdによる異常判定
# bool値のNumPy配列を作ってください(正常ならFalse, 異常ならTrue)
y_pred = np.array(y_score < threshold)

# 正常データを青色でプロットしてください
plt.plot(data[y_pred == False][:,0], data[y_pred == False][:,1], "o", color="b")
# 異常データを赤色でプロットしてください
plt.plot(data[y_pred][:,0], data[y_pred][:,1], "o", color="r")
plt.title("データの異常分類")
plt.show()
```
## 2.3 方向データの異常検知
### 2.3.1 方向データについて
- 手続き（ホテリング法の応用）
    1. データの規格化
    2. 標本データの異常度を計算
    3. 標本の分布を求め、閾値を設定
    4. 観測データについて異常判定
- 長所
    - 古典的で実績のある手法
    - 規格化の方法によって様々なデータに対応できる
    - 閾値の信頼性がマハラノビス法より高い
- 短所
    - データの制約が大きい(=最適な規格化の方法を選ぶ必要がある)
    - 閾値の信頼を高めるため、訓練データが必要
### 2.3.2 データの規格化
- 規格化：大きさを一定（たとえば1）に統一すること
- 実装例
```
import numpy as np
# データの読み込み

data = np.load("./anomaly_detection_data/direction_sample.npy")

# データを規格化し、規格化された各データの長さを求めて出力してください
# 各データの長さを求めてください
norm = np.linalg.norm(data, ord=2, axis=1)[:, np.newaxis]

# データを長さで割って規格化してください
normalized_data = data / norm

# 規格化されたデータの長さを求めてください
normalized_norm = np.linalg.norm(normalized_data, ord=2, axis=1)

# 出力
print(normalized_norm)
```
### 2.3.3 方向データの異常検知（異常度の計算）
- 異常度は `1 - (data と dataの平均の内積)` で求める
- 実装例
```
import numpy as np

# データの読み込み
data = np.load("./anomaly_detection_data/direction_sample.npy")

# データの規格化
norm = np.linalg.norm(data, ord=2, axis=1)[:, np.newaxis]
normalized_data = data / norm

# 方向データの平均を求めてください
mean = np.mean(normalized_data, axis=0)

# meanを規格化してください
normalized_mean = mean / np.linalg.norm(mean)

# 各データの異常度を求めてください
abnormalities = 1 - np.matmul(normalized_data, normalized_mean)

# 異常度の平均を出力しています
print(abnormalities.mean())
```
### 2.3.4 方向データの異常検知（閾値の設定）
- 異常度がカイ二乗分布に従うという定理を利用する
- 以下のパラメータを決定
    - 自由度 m = (2 * (異常度の平均)^2) / ((異常度の二乗平均) - (異常度の平均)^2)
    - スケール因子 s = (異常度の平均) / m
- 実装例
```
import numpy as np
import scipy.stats as st

# データの読み込み
data = np.load("./anomaly_detection_data/direction_sample.npy")

# データの規格化
norm = np.linalg.norm(data, ord=2, axis=1)[:, np.newaxis]
normalized_data = data / norm

# 異常度の計算
mean = np.mean(normalized_data, axis=0)
normalized_mean = mean / np.linalg.norm(mean)
abnormalities = 1 - np.matmul(normalized_data, normalized_mean)

# 誤報率a
a = 0.1

# カイ二乗分布のステータスを計算し、閾値を設定、出力してください
norm_ab = abnormalities.mean()
norm2_ab = (abnormalities ** 2).mean()
m = 2 * norm_ab ** 2 / (norm2_ab - norm_ab ** 2)
s = norm_ab / m

# 閾値の設定
threshold = st.chi2.ppf(1 -a, df=m, scale=s)

# 出力
print(threshold)
```
### 2.3.5 方向データの異常検知（異常判定）
-　実装例
```
import numpy as np
import scipy.stats as st

# データの読み込み
data = np.load("./anomaly_detection_data/direction_sample.npy")
# Xの読み込み
X = np.load("./anomaly_detection_data/direction_sample_train.npy")

# データの規格化
norm = np.linalg.norm(data, ord=2, axis=1)[:, np.newaxis]
normalized_data = data / norm

# 異常度の計算
mean = np.mean(normalized_data, axis=0)
normalized_mean = mean / np.linalg.norm(mean)
abnormalities = 1 - np.matmul(normalized_data, normalized_mean)

# 閾値を設定
norm_ab = abnormalities.mean()
norm2_ab = (abnormalities ** 2).mean()
m = 2 * norm_ab ** 2 / (norm2_ab - norm_ab ** 2)
s = norm_ab / m
threshold = st.chi2.ppf(0.9, df=m, scale=s)

# Xの異常度を計算してください
y_score = 1 - np.matmul(X, normalized_mean)

# 異常判定しラベルを作成してください
y_pred = np.array(y_score > threshold, dtype="int")

# 出力
print(y_pred)
```
# 3. 変化点検知
## 3.1 累積和法による変化検知
### 3.1.1 累積和法とは
- 累積和法: 異常である状態(変化度)を時間に沿ってカウントし、そのカウント(累積和)が閾値を超えたときに異常と判別する手法
    - 「継続して何らかの異常状態が発生している」ということを検出可能
- 管理図: 観測値(data)と変化度(change score)、累積和(cumulative sum)をまとめたプロット
### 3.1.2 変化度
- 時系列データの変化検知のフロー
    1. 正常時の目安、上振れの目安を与え、変化度を定義する。
    2. 変化度の上側累積和を求める。
    3. 与えられた閾値を超えていたら異常と判定する。
- 変化度(対数尤度比): データがどれくらい異常状態へ変化しているかを示す数値
- 実装例
```
# 正常時の平均値mu_x、異常時の振れ幅nu_x
mu_x = 10
nu_x = 14

# xの標準偏差
std_x = np.std(x)

# 変化度
change_score = (nu_x / std_x) * ((x - mu_x - nu_x / 2) / std_x)
```
### 3.1.3 上側累積和
- 上振れした時の変化度の累積和：上側累積和
- 実装例
```
def upper_cumsum(x):
    # 累積和を格納する配列x_cumsum
    x_cumsum = np.array(x)
    # xの要素数分ループ
    for i in range(x.size - 1):
        # 累積和が負にならないようにする(0とx_cumsum[i]の大きい方を格納する)
        x_cumsum[i] = max(0, x_cumsum[i])
        # 累積
        x_cumsum[i + 1] = x_cumsum[i] + x[i + 1]
    # 上側累積和を返す
    return x_cumsum
```
### 3.1.4 異常判定
- 実装例
```
# 上側累積和score_cumsumの読み込み
score_cumsum = np.load("./anomaly_detection_data/sample1_cumsum.npy")

# 閾値
threshold = 10

# 閾値を超えているデータのラベルを1、それ以外を0としてラベル付けしてください
pred = np.array(score_cumsum > threshold, dtype="int")

# 初めて閾値を超えた点のインデックスを求めてください
ind_err = np.where(pred==1)[0][0]

# 出力
print("変化時点:" + str(ind_err))
```
## 3.2 近傍法による異常部位検出
### 3.2.1 スライド窓と部分時系列
- 実装例
```
# xの軸を調整データの型を(550,)→(1, 550)に変換(1次元から2次元に変換)
x = x.reshape(1, -1)
# 部分時系列データの作成(この時点でのXは(1, 541)のリストです)
X = x[:, :x.shape[1]-M+1]
for i in range(1, M):
    # Xiを切り抜く(サイズMの窓を取り出す)
    Xi =  x[:, i:x.shape[1]-M+i+1]
    # X, Xi配列を結合(この時点でのXは(i, 541)のリスト)
    X = np.concatenate((X, Xi), axis=0)
# 配列を転置し(541, 10)のリストに変換
X = np.transpose(X)
```
### 3.2.2 ラベル無しデータの最近傍法（異常度の計算）
- 実装例
```
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# データXの読み込み
X = np.load("./anomaly_detection_data/sample1_pts.npy")

# 分類器を用意してください
clf = KNeighborsClassifier(n_neighbors = 2)
clf.fit(X, np.zeros(X.shape[0]))

# 近傍距離を取得してください
dist, ind = clf.kneighbors(X)
distances = dist[:, 1]

# 平均値を計算し、出力してください
mean_distances = distances.mean()
print(round(mean_distances, 3))
```
### 3.2.3 ラベル無しデータの最近傍法（閾値の設定、異常判定）
- 実装例
```
import scipy.stats as st
import numpy as np

# 近傍距離distancesの読み込み
distances = np.load("./anomaly_detection_data/sample1_dist.npy")

# 閾値を設定し、出力してください
threshold = st.scoreatpercentile(distances, 100 - 30)
print(threshold)
```
## 3.3 特異スペクトル変換法
### 3.3.1 履歴行列とテスト行列
- 実装例
```
import numpy as np

ecg = np.loadtxt("./anomaly_detection_data/sample2.csv", delimiter=",")

# 部分時系列データの作成
ecg = ecg.reshape(1, -1)
M = 50
ecg_pts = ecg[:, :-M]
for m in range(M - 1):
    ecg_pts = np.concatenate((ecg_pts, ecg[:, m + 1: m + 1 - M]), axis=0)
ecg_pts = np.transpose(ecg_pts)
# 履歴行列の列数n、テスト行列の列数k、ラグL
n = 25
k = 25
L = 13

# 部分時系列データを連結するため、軸を追加
ecg_pts_nest = ecg_pts[np.newaxis, :]

# 履歴行列を作成してください
ecg_hist = np.array(ecg_pts_nest[:, :-(n + L), :])
for i in range(n - 1):
    ecg_hist = np.concatenate((ecg_hist,
                               ecg_pts_nest[:, i + 1: i + 1 - (n + L), :]),
                              axis=0)
ecg_hist = np.transpose(ecg_hist, axes=(1, 2, 0))

# テスト行列を作成してください
ecg_test = np.array(ecg_pts_nest[:, n + L - k: -k, :])
for i in range(k - 1):
    ecg_test = np.concatenate((ecg_test,
                               ecg_pts_nest[:, i + 1 + (n + L - k): i + 1 - k, :]),
                              axis=0)
ecg_test = np.transpose(ecg_test, axes=(1, 2, 0))

# 両配列のshapeを出力してください
print(ecg_hist.shape)
print(ecg_test.shape)
```
### 3.3.2 特異値分解
- 実装例
```
import numpy as np

# 履歴行列、テスト行列の読み込み
ecg_hist = np.load("./anomaly_detection_data/sample2_ht.npz")["h"]
ecg_test = np.load("./anomaly_detection_data/sample2_ht.npz")["t"]

# 各行列の近似の階数 r: 履歴行列用、m: テスト行列用
r = 3
m = 2

# 履歴行列の特異ベクトルを計算してください
U_hist = np.linalg.svd(ecg_hist)[0][:, :, :r]

# テスト行列の特異ベクトルを計算してください
U_test = np.linalg.svd(ecg_test)[0][:, :, :m]

# shapeを出力しています
print(U_hist.shape)
print(U_test.shape)
```
### 3.3.3 変化度の計算
- 実装例
```
import numpy as np
import matplotlib.pyplot as plt

# U_hist, U_testの読み込み
U_hist = np.load("./anomaly_detection_data/sample2_Uht.npz")["h"]
U_test = np.load("./anomaly_detection_data/sample2_Uht.npz")["t"]

# 各時間における変化度を計算し、配列に格納してください
get_score = lambda X, Y: 1 - np.linalg.norm(np.matmul(X.T, Y), ord=2)
score = [get_score(h, t) for h, t in zip(U_hist, U_test)]

# 折れ線グラフにプロットしてください
plt.plot(range(len(score)), score)
plt.title("各時間における変化度")
plt.show()

# 回答チェック用です
print("変化度の最大値:" + str(np.argmax(score)))
```