# サマリ

# メモ
## データのランダム生成
- 分類に適した架空のデータを作成する `scikit-learn.datasets.make_classification()`
```
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=50, n_classes=2, n_features=2, n_redundant=0, random_state=0)
```

# 分類モデル
## ロジスティック回帰
- `from sklearn.linear_model import LogisticRegression`
### 特徴
- 直線でデータのカテゴリーのグループに分ける
- データがクラスに分類される確率も計算することが可能
- 一般化した境界線になりにくい（汎化能力が低い）
### ハイパーパラメーター
- `C`: モデルが学習する識別境界線が教師データの分類間違いに対してどのくらい厳しくするのかという指標(初期値: 1.0)
- `penalty`: 複雑さに対するペナルティ。L1 か L2 を使用可能。通常L2を使っておけば良い。
    - `L1`: データの特徴量を削減することで識別境界線の一般化を図るペナルティ
    - `L2`: データ全体の重みを減少させることで識別境界線の一般化を図るペナルティ
- `multi_class`: 多クラス分類を行う際のモデルの動作（二値分類では不能）
    - `ovr`: 各クラスに「属する/属さない」だけを考える
    - `multinomial`: 各クラスに分類される確率も考える
- `random_state`: データの処理順を制御するパラメータ
    - 処理順によって、大きく結果が異なることがある
    - 学習結果を保存するという意味でも、この値が必要になる
### サンプル(引用)
```
# モジュールのインポート
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# max_depthの値の範囲(1から10)
depth_list = [i for i in range(1, 11)]

# 正解率を格納するからリストを作成
accuracy = []

# 以下にコードを書いてください
# max_depthを変えながらモデルを学習
for max_depth in depth_list:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(train_X, train_y)
    accuracy.append(model.score(test_X, test_y))

# コードの編集はここまでです。
# グラフのプロット
plt.plot(depth_list, accuracy)
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("accuracy by changing max_depth")
plt.show()
```

## 線形SVM
- `from sklearn.svm import LinearSVC`
### 特徴
- 線形分離
- 分類する境界線が2クラス間の最も離れた場所に引かれるためロジスティック回帰と比べて一般化されやすい
### ハイパーパラメーター
- `C`: モデルが学習する識別境界線が教師データの分類間違いに対してどのくらい厳しくするのかという指標(初期値: 1.0)
- `penalty`: 複雑さに対するペナルティ。L1 か L2 を使用可能。通常L2を使っておけば良い。
    - `L1`: データの特徴量を削減することで識別境界線の一般化を図るペナルティ
    - `L2`: データ全体の重みを減少させることで識別境界線の一般化を図るペナルティ
- `multi_class`: 多クラス分類を行う際のモデルの動作（二値分類では不能）
    - `ovr`: 基本的にこっちの方が軽くて結果が良い
    - `crammer_singer`: ？？？
- `random_state`: データの処理順を制御するパラメータ
    - 処理順によって、大きく結果が異なることがある
    - 学習結果を保存するという意味でも、この値が必要になる
### サンプル(引用)
```
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(
    n_samples=1250, n_features=4, n_informative=2, n_redundant=2, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# Cの値の範囲を設定(今回は1e-5,1e-4,1e-3,0.01,0.1,1,10,100,1000,10000)
C_list = [10 ** i for i in range(-5, 5)]

# グラフ描画用の空リストを用意
svm_train_accuracy = []
svm_test_accuracy = []
log_train_accuracy = []
log_test_accuracy = []

# 以下にコードを書いてください。
for C in C_list:
    # 線形SVMのモデルを構築してください
    model1 = LinearSVC(C=C)
    model1.fit(train_X, train_y)
    svm_train_accuracy.append(model1.score(train_X, train_y))
    svm_test_accuracy.append(model1.score(test_X, test_y))
    
    # ロジスティック回帰のモデルを構築してください
    model2 = LogisticRegression(C=C)
    model2.fit(train_X, train_y)
    log_train_accuracy.append(model2.score(train_X, train_y))
    log_test_accuracy.append(model2.score(test_X, test_y))
    
# グラフの準備
# semilogx()はxのスケールを10のx乗のスケールに変更する

fig = plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
ax = fig.add_subplot(1, 1, 1)
ax.grid(True)
ax.set_title("SVM")
ax.set_xlabel("C")
ax.set_ylabel("accuracy")
ax.semilogx(C_list, svm_train_accuracy, label="accuracy of train_data")
ax.semilogx(C_list, svm_test_accuracy, label="accuracy of test_data")
ax.legend()
ax.plot()
plt.show()
fig2 =plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.grid(True)
ax2.set_title("LogisticRegression")
ax2.set_xlabel("C")
ax2.set_ylabel("accuracy")
ax2.semilogx(C_list, log_train_accuracy, label="accuracy of train_data")
ax2.semilogx(C_list, log_test_accuracy, label="accuracy of test_data")
ax2.legend()
ax2.plot()
plt.show()
```

## 非線形SVM
- `from sklearn.svm import SVC`
### 特徴
- 非線形に分離
### ハイパーパラメーター
- `C`: モデルが学習する識別境界線が教師データの分類間違いに対してどのくらい厳しくするのかという指標(初期値: 1.0)
- `kernel`: データ変形のための関数を決定する
    - `linear`: 線形SVCとほぼ同じ。これを使うならLinearSVCを使えば良い
    - `rbf, poly`: 立体投影のようなもの。通常 `rbf` が良い結果が出やすい
    - `precomputed`: データが前処理によって既に整形済みの場合
    - `sigmoid`: ほぼロジスティック回帰モデルと同じになる
- `decision_function_shape`: SVCにおける `multi_class` のようなもの。
    - `ovo`: クラス同士のペアを作り、そのペアでの二項分類を行い、多数決で属するクラスを決定する。
        - 計算量が多いので、データ量が多い時には動作が重くなる。
    - `ovr`: 一つのクラスとそれ以外、という分類で多数決を行う。
- `random_state`: データの処理順を制御するパラメータ
    - 処理順によって、大きく結果が異なることがある
    - 学習結果を保存するという意味でも、この値が必要になる
### サンプル(引用)
```
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_gaussian_quantiles(n_samples=1250, n_features=2, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# Cの値の範囲を設定(今回は1e-5,1e-4,1e-3,0.01,0.1,1,10,100,1000,10000)
C_list = [10 ** i for i in range(-5, 5)]

# グラフ描画用の空リストを用意
train_accuracy = []
test_accuracy = []

# 以下にコードを書いてください。
for C in C_list:
    model = SVC(C=C)
    model.fit(train_X, train_y)
    train_accuracy.append(model.score(train_X, train_y))
    test_accuracy.append(model.score(test_X, test_y))

# グラフの準備
# semilogx()はxのスケールを10のx乗のスケールに変更する
plt.semilogx(C_list, train_accuracy, label="accuracy of train_data")
plt.semilogx(C_list, test_accuracy, label="accuracy of test_data")
plt.title("accuracy with changing C")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend()
plt.show()
```

## 決定木
- `from sklearn.tree import DecisionTreeClassifier`
### 特徴
- 各説明変数で分岐を作っていく
- 説明可能（各説明変数の寄与度を測れる）
    - 先に使用される（根に近い）変数ほど影響力が大きい
- 外れ値の影響が低い
- 非線形分離は苦手
- 汎化性能は低い
### ハイパーパラメーター
- `max_depth`: モデルが学習する木の深さの最大値
    - 設定しないと、学習が教師データを完全に分類し切るまで終わらない。
        - 教師データを過剰に信頼し学習した一般性の低いモデルとなってしまう。
    - max_depth を設定し木の高さを制限することを**決定木の枝刈り**と呼ぶ。
- `random_state`: データの処理順を制御するパラメータ
    - 処理順によって、大きく結果が異なることがある(特に決定木では学習過程に直接関わる)
    - 学習結果を保存するという意味でも、この値が必要になる
### サンプル(引用)
```
# モジュールのインポート
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# max_depthの値の範囲(1から10)
depth_list = [i for i in range(1, 11)]

# 正解率を格納するからリストを作成
accuracy = []

# 以下にコードを書いてください
# max_depthを変えながらモデルを学習
for max_depth in depth_list:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(train_X, train_y)
    accuracy.append(model.score(test_X, test_y))

# コードの編集はここまでです。
# グラフのプロット
plt.plot(depth_list, accuracy)
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("accuracy by changing max_depth")
plt.show()
```

## ランダムフォレスト
- `from sklearn.ensemble import RandomForestClassifier`
### 特徴
- 決定木を複数作り、多数決で決定する
- アンサンブル学習の一つ
- 決定木では使用する説明変数は全て使用していたのに対し、ランダムフォレストの一つ一つの決定木はランダムに決められた少数の説明変数だけを用いる
- 非線形分離可能
### ハイパーパラメータ
- `n_estimators`: 多数決をおこなう簡易決定木の個数
- `max_depth`: モデルが学習する木の深さの最大値
    - 決定木よりも小さな値とすることが多い。
        - 簡易決定木の分類の多数決というアルゴリズムであるため、一つ一つの決定木に対して厳密な分類を行うより着目要素を絞り俯瞰的に分析を行うことで学習の効率の良さと高い精度を保つことができる
- `random_state`: データの処理順を制御するパラメータ
    - 処理順によって、大きく結果が異なることがある(特に決定木では学習過程に直接関わる)
    - 学習結果を保存するという意味でも、この値が必要になる
### サンプル(引用)
```
# モジュールのインポート
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=3, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# n_estimatorsの値の範囲(1から20)
n_estimators_list = [i for i in range(1, 21)]

# 正解率を格納するからリストを作成
accuracy = []

# 以下にコードを書いてください
# n_estimatorsを変えながらモデルを学習
for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(train_X, train_y)
    accuracy.append(model.score(test_X, test_y))

# グラフのプロット
plt.plot(n_estimators_list, accuracy)
plt.title("accuracy by n_estimators increasement")
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.show()
```

## k-NN(k近傍法)
- `from sklearn.neighbors import KNeighborsClassifier`
### 特徴
- 予測をするデータと類似したデータをいくつか見つけ、多数決により分類結果を決める手法
- 怠惰学習と呼ばれる学習の種類の一手法であり、学習コスト（学習にかかる計算量）が0
    - 教師データから学習するわけではなく、予測時に教師データを直接参照してラベルを予測する
- 非線形も可能
- 予測時の計算量が大きくなりやすい
### ハイパーパラメータ
- `n_neighbors`: k近傍法におけるkの値
    - 未知のデータを分類するときに、そのデータ自身からいくつのデータを予測に使うか
    - 数が多すぎると類似データとして選ばれるデータの類似度に幅が出るため、分類範囲の狭いカテゴリーがうまく分類されない
### サンプル(引用)
```
# モジュールのインポート
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=3, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# n_neighborsの値の範囲(1から10)
k_list = [i for i in range(1, 11)]

# 正解率を格納するからリストを作成
accuracy = []

# 以下にコードを書いてください

# n_neighborsを変えながらモデルを学習
for k in k_list:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_X, train_y)
    accuracy.append(model.score(test_X, test_y))

# グラフのプロット
plt.plot(k_list, accuracy)
plt.xlabel("n_neighbor")
plt.ylabel("accuracy")
plt.title("accuracy by changing n_neighbor")
plt.show()
```


# パラメータの設定方法
## グリッドサーチ
- 調整したいハイパーパラメーターの値の候補を明示的に複数指定し、パラメーターセットを作成し、その時のモデルの評価を繰り返す
    - 値の候補を明示的に指定するためパラメーターの値に文字列や整数、True or Falseといった数学的に連続ではない値をとるパラメーターの探索に向く
    - パラメーターの候補を網羅するようにパラメーターセットが作成されるため多数のパラメーターを同時にチューニングするのには不向き
- サンプル
```
import scipy.stats
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_digits()
train_X, test_X, train_y, test_y = train_test_split(data.data, data.target, random_state=42)

# パラメーターの値の候補を設定
model_param_set_grid = {SVC(): {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                                "C": [10 ** i for i in range(-5, 5)],
                                "decision_function_shape": ["ovr", "ovo"],
                                "random_state": [42]}}
              
max_score = 0
best_param = None

# グリッドサーチでパラメーターサーチ
for model, param in model_param_set_grid.items():
    clf = GridSearchCV(model, param)
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    score = accuracy_score(test_y, pred_y)
    if max_score < score:
        max_score = score
        best_param = clf.best_params_
                        
print("パラメーター:{}".format(best_param))
print("ベストスコア:",max_score)
svm = SVC()
svm.fit(train_X, train_y)
print()
print('調整なし')
print(svm.score(test_X, test_y))
```
## ランダムサーチ
- 取りうる値の範囲を指定し、確率で決定されたパラメーターセットを用いてモデルの評価を行うことを繰り返すことによって最適なパラメーターセットを探す
    - 値の範囲の指定にはパラメーターの確率関数を指定する
        - パラメーターの確率関数としてscipy.statsモジュールの確率関数がよく用いらる
- サンプル
```
import scipy.stats
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_digits()
train_X, test_X, train_y, test_y = train_test_split(data.data, data.target, random_state=42)

# パラメーターの値の候補を設定
model_param_set_random =  {SVC(): {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": scipy.stats.uniform(0.00001, 1000),
        "decision_function_shape": ["ovr", "ovo"],
        "random_state": scipy.stats.randint(0, 100)
    }}

max_score = 0
best_param = None

# ランダムサーチでパラメーターサーチ
for model, param in model_param_set_random.items():
    clf = RandomizedSearchCV(model, param)
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    score = accuracy_score(test_y, pred_y)
    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        
print("パラメーター:{}".format(best_param))
print("ベストスコア:",max_score)
svm = SVC()
svm.fit(train_X, train_y)
print()
print('調整なし')
print(svm.score(test_X, test_y))
```