# サマリ
- さらっと。
- 一番知っていると思っていたが、、ラッソ回帰やリッジ回帰は全然理解してなかった。
    - そもそもやったことのある回帰モデルって、予測モデルの生成というよりも、分析のためのそれだったから、少し毛色が違ったのか。
    - あるいは、ここ数年で大きくレベルが上がって、このレベルのことも簡単にできるようになってしまった、ということか。。

# メモ
## 正則化
- 回帰分析を行うモデルに対し、モデルが推定したデータ同士の関係性の複雑さに対してペナルティを加え、汎化させる
### L1正則化
- 「予測に影響を及ぼしにくいデータ」にかかる係数をゼロに近づける手法
- 主に余分な情報がたくさん存在するようなデータに対して使用
- 特徴量削減の手法として用いることも可能
### L2正則化
- 係数が大きくなりすぎないように制限する手法
- 過学習を抑えるために用いる
- 学習の結果、得られる係数が大きくならないので汎化しやすい

## 回帰モデル
- 線形回帰: `sklearn.linear_model.LinearRegression`
- ラッソ回帰: `sklearn.linear_model.Lasso`
    - L1正則化を行いながら線形回帰の適切なパラメータを設定する回帰モデル
- リッジ回帰: `sklearn.linear_model.Ridge`
    - L2正則化を行いながら線形回帰の適切なパラメータを設定する回帰モデル
- ElasticNet回帰: `sklearn.linear_model.ElasticNet(l1_ratio=0.3)`
    - ラッソ回帰とリッジ回帰を組み合わせて正則化項を作るモデル

## 基本的な流れの例
```
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# データを生成
X, y = make_regression(n_samples=100, n_features=100, n_informative=60, n_targets=1, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 線形回帰
model = LinearRegression()
model.fit(train_X, train_y)

print(model.score(test_X, test_y))
print(model.predict(test_X))
```