# サマリ
-　例題でガイドに従ってやっているだけではなかなか身に付かない
- メモをチートシートにしつつ、実際の分析の作業で使っていくことで慣れていきたいところ

# メモ
## データ分析の標準プロセス
- CRISP-DM(CRoss Industry Standard Process for Data Mining)(Shearer他)
> 1. ビジネス理解でビジネスにおける課題を明確にし、データ分析プロジェクトの計画を立てます。2. データ理解でデータを取得し、そのデータが分析に使える状態であるか確かめるなどして現状のデータを理解します。3. データ準備では、後続のモデリングで要求される形式にデータを整形します。4. モデリングで得た分析結果を 5. 評価し、十分な結果が出ていれば業務に分析結果を6. 適用します。

- KDD(Knowledge Discovery in Databases)(Fayyad他)
> 1. データ取得
> 2. データ選択
> 3. データクレンジング
> 4. データ変換
> 5. データマイニング
> 6. 解釈・評価

## Excel からのデータ読み込み
```
xlsx = pd.ExcelFile(ファイルパス)
df = pd.read_excel(xlsx, 対象シート)
```

## DBからのデータ読み込み
```
engine = sqla.create_engine('接続データベース+ドライバー名://接続ユーザー名:パスワード@ホスト名:ポート番号/データベース名?charset=文字コード')
print(pd.read_sql('''
SELECT
  列名1,
  列名2,
  ...
FROM テーブル名
''', engine))
```

## 欠損値が発生するパターンと対処法
- MCAR(Missing Completely At Random): 欠損確率が他の項目と無関係。
    - 誤記など
    - リストワイズ削除（NaNを含むデータを削除）か、代入法などで対処。
- MAR(Missing At Random): 欠損確率が他の項目と相関がある。
    - 性別が女性であると、体重という項目が欠損しやすい。
    - リストワイズ削除すると、相関する項目のデータに偏りが発生するので、代入法など。相関する項目から推測した代入法などで対処。
- NMAR(Not Missing At Random): 欠損確率が、その項目自体の値のみに依存する。
    - 体重という項目が、体重が重くなるほど欠損しやすい（答えてもらいづらい）
    - リストワイズ削除すると、その項目に偏りが発生する。他項目からの推測も難しいので、再収集などが必要になる、


## 欠損値の対処
## 欠損値の可視化
- `missingno.matrix(data)` を使えば、画像で欠損の状況を確認できる。

### 単一代入法(ホットデック法)
- 欠損値を含むデータ行(レシピエントと呼ぶ)の欠損値を、欠損しているそのデータ項目の値を持っている別のデータ行(ドナーと呼ぶ)の値を使って補完
- `knnimpute.knn_impute_few_observed(matrix, missing_mask, k)`
- 結果は `np.ndarray` 型になっているため、`pd.DataFrame(result, colums=)` でdfに変換

### 多重代入法(MICE)
- 欠損値を、他のデータによる回帰モデルで推定

```
import statsmodels.api as sm
from statsmodels.imputation import mice

imp_data = mice.MICEData(data)
formula = 'price ~ distance + age + m2'
model = mice.MICE(formula, sm.OLS, imp_data)
results = model.fit(3, 3)
print(results.summary())
```

## 外れ値の対処
### 外れ値の可視化
- `seaborn.boxplot(y=Series)` による箱ひげ図
- ２軸なら、`seaborn.jointplot('x_key', 'y_key', data=df)` もよい。

### 外れ値の検出(LOF:Local Outlier Factor)
- LOFとは、以下の考え方で外れ値を検出する方法
    - 近くにデータ点が少ないのが外れ値であると考える
    - k個の近傍点を使ってデータの密度を推定する
    - 上記の密度が、周囲と相対的に低い点を外れ値と判定する
```
clf = sklearn.neighbors.LocalOutlierFactor(n_neighbors=20)
predictions = clf.fit_predict(data)
data[predictions == -1]
```

### 外れ値の検出(Isolation Forest)
- Isolation Forestは、以下の特徴がある検出方法
    - 距離や密度に依存しないため、それらの指標を計算するコストが不要
    - 計算が複雑でなく、省メモリである
    - 大規模データであっても計算をスケールさせやすい
```
clf = sklearn.ensemble.IsolationForest()
clf.fit(data)
predictions = clf.predict(data)
data[predictions == -1]
```

## 不均一データの対処
### データの均一性を確認
- `df["key"].vales_counts()` でカテゴリ値の偏りを確認できる

### アンダーサンプリング
- 多い方を減らして、バランスを取る。
```
rus = imblearn.under_sampling.RandomUnderSampler(ratio= 'majority')
X_resampled, y_resampled = rus.fit_sample(X, y)
```

### オーバーサンプリング
- 少ない方を水増しする。
- 例えば、ランダムに増やす。
```
ros = imblearn.under_sampling.RandomOverSampler(ratio = 'minority')
X_resampled, y_resampled =ros.fit_sample(X, y)
```

### (アンダー/オーバー)サンプリングの組み合わせ: SMOTE-ENN
- オーバーサンプリングにSMOTE(Synthetic minority over-sampling technique、*1)、アンダーサンプリングにENN(Edited Nearest Neighbours) を使う
- SMOTE: 
    - 正例を、そのまま水増しするのではなく最近傍法(kNN)を使って、近傍点から増やすべきデータの値を推測してオーバーサンプリングを行う
- ENN:
    - 負例を削除する際、各データ点の近傍の点を考慮するようにし、近くにあるデータが少なくなるようにアンダーサンプリングを行う
```
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

sm_enn = SMOTEENN(smote=SMOTE(k_neighbors=3), enn=EditedNearestNeighbours(n_neighbors=3))
X_resampled, y_resampled =  sm_enn.fit_sample(X, y)
(X_resampled, y_resampled)
```

## 連続値のカテゴリ化
- 開空間にするには `right=False` など
- `pd.cut(Series, bins=[], labels=[], right=False)`

## カテゴリデータのダミー変数化
- `pd.get_dummies(Series)`

## データスケールの変換
- 平均0, 分散1 に標準化するには `sklearn.preprocessing.scale(data)`

## 縦持ち横持ち変換
- `pd.pivot`
```
pivoted_data = data.pivot(index='', columns='', values='')
pivoted_data = pivoted_data.reset_index()
pivoted_data.columns = pivoted_data.columns.set_names(None)
```
- `pd.melt(df, id_vars=, value_vars=, var_name=, value_name=)`
    - id_vars: 変換のキーに使う列を配列で指定する
    - value_vars: 変換後に値として使う列を配列で指定する
    - var_name: 変換後に、横持ちデータの列をまとめるための変数列の名前
    - value_name: 変換後に、値となる列の名前