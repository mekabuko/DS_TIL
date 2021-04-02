# サマリ
- 全体として手で覚えないと、しばらくはチートシートが必要になりそう。100本ノックで鍛えるしかないか。
- https://github.com/Gedevan-Aleksizde/pandas-cheat-sheet-ja/blob/master/doc/Pandas_Cheat_Sheet_ja.pdf
    - この日本語訳されたチートシートがあればほとんどことたりそう。
    
# メモ
## NumPy との使い分け
- NumPy：行列として扱うことができ、科学計算に特化
- Pandas：一般的なDBで行える操作が可能。数値以外にも文字列なども扱うことが可能。

## Pandasのデータ
### DataFrame: 表形式のデータ
- 辞書型から変換可能。

```
# 元になる辞書型
data_d = {
    "fruits": ["apple", "orange", "banana", "strawberry"],
    "year": [2001, 2002, 2001, 2008],
    "time": [1, 4, 5, 6]}
# データフレームへの変換(columns で ソート順を指定)
df = pd.DataFrame(data_d, columns=["year", "time", "fruits"])
```

### Series: DataFrameの１行/１列
- list から変換可能

```
# 元となるリスト
data_s = [10, 5, 8, 12, 3]

# Seriesへの変換
series = pd.Series(data_s)
```

- index 付きも可能
```
# 元となるリスト
data_s = [10, 5, 8, 12, 3]
index = ["a","b","c","d","e"]

# Seriesへの変換
series = pd.Series(data_s, index=index)
```

## DataFrame からの取り出し
- index番号指定：`df[1:2]`
- index名指定： `df[["key_1","key_2"]]`
- valueのみ: `df.values`
- indexのみ: `df.index`

## DataFrame の操作
###  行追加
- `append(Series s)`
```
new_data = {"xxx": 1}
df = df.append(pd.Series(new_data)) 
```
- インデックスが元のDFと異なる時には `ignore_index=True` を指定すること

### 列追加
- `df["index"] = series`

### 削除
- `df.drop("index", axis=n)`

### 参照
- 名前参照: loc
```
df.loc[1:5,["xxx", "yyy"]] # loc[index_num, [column_name]]で取得
```
- 番号参照: iloc
``` 
df = df.iloc[1:5, [2,4]] # loc[index_num, [column_num]]で取得
```
- `head()/tail()` も可能

### ソート
- sort_values(by="カラム名")
```
df.sort_values(by="xxx", ascending=False) #ascending= で昇降設定
```
- sort_values(by="カラムのリスト")
```
df.sort_values(by=["xxx","yyy])
```

### フィルタ
- `df.loc[条件式]`
    - インデックス名を使う
    ```
    df = df.loc[df["xxx"] >= 5]
    ```
    - インデックス番号を使う
    ```
    df = df.loc[df.index >= 5]
    ```

### 結合
- `pd.concat([df_1, df_2], axis=0)`
    - index や columnが一致していない場合、抜けている部分は `NaN` になる。
- 複合インデックス
    - `pd.concat([df_1, df_2], axis=1, keys=["Data1","Data2"])`
    - `df["Data1", "xxx"]` で複合インデックスの項目にアクセス可能

### 内部結合/外部結合
- 内部結合
    - 同じ key を持たないものは落とされる
    - `pd.merge(df1, df2, on="xxx", how="inner")`
- 内部結合
    - key がなくてもいい
         - 不足部分は `NaN` になる
    - `pd.merge(df1, df2, on="xxx", how="outer")`
- Key の同一みなし（同名でない列をKeyとした結合）
    - `pd.merge(df1, df2, left_on="df1のkey", right_on="df2のkey", how="結合方法")`
- indexをKeyとして使う時
    - `left(right)_on="xxx"` の代わりに `left(right)_index=True` とする
    - `pd.merge(df1, df2, left_on="df1のkey", right_index=True, how="結合方法")`

### 差分
- 時系列分析などで特に用いられる、行間の差を求める方法
```
df.diff(periods=2, axis=0) # n行前、で periods=n。後なら、-n。
```
### グループ化
- ある列の値が同じものでグループ化。その結果はに対して、mean()などをできる。
```
df.groupby("XXX").mean()
```

## Seriesの操作

### 追加
- `append(Series s)`
```
new_data = {"xxx": 1}
series = series.append(pd.Series(new_data)) 
```
- インデックスが元のDFと異なる時には `ignore_index=True` を指定すること

### 削除
- `df.drop("index")`

### ソート
- インデックス：`series.sort_index()`
- 値：`series.sort_values()`
    - `ascending=False` を引数にすると降順になる

### フィルタ
- `&, |` を使用可能
```
series = series[(x > 1) & (x < 10)]
```