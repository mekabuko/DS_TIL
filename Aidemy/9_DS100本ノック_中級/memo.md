# サマリ
- 初級に続き、回答をメモで残す

# 回答
## 10
> 店舗データフレーム（df_store）から、店舗コード（store_cd）が"S14"で始まるものだけ全項目抽出し、10件だけ表示せよ。
- `df.query('key.str.startswith("")', engine='python')`
```
print(df_store.query('store_cd.str.startswith("S14")', engine='python').head(10))
```

## 13
> 顧客データフレーム（df_customer）から、ステータスコード（status_cd）の先頭がアルファベットのA〜Fで始まるデータを全項目抽出し、10件だけ表示せよ。
- `df.query('key.str.contains(Regex, regex=True)', engine='python')`
- 他には、`df.key.str.contains(Regex)` で mask を作って、`df[mask]` で抽出する方法も。
```
mask = df_customer.status_cd.str.contains('^[A-F].*')
print(df_customer[mask].head(10))
```

## 11
> 顧客データフレーム（df_customer）から顧客ID（customer_id）の末尾が1のものだけ全項目抽出し、10件だけ表示せよ。
- `df.query('key.str.endswith("")', engine='python')`
```
print(df_customer.query('customer_id.str.endswith("1")', engine='python').head(10))
```

## 14
> 顧客データフレーム（df_customer）から、ステータスコード（status_cd）の末尾が数字の1〜9で終わるデータを全項目抽出し、10件だけ表示せよ。
```
print(df_customer.query('status_cd.str.contains(".*[1-9]$", regex=True)', engine='python').head(10))
```

## 12
> 店舗データフレーム（df_store）から横浜市の店舗だけ全項目表示せよ。
```
mask = df_store.address.str.contains("^神奈川県横浜市.*")
print(df_store[mask])
```

## 15
> 顧客データフレーム（df_customer）から、ステータスコード（status_cd）の先頭がアルファベットのA〜Fで始まり、末尾が数字の1〜9で終わるデータを全項目抽出し、10件だけ表示せよ。
```
print(df_customer.query('status_cd.str.contains("^[A-F].*[1-9]$", regex=True)', engine='python').head(10))
```

## 17
> 顧客データフレーム（df_customer）を生年月日（birth_day）で高齢順にソートし、先頭10件を全項目表示せよ。
- `df.sort_values("key")`
```
print(df_customer.sort_values("birth_day", ascending=True).head(10))
```

## 18
> 顧客データフレーム（df_customer）を生年月日（birth_day）で若い順にソートし、先頭10件を全項目表示せよ。
```
print(df_customer.sort_values("birth_day", ascending=False).head(10))
```

## 19
> レシート明細データフレーム（df_receipt）に対し、1件あたりの売上金額（amount）が高い順にランクを付与し、先頭10件を抽出し表示せよ。項目は顧客ID（customer_id）、売上金額（amount）、付与したランクを表示させること。なお、売上金額（amount）が等しい場合は同一順位を付与するものとする。
- `series.rank(method="min")` : method="min" で、同値には最小の同一順位となる。
```
# ランクづけ
df_tmp = df_receipt["amount"].rank(method='min', ascending=False)
# 元の表とランクを結合
df_tmp = pd.concat([df_receipt[["customer_id", "amount"]], df_tmp], axis=1)
# column 名を設定
df_tmp.columns = ['customer_id', 'amount', 'ranking']
# rankの昇順で出力
print(df_tmp.sort_values("ranking", ascending=True).head(10))
```

## 20
> レシート明細データフレーム（df_receipt）に対し、1件あたりの売上金額（amount）が高い順にランクを付与し、先頭10件を抽出し表示せよ。項目は顧客ID（customer_id）、売上金額（amount）、付与したランクを表示させること。なお、売上金額（amount）が等しい場合でも別順位を付与すること。
- `series.rank(method="first")` : method="first" で、同値には出現順に順位が振られる。
```
df_tmp = df_receipt["amount"].rank(method="first", ascending=False)
df_tmp = pd.concat([df_receipt[["customer_id", "amount"]], df_tmp], axis=1)
df_tmp.columns = ["customer_id", "amount", "ranking"]
print(df_tmp.sort_values("ranking", ascending=True).head(10))
```

## 21
> レシート明細データフレーム（df_receipt）に対し、件数をカウントせよ。
```
print(len(df_receipt))
```

## 22
> レシート明細データフレーム（df_receipt）の顧客ID（customer_id）に対し、ユニーク件数をカウントせよ。
- `drop_duplicates(key)` の他にも `series.unique()` を使う方法も。
```
print(len(df_receipt.drop_duplicates("customer_id")))
```

## 23
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）と売上数量（quantity）を合計し、結果を5件表示せよ。
```
df_tmp = df_receipt.groupby("store_cd").sum().reset_index()
print(df_tmp[["store_cd", "amount", "quantity"]].head())
```
- agg を使用した最適解
```
print(df_receipt.groupby('store_cd').agg({'amount':'sum', 'quantity':'sum'}).reset_index().head())
```

 ## 24
> レシート明細データフレーム（df_receipt）に対し、顧客ID（customer_id）ごとに最も新しい売上日（sales_ymd）を求め、10件表示せよ。
```
print(df_receipt.groupby("customer_id").agg({"sales_ymd": "max"}).reset_index().head(10))
```

 ## 25
> レシート明細データフレーム（df_receipt）に対し、顧客ID（customer_id）ごとに最も古い売上日（sales_ymd）を求め、10件表示せよ。
```
print(df_receipt.groupby("customer_id").agg({"sales_ymd": "min"}).reset_index().head(10))
```

## 26
> レシート明細データフレーム（df_receipt）に対し、顧客ID（customer_id）ごとに最も新しい売上日（sales_ymd）と古い売上日を求め、両者が異なるデータを10件表示せよ
- `df.groupby(key).agg({key:['max','min']})` で複数の処理結果を返すことができる
```
df_tmp = df_receipt.groupby('customer_id').agg({'sales_ymd':['max','min']}).reset_index()
#  カラム名を指定します。書き換える必要はありません
df_tmp.columns = ["_".join(pair) for pair in df_tmp.columns]
print(df_tmp.query('sales_ymd_max != sales_ymd_min').head(10))
```

## 27
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の平均を計算し、降順でTOP5を表示せよ。
```
print(df_receipt.groupby("store_cd").agg({"amount": "mean"}).reset_index().sort_values("amount", ascending=False).head())
```

## 28 
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の中央値を計算し、降順でTOP5を表示せよ。
```
print(df_receipt.groupby("store_cd").agg({"amount": "median"}).reset_index().sort_values("amount", ascending=False).head())
```

## 29
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに商品コード（product_cd）の最頻値を求めよ。
- groupby 後の処理に最頻値(mode)はないので、`df.groupby(key).key.apply(lambda x:func(x))` を使う
- `mean()`
```
print(df_receipt.groupby("store_cd").product_cd.apply(lambda cd: cd.mode()).reset_index())
```

## 30
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の標本分散を計算し、降順でTOP5を表示せよ。
- `var()`
```
print(df_receipt.groupby("store_cd").amount.apply(lambda x: x.var(ddof=0)).reset_index().sort_values("amount", ascending=False).head())
```

## 31 
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の標本標準偏差を計算し、降順でTOP5を表示せよ。
- `std()`
```
print(df_receipt.groupby("store_cd").amount.apply(lambda x: x.std(ddof=0)).reset_index().sort_values("amount", ascending=False).head())
```

## 32
> レシート明細データフレーム（df_receipt）の売上金額（amount）について、25％刻みでパーセンタイル値を求めよ。
- `pd.percentile(series, [25,50,75,100], axis=0)`
```
print(np.percentile(df_receipt["amount"], [25,50,75,100], axis=0))
```

## 33
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の平均を計算し、330以上のものを抽出し出力せよ。
```
df_tmp = df_receipt.groupby("store_cd").amount.mean().reset_index();
print(df_tmp.query('amount >= 330'))
```

