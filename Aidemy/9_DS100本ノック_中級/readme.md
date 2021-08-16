# サマリ
- 初級に続き、回答をメモで残す

# 回答
## 10 前方一致
> 店舗データフレーム（df_store）から、店舗コード（store_cd）が"S14"で始まるものだけ全項目抽出し、10件だけ表示せよ。
- `df.query('key.str.startswith("")', engine='python')`
```
print(df_store.query('store_cd.str.startswith("S14")', engine='python').head(10))
```

## 13 前方一致
> 顧客データフレーム（df_customer）から、ステータスコード（status_cd）の先頭がアルファベットのA〜Fで始まるデータを全項目抽出し、10件だけ表示せよ。
- `df.query('key.str.contains(Regex, regex=True)', engine='python')`
- 他には、`df.key.str.contains(Regex)` で mask を作って、`df[mask]` で抽出する方法も。
```
mask = df_customer.status_cd.str.contains('^[A-F].*')
print(df_customer[mask].head(10))
```

## 11 後方一致
> 顧客データフレーム（df_customer）から顧客ID（customer_id）の末尾が1のものだけ全項目抽出し、10件だけ表示せよ。
- `df.query('key.str.endswith("")', engine='python')`
```
print(df_customer.query('customer_id.str.endswith("1")', engine='python').head(10))
```

## 14 後方一致
> 顧客データフレーム（df_customer）から、ステータスコード（status_cd）の末尾が数字の1〜9で終わるデータを全項目抽出し、10件だけ表示せよ。
```
print(df_customer.query('status_cd.str.contains(".*[1-9]$", regex=True)', engine='python').head(10))
```

## 12 部分一致
> 店舗データフレーム（df_store）から横浜市の店舗だけ全項目表示せよ。
```
mask = df_store.address.str.contains("^神奈川県横浜市.*")
print(df_store[mask])
```

## 15 部分一致
> 顧客データフレーム（df_customer）から、ステータスコード（status_cd）の先頭がアルファベットのA〜Fで始まり、末尾が数字の1〜9で終わるデータを全項目抽出し、10件だけ表示せよ。
```
print(df_customer.query('status_cd.str.contains("^[A-F].*[1-9]$", regex=True)', engine='python').head(10))
```

## 17 並び替え
> 顧客データフレーム（df_customer）を生年月日（birth_day）で高齢順にソートし、先頭10件を全項目表示せよ。
- `df.sort_values("key")`
```
print(df_customer.sort_values("birth_day", ascending=True).head(10))
```

## 18 並び替え
> 顧客データフレーム（df_customer）を生年月日（birth_day）で若い順にソートし、先頭10件を全項目表示せよ。
```
print(df_customer.sort_values("birth_day", ascending=False).head(10))
```

## 19 順位
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

## 20 順位
> レシート明細データフレーム（df_receipt）に対し、1件あたりの売上金額（amount）が高い順にランクを付与し、先頭10件を抽出し表示せよ。項目は顧客ID（customer_id）、売上金額（amount）、付与したランクを表示させること。なお、売上金額（amount）が等しい場合でも別順位を付与すること。
- `series.rank(method="first")` : method="first" で、同値には出現順に順位が振られる。
```
df_tmp = df_receipt["amount"].rank(method="first", ascending=False)
df_tmp = pd.concat([df_receipt[["customer_id", "amount"]], df_tmp], axis=1)
df_tmp.columns = ["customer_id", "amount", "ranking"]
print(df_tmp.sort_values("ranking", ascending=True).head(10))
```

## 21 カウント
> レシート明細データフレーム（df_receipt）に対し、件数をカウントせよ。
```
print(len(df_receipt))
```

## 22 カウント
> レシート明細データフレーム（df_receipt）の顧客ID（customer_id）に対し、ユニーク件数をカウントせよ。
- `drop_duplicates(key)` の他にも `series.unique()` を使う方法も。
```
print(len(df_receipt.drop_duplicates("customer_id")))
```

## 23 合計
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）と売上数量（quantity）を合計し、結果を5件表示せよ。
```
df_tmp = df_receipt.groupby("store_cd").sum().reset_index()
print(df_tmp[["store_cd", "amount", "quantity"]].head())
```
- agg を使用した最適解
```
print(df_receipt.groupby('store_cd').agg({'amount':'sum', 'quantity':'sum'}).reset_index().head())
```

 ## 24 Max/Min
> レシート明細データフレーム（df_receipt）に対し、顧客ID（customer_id）ごとに最も新しい売上日（sales_ymd）を求め、10件表示せよ。
```
print(df_receipt.groupby("customer_id").agg({"sales_ymd": "max"}).reset_index().head(10))
```

 ## 25 Max/Min
> レシート明細データフレーム（df_receipt）に対し、顧客ID（customer_id）ごとに最も古い売上日（sales_ymd）を求め、10件表示せよ。
```
print(df_receipt.groupby("customer_id").agg({"sales_ymd": "min"}).reset_index().head(10))
```

## 26 Max/Min
> レシート明細データフレーム（df_receipt）に対し、顧客ID（customer_id）ごとに最も新しい売上日（sales_ymd）と古い売上日を求め、両者が異なるデータを10件表示せよ
- `df.groupby(key).agg({key:['max','min']})` で複数の処理結果を返すことができる
```
df_tmp = df_receipt.groupby('customer_id').agg({'sales_ymd':['max','min']}).reset_index()
#  カラム名を指定します。書き換える必要はありません
df_tmp.columns = ["_".join(pair) for pair in df_tmp.columns]
print(df_tmp.query('sales_ymd_max != sales_ymd_min').head(10))
```

## 27 統計量
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の平均を計算し、降順でTOP5を表示せよ。
```
print(df_receipt.groupby("store_cd").agg({"amount": "mean"}).reset_index().sort_values("amount", ascending=False).head())
```

## 28 統計量
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の中央値を計算し、降順でTOP5を表示せよ。
```
print(df_receipt.groupby("store_cd").agg({"amount": "median"}).reset_index().sort_values("amount", ascending=False).head())
```

## 29 統計量
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに商品コード（product_cd）の最頻値を求めよ。
- groupby 後の処理に最頻値(mode)はないので、`df.groupby(key).key.apply(lambda x:func(x))` を使う
- `mean()`
```
print(df_receipt.groupby("store_cd").product_cd.apply(lambda cd: cd.mode()).reset_index())
```

## 30 統計量
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の標本分散を計算し、降順でTOP5を表示せよ。
- `var()`
```
print(df_receipt.groupby("store_cd").amount.apply(lambda x: x.var(ddof=0)).reset_index().sort_values("amount", ascending=False).head())
```

## 31 統計量
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の標本標準偏差を計算し、降順でTOP5を表示せよ。
- `std()`
```
print(df_receipt.groupby("store_cd").amount.apply(lambda x: x.std(ddof=0)).reset_index().sort_values("amount", ascending=False).head())
```

## 32 統計量
> レシート明細データフレーム（df_receipt）の売上金額（amount）について、25％刻みでパーセンタイル値を求めよ。
- `np.percentile(series, [25,50,75,100], axis=0)`
```
print(np.percentile(df_receipt["amount"], [25,50,75,100], axis=0))
```

## 33 統計量
> レシート明細データフレーム（df_receipt）に対し、店舗コード（store_cd）ごとに売上金額（amount）の平均を計算し、330以上のものを抽出し出力せよ。
```
df_tmp = df_receipt.groupby("store_cd").amount.mean().reset_index();
print(df_tmp.query('amount >= 330'))
```

## 34 検索結果からのサブクエリ
> レシート明細データフレーム（df_receipt）に対し、顧客ID（customer_id）ごとに売上金額（amount）を合計して全顧客の平均を求めよ。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。
```
print(df_receipt.query('not customer_id.str.startswith("Z")', engine='python').groupby("customer_id").amount.sum().mean())
```

## 35 条件指定でのサブクエリ
> レシート明細データフレーム（df_receipt）に対し、顧客ID（customer_id）ごとに売上金額（amount）を合計して全顧客の平均を求め、平均以上に買い物をしている顧客を抽出せよ。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。なお、データは10件だけ表示させれば良い。
- `df.query('')` 内で変数を使いたいなら `@x` とする
```
df_amount_sum = df_receipt.query('not customer_id.str.startswith("Z")', engine='python').groupby("customer_id").amount.sum().reset_index()
mean_amount = df_amount_sum.amount.mean()
print(df_amount_sum.query('amount >= @mean_amount').head(10))
```

## 75 ランダム
> 顧客データフレーム（df_customer）からランダムに1%のデータを抽出し、先頭から10件データを抽出せよ。なお、random_stateは42としなさい。
- `df.sample()`
```
df_sample = df_customer.sample(frac = 0.01, random_state=42)
print(df_sample.head(10))
```

## 76 層化
> 顧客データフレーム（df_customer）から性別（gender_cd）の割合に基づきランダムに10%のデータを層化抽出データし、性別ごとに件数を集計せよ。なお、random_stateは42としなさい。
- **層化抽出** : 元データの分布を維持したまま、ランダム抽出する方法
- `sklearn.model_selection.train_test_split` を使うと、層化抽出が可能
```
_, df_tmp = train_test_split(df_customer, test_size=0.1, stratify=df_customer["gender_cd"], random_state=42)
print(df_tmp.groupby("gender_cd").agg({"customer_id":"count"}))
```

# 84 除算エラー対応
> 顧客データフレーム（df_customer）の全顧客に対し、全期間の売上金額に占める2019年売上金額の割合を計算せよ。ただし、販売実績のない場合は0として扱うこと。そして計算した割合が0超のものを抽出せよ。 結果は10件表示させれば良い。
```

# 1.レシート明細データフレーム（df_receipt）からqueryメソッドにて該当の期間のデータを抽出する
df_tmp_1 = df_receipt.query('20191231 >=sales_ymd >= 20190101')
# 2. "１"で抽出したデータを顧客データフレーム（df_customer）に結合する
df_tmp_1 = pd.merge(df_customer[["customer_id"]], df_tmp_1[["customer_id", "amount"]], how="left", on="customer_id"). groupby('customer_id').sum().reset_index().rename(columns={'amount':'amount_2019'})

# 3. レシート明細データフレーム（df_receipt）を顧客データフレーム（df_customer）に結合する
df_tmp_2 = pd.merge(df_customer[["customer_id"]], df_receipt[["customer_id", "amount"]], how="left", on="customer_id"). groupby('customer_id').sum().reset_index()

# 4. "2"と"3"で得たデータを内部結合する
df_tmp =pd.merge(df_tmp_1, df_tmp_2, how="inner", on="customer_id")
# 5. "4"の結合時に生じた欠損値を補完する
df_tmp['amount_2019'] = df_tmp['amount_2019'].fillna(0)
df_tmp['amount'] = df_tmp['amount'].fillna(0)
# 6. 2019の売り上げ金額 / 全期間の売上金額を行い割合をデータフレームに追加する 
df_tmp['amount_rate'] = df_tmp['amount_2019'] / df_tmp['amount']
# 7. "6"で生じた欠損値を補完する
df_tmp['amount_rate'] = df_tmp['amount_rate'].fillna(0)
# 8. queryメソッドにて条件に基づいて取得する
print(df_tmp.query('amount_rate > 0').head(10))
```

## 87 完全一致
> 顧客データフレーム（df_customer）では、異なる店舗での申込みなどにより同一顧客が複数登録されている。名前（customer_name）と郵便番号（postal_cd）が同じ顧客は同一顧客とみなし、1顧客1レコードとなるように名寄せした名寄顧客データフレーム（df_customer_u）を作成せよ。ただし、同一顧客に対しては売上金額合計が最も高いものを残すものとし、売上金額合計が同一もしくは売上実績の無い顧客については顧客ID（customer_id）の番号が小さいものを残すこととする。
- `sort_values(by=["key1", "key2"], ascending=[True, False])`
```
#　顧客ごとの売上金額合計を算出する
df_tmp = df_receipt.groupby("customer_id").agg({"amount": "sum"}).reset_index()
# 顧客データフレーム（df_customer）に売上金額合計を追加し、売上金額合計、顧客IDでソートする
df_customer_u = pd.merge(df_customer, df_tmp, how="left", on="customer_id").sort_values(by=["amount","customer_id"], ascending=[False, True])
# 同一顧客に対しては売上金額合計が最も高いものを残すように削除する
df_customer_u.drop_duplicates(subset=["customer_name", "postal_cd"], keep='first', inplace=True)

print('減少数: ', len(df_customer) - len(df_customer_u))
```

## 88 変換データ作成
> 前設問で作成したデータ（df_customer_u）を元に、顧客データフレームに統合名寄ID（integration_id）を付与したデータフレーム（df_customer_n）を作成せよ。ただし、統合名寄IDは以下の仕様で付与するものとする。
> - 重複していない顧客：顧客ID（customer_id）を設定
> - 重複している顧客：前設問で抽出したレコードの顧客IDを設定
```
# 顧客データフレーム(df_customer)と名寄顧客データフレーム（df_customer_u）を内部結合する
df_customer_n = pd.merge(df_customer, df_customer_u, how="inner", on =['customer_name', 'postal_cd'])
# カラム名を変更する
df_customer_u.drop_duplicates(subset=["customer_name", "postal_cd"], keep='first', inplace=True)
df_customer_n.rename(columns={"customer_id_y": "integration_id"} , inplace=True)

print('ID数の差', len(df_customer_n['customer_id_x'].unique()) - len(df_customer_n['integration_id'].unique()))
```

## 89 レコードデータ
> 売上実績のある顧客に対し、予測モデル構築のため学習用データとテスト用データに分割したい。それぞれ8:2の割合でランダムにデータを分割せよ。また、random_stateは71とせよ。
- `train_test_split(df, test_size=nn)`
```
#　顧客ごとの売上金額合計を算出します
df_sales= df_receipt.groupby("customer_id").agg({"amount": "sum"}).reset_index()
#  df_salesにある顧客のみを抽出します
df_tmp = pd.merge(df_customer, df_sales['customer_id'], how='inner', on='customer_id')
# 8:2の割合でランダムにデータを分割します
df_train, df_test = train_test_split(df_tmp, test_size=0.2, train_size=0.8, random_state=71)
print('学習データ割合: ', len(df_train) / len(df_tmp))
print('テストデータ割合: ', len(df_test) / len(df_tmp))
```

## 90 時系列データ
> レシート明細データフレーム（df_receipt）は2017年1月1日〜2019年10月31日までのデータを有している。売上金額（amount）を月次で集計し、学習用に12ヶ月、テスト用に6ヶ月のモデル構築用データを3セット作成せよ。
```
df_tmp = df_receipt[['sales_ymd', 'amount']].copy()
# 西暦と月のみにし、"sales_ym"に代入します
df_tmp['sales_ym'] = df_tmp["sales_ymd"].map(lambda x: int(x/100))
# 月毎の"amount"を算出します
df_tmp = df_tmp.groupby("sales_ym").amount.sum().reset_index()

#  「train_size, test_size」はデータの長さ, 「slide_window,start_point」はtrainデータの始まりを決定するのに使用します
def split_data(df, train_size, test_size, slide_window, start_point):
    train_start = start_point * slide_window
    test_start = train_start + train_size
    return df[train_start: test_start], df[test_start: test_start + slide_window]

df_train_1, df_test_1 = split_data(df_tmp, train_size=12, test_size=6, slide_window=6, start_point=0)
df_train_2, df_test_2 = split_data(df_tmp, train_size=12, test_size=6, slide_window=6, start_point=1)
df_train_3, df_test_3 = split_data(df_tmp, train_size=12, test_size=6, slide_window=6, start_point=2)
print(df_train_3)
```