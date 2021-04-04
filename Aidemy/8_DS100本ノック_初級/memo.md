# サマリ
- 手元で環境作ってやろうかと思ったけど、面倒で断念。
- 回答を一通り、メモで残す形にしておいた。
- リファレンス片手に大体行けたものの、ガイドラインがないと実際には何をしていいかわからない時も。。
- あと、求められていることはできていても、出力が完全一致しないとダメな場合があり（わかりやすくするために column名を付け直した時など）、無駄にハマった。出力を見ても何が間違えているのかわからないような時には、さっさと答えを確認してよさそう。

# 回答
## Series 1
共通部は以下の通り
```
import pandas as pd

# データをcsvファイルから読み込みます。書き換える必要はありません
df_customer = pd.read_csv('./100knocks-preprocess/customer.csv')
df_category = pd.read_csv('./100knocks-preprocess/category.csv')
df_product = pd.read_csv('./100knocks-preprocess/product.csv')
df_receipt =pd.read_csv('./100knocks-preprocess/receipt.csv')
df_store = pd.read_csv('./100knocks-preprocess/store.csv')
df_geocode = pd.read_csv('./100knocks-preprocess/geocode.csv')
```
## 1
> レシート明細のデータフレーム（df_receipt）から全項目の先頭10件を表示し、どのようなデータを保有しているか目視で確認せよ。
```
print(df_receipt.head(10))
```

## 2 
> レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、10件表示させよ。
```
print(df_receipt[["sales_ymd", "customer_id", "product_cd", "amount"]].head(10))
```

## 3
> レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、10件表示させよ。
> - ただし、sales_ymdはsales_dateに項目名を変更しながら抽出すること。
- `rename(columns={before:after})`
```
print(df_receipt[["sales_ymd", "customer_id", "product_cd", "amount"]].rename(columns={"sales_ymd":"sales_date"}).head(10))
```

## 4 
> レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
> - 顧客ID（customer_id）が"CS018205000001"
- `query('条件')`
```
print(df_receipt[["sales_ymd", "customer_id", "product_cd", "amount"]].query('customer_id == "CS018205000001"'))
```

## 5
>レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
> - 顧客ID（customer_id）が"CS018205000001"
> - 売上金額（amount）が1,000以上
- `query('条件1 & 条件2')`
```
print(df_receipt[["sales_ymd", "customer_id", "product_cd", "amount"]].query('customer_id == "CS018205000001" & amount >= 1000'))
```

## 6
>レシート明細データフレーム「df_receipt」から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上数量（quantity）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
> - 顧客ID（customer_id）が"CS018205000001"
> - 売上金額（amount）が1,000以上または売上数量（quantity）が5以上
- `query('条件1 & (条件2 | 条件3)')`
```
print(df_receipt[["sales_ymd", "customer_id", "product_cd", "quantity", "amount"]].query('customer_id == "CS018205000001" & (amount >= 1000 | quantity >= 5)'))
```

## 7
> レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
> - 顧客ID（customer_id）が"CS018205000001"
> - 売上金額（amount）が1,000以上2,000以下
- `query('a < x < b')`
```
print(df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'amount']].query('customer_id == "CS018205000001" & 1000 <= amount <= 2000'))
```

## 8
> レシート明細のデータフレーム（df_receipt）から売上日（sales_ymd）、顧客ID（customer_id）、商品コード（product_cd）、売上金額（amount）の順に列を指定し、以下の条件を満たすデータを抽出せよ。
> - 顧客ID（customer_id）が"CS018205000001"
> - 商品コード（product_cd）が"P071401019"以外
```
print(df_receipt[['sales_ymd', 'customer_id', 'product_cd', 'amount']].query('customer_id == "CS018205000001" & product_cd != "P071401019"'))
```

## 9 
>以下の処理において、出力結果を変えずにORをANDに書き換えよ。
> `df_store.query('not(prefecture_cd == "13" | floor_area > 900)')`
- `not(A or B) = not(A) and not(B)`
```
print(df_store.query('prefecture_cd != "13" & floor_area <= 900'))
```

## 36
> レシート明細データフレーム（df_receipt）と店舗データフレーム（df_store）を内部結合し、レシート明細データフレームの全項目と店舗データフレームの店舗名（store_name）を10件表示させよ。
- `pd.merge(df1, df2, how="inner", on="key")`
```
print(pd.merge(df_receipt, df_store[['store_cd','store_name']], how='inner', on='store_cd').head(10))
```

## 37
> 商品データフレーム（df_product）とカテゴリデータフレーム（df_category）を内部結合し、商品データフレームの全項目とカテゴリデータフレームの小区分名（category_small_name）を10件表示させよ。
- `pd.merge(df1, df2, how="inner", on="key")`
```
# print(df_product.columns)
# print(df_category.columns)
print(pd.merge(df_product, df_category[['category_small_cd','category_small_name']], how='inner', on='category_small_cd').head(10))
```

## 38
> 顧客データフレーム（df_customer）とレシート明細データフレーム（df_receipt）から、各顧客ごとの売上金額合計を求めよ。ただし、買い物の実績がない顧客については売上金額を0として表示させること。また、顧客は性別コード（gender_cd）が女性（1）であるものを対象とし、非会員（顧客IDが'Z'から始まるもの）は除外すること。なお、結果は10件だけ表示させれば良い。
- `pd.merge(df1, df2, how="left", on="key")`
- `df.groupby("key").key.sum()`
- `df.reset_index()`
- `df.query('key.str.startswith("")', engine='python')`
```
# 各顧客idでグループ化して、amountを合計し、indexを再度振り直す
df_amount_sum = df_receipt.groupby("customer_id").amount.sum().reset_index()
# 条件に基づいた絞り込み。文字列一致のみ少し特殊。
df_tmp = df_customer.query('gender_cd == 1 & not(customer_id.str.startswith("Z"))', engine='python')
# 左側結合で全顧客データを出す。条件通り、購入していない人は 0 で埋まるようにする。
print(pd.merge(df_tmp, df_amount_sum, on="customer_id", how="left")[["customer_id", "amount"]].fillna(0).head(10))
```

## 39
>レシート明細データフレーム（df_receipt）から売上日数の多い顧客の上位20件と、売上金額合計の多い顧客の上位20件を抽出し、完全外部結合せよ。ただし、非会員（顧客IDが'Z'から始まるもの）は除外すること。
- `pd.merge(df1, df2, how="outer", on="key")`
- `df.groupby("key").count()`
- `df.drop_duplicates(subset=["key1","key2"])`
- `df.sort_values(by="key", ascending=False)`

```
df_receipt_only_member =\
    df_receipt.query('not(customer_id.str.startswith("Z"))', engine='python')

df_amount_top_customers =\
    df_receipt_only_member\
    .groupby("customer_id")\
    .amount.sum()\
    .reset_index()\
    .sort_values(by="amount", ascending=False)[:20]

df_count_top_customers =\
    df_receipt_only_member\
    .drop_duplicates(subset=["customer_id", "sales_ymd"])\
    .groupby("customer_id")\
    .sales_ymd.count().reset_index()\
    .sort_values(by="sales_ymd", ascending=False)[:20]

print(\
    pd.merge(\
        df_amount_top_customers,\
        df_count_top_customers,\
        on="customer_id",\
        how="outer"\
    )\
)
```

- こういうように、処理が長くなる時には、複数行に分けて、順番に処理していくほうがいいみたい。ワンライナーは見づらい。
    - バックスラッシュで改行分割だと、コメント入れられないし、順次変数更新していったほうがいい？参考解答はそうなっていた。
    ```
    df_sum = df_receipt.groupby('customer_id').amount.sum().reset_index()
    df_sum = df_sum.query('not customer_id.str.startswith("Z")', engine='python')
    df_sum = df_sum.sort_values('amount', ascending=False).head(20)
    ```

## 40
> 全ての店舗と全ての商品を組み合わせると何件のデータとなるか調査したい。店舗（df_store）と商品（df_product）を直積した件数を計算せよ。
- `df.copy()`
```
tmp_df_store =  df_store.copy()
tmp_df_product = df_product.copy()
tmp_df_store["key"] = 1
tmp_df_product["key"] = 1
# 外部結合で、keyが同じものに分配されるので、結果的に全店舗に全商品が分配されたデータになる
print(len(pd.merge(tmp_df_store, tmp_df_product, how="outer", on="key")))
```

## 41
> レシート明細データフレーム（df_receipt）の売上金額（amount）を日付（sales_ymd）ごとに集計し、前日からの売上金額増減を計算せよ。なお、計算結果は10件表示すればよい。
- df.shift()
- これは `df.diff()` を使っても同じことができそう
```
# "df_sales_amount_by_date "に処理後のデータを代入してください
df_sales_amount_by_date = df_receipt.groupby("sales_ymd").amount.sum().reset_index()
df_sales_amount_by_date = pd.concat([df_sales_amount_by_date, df_sales_amount_by_date.shift()], axis=1)

# カラム名を指定します。書き換える必要はありません
df_sales_amount_by_date.columns = ['sales_ymd','amount','lag_ymd','lag_amount']

# "diff_amount"カラムに売上金額増減を代入します
df_sales_amount_by_date['diff_amount'] =\
 df_sales_amount_by_date['amount'] - df_sales_amount_by_date['lag_amount']
print(df_sales_amount_by_date.head(10))
```
- `df.diff()` を使ったら、シフトなんてしなくていい
```
df_sales_amount_by_date = df_receipt.groupby("sales_ymd").amount.sum().reset_index()
df_sales_amount_by_date["diff_amount"] = df_sales_amount_by_date.diff(periods=1, axis=0)["amount"]
```

## 42
> レシート明細データフレーム（df_receipt）の売上金額（amount）を日付（sales_ymd）ごとに集計し、各日付のデータに対し、１日前、２日前、３日前のデータを結合せよ。結果は10件表示すればよい。
- df.dropna()
```
df_sales_amount_by_date = df_receipt.groupby("sales_ymd").amount.sum().reset_index()
df_lag = df_sales_amount_by_date
for i in range(1,4):
    df_lag = pd.concat([df_lag, df_sales_amount_by_date.shift(i)],axis=1)

# カラム名を指定します。書き換える必要はありません
df_lag.columns = ['sales_ymd', 'amount', 'lag_ymd_1', 'lag_amount_1', 'lag_ymd_2', 'lag_amount_2', 'lag_ymd_3', 'lag_amount_3']
print(df_lag.dropna().head(10))
```

## 43
> レシート明細データフレーム（df_receipt）と顧客データフレーム（df_customer）を結合し、性別（gender）と年代（ageから計算）ごとに売上金額（amount）を合計した売上サマリデータフレーム（df_sales_summary）を作成せよ。
性別は0が男性、1が女性、9が不明を表すものとする。
ただし、項目構成は年代、女性の売上金額、男性の売上金額、性別不明の売上金額の4項目とすること（縦に年代、横に性別のクロス集計）。また、年代は10歳ごとの階級とすること。
- `series.map(function)`
    - `df.apply(function)` はdfに対して操作できる。Series に対しては map。
- `pd.pivot_table(df, index, column, key, 操作)`

```
# 結合
df_sales_summary = pd.merge(df_receipt, df_customer, on="customer_id", how="inner")[["gender_cd", "age", "amount"]]

# 年代処理
def calc_floor(x):
   return math.floor(x / 10) * 10
df_sales_summary["era"] = df_sales_summary["age"].map(calc_floor)

# 性別・年代別に集計
df_sales_summary = pd.pivot_table(df_sales_summary, index='era', columns='gender_cd', values='amount', aggfunc='sum').reset_index()

# カラム名を指定します。書き換える必要はありません
df_sales_summary.columns = ['era', 'male', 'female', 'unknown']
print(df_sales_summary.head(10))
```

## 44
> 前設問(43問目)で作成した売上サマリデータフレーム（df_sales_summary）は性別の売上を横持ちさせたものであった。
このデータフレームから性別を縦持ちさせ、年代、性別コード、売上金額の3項目に変換し、変換後の売上サマリデータフレーム（df_sales_summary）を出力せよ。
ただし、性別コードは男性を'00'、女性を'01'、不明を'99'とする。
- `df.set_index("key")`
- `df.stack()`
- `df.rename()`
- `df.replace()`
```
# 43の続きから

df_sales_summary = df_sales_summary.set_index("era")
df_sales_summary = df_sales_summary.stack().reset_index()
df_sales_summary = df_sales_summary.rename(columns={"level_1": "gender_cd", 0:"amount"})
df_sales_summary = df_sales_summary.replace({"male": "00", "female": "01", "unknown": "99"})
print(df_sales_summary)
```

## 63
> 商品データフレーム（df_product）の単価（unit_price）と原価（unit_cost）から、各商品の利益額を算出せよ。結果は10件表示させれば良い。
```
tmp_df_product = df_product.copy()
tmp_df_product['unit_profit'] = df_product['unit_price'] - df_product['unit_cost']
print(tmp_df_product.head(10))
```

## 64
> 商品データフレーム（df_product）の単価（unit_price）と原価（unit_cost）から、各商品の利益率の全体平均を算出せよ。 ただし、単価と原価にはNULLが存在することに注意せよ。
```
tmp_df_product = df_product.copy()
tmp_df_product['unit_profit'] = df_product['unit_price'] - df_product['unit_cost']
# index毎に(単価-原価)/単価で利益率を求める
tmp_df_product['unit_profit_rate'] = tmp_df_product['unit_profit']/df_product['unit_price'] 
# 平均を求める
df_mean = tmp_df_product['unit_profit_rate'].mean()
print(df_mean)
```

## 65
> 商品データフレーム（df_product）の各商品について、利益率が30%となる新たな単価を求めよ。ただし、1円未満は切り捨てること。そして結果を10件表示させ、利益率がおよそ30％付近であることを確認せよ。ただし、単価（unit_price）と原価（unit_cost）にはNULLが存在することに注意せよ。
- `np.floor()`: NaN があってもOK！
```
df_tmp = df_product.copy()
# 利益率が30%となる価格を"new_price"カラムに代入します
df_tmp['new_price'] = np.floor(df_tmp['unit_cost'] / 0.7)
# "new_price"で利益率を算出し、"new_profit_rate"カラムに代入します
df_tmp['new_profit_rate'] = (df_tmp['new_price'] - df_tmp['unit_cost'])/df_tmp['new_price']
# print関数を使用し、df_tmpを10件表示させます
print(df_tmp.head(10))
```

# 66
> 商品データフレーム（df_product）の各商品について、利益率が30%となる新たな単価を求めよ。今回は、1円未満を四捨五入すること（0.5については偶数方向の丸めで良い）。そして結果を10件表示させ、利益率がおよそ30％付近であることを確認せよ。ただし、単価（unit_price）と原価（unit_cost）にはNULLが存在することに注意せよ。
- `np.round()`
```
df_tmp = df_product.copy()
# 利益率が30%となる価格を"new_price"カラムに代入します
df_tmp['new_price'] = np.round(df_tmp['unit_cost'] / 0.7)
# "new_price"で利益率を算出し、"new_profit_rate"カラムに代入します
df_tmp['new_profit_rate'] = (df_tmp['new_price'] - df_tmp['unit_cost'])/df_tmp['new_price']
# print関数を使用し、df_tmpを10件表示させます
print(df_tmp.head(10))
```

# 67
> 商品データフレーム（df_product）の各商品について、利益率が30%となる新たな単価を求めよ。今回は、1円未満を切り上げること。そして結果を10件表示させ、利益率がおよそ30％付近であることを確認せよ。ただし、単価（unit_price）と原価（unit_cost）にはNULLが存在することに注意せよ。
- `np.ceil()`
```
df_tmp = df_product.copy()
# 利益率が30%となる価格を"new_price"カラムに代入します
df_tmp['new_price'] = np.ceil(df_tmp['unit_cost'] / 0.7)
# "new_price"で利益率を算出し、"new_profit_rate"カラムに代入します
df_tmp['new_profit_rate'] = (df_tmp['new_price'] - df_tmp['unit_cost'])/df_tmp['new_price']
# print関数を使用し、df_tmpを10件表示させます
print(df_tmp.head(10))
```

# 68
> 商品データフレーム（df_product）の各商品について、消費税率10%の税込み金額を求めよ。 1円未満の端数は切り捨てとし、結果は10件表示すれば良い。ただし、単価（unit_price）にはNULLが存在することに注意せよ。
```
df_tmp = df_product.copy()
# 税込価格を"price_tax"カラムに代入します
df_tmp['price_tax'] = np.floor(df_tmp['unit_price'] * 1.10)
# print関数を使用し、df_tmpを10件表示させます
print(df_tmp.head(10))
```
# 69
> レシート明細データフレーム（df_receipt）と商品データフレーム（df_product）を結合し、顧客毎に全商品の売上金額合計と、カテゴリ大区分（category_major_cd）が"07"（瓶詰缶詰）の売上金額合計を計算の上、両者の比率を求めよ。抽出対象はカテゴリ大区分"07"（瓶詰缶詰）の購入実績がある顧客のみとし、結果は10件表示させればよい。

```
# 顧客毎に全商品の売上金額合計のデータフレーム1を作成する
df_tmp_1 = pd.merge(df_receipt, df_product, how="inner", on="product_cd").groupby("customer_id").amount.agg("sum").reset_index()
# 顧客毎に瓶詰缶詰の売上金額合計のデータフレーム2を作成する
df_tmp_2 = pd.merge(df_receipt, df_product.query('category_major_cd == "07"'), how="inner", on="product_cd").groupby("customer_id").amount.agg("sum").reset_index()
# データフレーム1、データフレーム２よりデータフレーム３を作成する
df_tmp_3 = pd.merge(df_tmp_1, df_tmp_2, how="inner", on="customer_id")
# ラベルの付け直し
df_tmp_3.columns = ['customer_id', 'amount_all', 'amount_07']
# 瓶詰缶詰の売上金額合計の比率を"rate_07"カラムに代入する
df_tmp_3['rate_07'] = df_tmp_3["amount_07"] / df_tmp_3["amount_all"]
# print関数を使用し、df_tmp_3を10件表示させます
print(df_tmp_3.head(10))
```

## 92
> 顧客データフレーム（df_customer）では、性別に関する情報が非正規化の状態で保持されている。これを第三正規化せよ。
- `df.drop()`
- `df.drop_duplicates()`
```
# "gender_cd"と"gender"の依存関係をdf_genderに代入します
df_gender = df_customer[['gender_cd', 'gender']].drop_duplicates()
# "gender"カラムを削除します
df_customer_s = df_customer.drop(columns='gender')
print(df_gender)
print(df_customer_s)
```

## 93
> 商品データフレーム（df_product）では各カテゴリのコード値だけを保有し、カテゴリ名は保有していない。カテゴリデータフレーム（df_category）と組み合わせて非正規化し、カテゴリ名を保有した新たな商品データフレームを作成せよ。
```
df_product_full = pd.merge(df_product, df_category[["category_small_cd", "category_major_name","category_medium_name","category_small_name"]], how="inner", on="category_small_cd")
print(df_product_full)
```

## 94
> 93問目で作成したカテゴリ名付き商品データ（df_product_full ）を以下の仕様でファイル出力せよ。
なお、出力先のパスは100knocks-preprocess配下とし、ファイル名はP_df_product_full_UTF-8_header.csvとせよ。
> - ファイル形式はCSV（カンマ区切り）
> - ヘッダ有り
> - 文字コードはUTF-8
> - インデックスはなし
- `df.to_csv(path, header=, index=, encoding=)`
```
# 93 の続きから
# ファイルへ出力
df_product_full.to_csv('./100knocks-preprocess/P_df_product_full_UTF-8_header.csv', header=True, index=False, encoding='utf-8')
```
## 95
> 93問目で作成したカテゴリ名付き商品データ（df_product_full ）を以下の仕様でファイル出力せよ。
なお、出力先のパスは100knocks-preprocess配下とし、ファイル名はP_df_product_full_CP932_header.csvとせよ。
> - ファイル形式はCSV（カンマ区切り）
> - ヘッダ有り
> - 文字コードはCP932
```
# 93 の続きから
# ファイルへ出力
df_product_full.to_csv('./100knocks-preprocess/P_df_product_full_CP932_header.csv',header=True, encoding="CP932")
```

## 96
> 93問目で作成したカテゴリ名付き商品データ（df_product_full ）を以下の仕様でファイル出力せよ。
なお、出力先のパスは100knocks-preprocess配下とし、ファイル名はP_df_product_full_UTF-8_noh.csvとせよ。
> - ファイル形式はCSV（カンマ区切り）
> - ヘッダ無し
> - 文字コードはUTF-8
> - インデックスはなし
```
# 93 の続きから
# ファイルへ出力
df_product_full.to_csv('./100knocks-preprocess/P_df_product_full_UTF-8_noh.csv',header=False, index=False, encoding="utf-8")
```

## 97 
> 94問目で作成した以下形式のファイルを読み込み、データフレームを作成せよ。また、先頭10件を表示させ、正しくとりまれていることを確認せよ。
> - ファイル形式はCSV（カンマ区切り）
> - ヘッダ有り
> - 文字コードはUTF-8
- `pd.read_csv()`
```
df_tmp = pd.read_csv('./100knocks-preprocess/P_df_product_full_UTF-8_header.csv', encoding="utf-8")
print(df_tmp.head(10))
```

## 98
> 96問目で作成した以下形式のファイルを読み込み、データフレームを作成せよ。また、先頭10件を表示させ、正しくとりまれていることを確認せよ。
> ファイル形式はCSV（カンマ区切り）
> ヘッダ無し
> 文字コードはUTF-8
```
df_tmp = pd.read_csv("./100knocks-preprocess/P_df_product_full_UTF-8_noh.csv", header=None, encoding="utf-8")
print(df_tmp.head(10))
```

## 99
> （df_product_full ）を以下の仕様でファイル出力せよ。
なお、出力先のパスは100knocks-preprocess配下とし、ファイル名はP_df_product_full_UTF-8_header.tsvとせよ。
> - ファイル形式はTSV（タブ区切り）
> - ヘッダ有り
> - 文字コードはUTF-8
> - インデックスはなし
```
df_product_full.to_csv("./100knocks-preprocess/P_df_product_full_UTF-8_header.tsv",header=True, index=False, encoding="utf-8", sep='\t')
```

## 100
> 99問目で作成した以下形式のファイルを読み込み、データフレームを作成せよ。また、先頭10件を表示させ、正しくとりまれていることを確認せよ。
> - ファイル形式はTSV（タブ区切り）
> - ヘッダ有り
> - 文字コードはUTF-8
- `pd.read_table()`
```
df_tmp = pd.read_table("./100knocks-preprocess/P_df_product_full_UTF-8_header.tsv", encoding="utf-8")
print(df_tmp.head(10))
```

# 環境構築（備忘)
## 従来環境
- 以前は色々試行錯誤したがうまくいかず、結局 Google Colab に逃げた
- 今回はとりあえず jupyter notebook を手元で使えるようにしておいて、ダメになった時点で colab に移行するか
    - ここら辺を参考にする https://qiita.com/kurilab/items/f6f4374d7b1980060de7

## 100本ノック公式
- https://github.com/The-Japan-DataScientist-Society/100knocks-preprocess
- Docker 起動だが、これだとオンライン環境と同じ話で、手元での再現性がない

## (M1) Mac で jupyter notebook の環境構築
- 参考 https://degitalization.hatenablog.jp/entry/2020/12/07/154628
    - とりあえずこれで行けそうなのでこれでいく。
