# サマリ
- 手元で環境作ってやろうかと思ったけど、面倒で断念。
- 回答を一通り、メモで残す。これも手間だけど。

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

## 5, 6
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
- `df.drop_duplicated(subset=["key1","key2"])`
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
