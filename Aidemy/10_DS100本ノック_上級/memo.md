# サマリ

# 回答
## 45 日付型からの変換
> 顧客データフレーム（df_customer）の生年月日（birth_day）は日付型（Date）でデータを保有している。これをYYYYMMDD形式の文字列に変換し、顧客ID（customer_id）とともに抽出せよ。データは10件を抽出すれば良い。
- `to_datetime(Series)` : datatime変換
- `Series.dt` : 要素をdatatime として操作
- `strftime('')` : 表記変換
```
series_datetime = pd.to_datetime(df_customer["birth_day"]).dt.strftime('%Y%m%d')
print(pd.concat([df_customer["customer_id"], series_datetime], axis=1).head(10))
```

## 46 日付型からの変換
> 顧客データフレーム（df_customer）の申し込み日（application_date）はYYYYMMD形式の文字列型でデータを保有している。これを日付型（dateやdatetime）に変換し、顧客ID（customer_id）とともに抽出せよ。データは10件を抽出すれば良い。
```
print(pd.concat([df_customer["customer_id"], pd.to_datetime(df_customer["application_date"], format='%Y%m%d', errors="ignore")], axis=1).head(10))
```

## 47 日付型からの変換
> レシート明細データフレーム（df_receipt）の売上日（sales_ymd）はYYYYMMDD形式の数値型でデータを保有している。これを日付型（dateやdatetime）に変換し、レシート番号(receipt_no)、レシートサブ番号（receipt_sub_no）とともに抽出せよ。データは10件を抽出すれば良い。
```
series_sales_ymd = pd.to_datetime(df_receipt["sales_ymd"], format="%Y%m%d", errors="ignore")
print(pd.concat([df_receipt[["receipt_no","receipt_sub_no"]], series_sales_ymd], axis=1).head(10))
```

## 48 日付型からの変換
> レシート明細データフレーム（df_receipt）の売上エポック秒（sales_epoch）は数値型のUNIX秒でデータを保有している。これを日付型（dateやdatetime）に変換し、レシート番号(receipt_no)、レシートサブ番号（receipt_sub_no）とともに抽出せよ。データは10件を抽出すれば良い。
```
series_sales_date = pd.to_datetime(df_receipt["sales_epoch"], unit='s')
print(pd.concat([df_receipt[["receipt_no","receipt_sub_no"]],series_sales_date], axis=1).head(10))
```

## 51 日付要素の取り出し
> レシート明細データフレーム（df_receipt）の売上エポック秒（sales_epoch）を日付型（timestamp型）に変換し、"日"だけ取り出してレシート番号(receipt_no)、レシートサブ番号（receipt_sub_no）とともに抽出せよ。なお、"日"は0埋め2桁で取り出すこと。データは10件を抽出すれば良い。
```
df_tmp = pd.concat([df_receipt[['receipt_no', 'receipt_sub_no']], pd.to_datetime(df_receipt["sales_epoch"], unit='s').dt.strftime('%d')],axis=1).head(10)
print(df_tmp)
```

## 52 二値化
> レシート明細データフレーム（df_receipt）の売上金額（amount）を顧客ID（customer_id）ごとに合計の上、売上金額合計に対して2000円以下を0、2000円超を1に2値化し、index番号、顧客ID、売上金額合計とともに10件表示せよ。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。
```
# 顧客IDが"Z"から始まるのもの以外を抽出する
df_sales_amount = df_receipt.query('not customer_id.str.startswith("Z")', engine='python')
# "customer_id"でグループ化し、各合計を算出する（index番号を振り直しも行ってください）
df_sales_amount = df_sales_amount.groupby("customer_id").sum().reset_index()
# applyを使用して、2値化を行い、"sales_flg"カラムへ代入してください
def amount_to_binary(x):
    if x <= 2000:
        return 0
    return 1
df_sales_amount['sales_flg'] = df_sales_amount.amount.apply(amount_to_binary)
# 解答例：df_sales_amount['sales_flg'] = df_sales_amount['amount'].apply(lambda x: 1 if x > 2000 else 0)

sprint(df_sales_amount[["customer_id", "amount", "sales_flg"]].head(10))
```

## 53 二値化
> 顧客データフレーム（df_customer）の郵便番号（postal_cd）に対し、東京（先頭3桁が100〜209のもの）を1、それ以外のものを0に2値化せよ。さらにレシート明細データフレーム（df_receipt）と結合し、全期間において買い物実績のある顧客数を、作成した2値ごとにカウントせよ。
- `df.groupby("key").agg({"key":"nunique"})`: ユニークな数をカウント
```
# コピーを作成する
df_tmp = df_customer[['customer_id', 'postal_cd']].copy()
# applyを使用し、2値化したものを"postal_flg"へ代入する
df_tmp['postal_flg'] = df_tmp.postal_cd.apply(lambda code: 1 if 100 <= int(code[0:3]) <= 209 else 0)
# 2つのDataFrameを内部結合し、作成した2値ごとに"customer_id"のユニークな値の数を取得する
print(pd.merge(df_tmp, df_receipt, on='customer_id', how="inner").groupby("postal_flg").agg({"customer_id":"nunique"}))
```

## 54 カテゴリ化
> 顧客データデータフレーム（df_customer）の住所（address）は、埼玉県、千葉県、東京都、神奈川県のいずれかとなっている。都道府県毎にコード値を作成し、顧客ID、住所とともに抽出せよ。値は埼玉県を11、千葉県を12、東京都を13、神奈川県を14とすること。結果は10件表示させれば良い。
```
def address_to_code(address):
    if "埼玉県" in address:
        return "11"
    if "千葉県" in address:
        return "12"
    if "東京都" in address:
        return "13"
    if "神奈川県" in address:
        return "14"
    return "99"
    
df_address_code = df_customer.address.apply(address_to_code)
# 「顧客ID、住所」のデータフレームと「都道府県毎にコード値」のデータフレームを連結させます
print(pd.concat([df_customer[["customer_id", "address"]],df_address_code], axis=1).head(10))
```
- `map`を使えばもっと簡潔にかける
```
print(pd.concat([df_customer[['customer_id', 'address']], df_customer['address'].str[0:3].map({'埼玉県': '11','千葉県':'12', '東京都':'13', '神奈川':'14'})], axis=1).head(10))
```

## 55 カテゴリ化
> レシート明細データフレーム（df_receipt）の売上金額（amount）を顧客ID（customer_id）ごとに合計し、その合計金額の四分位点を求めよ。その上で、顧客ごとの売上金額合計に対して以下の基準でカテゴリ値を作成し、顧客ID、売上金額と合計ともに表示せよ。カテゴリ値は上から順に1〜4とする。結果は10件表示させれば良い。
> - 最小値以上第一四分位未満
> - 第一四分位以上第二四分位未満
> - 第二四分位以上第三四分位未満
> - 第三四分位以上
```
# 売上金額（amount）を顧客ID（customer_id）ごとの合計を算出してください
df_sales_amount = df_receipt[['customer_id', 'amount']].groupby('customer_id').sum().reset_index()
# pctには、それぞれの四分位を代入してください
pct25 = np.percentile(df_sales_amount['amount'], 25, axis=0)
pct50 = np.percentile(df_sales_amount['amount'], 50, axis=0)
pct75 = np.percentile(df_sales_amount['amount'], 75, axis=0)

# xのデータがどの四分位点に該当するかを返す関数
def pct_group(x):
    if x < pct25:
        return "1"
    if x < pct50:
        return "2"
    if x < pct75:
        return "3"
    return "4"
    
df_sales_amount['pct_group'] = df_sales_amount['amount'].apply(lambda x: pct_group(x))
print(df_sales_amount.head(10))
```

## 56 カテゴリ化
> 顧客データフレーム（df_customer）の年齢（age）をもとに10歳刻みで年代を算出し、顧客ID（customer_id）、生年月日（birth_day）とともに抽出せよ。ただし、60歳以上は全て60歳代とすること。年代を表すカテゴリ名は任意とする。先頭10件を表示させればよい。
```
tmp_df = df_customer.age.apply(lambda x: 60 if x >= 60 else int(x/10) * 10).reset_index()
# 「顧客ID、生年月日」のデータフレームと「年代」のデータフレームを連結させます
df_customer_era = pd.concat([df_customer[["customer_id","birth_day"]],tmp_df["age"]],axis=1)

print(df_customer_era.head(10))
```

## 57 カテゴリ化
>前問題の抽出結果と性別（gender）を組み合わせ、新たに性別×年代の組み合わせを表すカテゴリデータを作成せよ。なお、組み合わせを表すカテゴリの値はgender_cd+ageとする。
> 
> 例
> - gender_cd ... 1
> - age ... 30
> 
> なら
era_genderは130となる
なお、表示は10件とせよ。
- `series.astype('str')` 型変換するならこれ。
```
# era_genderカラムに性別×年代の組み合わせを表すカテゴリデータを代入してください
df_customer_era['era_gender'] = df_customer['gender_cd'].astype('str') + df_customer_era['age'].astype('str')
print(df_customer_era.head(10))
```

## 58 ダミー変数化
> 顧客データフレーム（df_customer）の性別コード（gender_cd）をダミー変数化し、顧客ID（customer_id）とともに抽出せよ。結果は10件表示させれば良い。
- `pd.get_dummies(df, columns=["key"])` でダミー変数(0/1変数)が作成できる。
```
print(pd.get_dummies(df_customer[["customer_id", "gender_cd"]], columns=['gender_cd']).head(10))
```

## 59 正規化(z-score)
> レシート明細データフレーム（df_receipt）の売上金額（amount）を顧客ID（customer_id）ごとに合計し、合計した売上金額を平均0、標準偏差1に標準化して顧客ID、売上金額合計とともに表示せよ。標準化に使用する標準偏差は、不偏標準偏差と標本標準偏差のどちらでも良いものとする。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。結果は10件表示させれば良い。
- `sklearn.preprocessing.StandardScaler()` で、データの標準化が可能
```
# "df_receipt"の売上金額（amount）を顧客ID（customer_id）ごとに合計したデータを代入する
df_sales_amount = df_receipt.query('not customer_id.str.startswith("Z")', engine='python'). \
    groupby("customer_id").agg({"amount":"sum"}).reset_index()
# StandardScalerクラスのインスタンスを生成する
scaler = preprocessing.StandardScaler()
# 変換モデルを学習する
scaler.fit(df_sales_amount[["amount"]])
# 標準化したデータを"amount_ss"カラムに代入する
df_sales_amount['amount_ss'] = scaler.transform(df_sales_amount[["amount"]])
print(df_sales_amount.head(10))
```

## 正規化(Min-Max normalization)
> レシート明細データフレーム（df_receipt）の売上金額（amount）を顧客ID（customer_id）ごとに合計し、合計した売上金額を最小値0、最大値1に正規化して顧客ID、売上金額合計とともに表示せよ。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。結果は10件表示させれば良い。
- 最小0, 最大1に変換するMin-Max正規化を行うには`sklearn.preprocessing.MinMaxScaler()`
```
# "df_receipt"の売上金額（amount）を顧客ID（customer_id）ごとに合計したデータを代入する
df_sales_amount = df_receipt.query('not customer_id.str.startswith("Z")', engine='python'). \
    groupby("customer_id").agg({"amount":"sum"}).reset_index()
# MinMaxScalerクラスのインスタンスを生成する
scaler = preprocessing.MinMaxScaler()
# 変換モデルを学習する
scaler.fit(df_sales_amount[["amount"]])
# Min-Max正規化したデータを"amount_mm"カラムに代入する
df_sales_amount['amount_mm'] = scaler.transform(df_sales_amount[["amount"]])
print(df_sales_amount.head(10))
```

## 61 対数化
> レシート明細データフレーム（df_receipt）の売上金額（amount）を顧客ID（customer_id）ごとに合計し、合計した売上金額を常用対数化（底=10）して顧客ID、売上金額合計とともに表示せよ。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。結果は10件表示させれば良い。
- これだと不正解。
```
# "df_receipt"の売上金額（amount）を顧客ID（customer_id）ごとに合計したデータを代入する
df_sales_amount = df_receipt.query('not customer_id.str.startswith("Z")', engine='python'). \
    groupby("customer_id").agg({"amount":"sum"}).reset_index()
# 売上金額（amount）を常用対数に変換し、"amount_log10"カラムに代入する
df_sales_amount['amount_log10'] = np.log10(df_sales_amount["amount"])
print(df_sales_amount.head(10))
```
- 模範解答、なんで +1 している？？ 0 だと error になるから？（問い合わせ中）
```
df_sales_amount['amount_log10'] = np.log10(df_sales_amount['amount'] + 1)
```

## 62 対数化
> レシート明細データフレーム（df_receipt）の売上金額（amount）を顧客ID（customer_id）ごとに合計し、合計した売上金額を自然対数化(底=e）して顧客ID、売上金額合計とともに表示せよ。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。結果は10件表示させれば良い。
```
# "df_receipt"の売上金額（amount）を顧客ID（customer_id）ごとに合計したデータを代入する
df_sales_amount = df_receipt.query('not customer_id.str.startswith("Z")', engine='python'). \
    groupby("customer_id").agg({"amount":"sum"}).reset_index()
# 売上金額（amount）を常用対数に変換し、"amount_log10"カラムに代入する
df_sales_amount['amount_loge'] = np.log(df_sales_amount["amount"] + 1)
print(df_sales_amount.head(10))
```

## 70 経過日数の計算
> レシート明細データフレーム（df_receipt）の売上日（sales_ymd）に対し、顧客データフレーム（df_customer）の会員申込日（application_date）からの経過日数を計算し、顧客ID（customer_id）、売上日、会員申込日とともに表示せよ。結果は10件表示させれば良い（なお、sales_ymdは数値、application_dateは文字列でデータを保持している点に注意）。
```
# "df_receipt"と"df_customer"の必要なカラムのみで内部結合させます
df_tmp = pd.merge(df_receipt[["customer_id","sales_ymd"]],df_customer[["customer_id", "application_date"]], how="inner", on="customer_id")

# 重複した行は削除します
df_tmp = df_tmp.drop_duplicates(subset=["customer_id", "sales_ymd"])

# 日付型に変換後、経過日数を計算し、"elapsed_date"カラムに代入します
df_tmp['sales_ymd'] = pd.to_datetime(df_tmp['sales_ymd'], format="%Y%m%d")
df_tmp['application_date'] = pd.to_datetime(df_tmp['application_date'], format="%Y%m%d")
df_tmp['elapsed_date'] = df_tmp['sales_ymd'] - df_tmp['application_date']
print(df_tmp.head(10))
```

## 71 経過日数の計算
> レシート明細データフレーム（df_receipt）の売上日（sales_ymd）に対し、顧客データフレーム（df_customer）の会員申込日（application_date）からの経過月数を計算し、顧客ID（customer_id）、売上日、会員申込日とともに表示せよ。結果は10件表示させれば良い（なお、sales_ymdは数値、application_dateは文字列でデータを保持している点に注意）。1ヶ月未満は切り捨てること。
- `dateutil.relativedelta(a,b)` で経過日数が取れる。年別、月別、日別で取得可能。
```
# "df_receipt"と"df_customer"の必要なカラムのみで内部結合させます
df_tmp = pd.merge(df_receipt[["customer_id", "sales_ymd"]], df_customer[["customer_id", "application_date"]], how="inner", on="customer_id")

# 重複した行は削除します
df_tmp = df_tmp.drop_duplicates(subset=["customer_id", "sales_ymd"])

# それぞれ日付型に変換します
df_tmp['sales_ymd'] = pd.to_datetime(df_tmp['sales_ymd'], format="%Y%m%d")
df_tmp['application_date'] = pd.to_datetime(df_tmp['application_date'], format="%Y%m%d")

# applyで対象のカラムを取り出し、relativedeltaを使用して経過月数を求めます
df_tmp['elapsed_date'] = df_tmp[["sales_ymd","application_date"]].apply(lambda x: 
                                    relativedelta(x[0], x[1]).years * 12 + relativedelta(x[0], x[1]).months, axis=1)

print(df_tmp.sort_values('customer_id').head(10))
```


## 72 経過日数の計算
> レシート明細データフレーム（df_receipt）の売上日（sales_ymd）に対し、顧客データフレーム（df_customer）の会員申込日（application_date）からの経過年数を計算し、顧客ID（customer_id）、売上日、会員申込日とともに表示せよ。結果は10件表示させれば良い。（なお、sales_ymdは数値、application_dateは文字列でデータを保持している点に注意）。1年未満は切り捨てること。
```
# "df_receipt"と"df_customer"の必要なカラムのみで内部結合させます
df_tmp = pd.merge(df_receipt[["customer_id", "sales_ymd"]], df_customer[["customer_id", "application_date"]], how="inner", on="customer_id")

# 重複した行は削除します
df_tmp = df_tmp.drop_duplicates(subset=["customer_id", "sales_ymd"])

# それぞれ日付型に変換します
df_tmp['sales_ymd'] = pd.to_datetime(df_tmp['sales_ymd'], format="%Y%m%d")
df_tmp['application_date'] = pd.to_datetime(df_tmp['application_date'], format="%Y%m%d")

# applyで対象のカラムを取り出し、relativedeltaを使用して経過月数を求めます
df_tmp['elapsed_date'] = df_tmp[["sales_ymd","application_date"]].apply(lambda x: 
                                    relativedelta(x[0], x[1]).years, axis=1)

print(df_tmp.head(10))
```