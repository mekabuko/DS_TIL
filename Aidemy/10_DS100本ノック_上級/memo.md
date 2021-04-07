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

## 73 経過時間の計算
> レシート明細データフレーム（df_receipt）の売上日（sales_ymd）に対し、顧客データフレーム（df_customer）の会員申込日（application_date）からのエポック秒による経過時間を計算し、顧客ID（customer_id）、売上日、会員申込日とともに表示せよ。結果は10件表示させれば良い（なお、sales_ymdは数値、application_dateは文字列でデータを保持している点に注意）。なお、時間情報は保有していないため各日付は0時0分0秒を表すものとする。
- タイムスタンプへの変換は`datetime.timestamp()`
- Seriesを変換するときには`Series.astype(np.int64)/10**9`が使える）（ナノ秒で得られるので、10**9で割って秒に戻している
```
# "df_receipt"と"df_customer"の必要なカラムのみで内部結合させます
df_tmp = pd.merge(df_receipt[["customer_id", "sales_ymd"]], df_customer[["customer_id", "application_date"]], how="inner", on="customer_id")

# 重複した行は削除します
df_tmp = df_tmp.drop_duplicates()

# それぞれ日付型に変換します
df_tmp['sales_ymd'] = pd.to_datetime(df_tmp['sales_ymd'], format="%Y%m%d")
df_tmp['application_date'] = pd.to_datetime(df_tmp['application_date'], format="%Y%m%d")

# エポック秒での経過時間を求めます
df_tmp['elapsed_date'] = df_tmp['sales_ymd'].astype(np.int64)/ 10**9 - df_tmp['application_date'].astype(np.int64)/ 10**9
print(df_tmp.head(10))
```

## 74 経過時間の計算
> レシート明細データフレーム（df_receipt）の売上日（sales_ymd）に対し、当該週の月曜日からの経過日数を計算し、売上日、当該週の月曜日付とともに表示せよ。結果は10件表示させれば良い（なお、sales_ymdは数値でデータを保持している点に注意）。
- `relativedelta(days=series)`
- `datetime.weekday()` で曜日を[0-6]で取得できる
```
# "df_receipt"の必要なカラムのみ抽出する
df_tmp = df_receipt[["customer_id","sales_ymd"]]
# 重複した行は削除します
df_tmp = df_tmp.drop_duplicates()
# "sales_ymd"カラムのデータを日付型に変換します
df_tmp['sales_ymd'] = pd.to_datetime(df_tmp['sales_ymd'], format="%Y%m%d")

# 当該週の月曜日を"monday"カラムに代入する
df_tmp['monday'] = df_tmp['sales_ymd'].map(lambda x: x - relativedelta(days=x.weekday()))
# 当該週の月曜日からの経過日数を"elapsed_weekday"カラムに代入する
df_tmp['elapsed_weekday'] = df_tmp['sales_ymd'] - df_tmp['monday']

print(df_tmp.head(10))
```

## 77 外れ値除外
> レシート明細データフレーム（df_receipt）の売上金額（amount）を顧客単位に合計し、合計した売上金額の外れ値を抽出せよ。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。なお、ここでは外れ値を平均から3σ以上離れたものとする。結果は10件表示させれば良い。
- 標準化した結果は、そのままσの値。3σ離れている外れ値は、標準化の結果から抽出可能。
```
# "df_receipt"の売上金額（amount）を顧客ID（customer_id）ごとに合計したデータを代入する
df_sales_amount = df_receipt.query('not customer_id.str.startswith("Z")', engine='python'). \
    groupby("customer_id").agg({"amount":"sum"}).reset_index()
# 標準化を行い、"amount_ss'"カラムに代入する
scaler = preprocessing.StandardScaler()
scaler.fit(df_sales_amount[["amount"]])
df_sales_amount['amount_ss'] = scaler.transform(df_sales_amount[["amount"]])
# 平均から3σ以上離れたものを10件抽出する
print(df_sales_amount.query('abs(amount_ss) >= 3').head(10))
```

## 78 外れ値除外
> レシート明細データフレーム（df_receipt）の売上金額（amount）を顧客単位に合計し、合計した売上金額の外れ値を抽出せよ。ただし、顧客IDが"Z"から始まるのものは非会員を表すため、除外して計算すること。なお、ここでは外れ値を第一四分位と第三四分位の差であるIQRを用いて、「第一四分位数-1.5×IQR」よりも下回るもの、または「第三四分位数+1.5×IQR」を超えるものとする。結果は10件表示させれば良い。
```
# "df_receipt"の売上金額（amount）を顧客ID（customer_id）ごとに合計したデータを代入する
df_sales_amount = df_receipt.query('not customer_id.str.startswith("Z")', engine='python'). \
    groupby("customer_id").agg({"amount":"sum"}).reset_index()
# 第一四分位数と第三四分位数を求める
pct75 = np.percentile(df_sales_amount["amount"], 75, axis=0)
pct25 = np.percentile(df_sales_amount["amount"], 25, axis=0)
# IQRを求め、外れ値を10件抽出する
IQR = pct75 - pct25
print(df_sales_amount.query('not @pct25 -1.5 * @IQR< amount < @pct75 + 1.5 * @IQR').head(10))
```

## 79 欠損列状況確認
> 商品データフレーム（df_product）の各項目に対し、欠損数を確認せよ。
- `df.isnull()` で欠損値は True として表示できる。
```
print(df_product.isnull().sum())
```

## 80 欠損レコード削除
> 商品データフレーム（df_product）のいずれかの項目に欠損が発生しているレコードを全て削除した新たなdf_product_1を作成せよ。なお、削除前後の件数を表示せよ。
- `df.dropna()` で欠損値を削除できる
```
df_product_1 = df_product.copy()
print('削除前:', len(df_product_1))
# 欠損df_product_1
df_product_1 = df_product_1.dropna()
print('削除後:', len(df_product_1))
```

## 81 数値補完(平均値)
> 単価（unit_price）と原価（unit_cost）の欠損値について、それぞれの平均値で補完した新たなdf_product_2を作成せよ。なお、平均値について1円未満は四捨五入とし、0.5については偶数寄せでかまわない。補完実施後、各項目について欠損が生じていないことも確認すること。
- `df.fillna({"key":"value})` で欠損値を埋める
- `np.nanmean(series)` で 欠損値ありの平均を求める
```
# 平均値で欠損値を補完します
df_product_2 = df_product.fillna({"unit_price": np.nanmean(df_product['unit_price']),\
    "unit_cost": np.nanmean(df_product['unit_cost'])})
print(df_product_2.isnull().sum())
```

## 82 数値補完(中央値)
> 単価（unit_price）と原価（unit_cost）の欠損値について、それぞれの中央値で補完した新たなdf_product_3を作成せよ。なお、中央値について1円未満は四捨五入とし、0.5については偶数寄せでかまわない。補完実施後、各項目について欠損が生じていないことも確認すること。
- `np.nunmedian(series)`
```
# 中央値で欠損値を補完します
df_product_3 = df_product.fillna({"unit_price" : np.nanmedian(df_product["unit_price"]),\
    "unit_cost" : np.nanmedian(df_product["unit_cost"])})
print(df_product_3.isnull().sum())
```

## 83 数値補完(カテゴリごとの中央値)
> 単価（unit_price）と原価（unit_cost）の欠損値について、各商品の小区分（category_small_cd）ごとに算出した中央値で補完した新たなdf_product_4を作成せよ。なお、中央値について1円未満は四捨五入とし、0.5については偶数寄せでかまわない。補完実施後、各項目について欠損が生じていないことも確認すること。
```
# 商品の小区分（category_small_cd）ごとに中央値を算出
df_tmp = df_product.groupby("category_small_cd")\
    .agg({"unit_price": "median", "unit_cost": "median"}).reset_index()
df_tmp.columns = ['category_small_cd', 'median_price', 'median_cost']

# "df_product"と"df_tmp"を内部結合
df_product_4 = pd.merge(df_product, df_tmp, how='inner', on='category_small_cd')

# applyを使用して、欠損値補完
df_product_4['unit_price'] = df_product_4[['unit_price', 'median_price']].apply(lambda x: x[1] if np.isnan(x[0]) else x[0], axis=1)
df_product_4['unit_cost'] = df_product_4[['unit_cost', 'median_cost']].apply(lambda x: x[1] if np.isnan(x[0]) else x[0], axis=1)

print(df_product_4.isnull().sum())
```

## 85 ジオコード
> 顧客データフレーム（df_customer）の全顧客に対し、郵便番号（postal_cd）を用いて経度緯度変換用データフレーム（df_geocode）を紐付け、新たなdf_customer_1を作成せよ。ただし、複数紐づく場合は経度（longitude）、緯度（latitude）それぞれ平均を算出すること。
```
#  "df_customer"と"df_geocode"を内部結合することで紐付けを行います
df_customer_1 = pd.merge(df_customer, df_geocode, how="inner", on="postal_cd")
# 複数紐づいたものに対しては平均を算出し、代入します
df_customer_1 = df_customer_1.groupby("customer_id"). \
    agg({"longitude": "mean", "latitude": "mean"}).reset_index()
# renameメソッドでは、カラム名の変更を行います
df_customer_1.rename(columns={'longitude':'m_longitude', 'latitude':'m_latitude'}, inplace=True)
# df_customer"と"df_customer_1 "を内部結合します
df_customer_1 = pd.merge(df_customer, df_customer_1, how='inner', on='customer_id')
print(df_customer_1.head(3))
```

## 86 ジオコード
> 前設問で作成した緯度経度つき顧客データフレーム（df_customer_1）に対し、申込み店舗コード（application_store_cd）をキーに店舗データフレーム（df_store）と結合せよ。そして申込み店舗の緯度（latitude）・経度情報（longitude)と顧客の緯度・経度を用いて距離（km）を求め、顧客ID（customer_id）、顧客住所（address）、店舗住所（address）とともに表示せよ。また、距離は下記の式に基づいて計算せよ。
> - (略)
```
#　緯度軽度から距離を算出する関数
def calc_distance (x1, y1, x2, y2):
    distance = 6371 * math.acos(math.sin(math.radians(y1)) * math.sin(math.radians(y2))\
    + math.cos(math.radians(y1)) * math.cos(math.radians(y2)) * math.cos(math.radians(x1) - math.radians(x2)))
    return distance

# 顧客データフレーム（df_customer_1）に対し、申込み店舗コード（application_store_cd）をキーに店舗データフレーム（df_store）と結合
df_tmp = pd.merge(df_customer_1.rename(columns={"application_store_cd":"store_cd"}), df_store, how="inner", on="store_cd" ) 

# applyメソッドを使用し、距離を求める
df_tmp['distance'] =   df_tmp[['m_longitude', 'm_latitude','longitude', 'latitude']].apply(lambda x: calc_distance(x[0], x[1], x[2], x[3]), axis=1)

print(df_tmp[['customer_id', 'address_x', 'address_y', 'distance']].head(10))
```

## 91 アンダーサンプリング
> 顧客データフレーム（df_customer）の各顧客に対し、売上実績のある顧客数と売上実績のない顧客数が1:1となるようにアンダーサンプリングで抽出せよ。なお、random_stateは71で設定せよ。
- データに偏りがあるときに、偏りをなくしてピックすることをアンダーサンプリングと呼ぶ。
- `imblearn.under_sampling.RandomUnderSampler` を使用することで、可能。
```
# 顧客ごとの売上金額合計を算出します
df_tmp = df_receipt.groupby("customer_id").agg({"amount":"sum"}).reset_index()
# 顧客情報に売上金額合計を追加します
df_tmp = pd.merge(df_customer, df_tmp, how="left", on="customer_id")
# 売上金額合計がない(NaN)のものは0とし、あるものは1とし、"buy_flg"カラムに結果を代入します
df_tmp['buy_flg'] = df_tmp['amount'].apply(lambda x : 0 if np.isnan(x) else 1)

print('0の件数', len(df_tmp.query('buy_flg == 0')))
print('1の件数', len(df_tmp.query('buy_flg == 1')))

# アンダーサンプリングを行います
rs = RandomUnderSampler(random_state=71)
df_sample, _ = rs.fit_sample(df_tmp, df_tmp['buy_flg'])

print('0の件数', len(df_sample.query('buy_flg == 0')))
print('1の件数', len(df_sample.query('buy_flg == 1')))
```