# サマリ
- 時系列データの分析では、ARIMAモデルや、季節変動があるならSARIMAモデルなどが適用できる
- どれも、原則として定常過程（ある軸を中心に一定の幅を振れるもの）に変換して考える
- 周期はデータの可視化で、各種のパラメータは統計量などを用いて設定する

# メモ
## 時系列データのパターン
- トレンド：長期的な傾向
- 周期変動：周期的な変化
- 不規則変動：時間の経過と関係のない変動

## 時系列データの変換
- 原系列：元の生データ
- 対数変換：値の変動の大きさを穏やかにする
    - `np.log(df)`
- 階差系列：一つ前の時間との差を取った系列
    - `df.diff()`
- 季節調整済み系列：季節変動を取り除いた系列
    - `statsmodels.api.seasonal_decompose(df, freq=)`で、トレンド、季節変動、不規則変動に分けることができる
    - 可視化：`statsmodels.api.tsa.seasonal_decompose(df, freq=).plot()`

## 統計量
- 平均
    - `np.mean(df)`
- 分散・標準偏差
- 自己共分散
    - 同じ時系列データ内の時点の離れたデータでの共分散
    - 例：N 年の身長と N+1 年の身長
    - k時点離れた自己共分散を k次の共分散という
- 自己相関係数
    - 異なる時系列データ間の時点の離れたデータ同士での共分散
    - 例：N 年の身長と N+1 年の体重
    - `statsmodels.api.tsa.stattools.acf(df, nlags=)`
    - 可視化：`statsmodels.api.graphics.tsa.plot_acf(df, nlags=)`

## 定常性
- 時間の経過によらず一定の値を軸に、同程度の幅で触れて変化する
    - 時間に依らず、データの確率分布が一定＝確率過程が定常過程である
### 弱定常性
- 期待値と自己共分散が時間を通して一定
- 基本的に「定常性」と言ったらこっちの定義

## ホワイトノイズ
- 期待値0、自己相関0 = 不規則な変動。
- 時系列もでるの不規則変動パターン（誤差）を担う

## MA(移動平均)モデル
- ある範囲 q期間の間に自己相関を持つ形のモデル
- ホワイトノイズの拡張で、１期前の誤差項を変数としてもつ
- `y(t) = 期待値 + 誤差(t) + θ * 誤差(t-1)`

## AR(自己回帰)モデル
- 規則的に値が変化していくモデル
- １期前の自身を変数として持つ
- `y(t) = c + φ * y(t-1) + 誤差`

## ARMAモデル
- ARモデルとMRモデルを組み合わせたモデル
- 定常過程にしか適用できない
- `ARMA(p,q)` として表現する
    `AR(p)` と `MA(q)` の組み合わせ

## ARIMAモデル
- ARMAモデルについて、原系列ではなく、階差系列に変換して適用したもの
- 非定常過程にも適用可能
- d時点前との差分を取って、`ARIMA(p,q,d)` として表現する
    - p: 自己相関度
    - q: 誘導
    - d: 移動平均

## SARIMAモデル
- ARIMAモデルをさらに季節周期を持つ時系列データにも拡張できるようにしたモデル
- (p, d, q)のパラメーターに加えて(sp, sd, sq, s)というパラメーターも持つ
    - sp: 季節性自己相関
    - sd: 季節性導出
    - sq: 季節性移動平均
    - s: 周期
- 周期については、自己相関係数や偏自己相関の可視化によって推定する
    - 例
    ```
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from pandas import datetime


    # データの読み込みと整理
    sales_sparkling = pd.read_csv("./5060_tsa_data/monthly-australian-wine-sales-th-sparkling.csv")
    index = pd.date_range("1980-01-31", "1995-07-31", freq="M")
    sales_sparkling.index=index
    del sales_sparkling["Month"]
    # 自己相関・偏自己相関の可視化
    fig=plt.figure(figsize=(12, 8))
    # 自己相関係数のグラフを出力します
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(sales_sparkling, lags=30, ax=ax1)
    # 偏自己相関係数のグラフを出力します
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(sales_sparkling, lags=30, ax=ax2)
    plt.show()
    ```
    
- 自動でパラメータ調整をする方法はないので、情報量規準によってどの値が最も適切か調べる必要がある
    - BIC（ベイズ情報量基準）では、この値が小さいほどパラメータのあてはまりが良いと言える
    - BICを算出するプログラム例（引用）
    ```
    def selectparameter(DATA,s):
    p = d = q = range(0, 1) # 今回は、各パラメータを 0 or 1
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    parameters = []
    BICs = np.array([])
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(DATA,
                                            order=param,
                seasonal_order=param_seasonal)
                results = mod.fit()
                parameters.append([param, param_seasonal, results.bic])
                BICs = np.append(y,results.bic)
            except:
                continue
    return print(parameters[np.argmin(BICs)])
    ```
- モデル作成と予測・可視化の例(引用)
```
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import datetime
import numpy as np

 
# データの読み込みと整理
sales_sparkling = pd.read_csv("./5060_tsa_data/monthly-australian-wine-sales-th-sparkling.csv")
index = pd.date_range("1980-01-31", "1995-07-31", freq="M")
sales_sparkling.index=index
del sales_sparkling["Month"]

# モデルの当てはめ
SARIMA_sparkring_sales = sm.tsa.statespace.SARIMAX(sales_sparkling,order=(0, 0, 0),seasonal_order=(0, 1, 1, 12)).fit()

# predに予測データを代入する
pred = SARIMA_sparkring_sales.predict("1994-7-31", "1997-12-31")

# predデータともとの時系列データの可視化
plt.plot(pred, color="r")
plt.plot(sales_sparkling)
plt.show()
```


## トレンド・季節変動の除去方法
- 対数変換：変動の分散を一様にする
    ```
    np.log(df)
    ```
- 移動平均によって、トレンドを推定し、トレンド成分を除去
    ```
    # 移動平均を求める
    co2_moving_avg = co2_tsdata2.rolling(window=51).mean()
    # トレンド成分を差し引く
    mov_diff_co2_tsdata = co2_tsdata2-co2_moving_avg 
    ```
- 階差系列にしてトレンド・季節変動を除去
    ```
    df.diff()
    ```
- 季節調整を利用
    ```
    res = statsmodels.api.tsa.seasonal_decompose(co2_tsdata2,freq=51)
    ```

