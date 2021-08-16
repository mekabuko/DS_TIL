# サマリ
- これはチートシートとライブラリをみながら、都度必要なものを使うしかなさそう。。ちょっと日常的に使わない限りはコマンド覚えられなそう。

# メモ
## 乱数におけるシード値の設定について
- シード値を一定にすれば、乱数の値も一定になる
```
np.random.seed(0) 
np,random.randn(5) # 一定の値！
```
- 指定しない時には、シードとしてPC時刻が使われるので、結果的に毎回異なる値になる。

## 乱数の種別
### 正規分布
- `np.random.randn()`
### 二項分布
- `np.random.binomial(n, p)`
    - n: 試行回数、 p: 確率
### ランダムピック
- `np.random.choice(list, pick_num)`

## 日時を扱うライブラリ datetime
```
import datetime as dt

base_date = dt.datetime(1992,10,22) # 日時を扱う datetime
elapsed_time = dt.timedelta(days=1) # 経過時間扱う timedelta

new_date = base_date + elapsed_time # こんな形で演算可能

s = "2021-4-2"
dt.datetime.strptime(s, "%Y-%m-%d") # fromStringはこんな形
```

##　数列の生成
- 等間隔: `np.arange(始まり, 終わり+1, 間隔)`
- 等間隔分割: `np.linspace(始まり, 終わり, 分割数)`


