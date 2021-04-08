# サマリ

# メモ
## list分割(re.split)
- 標準のsplit関数は、１つの文字でしかsplitできない
- 複数の文字でsplitするには、reモジュールの `re.split` を使う
```
import re

time_data = "2017/4/1_22:15"
time_list = re.split("[/_:]", time_data)
```

## 高次関数
- 引数に関数を取る関数のこと
### map
```
pick_h = lambda time: re.split("[/_:]",time)[3]
h_list = list(map(pick_h, time_list))
```
### filter
```
filter_month_1_6 = lambda time : int(re.split("[/_:]", time)[1]) < 7
month_1_6_list = list(filter(filter_month_1_6,time_list))
```
### sorted
- `list.sort()` よりも複雑な条件でソートしたいときには `sorted(list, key=func, reverse=bool)` を使う
- `key=` には、要素を取り出すための関数を渡す
```
time_data = sorted(time_data, key=lambda x: x[3])
```

## リスト内包表記
### for を使った生成
- `map` を使用するほかに `[func(x), for x in list]` を使用する方法がある
```
converter = lambda x: [int(x/60), x % 60]
h_m_data = [converter(minute) for minute in minute_data]
```
### if を使った生成
- `[func(X), for x in list if 条件]` を使うこともできる
```
h_m_data = [minute for minute in minute_data if minute % 60 == 0]
```

## 複数配列の同時ループ
- `zip(list1, list2)` を使って `for x,y in zip (list1, list2):` が使える

## 多重ループ
- 以下の形で多重ループができる
```
for x in a:
    for y in b:
        ...
```
- リスト内包でも可能: `[func(x,y,z) for x in a for y in b for z in c]`

## defaultdictを使った辞書
- 
```
from collections import defaultdict

sample_dict = defaultdict("keyの型")
for value in list:
    char_freq[key] = value
```
## defaultdictのサンプル
```
# まとめたいデータ price...(名前, 値段)
price = [
    ("apple", 50),
    ("banana", 120),
    ("grape", 500),
    ("apple", 70),
    ("lemon", 150),
    ("grape", 1000)
]
# defaultdict を定義して下さい
price_dict = defaultdict(list)

# 上記の例と同様にvalue の要素に値段を追加して下さい
for key, value in price:
    price_dict[key].append(value)

# 各value の平均値を計算し、配列にして出力して下さい
print([sum(x) / len(x) for x in price_dict.values()])
```

## Counter を使った要素の数え上げ
- `counter = Counter(list)` で数え上げ
- `counter.most_common(n))` でソートして、n個出力

## Pandas での CSV 読み込み
- ヘッダを独自につけたいときには `pd.read_csv(path, header=None)`で読み込み、`df.columns=["key1","key2",...]` でつける

## Pandas での CSV 書き出し
```
df = pd.DataFrame(data)
df.to_csv("path")
```

## NaNを含むデータの扱い
### 削除
- リストワイズ削除：Nanを含む行を全て削除
- ペアワイズ削除：欠損の少ない列を残して、Nanを含んでいても、一部のNanではないデータを使用する

### 補完
- 固定値: `fillna(0)`
- 一つ前のデータ: `fillna(method="ffill")`
- 平均: `fillna(df.mean())`

## データのビン分割
- `pd.cut(df, 分割数)` で分割が可能
- `pd.cut(df, [境界値1,境界値2,...]` でも可能

## OpenCv での画像処理
- 読込: `cv2.imread("ファイル名")`
- 表示: `cv2.imshow("ウィンドウ名", 読み込んだ画像データ)`
- 保存: `cv2.imwrite("ファイル名", 作成した画像データ)`
- リサイズ: `cv2.resize(画像データ, (リサイズ後の幅, リサイズ後の高さ))`
- 回転: `cv2.warpAffine(元の画像データ, 変換行列, 出力画像のサイズ)`
- 反転: `cv2.flip(画像データ, flipCode)`
- 色空間変更: `my_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)`
- 色反転：`img = cv2.bitwise_not(画像データ)`
- 閾値処理; `cv2.threshold(画像データ, 閾値, 最大値 maxValue, 閾値処理の種類)`
- マスク: `cv2.bitwise_and(元画像データ1, 元画像データ2(元画像データ1と同じで可), mask=マスク用画像)`
- ぼかし: `cv2.GaussianBlur(画像データ, カーネルサイズ(n, n), 0)` ※ n は奇数
- 膨張: `cv2.dilate(画像データ, フィルタ)`
- 収縮: `cv2.erode(画像データ, フィルタ)`

