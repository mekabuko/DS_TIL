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