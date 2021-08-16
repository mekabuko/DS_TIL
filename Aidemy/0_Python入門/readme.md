# 方針
- 基本的に全部残しはしない
- 忘れてたり、引っかかったところのみメモする

# サマリ
- 結構忘れている。。
- やはり TS や Java とは違うものの、書きやすい印象ではある
- クラスとかの書き方はだいぶ違うので気をつけたい。そんなオブジェクト指向でガリガリ書いたりしないと思うけど。

# メモ

## 型取得
`type(x)` で取得できる

## and, or, not
`and, or, not` が使える(`&, |, !` ではない)

## リストのスライスについて
- 始点に指定されたインデックスについてはスライスに**含まれる**
- 終点で指定したインデックスについてはスライスに**含まれない**

```
list = [0,1,2,3,4,5,6,7,8,9]
print(list[1,7]) # [1,2,3,4,5,6] と出力される
```
## リストの操作
- 追加: `list_1.append(x)`
- 削除: `del list_1[n]`
- コピー: `list2 = list_1.copy() / list_1[:] / list(list_1)`

## int to string
int は str とつなげないので注意
```
n = 1
print("n = " + n) # Error
print("n = " str(n)) # n = 1
```

## for loop 内で index を取得したい時
`enumerate(list_1)` を使う

```
animals = ["tiger", "dog", "elephant"]

for index, animal in enumerate(animals):
    print("index:" + str(index) + " " + animal)
```

## 文字列埋め込み
- いくつか方法があるが f-string `print(f"I am {x}")` が楽そう。
- あとは format関数 `"I am {}".format(x)` あたりか。

## メソッドの種類
### 通常メソッド
- 第一引数が `self` になるので注意
### クラスメソッド
- クラス全体の操作を行う
- 第一引数は `cls` として、その `cls` に対して操作する
### スタティックメソッド
- 引数不要。ただし`self`にもさわれない
    - インスタンスがなくても使えるということを明示的に示していると言えるのか

## format による書式設定
- １０進法表示：`"xxx {:d}".format(x)`
- 小数点 5 桁まで： `"sss {:.f5}".format(x)`
