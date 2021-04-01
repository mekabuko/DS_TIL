# サマリ
- 困った時は公式リファレンス：https://numpy.org/doc/1.13/search.html
    - でも人気だから日本語で検索しても十分情報出てくる
- 軸交換が本当に意味がわからなかった。。xy軸の交換（転置）はわかっても、３次元になった瞬間？？？に。
    - 図形的に考えるのは３次元が限界なので、そういう考え方自体が無理がある。

# メモ
## loadtext
- テキストファイルからnp配列を作成する
- [公式](https://numpy.org/doc/1.13/reference/generated/numpy.loadtxt.html)
```
np.loadtxt("path_to_file",
            delimiter=",", 
            skiprows=1,
            usecols=[1, 2])
```

## np配列のブロードキャスト
- 自動で行列サイズが計算可能な shape に変換される ＝ ブロードキャスト
```
arr_1 = np.array([x,x,x,x,...])

arr_for_mult_culc = np.array([1.8, 1,1,1])
arr_for_add_culc = np.array([32,0,0,0])

arr_2 = arr_1 * arr_for_mult_culc + arr_for_add_culc
```

## np配列の条件付き抽出
```
array1[array1 % 2 == 1]
```

## np配列の操作
- 転置: `arr_1.T`
- 軸変更: `arr_1.transpose(0,2,1)`
    - https://deepage.net/features/numpy-transpose.html
    - 2次元で言えば、x,y 軸を入れ替える <=> 0軸と1軸を入れ替えている、という感じ。
- 形状変更(要素数は不変): `arr_1.reshape(1, -1)` # -1 とすると、他の次元の要素数から推論してくれる
- 形状変更(要素数は可変): `arr_resize(30,1)` # 不足分の要素は 0 埋めされる

 ## np配列のデータ型
 - `np.array([x,x,x,...], dtype=np.int16)` などで指定可能
    - int: 符号付き整数
    - uint: 符号なし整数
    - float: 浮動小数
    - bool: 真偽値

## axis
- 軸
- 計算などでは、どの方向に計算を行うのか、`axis=n`で軸方向を決める

## 統計量計算
どれも、引数として axis を取れる。どの方向で、計算を行うかを指定可能。指定しないと、全ての要素に対して行われる。
- `sum()`
- `mean()`
- `var()`
- `max()`
- `min()`
- `argmax()` : 最大となるインデックスの取得
- `argmin()` : 最小となるインデックスの取得
- `argsort()` : 配列ソート時の、元々のインデックス

## 乱数生成
- `np.random.rand()`：一葉分布
- `np.random.randn()` : 正規分布
- 引数がないなら単一、引数があればその shape で乱数生成