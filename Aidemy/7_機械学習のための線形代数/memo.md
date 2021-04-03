# サマリ
- ほとんど忘れている、、特に、手計算が全然できない。
    - というか、理論もほとんど抜け落ちている。。
- 計算機は WolframAlpha を使えば、、大体、なんとかなる。大学の時のレポート作成みたい。
    - https://ja.wolframalpha.com/examples/mathematics/algebra/matrices/

# メモ
## 行列
- 逆行列: AX = E となるとき、 X を A の逆行列といい、A^(-1) と表す
    - 全ての行列に逆行列があるわけではない
    - 逆表列を持つ正方行列(n x n)を正則行列という
    - 逆行列を持つかどうかは 行列式 != 0 で調べられる
- 行列式
    - A の行列式は detAと表す
    - ２次、３次までは、サラスの方法(左からプラス、右からマイナス)で求める
    - 4次以降は、余因子展開を用いる(ref: https://risalc.info/src/determinant-four-by-four.html)
- 掃き出し法
    - 逆行列を求めるために、連立方程式を解いていく方法。
- 階数(rank): 線形独立なベクトルの個数
- 直行行列: AAT = E となるような A
    - 正規直交基底：各ベクトルが直行し、かつ、大きさが１となるもの
    - シュミットの正規直交化法：順番に全てのベクトルを処理して、ベクトルを正規化する方法
- 固有方程式
    - det[A - λE] = 0
    - これで固有値を求められる
- 固有ベクトルと固有空間の求め方
    - 固有ベクトル：Ax = x となる x 

## ベクトル
- 線形従属
    - あるベクトルが、他の２ベクトルの成す平面上に存在する状態（それらで、残りの１つが説明可能）
- 線型独立
    - ベクトルの個数＝ベクトル空間の階数、なら全て線型独立と言える
- 部分空間
    - あるベクトル空間に含まれるベクトルで、作られるベクトル空間。
    - 必要十分条件は以下の通り
        1. 0 ∈ W
        2. a,b ∈ W → a + b ∈ W
        3. a ∈ W → ca ∈ W
- 基底
    - そのベクトルの組み合わせで、その空間のあらゆるベクトルを表現できるもの。

# 問題集
## 2.1.2
1. https://ja.wolframalpha.com/input/?i=%7B%7B1%2C2%2C3%7D%2C%7B4%2C7%2C6%7D%2C%7B5%2C9%2C8%7D%7D
2. https://ja.wolframalpha.com/input/?i=%7B%7B3%2C1%2C4%2C1%7D%2C%7B5%2C9%2C2%2C6%7D%2C%7B5%2C3%2C5%2C8%7D%2C%7B9%2C7%2C9%2C3%7D%7D
3. https://ja.wolframalpha.com/input/?i=%7B%7B2%2C7%2C8%7D%2C%7B1%2C2%2C8%7D%2C%7B1%2C8%2C2%7D%7D

## 2.1.3
- https://ja.wolframalpha.com/input/?i=%7B%7B0%2C-2%2C1%7D%2C%7B-1%2C1%2C1%7D%2C%7B2%2C5%2C-5%7D%7D

## 2.1.4
1. https://ja.wolframalpha.com/input/?i=%7B%7B2%2C4%7D%2C%7B-1%2C-3%7D%7D
2. https://ja.wolframalpha.com/input/?i=%7B%7B1%2C0%2C0%7D%2C%7B1%2C1%2C-1%7D%2C%7B-2%2C0%2C3%7D%7D

## 2.1.5
1. https://ja.wolframalpha.com/input/?i=%7B%7B1%2C0%2C0%7D%2C%7B0%2C1%2C1%7D%7D+%E7%89%B9%E7%95%B0%E5%80%A4%E5%88%86%E8%A7%A3
2. https://ja.wolframalpha.com/input/?i=%7B%7B5%2C1%2C-2%7D%2C%7B1%2C6%2C-1%7D%2C%7B-2%2C-1%2C5%7D%7D+%E7%89%B9%E7%95%B0%E5%80%A4%E5%88%86%E8%A7%A3

