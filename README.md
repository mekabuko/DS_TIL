# Today I Learned, about Data Science
学習記録。


## Links
### 公式ドキュメント等
- Python公式リファレンス: https://docs.python.org/ja/3/reference/index.html
- NumPy公式リファレンス(v1.13)：https://numpy.org/doc/1.13/search.html
- Pandas公式リファレンス: https://pandas.pydata.org/pandas-docs/stable/reference/index.html
- matplotlib公式リファレンス: https://matplotlib.org/2.0.2/api/index.html
### チートシート(ちょっと古いかも)
- [pyhton](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf)
- [Numpy](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
- [Pandas](http://datacamp-community-prod.s3.amazonaws.com/dbed353d-2757-4617-8206-8767ab379ab3)
    - [和訳](https://github.com/Gedevan-Aleksizde/pandas-cheat-sheet-ja/blob/master/doc/Pandas_Cheat_Sheet_ja.pdf)
- [Matplotlib](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf)
- [scikit-learn](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)
    - [scikit-learn argorithm](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
### Utility
- [WolframAlpha(計算機)](https://ja.wolframalpha.com/examples/mathematics/algebra/matrices/)

## Kaggle コンペ
1. タイタニック: 2021/04/17 ~ 19
    - 機械学習を用いて、タイタニック号の悲劇からどのような人々が生き残る可能性が高いのかを予測
    - aidemy のコースの一環
    - https://www.kaggle.com/c/titanic
2. 住宅価格予測: 2021/04/20
    - 機械学習を用いて、どのような住宅がどのような価格になるのかを予測
    - aidemy のコースの一環
    - https://www.kaggle.com/c/house-prices-advanced-regression-techniques
3. レンタサイクル需要予測: 2021/04/29 ~ 2021/05/24
    - レンタサイクルの需要予測を、過去の利用データと気象データを用いたモデルで行う
    - 研修の一環
    - https://www.kaggle.com/c/bike-sharing-demand

## Aidemy Courses(https://aidemy.net)
### 受講必須コース(+α): 2021/04/01 ~ 2021/04/20, 所用時間 35時間程度。
0. Python入門 : 2021/04/01
    - **Python** の基礎
    - 文字の出力、変数の概要、条件分岐、ループ など、基本のき
1. NumPy基礎（数値計算）: 2021/04/01
    - ライブラリ **NumPy** の基礎
    - 効率的な科学技術計算を可能にする、DS分野では必須のライブラリ
2. Pandas基礎 : 2021/04/02
    - ライブラリ **Pandas** の基礎
    - 表や時系列データの計算を楽にする、定量データ解析には必須のライブラリ
3. Matplotlib基礎 : 2021/04/02
    - ライブラリ **Matplotlib** の基礎
    - データの可視化を行うライブラリ
4. 機械学習概論: 2021/04/02
    - 機械学習の基本や精度評価の方法など
    - G検定で学んだことで十分だと思ったので、テストのみさらっと。動画はスキップした。
5. 統計学基礎: 2021/04/02
    - 流石に統計が主専攻だったし、、スキップでいいかなと
    - とはいえ多分、忘れていることも多そう。。また必要なら戻ってくる。
6. 統計学標準: 2021/04/02
    - これもスキップ。
7. 機械学習のための線形代数: 2021/04/03
    - かなり忘れている。。とりあえずこなしただけ。
    - 特に手計算が辛すぎる。途中からWolframAlpha頼りに。
8. データサイエンス100本ノック（構造化データ加工編）（初級）: 2021/04/03 ~ 04
    - 「列や行に対する操作」「結合」「縦横変換」「四則演算」
9. データサイエンス100本ノック（構造化データ加工編）（中級）: 2021/04/04 ~ 05
    - 「あいまい条件」「ソート」「集計」「サンプリング」
10. データサイエンス100本ノック（構造化データ加工編）（上級）: 2021/04/06 ~ 07
    - 「データ変換」「数値変換」「日付型の計算」「外れ値・異常値」
11. スクレイピング入門: 2021/04/07
    - BeautifulSoup等を用いたスクレイピングの手法
    - 使用経験はあるので、ざっくりと。
12. ビジネスパーソンのためのAI入門: 2021/04/07
    - 機械学習概論 の範囲で十分だった。バズワード卒業、くらいのレベル。
13. データクレンジング: 2021/04/08
    - CSVデータの扱い方や欠損値の処理、OpenCVを用いた画像加工の方法
14. 機械学習におけるデータ前処理: 2021/04/09
    - CSV・Excel・DBからのデータの取得、欠損値への対応方法、不均衡データの調整方法、データのスケール調整や、縦持ち横持ち変換など
15. 教師あり学習(回帰): 2021/04/09
    - 数値予測を行うための「回帰」モデルの扱い方
    - 一番知識がある(はずの)所なので、ツールの使い方のみさっくり。
16. 教師あり学習(分類): 2021/04/10, 12
    - 画像や文章などをカテゴリ分けする「分類」モデルの扱い方
17. 教師なし学習: 2021/04/12
    - 正解ラベルが付いていないデータセットを使って機械学習モデルを作る手法
    - クラスタリングや主成分分析
18. サポートベクターマシン入門: 2021/04/13
    - パターン認識手法として根強い人気のあるSVM
    - 基本となる線形SVMにフォーカスして、概要から数式、ソースコードまで
19. 時系列解析1_統計学的モデル: 2021/4/14
    - 季節変動や曜日変動など定期的周期を持った時系列データの解析を行うためのアルゴリズム
    - このような変動を除去しながら数値予測を行う手法
20. ディープラーニング基礎: 2021/4/15
    - 最も基礎的なアルゴリズムであるDNN（ディープニューラルネットワーク）を用いて手書き文字認識に挑戦
21. 自然言語処理基礎: 2021/4/15, 17
    - 自然言語処理の方法について
    - 文章を数値に変換する手法を学び、教師あり学習（分類）を使ってカテゴリ分類
22. タイタニック（kaggleコンペ）: 2021/04/17 ~ 19
    - 機械学習を用いて、タイタニック号の悲劇からどのような人々が生き残る可能性が高いのかを予測
23. 住宅価格予測（kaggleコンペ）: 2021/04/20
    - 機械学習を用いて、どのような住宅がどのような価格になるのかを予測
24. 習熟テスト : 2021/04/20
    - 合格。完了。

### オプションコース
0. データベース入門: 2021/04/21
    - データベースの基本的な知識
    - 概論だけ。特にSQLなどに踏み込んだ内容ではない。基本情報を持っていれば十分カバーされる範囲。
1. SQL基礎: 2021/4/22
    - データベースからの読み出し、データベースへの書き込み等の基礎的なSQL文法
2. ネガ・ポジ分析: 2021/4/23
    - 文章などに含まれる評価・感情に関する表現を抽出して、文章中の感情を解析する感情分析の一種であるネガ・ポジ分析について学習
    - RNN を使用
3. 分散処理: 2021/4/26
    - 深層学習に演算に用いられるGPU、深層学習の処理時間を短縮する手法
4. 時系列解析2: 2021/4/28
    - 深層学習のネットワークである、RNNとLSTMについて学ぶ
    - 深層学習では、時系列データの分析ができない理由、RNNなどを使うことで、どのようにして時系列分析ができるようになるのか
    - 時系列データと自然言語処理の例を学習
5. 時系列解析3: 2021/5/26
    - LSTMを用いて、売上予測を実装
    - 時系列データの前処理から、ネットワークの構築・予測を学習
6. データハンドリング: 2021/5/26
    - テキストデータの整形やテキストファイルの入出力方法
    - テキストファイルの他にも様々な形式のデータをpandasライブラリを用いてpythonで扱う手法
7. ゼロトラストセキュリティ概論: 2021/5/27
    - これまで主流とされてきた情報セキュリティとは一線を画すゼロトラスト・セキュリティという考え方
    - その台頭の経緯と目的を理解しながら、自社での活用検討につなげられる知識を学習
8. DX時代のアジャイル適用術: 2021/5/28
    - DX時代に必要とされる「アジャイル思考」について
    - ソフトウェア開発領域におけるアジャイルを知り、ビジネスと結合したアジャイルを組織に適用する術について学ぶ
9. 深層学習ライブラリ: 2021/5/29
    - 深層学習ライブラリについて
10. 深層学習の適用（画像認識）: 2021/5/30
    - CNNの応用であるDenseNet、MobileNetの基礎
    - モデルの理論的な解説のみ。特に実装例などは無し。
11. 深層学習の適用（pix2pix): 2021/5/31
    - 画像のスタイル変換処理をするソフトウェア pix2pix
    - このソフトウェアが使う深層学習モデルの理論を解説
12. 深層学習の適用（WaveNet）: 2021/5/31
    - テキスト音声合成(text to speech: TTS)のディープラーニングモデルの一つ、WaveNetについて
13. 深層学習の適用（自然言語処理）: 2021/5/31
    - word embeddingとも呼ばれる単語の埋め込み表現
    - 翻訳モデルの Transformer とその基盤手法である Attention
14. モデル圧縮概論: 2021/6/1
    - エッジAIの普及もあり、ニーズの高まるモデル圧縮（軽量化）について
    - 代表的な手法である「蒸留」「プルーニング」「量子化」を解説
15. CNN概論: 2021/6/2
    - CNNの理論や応用、代表的なネットワークについて
16. CNNを用いた画像認識: 2021/6/3
    - 主に画像認識で用いられ活用の幅が広いCNN（Convolutional Neural Network）の実装を概観
    - CNNを用いて手書き文字認識や一般物体認識に挑戦し、精度向上のテクニックや転移学習の実装に関して触れる