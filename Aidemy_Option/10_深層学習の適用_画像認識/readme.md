# サマリ
- 基本的にモデルと理論の紹介のみ
- 画像認識では、ResMetの改良版、DenseNetが有力な選択肢
- さらに、エッジコンピューティングでは MobileNets が選択肢となる

# メモ
## DenseNet
- DenseNet: ResNetを改良したもの
   - ResNet: Microsoft Research Asia(MSRA)によって発表をされたモデル。国際的な画像認識精度のコンペティションであるILSVRCで2015年に優勝した。
   - パラメーター数当たりの認識精度を3倍改善
   - skip-connectionと呼ばれるデータの“迂回経路”を多く導入。この経路が損失関数の勾配を効率的に隠れ層へ伝えている
### ResNetとの違い
- 迂回経路付き畳み込み層の作り方が異なる
- 迂回したデータの接続数
    - ResNetは１本、DenseNetは５本
- 接続する位置
- 接続の方法
    - ResNet: 畳み込みレイヤーの入力と出力の和を計算
    - DenseNet: データを結合

## MobileNets
- 通常、AIの処理はサーバー上
- しかし、エッジコンピューティングの台頭によって、軽量な深層学習モデルが注目を集めるように
- MobileNetsは画像認識に特化した軽量なモデルの一つ。エッジコンピューティングが可能な軽量さ
- CNNの計算を、depthwise separable convolutionと1\times1×1 Convolutionという2つの畳み込みに分解して高速化を実現
### MobileNetsV2
- V1の欠点を改良
    - 活性化関数にReLUを使っていたものを、Liner/ReLU6に変更することで精度を向上
    - ストライドが1のCNNブロックにresidual connectionを追加して表現力を高めた
    - 入力側に 1 x 1 Convolution  を追加
### MobileNetV3
- V2をさらに改善
    - Squeeze-and-Excite というブロックを追加、活性化関数を ReLU6 から h-swish 関数に変更
    - NN の構造を自動探索する Neural Architecture Search でも止めたねえとワークに微修正を加えて作られた