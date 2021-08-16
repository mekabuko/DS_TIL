# サマリ

# メモ
## Define-and-run と Define-by-Run
### Define-and-run
- 静的グラフアプローチ（static-graph approach）とも
- ニューラルネットワークを作った時点で計算の仕方が固定され、訓練や推論の過程では変化しない
- 変化が少ないため計算を効率化させる工夫がしやすい
- フレームワークが最適化に対応することで計算時間が短く済む
#### フレームワーク例
- Theano
- Caffe
- TensorFlow(どちらも可能)
### Define-by-Run
- 動的グラフアプローチ（dynamic-graph approach）とも
- 計算する時に具体的なモデルを構築(学習や推論するたびに計算の仕方を決定)
- 入力するデータの内容によってニューラルネットワークの一部を変更するといったモデル設計がしやすくなる（固定も可能）
- モデル設計のデバッグがしやすい
#### フレームワーク例
- Chainer
- PyTorch

## 中間表現フォーマット
### IR（intermediate representation、中間表現）
- 何らかの変換をする途中の状態を指す表現
- 深層学習向けのIRを区別して「Graph IR」などと呼ぶことも
- 変換途中の状態を各フレームワークが共通して扱えるような表現のフォーマットに統一することで、フレームワーク間でモデルの移植をしやすくする
### Open Neural Network Exchange（ONNX）フォーマット
-  米MicrosoftとFacebookによって開発された、Graph IRの代表例