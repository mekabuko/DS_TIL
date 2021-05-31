# サマリ

# メモ
## WaveNetとは
- 文章からの音声生成モデルの一つ
- サンプル：https://www.deepmind.com/blog/article/wavenet-generative-model-raw-audio

## WaveNetの特性
- WaveNetより前に提案されてきたTTSと比べ、人間の発声に近い音が生成できる
- 音声の長期依存構造を取り扱うために、「dilated causal convolution」をベースにした新しいネットワーク構造を使っている
- 話者情報を条件付き確率モデルに加えて、一つのモデルで複数の話者の音声を合成できる
- TTSと同じ構造のネットワークで、音声の認識や音楽の合成を行うことができる

## TTS: text to speech
- テキストデータの文章から音声を生成する技術