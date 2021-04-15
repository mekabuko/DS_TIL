# サマリ

# メモ
## 単語集（引用）
- 自然言語(NL, Natural Language): 日本語や英語のような 自然発生的に生まれた言語
- 自然言語処理(NLP, Natural Language Processing): 人間が日常的に使っている 自然言語をコンピュータに処理させる技術。文書分類・機械翻訳・文書要約・質疑応答・対話などの活用がある。
- トークン：自然言語を解析する際、文章の最小単位として扱われる文字や文字列のこと。
- タイプ：単語の種類を表す用語。
- 文章：まとまった内容を表す文のこと。自然言語処理では一文を指すことが多い。
- 文書：複数の文章から成るデータ一件分を指すことが多い。
- コーパス：文書または音声データにある種の情報を与えたデータ。
- シソーラス：単語の上位/下位関係、部分/全体関係、同義関係、類義関係などによって単語を分類し、体系づけた類語辞典・辞書。
- 形態素：意味を持つ最小の単位。「食べた」という単語は、2つの形態素「食べ」と「た」に分解できる。
- 単語：単一または複数の形態素から構成される小さな単位。
- 表層：原文の記述のこと。
- 原形：活用する前の記述のこと。
- 特徴：文章や文書から抽出された情報のこと。
- 辞書：自然言語処理では、単語のリストを指す。
- 単語分割：文章を単語に分割すること
- 品詞タグ付け：単語を品詞に分類して、タグ付けをする処理のこと
- 形態素解析：形態素への分割と品詞タグ付けの作業をまとめたもの
- 表記揺れ：同じ文書の中で、同音・同義で使われるべき語句が異なって表記されていること
- 正規化：表記揺れを防ぐためルールベースで文字や数字を変換すること

## 自然言語処理における「日本語」
- 単語の区切りは難しい
- 単語ごとの解釈は少ないので、品詞や基本形を求めることは容易

## MeCab
- 形態素解析のツール
- `MeCab.Tagger("-Owakati").parse(str)`: わかちがき
- `MeCab.Tagger("-Ochasen").parse(str)`: 形態素解析

## janome
- 形態素解析のツール
- `Tokenizer(wakati=True).tokenize(str)`: わかちがき
- `Tokenizer().tokenize(str)`: 形態素解析
    - `Token.surface`： 表層形
    - `Token.part_of_speech`: 品詞部分
```
from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()
tokens = tokenizer.tokenize("明日は晴れるだろうか。")
for token in tokens:
    print(token)
```

## Ngram
- N文字ごとに単語を切り分ける 、または N単語ごとに文章を切り分ける 解析手法
- 単語のNgramを求めたい場合: 引数に単語と切り出したい数を入れる
- 文章のNgramを求めたい場合: janomeのtokenize関数を用いて分かち書きのリストを作成し、その分かち書きのリストと切り出したい数を引数に入れる
```
from janome.tokenizer import Tokenizer
t = Tokenizer()
tokens = t.tokenize("太郎はこの本を二郎を見た女性に渡した。", wakati=True)

def gen_Ngram(words,N):
    # Ngramを生成してください
    ngram = []
    for i in range(len(words)-N+1):
        cw = "".join(words[i:i+N])
        ngram.append(cw)
    return ngram

print(gen_Ngram(tokens, 2))
print(gen_Ngram(tokens, 3))
```

## 正規化
- 全角を半角に統一や大文字を小文字に統一等、ルールベースで文字を変換すること
- `neologdn.normalize("正規化したい文字列")`
- 単純なものなら`str.lower()`, `str.upper()`なども可能

### 数字の置き換え
- 数字は多様だが出現頻度が高い＝自然言語処理のノイズになりがち
- そのため、記号などに置き換えてしまいことが多い
- `re.sub('\d', "!", text)`

## 文書のベクトル表現