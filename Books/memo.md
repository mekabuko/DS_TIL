#　メモ
## 1章 自然言語処理の基礎
### 自然言語処理のタスク
1. 形態素解析
- テキストを形態素に分割して品詞を付与する処理
- ツール
    - MeCab: 高性能かつ高速
    - janome: MeCabより遅いが使いやすい
2. 構文解析
- 文の構造（係り受けなど）を解析する
- ツール
    - CaboCha
    - KNP
3. 意味解析
- テキストの意味を解析
- 同時に単語の意味の曖昧性解消を含む
    - bank が「銀行」なのか「土手」なのか、のようなもの
4. 文脈解析
- 複数の文のつながりに関する解析を行う
### 自然言語処理の応用
- 機械翻訳システム
- 質問応答システム
- 情報抽出
- 対話システム
## ２章 機械学習
- 特にメモなし
## 3章 コーパス
### コーパスとは
- 自然言語のデータ
    - ラベル付き、ラベルなしに大分できる
- フリーの公開コーパスも多い
    - テキスト分類用: 20 Newsgroups, IMDb
    - 固有表現認識用: CoNLL-2003
    - 質問応答用: SQuAD, MS MACRO
### コーパス読み込みの実装例
ファイルを読み書きするなら`with`を使うと、open/close が不要
```
with open('example.txt', 'r', encoding='utf-8') as f:
    f.read()
    f.readline()
    for line in f:
        print(line)
```
CSVを読み込むには pandas が便利
```
import pandas as pd
df = pd.read_csv('example.csv', encoding='utf-8')
```
JSONにも
```
pd.read_json('example.json', encoding='utf-8')
```
### コーパスの作成（クローリング/スクレイピング）
- ぐるなびのAPIなど、情報を取得できるAPIを使用することもできるし、クローリングなどの手法もある
    - ぐるなびAPI: 2021/06で終了していた。。
- ライブラリには、 requests や Scrapy などがある
### コーパスの作成（アノテーションツール）
アノテーションツール: データにラベルをつけるためのツール
- OSSの例
    - brat: 系列ラベリングと関係認識用ラベリングが可能
    - doccano: テキスト分類、系列ラベリング、系列変換が可能
- サービスの例(有料のものもあり)
    - prodigy
    - tagtog
    - LightTag
## 4章　テキストの前処理
[コード(Colab)](https://colab.research.google.com/drive/1c0NBJQBKQ_YkVqQWzNUD69J5VEPyV9BR?usp=sharing)
[実践サンプル](https://colab.research.google.com/drive/18OM5FoF6Ep4QxSML7pqaf3cd3KPt0wUr?usp=sharing)
### クリーニング
HTMLタグやjavascriptコードなどのノイズを除去する処理。
BeautifulSoup や Python標準の re パッケージが活用可能。
- 正規表現を試すことのできるオンラインエディタ: [Regex101](https://regex101.com/)
### 単語分割
テキストの分割や、品詞の抽出を行う
- 分割: janome.tokenizer
- フィルタ: janome.analyzer & janome.tokenfilter
場合によっては、辞書の修正も有効な方法になる
### 単語の正規化
表記揺れなどを統一する
- 文字種の統一
    - 大文字、小文字に統一など
- 数字の置き換え
    - 数字は邪魔になるので、全て置き換えてしまった方が良い
- 辞書を用いた単語の統一
    - 辞書を用いた表記揺れの修正は、地道な手作業となることが多い
    - 例: ソニー、Sony　→ Sony に統一、など
### スワップワードの除去
一般的で役に立たないものなど、処理対象外となる単語（ストップワード）を除去する
- 辞書方式:
    - [SlothLibの日本語データ](http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt)
- 出現頻度方式
    - カウントして、上位 n 個の文字をストップワード認定する、という方法
### 単語のID化
単語にIDを割り振って、無駄な情報を落とす
### パディング
データの系列長を合わせる操作: 機械学習フレームワークが、一定長を求める場合が多いから。
# Links
- [amazon](https://www.amazon.co.jp/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%BB%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92%E3%81%AB%E3%82%88%E3%82%8B%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E5%85%A5%E9%96%80-Compass-Books%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E4%B8%AD%E5%B1%B1-%E5%85%89%E6%A8%B9-ebook/dp/B084WPRT44)
- [公式サポート](https://book.mynavi.jp/supportsite/detail/9784839966607.html)
- [Google Colabのファイルリンク](https://gist.github.com/Hironsan/1f1cc629613cbd7de042a7ce269b91d6)
