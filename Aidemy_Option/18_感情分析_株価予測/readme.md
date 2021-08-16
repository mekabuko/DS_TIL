# サマリ
- TwitterAPIを用いて、Tweetデータを取得し、それを感情分析する
- そのデータと株価の終値を合わせて、株価予測を行うモデルを構築
- コードコピペになってしまったので、もう一度ちゃんと内容を抑えながら実行、改善などに取り組みたい
- コードの詳細は notebook を参照のこと

# メモ
## 1. 感情分析
### 1.1 はじめに
#### 1.1.1 このコースの進め方
### 1.2 ツイートの取得
#### 1.2.1 ツイートの取得（1）
- 投資判断の２手法
    - テクニカル分析: 今回の対象
    - ファンダメンタルズ分析
- Twitterを用いた日経平均株価の予測を行う流れ
    1. TwitterAPIを用いてTwitterからあるアカウントの過去のツイートを取得
    2. 極性辞書を用いて日毎のツイートの感情分析を実行
    3. 日経平均株価の時系列データを取得
    4. 日毎のsentimentから次の日の株価の上下の予測を機械学習を用いて実行
- 感情分析には形態素解析を使用
- サンプル（引用）
```
import MeCab
mecab = MeCab.Tagger()
print(mecab.parse("今日は良い天気ですね。"))
```
#### 1.2.2 ツイートの取得（2）
- notebook参考
#### 1.2.3 ツイートの取得（3）
- notebook参考
#### 1.2.4 ツイートの取得（4）
- notebook参考
### 1.3 感情分析
#### 1.3.1 感情分析（1）
- 感情分析: 自然言語処理を用いて、テキストがポジティブな意味合い、またはネガティブな意味合いを持つかを判断する技術
    - 文中に出現する単語がポジティブ・ネガティブ・ニュートラルな意味合いを持つかどうかで判断
    - 判断の基準となるものに極性辞書があり、あらかじめ形態素にポジティブかネガティブが定義された辞書中に定義されている
- サンプル（引用）
```
import MeCab
import re
# MeCabインスタンスの作成．引数を無指定にするとIPA辞書になります．
m = MeCab.Tagger('')

# テキストを形態素解析し辞書のリストを返す関数
def get_diclist(text):
    parsed = m.parse(text)      # 形態素解析結果（改行を含む文字列として得られる）
    lines = parsed.split('\n')  # 解析結果を1行（1語）ごとに分けてリストにする
    lines = lines[0:-2]         # 後ろ2行は不要なので削除
    diclist = []
    for word in lines:
        l = re.split('\t|,',word)  # 各行はタブとカンマで区切られてるので
        d = {'Surface':l[0], 'POS1':l[1], 'POS2':l[2], 'BaseForm':l[7]}
        diclist.append(d)
    return(diclist)

print(get_diclist("明日は晴れるでしょう。"))
```

判断の基準となるものに極性辞書があり、あらかじめ形態素にポジティブかネガティブが定義された辞書中に定義
#### 1.3.2 感情分析（2）
- 今回は極性辞書として単語感情極性対応表を使用
    - 「岩波国語辞書（岩波書店）」を参考に、-1から+1の実数値を割り当て。 -1に近いほどnegative、+1に近いほどpositive
- 辞書の作成(引用)
```
#word_list, pn_listにそれぞれリスト型でWordとPNを格納
import pandas as pd
pn_df = pd.read_csv('./6050_stock_price_prediction_data/pn_ja.csv', encoding='utf-8', names=('Word','Reading','POS', 'PN'))
#word_listにリスト型でWordを格納
word_list = pn_df["Word"]
#pn_listにリスト型でPNを格納
pn_list   = pn_df["PN"]
#pn_dictとしてword_list, pn_listを格納した辞書を作成
pn_dict = dict(zip(word_list, pn_list))
# 上位10個を取り出し
print(list(pn_dict.keys())[:10])
```
#### 1.3.3 感情分析（3）
- サンプル
```
import numpy as np
import pandas as pd
import MeCab
import re

m = MeCab.Tagger('')

def add_pnvalue(diclist_old, pn_dict):
    diclist_new = []
    for word in diclist_old:
        base = word['BaseForm']        # 個々の辞書から基本形を取得
        if base in pn_dict:
            pn = float(pn_dict[base]) 
        else:
            pn = 'notfound'            # その語がPN Tableになかった場合
        word['PN'] = pn
        diclist_new.append(word)
    return(diclist_new)

# 各ツイートのPN平均値を求める
def get_mean(dictlist):
    pn_list = []
    for word in dictlist:
        pn = word['PN']
        if pn!='notfound':
            pn_list.append(pn)
    if len(pn_list)>0:
        pnmean = np.mean(pn_list)
    else:
        pnmean=0
    return pnmean

def get_diclist(text):
    parsed = m.parse(text)     
    lines = parsed.split('\n')  
    lines = lines[0:-2]         
    diclist = []
    for word in lines:
        l = re.split('\t|,',word)  # 各行はタブとカンマで区切られてるので
        d = {'Surface':l[0], 'POS1':l[1], 'POS2':l[2], 'BaseForm':l[7]}
        diclist.append(d)
    return(diclist)

dl_old = get_diclist("明日は晴れるでしょう。")
pn_df = pd.read_csv('./6050_stock_price_prediction_data/pn_ja.csv', encoding='utf-8', names=('Word','Reading','POS', 'PN'))
word_list = list(pn_df['Word'])
pn_list   = list(pn_df['PN'])
pn_dict = dict(zip(word_list, pn_list))
# get_diclist("明日は晴れるでしょう。")を関数add_pnvalueに渡して働きを調べてください。
dl_new = add_pnvalue(dl_old, pn_dict)
print(dl_new)

# またそれを関数get_meanに渡してPN値の平均を調べてください。
pnmean =get_mean(dl_new)
print(pnmean)
```
#### 1.3.4 感情分析（4）
- notebook参考
#### 1.3.5 感情分析（5）
- notebook参考

## 2.株価予測
### 2.1 日経平均株価の時系列データの取得
#### 2.1.1 時系列データの取得（1）
- notebook参考
#### 2.1.2 時系列データの取得（2）
- notebook参考
#### 2.1.3 時系列データの取得（3）
- notebook参考
### 2.2 株価予測
#### 2.2.1 株価予測（1）
- notebook参考
#### 2.2.2 株価予測（2）
- notebook参考
#### 2.2.3 株価予測（3）
- notebook参考
#### 2.2.4 株価予測（4）
- notebook参考
