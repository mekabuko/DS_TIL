# サマリ
- まずは形態素解析
- それから、不要な文字列を削除、全て同じ長さの配列化
- それをInputとしたRNNを使って分析
- 正直、難度は一気に上がった印象、、
    - コマンドも、まだまだなれていない。検索しながら、になっている。

# メモ
## 極性辞書
- 極性：ネガポジのこと。
- 単語毎に極性を -1 ~ 1 などで表したもの
    - オープンで利用できるものも

## 極性辞書を用いた、形態素解析からネガポジ分析までのサンプル
```
import pandas as pd
import MeCab
import pandas as pd
import re

mecab = MeCab.Tagger('')
title = open('./6020_negative_positive_data/data/aidokushono_insho.txt')
file = title.read()
title.close()

# 辞書を読み込みます
pn_df = pd.read_csv('./6020_negative_positive_data/data/pn_ja.dic',\
                    sep=':',
                    encoding='utf-8',
                    names=('Word','Reading','POS', 'PN')
                   )

# PNTableを単語と極性値のみのdict型に変更します
word_list = list(pn_df['Word'])
pn_list = list(pn_df['PN'])
pn_dict = dict(zip(word_list, pn_list))

# 解析結果のリストからPNTableに存在する単語を抽出します
def add_pnvalue(diclist_old):
    diclist_new = []
    for word in diclist_old:
        baseword = word['BaseForm']        
        if baseword in pn_dict:
            # PNTableに存在していればその単語の極性値を追加します
            pn = pn_dict[baseword]
            
        else:
            # 存在していなければnotfoundを明記します
            pn = 'notfound'
        # 極性値を持つ単語に極性値を追加します
        word['PN'] = pn
        diclist_new.append(word)
    return(diclist_new)

def get_diclist(file):
    parsed = mecab.parse(file)
    # 解析結果を改行ごとに区切ります
    lines = parsed.split('\n')
    # 後ろ2行を削除した新しいリストを作ります
    lines = lines[0:-2]
    
    #解析結果のリストを作成します
    diclist = []
    for word in lines:
        # タブとカンマで区切ったデータを作成します
        data = re.split('\t|,',word)  
        datalist = {'BaseForm':data[0]}
        diclist.append(datalist)
    return(diclist)

wordlist = get_diclist(file)
pn_list = add_pnvalue(wordlist)

print(pn_list)
```

## ネガポジ分析に使われる手法
- 文章全体の流れによって、ネガポジ分析が必要になる -> RNN が利用される

## RNN を用いたネガポジ分析に実装例（引用）
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,LSTM
from tensorflow.keras.callbacks import EarlyStopping

import nltk
import numpy as np
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from tensorflow.keras.utils import to_categorical
#stopwordsでエラーが出たら↓を実行してください
nltk.download('stopwords')

# Tweetデータを読み込みます
Tweet = pd.read_csv('./6020_negative_positive_data/data/Airline-Sentiment-2-w-AA.csv', encoding='cp932')
tweetData = Tweet.loc[:,['text','airline_sentiment']]

# 英語Tweetの形態素解析を行います
def tweet_to_words(raw_tweet):

    # a~zまで始まる単語を空白ごとに区切ったリストをつくります
    letters_only = re.sub("[^a-zA-Z@]", " ",raw_tweet) 
    words = letters_only.lower().split()

    # '@'と'flight'が含まれる文字とストップワードを削除します
    stops = set(stopwords.words("english"))  
    meaningful_words = [w for w in words if not w in stops and not re.match("^[@]", w) and not re.match("flight",w)] 
    return( " ".join(meaningful_words)) 

cleanTweet = tweetData['text'].apply(lambda x: tweet_to_words(x))

# データベースを作成します
all_text = ' '.join(cleanTweet)
words = all_text.split()

# 単語の出現回数をカウントします
counts = Counter(words)

# 降順に並べ替えます
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

# 新しいリストに文字列を数値化したものを格納します
tweet_ints = []
for each in cleanTweet:
    tweet_ints.append([vocab_to_int[word] for word in each.split()])

# Tweetのnegative/positiveの文字列を数字に変換します
labels = np.array([0 if each == 'negative' else 1 if each == 'positive' else 2 for each in tweetData['airline_sentiment'][:]]) 

# Tweetの単語数を調べます
tweet_len = Counter([len(x) for x in tweet_ints])
seq_len = max(tweet_len)

# cleanTweetで0文字になってしまった行を削除します
tweet_idx  = [idx for idx,tweet in enumerate(tweet_ints) if len(tweet) > 0]
labels = labels[tweet_idx]
tweet_ints = [tweet for tweet in tweet_ints if len(tweet) > 0]

# i行ごとの数値化した単語を右から書いたフレームワークを作ります。他の数字は0にします。
features = np.zeros((len(tweet_ints), seq_len), dtype=int)
for i, row in enumerate(tweet_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
    
#-----ここまでchapter2-----

# 元データから学習用データを作成します
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

# テスト用データを作成します
test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

#学習データを(サンプル数, シーケンス長, 次元数) の形に変換する
train_x = np.reshape(train_x, (train_x.shape[0], 1,train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1 ,test_x.shape[1]))
val_x =  np.reshape(val_x, (val_x.shape[0], 1,val_x.shape[1]))
#正解ラベルをone-hotベクトルに変換
train_y = to_categorical(train_y, 3)
test_y = to_categorical(test_y, 3)
val_y = to_categorical(val_y, 3)

negaposi = ["negative","positive","neutral"]

text = "The plane was constantly shaking due to bad weather and it was not very comfortable."

# 英語textの形態素解析を行い、データベースを作成します。
text = tweet_to_words(text)
text = text.split()

# 入力された単語を、学習データと同じように数値に変換します
# 学習したデータベースにない単語は0としています
words_int = []
for word in text:
    try:
        words_int.append(vocab_to_int[word])
    except :
        words_int.append(0)

# モデルが学習したデータの形に加工します
features = np.zeros((1, seq_len), dtype=int)
features[0][-len(words_int):] = words_int
features = np.reshape(features, (1, features.shape[0], features.shape[1]))

#モデルの構築
lstm_model = Sequential()
lstm_model.add(LSTM(units=64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
lstm_model.add(LSTM(units=32, return_sequences=False))
lstm_model.add(Dense(units=3,activation='softmax'))

lstm_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

lstm_model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=2, batch_size=32)
print()

#入力データのネガポジを予測します
predict = lstm_model.predict([features])
answer = np.argmax(predict)
print(predict)
print("pred answer: ", negaposi[answer])
```