# サマリ
- データのファイル入出力方法と、簡単な整形方法
- 特殊なデータ形式はサンプルを見る程度。実際に使いたいタイミングで、ドキュメントなどを見た方が良さそう。

# メモ
## 1.1 テキストデータの整形
### 1.1.1 オブジェクトを文字列に変換する
- `str()`
### 1.1.2 フォーマットに文字を埋め込む
- `"xxx {} xxx".format("yyy")`
### 1.1.3 複数のフォーマットに文字を埋め込む
- `"xxx {} xxx {}".format("yy1", "yy2")`
### 1.1.4 順番を指定して文字を埋め込む
- `"xxx {1} xxx {0}".format("yy1", "yy2")`
### 1.1.5 名前を指定して文字を埋め込む
- `print("xxx {0[key]} xxx".format(dic))`
- `format` は辞書型を変数に取ることもできる
### 1.1.6 文字を中央寄せ、左寄せ、右寄せにする
- `"xxx {:^20} xxx".format("yyy")`
    - 中央寄せ `{:^nn}`
    - 左寄せ `{:<nn}`
    - 右寄せ `{:>nn}`

## 1.2 文字列の分割・結合
### 1.2.1 文字列をリストに分割する
- `str.split("-")`
### 1.2.2 リストの要素を結合して一つの文字列にする
- `"-".join(list)`

## 1.3 テキストファイルの入出力
### 1.3.1 ファイルを開く
- `open("path", "r")`
    - "r", "w", "r+"
### 1.3.2 ファイルを閉じる
- `file.close()`
### 1.3.3 ファイルにテキストを書き込む
- `file.write("xxx")`
### 1.3.4 ファイルからテキストを読み込む
- `file.read()`
- 1行だけなら `file.readline()`
### 1.3.5 withを利用してより簡単にファイルを開く
- `with open("path", "w") as f:`

## 2.2 DataFrameと各データフォーマットの変換
### 2.2.1 ファイルをDataFrameで読み込む
- HTML: `read_html()`
- JSON: `read_json()`
- CSV: `read_csv()`
- Excel: `read_excel()`
### 2.2.2 DataFrameからファイルに書き出す
- HTML: `to_html()`
- JSON: `to_json()`
- CSV: `to_csv()`
- Excel: `to_excel()`

## 3. Python, Keras, Tensorflowで用いられるデータ形式
- Protocol Buffers: Python 利用される
- hdf5: Keras で利用される
    - 学習したモデルを保存する場合はhdf5形式で出力される
    - hdf5の大きな特徴は、階層的な構造を1つのファイル内で完結できること
```
import h5py
import numpy as np
import os

np.random.seed(0)

# A県のうち、X市、Y市、Z市について考えます
# X市は1丁目〜3丁目、Y市は1丁目〜5丁目、Z市は1丁目のみあるとします

# それぞれの市の人口の定義
population_of_X = np.random.randint(50, high=200, size=3)
population_of_Y = np.random.randint(50, high=200, size=5)
population_of_Z = np.random.randint(50, high=200, size=1)

# 人口をリストにまとめる
population = [population_of_X, population_of_Y, population_of_Z]

# 既にファイルが存在した場合削除する
if os.path.isfile('./4080_data_handling_data/population.hdf5'):
    os.remove('./4080_data_handling_data/population.hdf5')

# ファイルを開く
hdf_file = h5py.File('./4080_data_handling_data/population.hdf5')

# 'A'という名前のグループを作成(A県の意味)
prefecture = hdf_file.create_group('A')

for i in range(3):
    # たとえばA/X/1は、A県X市1丁目のイメージ
    # Aディレクトリの中のXディレクトリの1という名前のファイルにデータを入れるイメージ
    for j in range(len(population[i])):
        city = hdf_file.create_dataset('A/' + ['X', 'Y', 'Z'][i] + '/' + str(j + 1), data=population[i][j])

# 書き込み
hdf_file.flush()

# 閉じる
hdf_file.close()
```
```
import pandas as pd
import h5py
import numpy as np

# 開きたいファイルのパス
path = './4080_data_handling_data/population.hdf5'

# ファイルを開く
# 'r'は読み取りモードの意味
population_data = h5py.File(path, 'r')

for prefecture in population_data.keys():
    for city in population_data[prefecture].keys():
        for i in population_data[prefecture][city].keys():
            print(prefecture + '県' + city + '市' + i + '丁目: ',
                  int(population_data[prefecture][city][i].value))

# 閉じる
population_data.close()
```
- TFRecord: TensorFlowで用いられている
    - 一度この形式にデータを保存しておくことで機械学習にかかるコストが少なくなることがある
    - TFRecordはproto3で実装されている
```
import numpy as np
import tensorflow as tf
from PIL import Image

# 画像を読み込む
image = Image.open('./4080_data_handling_data/hdf5_explain.png')

# 書き出すデータの定義
# tf.train.Exampleというクラスを用いる
# tf.train.Featuresというクラスの"まとまり"
# 各tf.train.Featureの要素はbytes
# 今回はimage, label, height, widthをデータとして採用
my_Example = tf.train.Example(features=tf.train.Features(feature={
    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.ndarray([1000]).tobytes()])),
    'height': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.ndarray([image.height]).tobytes()])),
    'width': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.ndarray([image.width]).tobytes()])),
}))


# TFRecoder形式のファイルを書き出すためのTFRecordWriterオブジェクトを生成
fp = tf.io.TFRecordWriter('./4080_data_handling_data/sample.tfrecord')

# Exampleオブジェクトをシリアライズして書き込み
fp.write(my_Example.SerializePartialToString())

# 閉じる
fp.close()
```
```
import numpy as np
import tensorflow as tf
from PIL import Image

# インスタンスの生成
my_Example = tf.train.SequenceExample()

# データの文字列
greeting = ["Hello", ", ", "World", "!!"]
fruits = ["apple", "banana", "grape"]

for item in {"greeting": greeting, "fruits": fruits}.items():
    for word in item[1]:
        # my_Example内の、feature_listsの、feature_listにキーとして"word"を持つ要素を追加
        words_list = my_Example.feature_lists.feature_list[item[0]].feature.add()
        # word_list内の、bytes_listのvalueへの参照
        new_word = words_list.bytes_list.value
        # utf-8にエンコードして要素を追加
        new_word.append(word.encode('utf-8'))

print(my_Example)
```