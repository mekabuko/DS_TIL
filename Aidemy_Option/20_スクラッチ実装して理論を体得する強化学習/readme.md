# メモ
## 1. 強化学習入門
### 1.1 強化学習とは
#### 1.1.1 機械学習の分類
- 教師あり学習
    - 目的: 正解ラベル付きのトレーニングデータからモデルを学習し、未知のデータに対して予想すること
- 教師なし学習
    - 目的: 正解ラベルのないデータや構造が不明なデータに対し、データの構造や関係性を見出すこと
- 強化学習
    - 目的: エージェントと環境が相互作用するような状況下で、もっとも最適な行動を発見すること
#### 1.1.2 強化学習というタスク
- 前提とする状況：
    - エージェントが、状態sにあるとして、環境に対して行動aを取る。その結果、環境はその行動の評価として報酬Rを返し、エージェントは次の状態s'に移行する。
    - 上記を繰り返し、行動を強化しながらタスクを進める
- 報酬は、即得られる即時報酬だけではなく、長期的な報酬（遅延報酬）も含めた「収益」全体を最大化することを目的とする必要がある。
    - 本質的な将来の価値を最大化することを目的とした問題は「強化学習に適したタスク」
- 不確実性のある状況において、真価を発揮する
    - ゲームやロボットの制御、ファイナンスなどに活用される
#### 1.1.3 N腕バンディット問題
- N腕バンディット問題とは:
    - 事前に当たる確率の定義されているスロットマシーンがN台ならんでいて、ユーザーはどのスロットマシーンがどのくらい当たるか知らされていない。
    - 一度スロットマシーンを引くと、それぞれ事前に設定されていた確率に基づき、当たりならば1、外れならば0という報酬が支払われる。ここでは簡単化のため、当たりの確率は変わらないものとする。
    - ユーザーは1回の試行につき、どれか1つのスロットマシーンを引ける。
- 試行回数あたりの平均報酬量を最大化するためにはどのようにするべきかを考える問題
    - エージェント自身はそれぞれの内部の確率を知らないために、実際に引くことによって得た報酬量からそれぞれの確率を推測しなければならないことが重要な点
#### 1.1.4 エージェントの作成
- エージェント: 環境の中で行動を決定し、環境に対して影響を与えるもの
    - N腕バンディット問題では、どのスロットマシーンを使用するかを判断し報酬を受け取り、次の判断をするユーザー
- 方策：取得した報酬から、どのようなアルゴリズムに基づいて次の腕を決めるかという指標のこと
#### 1.1.5 環境の作成
- 環境: エージェントが行動をおこす対象
    - エージェントの行動を受け、状況を観測し、報酬をエージェントに送信し、時間を1つ進めるという役割
    - N腕バンディッド問題においては、エージェントがあるスロットマシーンを引いた時、そのスロットの確率によって当たりか外れかを出すプロセス
#### 1.1.6 報酬の定義
- 報酬(reward): 環境からエージェントに与えられる信号のことで、エージェントの一連の行動の望ましさを評価する指標
    - N腕バンディッド問題において言えばスロットマシーンから得られた返り値
    - これは、即与えられるので「即時報酬」
#### 1.1.7 まとめ
- 配列の累積話を出す関数：`np.cumsum()`
- 累積和の遷移をみる方法: `np.cumsum(record) / np.arrange(1,record.size+1)`
    - 分子は累積和の配列（0までの累積、1までの累積, ... , nまでの累積)
    - 分母は(1,2,3, ... , n)
    - 累積和の平均の配列が求められる
- 実装例(引用):
```
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# 手法を定義する関数です
def randomselect():
    slot_num = np.random.randint(0, 5)
    return slot_num

# 環境を定義する関数です
def environments(band_number):
    coins_p = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    results = np.random.binomial(1, coins_p)
    result = results[band_number]
    return result

# 報酬を定義する関数です
def reward(record, results, slot_num, time):
    result = environments(slot_num)
    record[time - 1] = result
    results[slot_num][1] += 1
    results[slot_num][2] += result
    results[slot_num][3] = results[slot_num][2] / results[slot_num][1]
    return results, record

    
# 初期変数を設定しています
times = 10000
results = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]
record = np.zeros(times)

# slot_numを取得して、results,recordを書き換えてください
for time in range(0, times):
    slot_num = randomselect()
    results, record = reward(record, results, slot_num, time)

# 各マシーンの試行回数と結果を出力しています
print(results)

# recordを用いて平均報酬の推移をプロットしてください
plt.plot(np.cumsum(record) / np.arange(1, record.size + 1))

# 表を出力しています
plt.xlabel("試行回数")
plt.ylabel("平均報酬")
plt.title("試行回数と平均報酬の推移")
plt.show()
```
### 1.2 N腕バンディッド問題における方策
#### 1.2.1 greedy手法
- 最初はランダムに、ある程度情報が溜まれば最も良さそうなものを選び続ける
    - 良さそうなものを選び続けることを「利用」と呼ぶ
    - 試行回数が少ないうちは確率の偏りがあるため、一見成功率の低く見える腕も選択する(「探索」と呼ぶ)必要があある
- greedy手法
    - これまでの結果から最も期待値の大きいスロットマシーンを選択する手法
    - 探索：N腕を、n回ずつ動かす探索を行う
    - n が少なくなるほど、間違った（最適ではない）選択をする可能性が高くなる
    - しかし、nを大きくしすぎれば、無駄が多くなる
- 実装例（引用）
```
import numpy as np

# greedyアルゴリズム
def greedy(results, n):
    # 試行回数がnより少ないマシンがある場合、slot_numをそのマシンの腕番号にする 
    slot_num = None
    for i, d in enumerate(results):
        if d[1] < n:
            slot_num = i
            break
        
    # どのマシンの試行回数もnより大きい場合、slot_numを報酬の期待値の高いものにする
    if slot_num == None:
        slot_num = np.array([row[3] for row in results]).argmax()
        
    return slot_num

# 環境を定義する関数
def environments(band_number):
    coins_p = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    results = np.random.binomial(1, coins_p)
    result = results[band_number]
    return result

# 報酬を定義する関数
def reward(record, results, slot_num, time):
    result = environments(slot_num)
    record[time] = result
    results[slot_num][1] += 1
    results[slot_num][2] += result
    results[slot_num][3] = results[slot_num][2] / results[slot_num][1]
    return results, record


# 初期変数を設定
times = 10000
results = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]
record = np.zeros(times)
n = 100

for time in range(0, times):
    slot_num = greedy(results,n)
    results, record = reward(record, results, slot_num, time)
    
print(results)
```
#### 1.2.2 ε-greedy手法
- greedy手法の欠点を解決：探索と選択を織り交ぜることで、探索コストを下げつつ、間違ったマシンの選択確率を減らす
    1. まだ選択したことがないマシンがある場合、それを選択する 
    2. 確率εで全てのマシンの中からランダムに選択する(探索) ※ εはハイパーパラメータで 0 ~ 1
    3. 確率1-εでこれまでの報酬の平均が最大のマシンを選択する
- 長所
    - 見積もりが不確かな場合でも、期待値が高いマシンに選択を集中させる
    - かつ、確率的に全てのマシンを探索できるので、間違えたマシンを選択し続けるリスクも減らせる
- 短所
    - 真の期待値が大きいものを、誤って期待値が小さいと予測した場合（序盤で運悪く低いリターンが偏った場合など）、それが終盤まで選ばれない
        - 一度、ダメだと判断されると取り返すのが非常に困難
    - N腕バンディッド問題においていうならば、全ての腕をn回試行した段階で、真の確率が最も高い腕を選択できない可能性がある
- 実装例
```
import random
import numpy as np
from matplotlib import pyplot as plt

# ε-greedy手法
def epsilon_greedy(results,epsilon):
    # まだ選択したことがないマシンがある場合、そのマシンを選択
    for i, d in enumerate(results):
        if d[1] == 0:
            slot_num = i
            return slot_num
    
    # 確率εで全てのマシンの中からランダムに選択
    if np.random.binomial(1, epsilon):
        slot_num = np.random.randint(0, len(results) - 1)
        
    # 確率1-εでn=0のgreedy手法
    else:
        slot_num = np.array([row[3] for row in results]).argmax()
            
    return slot_num

# 環境を定義する関数
def environments(band_number):
    coins_p = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    results = np.random.binomial(1, coins_p)
    result = results[band_number]
    return result

# 報酬を定義する関数
def reward(record, results, slot_num, time):
    result = environments(slot_num)
    record[time] = result
    results[slot_num][1] += 1
    results[slot_num][2] += result
    results[slot_num][3] = results[slot_num][2] / results[slot_num][1]
    return results, record

    
# 初期変数を設定
times = 10000
record = np.zeros(times)
epsilons = [0.01, 0.1, 0.2]

for epsilon in epsilons:
    results = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]
    for time in range(0, times):
        slot_num = epsilon_greedy(results, epsilon)
        results, record = reward(record, results, slot_num, time)

    #epsilon を変更させて収束の度合いをプロット
    plt.plot(np.cumsum(record) / np.arange(1, record.size + 1), label ="epsilon = %s" %str(epsilon))
    

#グラフの出力     
plt.legend()
plt.xlabel("試行回数")
plt.ylabel("平均報酬")
plt.title("ε-greedyのεによる収束の違い")
plt.show()
```
#### 1.2.3 楽観的初期値法
- ε-greedyの、序盤で評価を低く見積もられると取り返せない、という問題を解決するために、「不確かな時は楽観的に」評価する
    - 予測した期待値に不確実性があるほど、その期待値を大きく見積もる
- 楽観的初期値法は、学習前から全ての選択肢に対してプラスの評価から始める
    1. 報酬の上界(最大値)を r_sup で定義(スロット問題においては1)
    2. すでに各マシンでK回上界が観測され、データとして蓄積しているとする
    3. この状態からgreedy法を始める
- 短所
    - 探査を行う際にすべての行動を等しい確率で選択してしまう点
    - ほぼ最悪と思われる選択を取る確率とほぼ最適と思われる選択を取る確率が同程度になっている
- 実装例
```
import numpy as np
import matplotlib.pyplot as plt

# 楽観的初期値法
def optimistic(results, K, rsup):
    # スロットごとの報酬の期待値を計算して、配列に格納してください
    optimistic_mean = np.array([((row[2] + K * rsup)/(row[1] + K)) for row in results])
    # 最も報酬の期待値が大きいスロットを選択してください
    slot_num = optimistic_mean.argmax()     
    return slot_num

# (参考)１つ前のchapterで実装したepsilon_greedy法
def epsilon_greedy(results,epsilon):
    if np.random.binomial(1, epsilon):
        slot_num = random.randint(0, 4)
    else:
        slot_num = None
        for i, d in enumerate(results):
            if d[1] == 0:
                slot_num = i
                break
        if slot_num == None:
            slot_num = np.array([row[3] for row in results]).argmax()     
    return slot_num

# 環境を定義する関数
def environments(band_number):
    coins_p = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    results = np.random.binomial(1, coins_p)
    result = results[band_number]
    return result

# 報酬を定義する関数
def reward(record, results, slot_num, time):
    result = environments(slot_num)
    record[time] = result
    results[slot_num][1] += 1
    results[slot_num][2] += result
    results[slot_num][3] = results[slot_num][2] / results[slot_num][1]
    return results, record

# 初期変数を設定
times = 10000
results = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]
record = np.zeros(times)
Ks = range(1,6)

# Kを変更させて収束の度合いをプロット
for K in Ks:
    for time in range(times):
        slot_num = optimistic(results, K, 1)
        results, record = reward(record, results, slot_num, time)
    plt.plot(np.cumsum(record) / np.arange(1, record.size + 1), label ="K = %d" % K)
plt.legend()
plt.xlabel("試行回数")
plt.ylabel("平均報酬")
plt.title("楽観的初期値法のKの変動による結果の差異")
plt.show()
```
#### 1.2.4 soft-max法
- 推定報酬によって選択する確率に重み付けをすることが可能になる手法
    - 推定される期待報酬が高い行動が選ばれやすくなる（全く選ばれなくなることはない）
- アルゴリズム：
    1. 今までのデータがない場合、全ての手法の報酬を1で仮定
    2. 各マシンiの選択確率を (exp(報酬の期待値)/τ) / (sum(exp(報酬の期待値)/τ))で定義(※τはハイパーパラメータ) 
    3. 2で定義した確率分布に基づき、選択を行う
    4. 選択で得られた報酬に基づき、報酬関数を更新
- 実装例
```
import numpy as np
tau = 0.1

# soft-max法
def softmax(results, tau):
    # 各マシンの予想期待値の配列
    q = np.array([row[3] for row in results])
    
    # 今までのデータがない場合、全ての手法の報酬を1で仮定
    if np.sum(q) == 0:
        q = np.ones(len(results))
        
    # 各マシンの選択確率を設定
    probability = np.exp(q/tau)/sum(np.exp(q/tau))
    # 確率分布に基づいて、slot_numを決定
    slot_num = np.random.choice(np.arange(len(results)),p = probability)
    
    return slot_num

# 環境を定義する関数です
def environments(band_number):
    coins_p = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    results = np.random.binomial(1, coins_p)
    result = results[band_number]
    return result

# 報酬を定義する関数です
def reward(record, results, slot_num, time):
    result = environments(slot_num)
    record[time] = result
    results[slot_num][1] += 1
    results[slot_num][2] += result
    results[slot_num][3] = results[slot_num][2] / results[slot_num][1]
    return results, record

results = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]
times = 1000
record = np.zeros(times)
for time in range(0, times):
    slot_num = softmax(results,tau)
    results, record = reward(record, results, slot_num, time)
print(results)
```
#### 1.2.5 UCB1アルゴリズム
- 楽観的初期値法を改善した手法
    - greedy手法などで見られたような今までのデータから導出される期待値にバイアス(試行回数が少ない場合に大きくなる)を加えて最大となるマシンを選択
    - そのマシンがどれほど当たってきたか(成功率)という情報にそのマシンについてどれだけ知っているか(偶然による成功率のばらつきの大きさ)という情報を付け加えていくということ
    - あまり探索されていないマシンを積極的に探索するとともに、データが集まってくるにつれて最も当選確率の高いと見られるマシンを選択するということが同時にできる
- アルゴリズム：
    1. R = 報酬の最大値と最小値の差とする
    2. まだ選んだことのないマシンがあればそれを選択
    3. マシンごとに今までの結果から得られた報酬の期待値 ui を計算
        - ui = これまでのマシンiの報酬の和/これまでのマシンiのプレイ回数
    4. マシンごとの偶然による成功率のばらつきの大きさ xi を計算
        - xi = R(2*log(これまでの総プレイ回数)/これまでのマシンiのプレイ回数)^(1/2)
    5. UCB1 = ui + xi が最大になるマシンiをプレイ
- 実装例
```
import numpy as np
import math
R = 1

#　UCB1法
def UCB(results, R):
    slot_num = None
    # まだ選んだことのないマシンがあればそれをslot_numに
    for i, d in enumerate(results):
        if d[1] == 0:
            slot_num = i
            break
        
    # 全て一度は選んでいる場合
    if slot_num == None:
        
        # これまでの総試行回数を計算
        times = sum([row[1] for row in results])
        
        # マシンごとに今までの結果から得られた成功率uiを取り出してリストに格納
        ui = np.array([row[3] for row in results])
        
        # マシンごとの偶然による成功率のばらつきの大きさxi を計算してリストに格納
        xi = R * ((2 * np.log(times)/np.array([row[1] for row in results])) ** (1/2))
        # ui+xiを計算してリストに格納
        uixi = ui + xi
        
        # uixiが最大値となるマシンをslot_numに
        slot_num = uixi.argmax()
        
    return slot_num

# 環境を定義する関数
def environments(band_number):
    coins_p = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    results = np.random.binomial(1, coins_p)
    result = results[band_number]
    return result

# 報酬を定義する関数
def reward(record, results, slot_num, time):
    result = environments(slot_num)
    record[time] = result
    results[slot_num][1] += 1
    results[slot_num][2] += result
    results[slot_num][3] = results[slot_num][2] / results[slot_num][1]
    return results, record


results = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]
times = 1000
record = np.zeros(times)
for time in range(0, times):
    slot_num = UCB(results,R)
    results, record = reward(record, results, slot_num, time)
print(results)
```
#### 1.2.6 まとめ
- ある一つの問題について各手法を比較しても、手法の優劣を議論する根拠にはならない
- これらの手法の間に明確な優劣があるというわけではなく、問題設定に応じて最適な手法を使い分ける必要がある
- リグレット：最初から全ての成功率がわかっている時＝最前手を知っている時と、ある方策との報酬の合計の差分
    - 探索が必要な以上、リグレットを０にはできない
    - UBC1はこのリグレットを最小化できる
- 比較の実装例
```
import random
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(0)

def greedy(results, n):
    slot_num = None
    for i, d in enumerate(results):
        if d[1] < n:
            slot_num = i

    if slot_num == None:
        slot_num = [row[3] for row in results].index(max([row[3] for row in results]))
    return slot_num


def optimistic(results, K, rsup):
    optimistic_mean = [(row[2] + K*rsup) / (row[1] + K) for row in results]
    slot_num = optimistic_mean.index(max(optimistic_mean))
    return slot_num


def epsilon_greedy(results, epsilon):
    if np.random.binomial(1, epsilon):
        slot_num = random.randint(0, 4)
    else:
        slot_num = None
        for i, d in enumerate(results):
            if d[1] == 0:
                slot_num = i
        if slot_num == None:
            slot_num = np.array([row[3] for row in results]).argmax()           
    return slot_num


def UCB(results, R):
    slot_num = None
    for i, d in enumerate(results):
        if d[1] == 0:
            slot_num = i
    if slot_num == None:
        times = sum([row[1] for row in results])
        u = [row[3] for row in results]
        xi = [R*math.sqrt(2*math.log(times) / row[1]) for row in results]
        uixi = [x+u for x, u in zip(xi, u)]
        slot_num = uixi.index(max(uixi))
    return slot_num


def environments(band_number):
    coins_p = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    results = np.random.binomial(1, coins_p)
    result = results[band_number]
    return result


def reward(record, results, slot_num, time):
    result = environments(slot_num)
    
    record[time] = result
    results[slot_num][1] += 1
    results[slot_num][2] += result
    results[slot_num][3] = results[slot_num][2]/results[slot_num][1]
    return results,record


times = 10000
n = 20
K = 10
rsup = 1
R = 1
epsilon = 0.2
results = [[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]] for _ in range(4)]
slot_num = np.zeros(4,dtype=np.int)
record = np.zeros((4,times))

# resultsからそれぞれの方策でslot_numを決定
for time in range(0, times):
    # greedy手法
    slot_num[0] = greedy(results[0], n)
    # ε-greedy手法
    slot_num[1] = epsilon_greedy(results[1], epsilon)
    # 楽観的初期値法
    slot_num[2] = optimistic(results[2], K, rsup)
    # UCB1アルゴリズム
    slot_num[3] = UCB(results[3], R)
    
    # results、recordを上書きしてください
    for i in range(4):
        results[i], record[i] = reward(record[i], results[i], slot_num[i], time)

# 凡例を以下のように設定
labels = ["greedy","epsilon_greedy","optimistic","UCB"]
for i in range(4):
    # 4種の手法の結果をプロット
     plt.plot(np.cumsum(record[i]) / np.arange(1, record[i].size + 1),label=labels[i])

plt.legend(loc="lower right")
plt.xlabel("試行回数")
plt.ylabel("平均報酬")
plt.title("N腕バンディッド問題の各手法における報酬の収束")
plt.show()
```
#### 1.2.7 探索と利用のトレードオフ
- 探索の割合を増やすと、探索中は最善の選択を選ばないため、そのぶん損失が発生してしまう
- 利用の割合を増やすと、最善の選択が今までされてこなかった場合にそれを見落とすリスクが発生してしまう
- つまり、トレードオフの関係にある

## 2. マルコフ決定過程とベルマン方程式
### 2.1 強化学習の構成要素
#### 2.1.1 強化学習の構成要素
- n腕バンディッド問題では、状態や時間を考える必要がなかった
    - 環境は変わらず、報酬は即時
- より一般的には、上記のようなものを考えないといけない
#### 2.1.2 強化学習のモデル化
- 時間ステップ：開始してから何回目の動作であるか
    - 離散化された 0,1, ... , t で表現される
- 方策：エージェントの行動する指針
- エージェントと環境：何をエージェントと見做し、何を環境と見做すかは、モデル次第
#### 2.1.3 状態・行動・報酬
- 状態集合 S = {s0, s1, s2, ...}
    - エピソード内でとりうる全ての状態
    - St: 時間ステップ t における状態 S
- 行動集合 A(s) = {a0, a1, a2, ...}
    - ある状態で取りうる行動全て
    - 状態によって取れる行動が変わるので、状態 s を引数に取る
    - At: 時間ステップ t における行動
- 報酬 Rt+1 = r(St, At, St+1)
    - 報酬関数：現在の状態 St, 行動 At, 次の状態 St+1 によって一意に決まる関数
- マルコフ性：
    - 未来の状態は現在の状態・行動によって確率的に決定され、過去の挙動と無関係である、ということ
### 2.2 マルコフ決定過程
#### 2.2.1 マルコフ性
- マルコフ性：
    - 未来の状態は現在の状態・行動によって確率的に決定され、過去の挙動と無関係である、ということ
- マルコフ決定過程(MDF): マルコフ製を満たす強化学習過程
- マルコフ過程の構成要素
    - 状態空間 S
        - 環境の取りうる状態の全て
    - 行動空間 A(s)
        - エージェントが取りうる行動の集合
    - 初期状態分布　P0
        - エピソードの開始時点お状態を表す確率変数
    - 状態遷移確率 P(s'|s,a) 
        - 状態 s で 行動 a をしたときに、状態が s' へ遷移する確率
    - 報酬関数　r(s,a,s')
        - 状態 s で 行動 a をして、状態が s' になるときの報酬 r を求める関数
- 状態遷移図の実装表現：
```
# [s,s',a,P(s'|s,a),r(s,a,s')] で表現可能
state_transition = np.array(
    [[0, 0, 0, 0.3, 5],
     [0, 1, 0, 0.7, -10],
     [0, 0, 1, 1, 5],
     [1, 0, 1, 1, 5],
     [1, 2, 0, 0.8, 100],
     [1, 1, 0, 0.2, -10]]
)
```
#### 2.2.2 環境変化の記述-1
- Action は、stateによって取りうる値が変わる
- 実装例
```
def actions(state):
    A = state_transition[(state_transition[:,0] == state)] # 与えられた State が s と同じものだけに絞り込み
    return np.unique(A[:, 2]) # Action を表す 2列目を取り出し、重複排除して返す
```
#### 2.2.3 環境変化の記述-2
- 報酬関数は s, a, s' を引数に取る関数として定義できる
- 実装例
```
def R(state, action, after_state):
    if state in terminals:
        return 0
    
    X = state_transition[(state_transition[:, 0] == state) # 今の状態 s
    & (state_transition[:, 1] == after_state) # アクション a
    & (state_transition[:, 2] == action)] # 次の状態 s'
    return X[0, 4] # その時の報酬
```
#### 2.2.4 エピソード
- エピソード：タスク開始から終了までの間にかかる時間のこと
    - 行動→状態変化というサイクルが複数回なされることで1つのエピソードが構成される
- 一回のエピソードの中で、以下のようなステップを踏む
    1. 環境を初期化
    2. エージェントに行動させる
    3. 受け取った報酬を元に行動モデルを最適化
- エピソードを繰り返して、エージェントは学習を進める
- 実装例
    - エピソードの実装には、状態と行動を受け取って、遷移先と遷移確率のセットの全量を返す関数 T(s,a) を定義する
```
def T(state, action):
    if (state in terminals):
        return [(0, terminals)] # 終点なら移動する先はない
    
    X = state_transition[(state_transition[:, 0] == state)&(state_transition[:, 2] == action)] # stateとactionから、あり得る遷移の可能性
    
    A = X[:, [3, 1]] # そのうち、遷移確率と遷移先
    return [tuple(A[i, :]) for i in range(A.shape[0])] # タプルで重複排除して返す
```
### 2.3 価値・収益・状態
#### 2.3.1 報酬と収益
- 報酬は、直前の行動と状態だけで決定される＝長期的目線が反映されない
- そこで、収益＝ある期間で得られた報酬の合計、という概念を導入
- 報酬の計算方法
    - 時刻tからある期間Tまで報酬を合計する
    - 割引報酬和: 時刻tからある時間Tまでの収益の平均を計算し、Tの極限を取る
- 割引報酬和 が一般的に使用される
- 割引報酬和の実装例
```
import numpy as np
from markov import T, R
import random

def take_single_action(state, action):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for probability_state in T(state, action):
        # 考えうる確率と行動過程をprobalility, next_stateに代入
        probability, next_state = probability_state
        # 累積確率に確率を足す
        cumulative_probability += probability
        # 累積確率がxより大きくなれば終了
        if x < cumulative_probability:
            break
    # 次のstateを返す
    return next_state

# 報酬の計算
def calculate_value():
    # stateを定義
    state = init
    # actionを定義
    action = [0]
    # 割引報酬和を定義
    sumg = 0
    for i in range(2):
        # 次のstateを定義
        after_state = take_single_action(state, action)
        # sumgを計算
        sumg += (gamma**(i))*R(state, action, after_state)
        # stateを更新
        state = after_state
    return sumg

#　パラメータ
state_transition = np.array(
                    [[0, 0, 0, 0.3, 5],
                    [0, 1, 0, 0.7, -10],
                    [0, 0, 1, 1, 5],
                    [1, 0, 1, 1, 5],
                    [1, 2, 0, 0.8, 100],
                    [1, 1, 0, 0.2, -10]]
                    )
states = np.unique(state_transition[:, 0])
actlist = np.unique(state_transition[:, 2])
terminals = [2]
init = [0]
gamma = 0.8

# 実行
calculate_value()
```
#### 2.3.2 価値関数・状態価値関数
- 収益をそのまま評価基準とすると、状態が分岐するほど確率変数の形で複雑なものになってしまう
- そこで、期待値を用いて一定値としたい
- ある状態をスタート地点にした時の今後の行動すべて考慮に入れた報酬の期待値を状態価値または価値と呼ぶ
- これによって以下が実現される
    - ある方策においてどの状態が優れているかを比較できるようになる。
    - ある状態をスタート地点においた時、方策毎の良し悪しを比較できるようになる。
### 2.4 最適な方策の探求
#### 2.4.1 最適方策
- 状態価値関数の導入により、方策の比較が可能になった
- これを応用して「全ての状態において」の方策の比較も可能
    - 全ての状態において 方策1 >= 方策2 で、少なくとも１つの状態では 方策1 > 方策2
    - 上記を満たすなら、方策1 の方が良い方策、徒歩床できる
- 最適方策：全ての方策に対して比較した結果、最も良い方策となるもの
#### 2.4.2 行動価値
- 状態関数は、ある状態をスタート地点にした時の報酬の期待値
- 行動価値：ある状態から、ある行動を起こしたときに報酬の期待値
- 行動価値関数：行動価値を求める関数。各行動がどれだけ良いかを判断する関数と言える。
#### 2.4.3 最適状態価値関数・最適行動価値関数
- 強化学習の目的は、最適方策を見つけることと言える
- 最適状態価値関数：最適な方策に従った場合の状態価値関数
- 最適行動価値関数：行動価値関数の中で、期待値を最大化する行動価値関数
### 2.5 ベルマン方程式
#### 2.5.1 最適な状態価値
- 強化学習の目的は「環境に対する最適な方策」を取得すること
- つまり、一度最適状態価値関数がわかってしまえば、それに従って一番価値が大きくなるように行動すれば良い
- 最適状態価値関数を求めるためには、それぞれの方策における状態価値観数を計算して、比較する必要がある
- モンテカルロ法
    - 期待値を求めるために、試行錯誤を繰り返し続けることで、確率的に期待値を収束させる
    - この方法には以下の問題がある
        - 収益を計算する必要があるため、１つのエピソードが終了するまで学習ができない
        - 方策が変わるたびに全て学び直しをする必要がある
        - 計算のためには各状態sにおける期待値を保持し続ける必要がある
    - 上記の欠点は、状態数が増えるほど厳しい条件になる
#### 2.5.2 ベルマン方程式
- モンテカルロ法の欠点の解決のために、ある時点での状態sとそこから行動した結果移行する可能性のある状態s'との間に価値関数の関係式が成り立てば良い、という発想が生まれた
- これによって、状態価値関数と行動価値関数を漸化式のような形で表現し、逐次的に更新しながら価値関数を求めることが可能に
- ベルマン方程式：上記によって導出された方程式
#### 2.5.3 ベルマン最適方程式
- ベルマン最適方程式: 最適状態価値関数・最適行動価値関数に対応するベルマン方程式
## 3. 動的計画法とTD手法
### 3.1 動的計画法
#### 3.1.1 動的計画法とは
- 環境のモデルがMDP（マルコフ決定過程）として完全に与えられている時の解法アルゴリズムのこと
#### 3.1.2 方策評価
- 方策評価(policy evaluation): ある特定の手法πを取った時の価値関数Vπ(s)を計算する方法
    1. 閾値εを事前に設定。更新量がεより小さくなるまで計算を繰り返す
    2. 方策πを入力
    3. V(s) を全ての状態において 0 と仮定
    4. 5 ~ 6 を繰り返す
    5. δ = 0
    6. 全ての状態sについて7~10の動作を繰り返す
    7. v = V(s)
    8. V(s) = sum(π(a|s))sum(P(s'|s, a)(r(s,a,s') + γV(s')) で更新
    9. δ = max(δ|v-V(s)|)
        - 一番大きい更新量を保存
        - これによって、全ての方策について更新量がε以下になるまで繰り返す
    10. δ < ε ならば V(s) を Vπ の近似解として出力
- 実装例
```
from markov import *


def policy_evaluation(pi, states, epsilon=0.001):
    while True:
        # δを定義
        delta = 0
        
        # 全てのstateにおいてループを回す
        for s in states:
            # vにVを代入
            v = V.copy()

            # xという変数に計算をしておいてV[s]にxを代入
            x = 0
            
            # pはエピソードTで得られる遷移先
            for (p, s1) in T(s, pi[s]):
                # V[s]を計算
                x += p * (R(s, pi[s], s1) + gamma * V[s1])
            V[s] = x
            
            # δを計算
            delta = max(delta, abs(v[s] - V[s]))
            
        # δ<εの場合Vを返す
        if delta < epsilon:
            return V

state_transition = np.array(
    [[0, 0, 0, 0.3, 5],
    [0, 1, 0, 0.7, -10],
    [0, 0, 1, 1, 5],
    [1, 0, 1, 1, 5],
    [1, 2, 0, 0.8, 100],
    [1, 1, 0, 0.2, -10]]
)

def T(state, action):
    if (state in terminals):
        return [(0, terminals)]

    X = state_transition[(state_transition[:, 0] == state)&(state_transition[:, 2] == action)]

    A = X[:, [3, 1]]
    return [tuple(A[i, :]) for i in range(A.shape[0])]

states = np.unique(state_transition[:, 0])
actlist = np.unique(state_transition[:, 2])
terminals = [2]
gamma = 0.8

V = {s: 0 for s in np.hstack((states,terminals))}
pi = {s: 0 for s in states}
policy_evaluation(pi, states, epsilon =0.001)
```
#### 3.1.3 方策反復
- 方策反復: 改善と評価を繰り返すことで最適価値関数を導き出すこと
    1. 全ての状態 s に対して V(s) と π(s) を初期化
    2. イプシロンが閾値以下になるまで以下を繰り返す（方策評価）
        1. δ=0
        2. 各状態 s について
            1. v = V(s)
            2. V(s) = sum(π(a|s))sum(P(s'|s, a)(r(s,a,s') + γV(s')) で更新
            3. δ = max(δ|v-V(s)|)
    3. polity-flag = True
    4. 各状態 s について
        1. b = π(s)
        2. π(s) = arg max(sum(P(s'|s,a)(r(s,a,s') + γV(s'))))
            - 2. の方策評価を行なった後の価値関数を使ってすべての action における価値関数を求める
            - 方策の更新を合わせて行う
        3. b ≠ π(s) なら policy-flag = False
    5. policy-flag = True で終了。それ以外は 2. から繰り返し
- 実装例
```
from markov import *
random.seed(0)

# 方策評価関数です
def policy_evaluation(pi, V, states, epsilon=0.001):
    while True:
        delta = 0
        for s in states:
            v = V.copy()
            V[s] = sum([p * (R(s, pi[s], s1) + gamma * V[s1]) for (p, s1) in T(s,pi[s])])
            delta = max(delta,abs(v[s] - V[s]))
        if  delta < epsilon:
            return V

# 方策反復関数
def policy_iteration(states):
    # 1. 価値関数Vと方策πを初期化
    V = {s: 0 for s in np.hstack((states, terminals))}
    pi = {s: random.choice(actions(s)) for s in states}
    while True:
        # 2. 方策評価をしてください
        V = policy_evaluation(pi, V, states)
        policy_flag = True

        for s in states:
            # 4. 更新した価値関数Vから報酬が得られる順に行動をソートしてください
            action = sorted(actions(s), key=lambda a:sum([p * gamma * V[s1] + p * R(s, a, s1) for (p, s1) in T(s, a)]), reverse=True)
            if action[0] != pi[s]:
                pi[s] = action[0]
                policy_flag = False
        if policy_flag:
            return pi

state_transition = np.array(
    [[0, 0, 0, 0.3, 5],
    [0, 1, 0, 0.7, -10],
    [0, 0, 1, 1, 5],
    [1, 0, 1, 1, 5],
    [1, 2, 0, 0.8, 100],
    [1, 1, 0, 0.2, -10]]
)

states = np.unique(state_transition[:,0])
terminals = [2]
init = [0]
gamma = 0.8

# 関数を動かして結果を表示します
pi = policy_iteration(states)
print(pi)
```
#### 3.1.4 価値反復
- 方策反復でのアルゴリズムには、毎回全ての状態について価値関数を計算し直す事を複数回しなければならないという問題がある
- 価値反復: 価値関数の計算を１回にできるように改善
    1. すべての状態 s に対して V(s) を初期化
    2. 繰り返し
        1. δ = 0
        2. 各状態 s について
            1. v = V(s)
            2. V(s) = sum(π(a|s))sum(P(s'|s, a)(r(s,a,s') + γV(s')) で更新
            3. δ = max(δ|v-V(s)|)
            4. δ < ε ならば 終了
    3. π = arg max(sum(P(s'|s,a)(r(s,a,s') + γV(s')))) を出力
- 実装例
```
from markov import *
random.seed(0)

# 価値反復関数
def value_iteration(states, actlist, epsilon=0.001):
    # 1. 状態価値Vの初期化
    V = {s: 0 for s in np.hstack((states, terminals))}
    # 2. 価値関数の更新
    while True:
        delta = 0
        for s in states:
            v = V.copy()
            # V[s]を計算してください
            V[s] = max([sum([p * (R(s,a,s1) + gamma*V[s1]) for (p,s1) in T(s,a)]) for a in actlist])
            delta = max(delta, abs(v[s] - V[s]))
        if  delta < epsilon:
            break
    # 3. greedyな方策の計算
    pi = {}
    for s in states:
        # V[s]から最も報酬が得られる行動を計算
        action = sorted(actions(s), key=lambda a:sum([p * gamma * V[s1] + p * R(s,a,s1) for (p, s1) in T(s, a)]), reverse=True)
        pi[s] = action[0]
    return pi

state_transition = np.array(
    [[0, 0, 0, 0.3, 5],
    [0, 1, 0, 0.7, -10],
    [0, 0, 1, 1, 5],
    [1, 0, 1, 1, 5],
    [1, 2, 0, 0.8, 100],
    [1, 1, 0, 0.2, -10]]
)

states = np.unique(state_transition[:, 0])
actlist = np.unique(state_transition[:, 2])
terminals = [2]
init = [0]
gamma = 0.8

pi = value_iteration(states,actlist,epsilon=0.001)
print(pi)
```
### 3.2 TD手法
#### 3.2.1 TD手法とは
- 動的計画法の欠点：状態遷移確率があらかじめわかっていなければならない
- TD手法：Time Differenceの事で、最終結果を見ずに、現在の推定値を利用して次の推定値を更新していく方法
- TD手法の有名な例
    - Sarsa
    - Q-learning
#### 3.2.2 Sarsa
- ベルマン方程式を試行錯誤しながら解いていくアルゴリズムの一つ
- 実装例（引用）
```
import numpy as np
np.random.seed(0)

def T(state, direction, actions):
        return [(0.8, go(state, actions[direction])),
                (0.1, go(state, actions[(direction + 1) % 4])),
                (0.1, go(state, actions[(direction - 1) % 4]))]

def go(state, direction):
    return [s + d for s, d in zip(state, direction)]

# ランダムに次の行動を決定
def get_action(t_state, episode):
    next_action = np.random.choice(len(actions))
    return next_action


# chapter2.3 報酬と収益 で実装した関数を少し変更して使用
# 返し値にrewardを追加
def take_single_action(state, direction,actions):
    x = np.random.uniform(0, 1)
    cumulative_probability = 0.0
    for probability_state in T(state,direction,actions):
        probability, next_state = probability_state
        cumulative_probability += probability
        if x < cumulative_probability:
            break
    reward = situation[next_state[0],next_state[1]]
    if reward is None:
        return state, -0.04
    else:
        return next_state, reward


num_episodes = 30
max_steps = 100
total_reward = np.zeros(5)
goal_average_reward = 0.7

# 環境を定義
situation = np.array([[None, None, None, None, None, None],
                      [None, -0.04, -0.04, -0.04, -0.04, None],
                      [None, -0.04, None, -0.04, -1, None],
                      [None, -0.04, -0.04, -0.04, +1,None],
                      [None, None, None, None, None, None]])
            

terminals=[[2, 4], [3, 4]]
init = [1, 1]
actions = ((1, 0), (0, 1), (-1, 0), (0, -1))


# エピソードの繰り返しを定義
for episode in range(num_episodes):
    state = init
    # エピソードの繰り返しを定義
    action = get_action(state, episode)
    episode_reward = 0

    # 時間ステップのループを定義
    for t in range(max_steps):
        next_state, reward = take_single_action(state, action, actions)
        episode_reward += reward
        next_action = get_action(state, episode)
        state = next_state
        action = next_action
        if state in terminals :
            break
            
    #報酬を記録        
    total_reward = np.hstack((total_reward[1:],episode_reward))
    print(total_reward)
    print("Episode %d has finished. t=%d" %(episode+1, t+1))
    # 直近の5エピソードが規定報酬以上であれば成功
    if (min(total_reward) >= goal_average_reward):
        print('Episode %d train agent successfuly! t=%d' %(episode,t))
        break 
```
#### 3.2.3 SarsaにおけるQ関数の実装
- Q関数:　[全ての状態×全ての行動]　の配列
#### 3.2.4 Sarsaでのε-greedy手法の実装
- Q関数を使いつつ、より優れた方法をε-greedyで探索
- 実装例(引用)
```
import numpy as np
np.random.seed(0)


def T(state, direction, actions):
        return [(0.8, go(state, actions[direction])),
                (0.1, go(state, actions[(direction + 1) % 4])),
                (0.1, go(state, actions[(direction - 1) % 4]))]


def go(state, direction):
    return [s+d for s, d in zip(state, direction)]

# 空欄を埋めてepsilon-greedy手法により次の行動を決定してください
def get_action(t_state, episode):
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[t_state])
    else:
        next_action = np.random.choice(len(actions))
    return next_action



def take_single_action(state, direction, actions):
    x = np.random.uniform(0, 1)
    cumulative_probability = 0.0
    for probability_state in T(state, direction, actions):
        probability, next_state = probability_state
        cumulative_probability += probability
        if x < cumulative_probability:
            break
    reward = situation[next_state[0], next_state[1]]
    if reward is None:
        return state, -0.04
    else:
        return next_state, reward



def update_Qtable(q_table, t_state, action, reward, t_next_state, next_action):
    gamma = 0.8
    alpha = 0.4
    q_table[t_state, action] += alpha * (reward + gamma * q_table[t_next_state, next_action] - q_table[t_state, action])
    return q_table

def trans_state(state):
    return sum([n*(10**i) for i, n in enumerate(state)])

q_table = np.random.uniform(
    low=-0.01, high=0.01, size=(10 ** 2, 4))

num_episodes = 500
max_number_of_steps = 1000
total_reward_vec = np.zeros(5)
goal_average_reward = 0.7

situation = np.array([[None, None, None, None, None, None],
                      [None, -0.04, -0.04, -0.04, -0.04, None],
                      [None, -0.04, None, -0.04, -1, None],
                      [None, -0.04, -0.04, -0.04, +1, None],
                      [None, None, None, None, None, None]])
            

terminals=[[2, 4], [3, 4]]
init = [1,1]
actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
state = [n*(10**i) for i,n in enumerate(init)]

for episode in range(num_episodes):
    state = init
    t_state = trans_state(state)
    action = np.argmax(q_table[t_state])
    episode_reward = 0

    for t in range(max_number_of_steps): 
        next_state, reward = take_single_action(state, action, actions)
        episode_reward += reward  
        t_next_state = trans_state(next_state)
        next_action = get_action(t_next_state, episode)  
        q_table = update_Qtable(q_table, t_state, action, reward, t_next_state, next_action)
        state = next_state
        t_state = trans_state(state)
        action = next_action
        
        if state in terminals :
            break
    total_reward_vec = np.hstack((total_reward_vec[1:],episode_reward))  
    print(total_reward_vec)
    print("Episode %d has finished. t=%d" %(episode+1, t+1))
    print(min(total_reward_vec),goal_average_reward)
    if (min(total_reward_vec) >= goal_average_reward): 
        print('Episode %d train agent successfuly! t=%d' %(episode, t))
        break 

```
#### 3.2.5 Q学習
- Q学習も全体の流れはSarsaと変わりない。以下の点のみ異なる。
    1. q関数の更新を行ったあとに次の行動を決定すること 
    2. q関数の更新の方法
