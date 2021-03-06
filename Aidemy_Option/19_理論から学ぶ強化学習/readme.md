# サマリ

# メモ
## 1. 強化学習基礎
### 1.1 方策勾配に基づく強化学習
#### 1.1.1 方策反復法と価値反復法
- 方策反復法: エージェントの学習状態として方策を実現し、それを利用して価値関数を計算する形
    - メリット: 多様な方策を表現できる
    - デメリット: 方策評価を必要とするため計算量が多くなる
    - 具体例: 多様な方策を表現できる
    - 適用対象: 連続の状態、行動空間を取り扱う場合
    - 手順:
        1. 初期化
        2. 方策評価
        3. 方策改善
- 価値反復法: SarsaやQ学習による学習の毎ステップで価値関数が更新されるたびに、方策を更新
    - メリット: 価値関数だけで学習状態を表現できるため、簡潔な実装で実現できる
#### 1.1.2 方策勾配の概念
- 価値反復に基づく強化学習では、方策は行動価値関数Qを用いて表現 = 方策は行動価値関数から導出されるものとして定義
    - エージェントは行動価値関数を試行錯誤を通して学習し、最適な方策を求める
- 異なるアプローチとして、方策を行動価値関数とは別のパラメータで表現することも考えられる
    - 方策勾配に基づく強化学習アルゴリズム: 
        - 確率的方策をあるパラメータベクトルによってパラメタライズされた確率モデルπ(a|s)と考え、これを最適化することで強化学習問題を解く方法
        - アルゴリズムは「収益Gを最大化する方策のパラメータを求める」を目的とする
        - つまり、、行動価値関数が直接用いられない
    - 行動価値関数 Q(St,At) を別のモデルでの近似で求める
#### 1.1.3 方策勾配に基づく強化学習のアルゴリズム
- 方策勾配に基づく強化学習のアルゴリズムの手順:
    1. 方策による行動
    2. 方策の評価
    3. 方策の更新
- 方策勾配に基づく機械学習アルゴリズムは方策モデルを更新する
    - 期待収益を最大化する確率的方策のパラメータを、勾配法によって求めていく
### 1.2 Q学習
#### 1.2.1 Q学習の流れと問題点
- 状態sにおいて行動aをとる価値、行動価値関数Q(s,a)を学習
- 学習の流れ
    - 考えうる全ての(s,a)の組み合わせに対してQ(s,a)を0に初期化し、次の式に基いて更新していく
        - `Q(s,a) = Q(s,a) + (学習率α)(報酬 + (割引率)(最大となるQ(s',a')) - Q(s,a))`
    - 値が動かなくなるまで、上記の更新を繰り返す
- 問題点
    - 全ての状態と行動の組み合わせ(s,a)に対して行動価値関数Q(s,a)を常に保存し、更新していく必要がある
    - 上記は、現実的には組合せ爆発を起こし、メモリが足りなくなる
#### 1.2.2 Q関数の近似
- Q学習の問題点：Q(s,a)を超大量に保存しておかないといけない
- その解決のために、Q(s,a)を関数で近似してしまうことを考える(関数近似)
    - そうすれば、関数一つ保存しておけば良い（全てのQ(s,a)を保持する必要がなくなる）
#### 1.2.3 関数近似した場合の学習方法
- Q学習の流れは、Q(s,a)を更新していくことだったが、関数近似では、近似関数のパラメータを更新していけばいい
    - 誤差関数 `(1/2)((報酬 + (割引率)(最大となるQ(s',a')) - Q(s,a)))^2` を最小化していく
    - 勾配を用いて近似関数Q(s,a)のパラメータを更新していく​	
- 但し、近似関数を非線形のものとした場合、学習の収束が保証されない点には注意が必要
#### 1.2.4 ニューラルネットワークによる近似
- 深層強化学習: 近似関数に、NNを使用したもの(所詮はNNも関数)
- 但し、NNは非線形関数なので、収束保証がなく、実際にDQNの登場までには性能発揮するものはほとんどなかった
#### 1.2.5 DQN
- 学習の安定しない深層強化学習を改善し、ブレイクスルーを達成したのが DQN(Deep Q-Network)
    - 元々はAtari社のゲームの攻略を目的としたもの
    - 画面ピクセルから学習を行う（まさに人間が行うのと同じようなイメージ）
    - 画像認識の必要性もあり、近似関数としてCNNが用いられている
- Q学習に対して、以下の改善を加えている
    - 報酬、TD誤差のClipping
        - 「報酬のスケールが与えられたタスクによって大きく異なる」という問題に対して適用される
        - あらゆる報酬は、-1,0,1に振り分け（クリッピング）してしまう
        - 学習の安定性向上のためにも、TD誤差もクリッピング
    - Experienced Replay
        - 「入力データが時系列データであり、入力データ間に独立性がない」という問題に対して適用される
        - 基本的に、Stepにおける状態s,行動a,報酬R,次の状態s'の連続（時系列）を学習していくが、これだとデータ間の相関は極大
        - しかし、学習データ間には相関がない方が良い結果を得られる
        - そのため、並び順を変えて、さらにランダムにデータを抽出して学習する
        - そのため、最初のある程度（１万フレームほど）は学習せずにデータの蓄積を行う
    - Fixed Target Q-network
        - 「価値関数が少し更新されただけでも方策が大きく変わってしまう」という問題に対して適用される
        - 一定の期間ごと(1万フレームに1回など)に、パラメータ更新を行う
        - つまり、一定期間、価値関数を固定しする
        - 一定期間の学習の間、目標値の計算に用いる価値関数ネットワークのパラメータを固定して、一定周期で更新
        - 学習を安定させ、計算量を削減させることに寄与する
## 2.AlphaGo
### 2.1 AlphaGo
#### 2.1.1 人間を超えた囲碁AI
- 米Google傘下のDeep Mindが開発した囲碁AI
    - プロの囲碁プレーヤー（棋士）に勝った最初のAI
- 以下のような点で注目を集めた
    - コンピューターにとって囲碁はもっとも難しいゲームの一つとされてきたから
    - CNNを囲碁AIに応用し、成果を出したから
    - 強化学習を囲碁に応用し、成果を出したから
    - 囲碁の勝率を計算するモデルを作れたから
    - 複数のモデルを機能的に組み合わせ、プロを上回る成果を出したから
- 囲碁は「2人ゼロ和有限確定完全情報ゲーム」と呼ばれる分類に属するゲーム = 「理論的には必勝法があるゲーム」
    - すべてのパターンを考慮して互いに最善手を打ち続ければ、「先手必勝」か「後手必勝」のどちらかに収束する
    - しかしその組み合わせが多すぎるので、現実的な時間で計算可能な問題ではない
- 実装理論
    - ベースとなっているのは「モンテカルロ木探索」
    - 以下の４つの改善点を追加:
        1. 勝率の実測に、ロジスティック回帰のモデルを使う
        2. 次の一手を予測する深層学習モデルによる、探索のバイアス補正
            - 人間が打った囲碁データで学習した深層学習モデル
            - 強化学習モデルはモンテカルロ木探索と相性が悪かったため？不採用
        3. 次の一手による勝率を予測する深層学習による、探索の勝率補正
            - 選んだ場所に石を置いた時の勝率を予測するモデル
            - 学習に必要なデータをAIによって作るという工夫がされている
        4. 処理の非同期並列化
            - 計算スピードの改良
#### 2.1.2 3つの改良
- AlphaGoは「モンテカルロ木探索」「ロジスティック回帰」「CNN」といったモデルを組み合わせて使っている
- AlphaGoは、モンテカルロ木探索で選択肢を評価する「勝率」と「バイアス」に深層学習モデルを使った補正をしている。打った手の勝率の予測を勝率の補正に使い、次の一手の予測をバイアスの補正に使う。勝敗のカウントは、計算の速いロジスティック回帰が手を進めた結果を数える。
- モンテカルロ木探索
    - その名の通り「モンテカルロ法」を使った「木探索」
        - モンテカルロ法とはランダムに試行した結果を評価する方法
    - 囲碁のモンテカルロ木探索は、次の一手を「勝率」と「バイアス」という2つの値で評価
        - 勝率: ランダムに石を置くだけのゲームで勝敗を数えて計算した結果
        - バイアス: 勝敗を数えた回数が増えるほど小さくなる項
            - 回数が増えるほどに正確な勝率が得られていると仮定すれば、バイアスが大きい選択肢は勝率の確度が低いと解釈できる
            - 「今より勝率の良い手があるかもしれないから、確かめよう」と促す項
        - 評価は勝率とバイアスの合計
    - 評価が高い手は、その次の一手（合わせて2手目）まで考えて勝率とバイアスを更新
        - どこまで評価が高くなったら「その次の一手」を考慮するかは、調整するパラメーター
        - このアプローチは全てのパターンを考慮しきらないと計算が終わらないため、1手当たりの評価にかける時間を制限して、時間がきたらその時点で優秀そうな一手を選ぶ
- ロジスティック回帰による脱ランダムと評価補正
    - 「ランダムに石を置く」というゲームプレイを、ロジスティック回帰モデルを用いて「予測して石を置く」ように変更
    - 評価方法を、学習モデルでバイアスをかけて、最初からあり得ない手は探索を抑制
- 非同期化
    - モンテカルロ木探索は並列化しやすい
    - 勝率とバイアスの更新以外は、非同期並列できる

#### 2.1.3 強化学習で進化
- DeepMindは「次の一手」を深層ニューラルネットワークで予測するモデルを2つ作っている
    - SL policy network: 人間の対戦履歴を教師データに学習
    - RL policy network（RL：reinforcement learning）:SL policy networkを元に「強化学習」の手法で学習
- この強化学習の手法は、モンテカルロ木探索と相性が悪く、対局には用いられなかった
    - 代わりに、勝率を予測する value network を学習するための教師データ作成で威力を発揮した
- value network の学習用データとして欲しい条件
    - 1対戦から1盤面+勝敗のデータを使う
    - 人間には予想できないような手の勝率も学習する
    - 1手打ったあとの勝敗は妥当な決着がつく
- データ作成の手順
    - 1∼450までのランダムな数だけ、SL policy network同士で対戦して手を進める。これを学習する盤面とする。
    - ランダムに1マスを選び、石を置く。この一手に対する勝率を予測する。
    - 勝敗が付くまでRL policy network同士で対局を進めて、勝敗を決める。勝敗を正解ラベルとする。
