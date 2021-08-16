# メモ
## 1. Environmentクラスとメイン部分を作成する
### 1.1 Environmentクラス
#### 1.1.1 概要
- 「Environment」クラス
    - ゲーム進行を行う部分
        - 盤面の作成 (9つの要素を持つ配列を用意）
        - Agentが打つ
        - 勝利判定 (引き分けも判定）
        - Enemyが打つ
        - 勝利判定
- 「Agent」クラス
    - ニューラルネットワークに関連する部分
        - QネットワークとtargetQネットワークの初期化
        - Qネットワークの学習
        - targetQネットワークを更新
        - QネットワークからAgentの行動選択
- メイン部分
    - 各クラスをインスタンス化し、ゲームを進行
    - 出力とグラフへのプロット
#### 1.1.2 盤を初期化するstart関数
- 3x3 の盤面を 配列で表現
- 初期化は全要素を 0 に: `np.zeros(9, dtype=int)`
#### 1.1.3 勝ちを判定するterminal関数
- 実装例
```
    def terminal(self, Map, num_player):
        tempMap = np.where(Map == num_player,1,0)
        # 縦の積
        virtical = tempMap[0] * tempMap[3] * tempMap[6] + tempMap[1] * tempMap[4] * tempMap[7] +tempMap[2] * tempMap[5] * tempMap[8] 
        # 横の積
        horizon =  tempMap[0] * tempMap[1] * tempMap[2] + tempMap[3] * tempMap[4] * tempMap[5] +tempMap[6] * tempMap[7] * tempMap[8] 
        # 斜めの積
        cross =  tempMap[0] * tempMap[4] * tempMap[8] + tempMap[2] * tempMap[4] * tempMap[6]
        done = virtical + horizon + cross > 0
        return done 
```
#### 1.1.4 盤面を進めるstep関数
- 実装例
```
    def step(self, action):
        #ここから return self.Map, reward, done までコードを書きなさい
        reward = 0.0
        done = True
        # Agentが打ちます
        self.Map[action] = 1
        # 先手(Agent)が勝ちの場合
        if self.terminal(self.Map, 1):
            reward = 1.0
        #「勝ちが決まらない」かつ「9マス埋まっている」場合、引き分けで終了
        elif np.count_nonzero(Map) == 9:
            reward = -0.5
        # 先手が打って終了しなかった場合、Enemyが打ちます
        else:
            # get_action_Enemy
            takable_actions = np.where(self.Map == 0)[0]
            enemy_action = np.random.choice(takable_actions, 1)
            # enemy_action = 1
            self.Map[enemy_action] = 2
            if self.terminal(self.Map, 2):   
                reward = -1.0
            else: 
                done = False
        return self.Map, reward, done
```
#### 1.1.5 1エピソードを動かす
- 実装例
```
# 初期設定
episodes = 1

# インスタンス作成
env=Environment()

for num_episode in range(episodes):
    Map = env.start()
    print("",Map[0:3],"\n",Map[3:6],"\n",Map[6:9])
    done = False
    step = 1
    while not done:
        # actionをここで指定します。最終的にはactionを計算して返す関数に置き換えます
        action = 2*(step-1)
        new_Map,reward,done = env.step(action)
        Map = np.copy(new_Map)
        print("→",Map[0:3],"\n ",Map[3:6],"\n ",Map[6:9], "報酬", reward,"終了", done)
        step += 1
```
#### 1.1.6 Enemyの打つ場所を指定するget_action_Enemy関数
- 実装例
    - 25% でランダム
    - 75% で、
        - 勝てるなら、勝つ手を選ぶ
        - 負けそうなら、負けない手を選ぶ
        - 上記に当てはまらないならランダム
```
    def get_action_Enemy(self,Map):
        # 空白を取得するコードを書いてください
        empty_Map_list =np.where(self.Map == 0)[0]
        
        if np.random.rand() <= 0.25:
            action = np.random.choice(empty_Map_list, 1)
        else:
            #Agentが次の一手を打った場合に勝利するかどうか
            done_future_list1 = []
            #Enemyが次の一手を打った場合に勝利するかどうか
            done_future_list2 = []
            for j in range(len(empty_Map_list)):
                future_Map1 = np.copy(Map)
                future_Map2 = np.copy(Map)
                future_Map1[empty_Map_list[j]] = 1
                future_Map2[empty_Map_list[j]] = 2                
                done_future_list1.append(self.terminal(future_Map1,num_player=1))
                done_future_list2.append(self.terminal(future_Map2,num_player=2))
            done_future_list1 = np.array(done_future_list1)  
            done_future_list2 = np.array(done_future_list2) 
            
            # done_future_list2にTrueが存在する場合
            if len(np.where(done_future_list2==True)[0]) > 0:
                # done_future_list2からTrueのactionを抜き出し
                temp_list = np.where(done_future_list2 == True)[0]
                action = empty_Map_list[np.random.choice(temp_list)]
            # done_future_list1にTrueが存在する場合
            elif len(np.where(done_future_list1==True)[0]) > 0:
                # done_future_list1からTrueのactionを抜き出し
                temp_list = np.where(done_future_list1 == True)[0]
                action = empty_Map_list[np.random.choice(temp_list)]
            else:
                action = np.random.choice(empty_Map_list, 1)
            
        return action
```
### 1.2 メイン部分
#### 1.2.1 メイン部分の実装
## 2.Agentクラスを作成する
### 2.1 準備編
#### 2.1.1 行動選択のget_action関数
- 実装例
```
    def get_action(self,Map):
        # 空白取得
        empty_Map_list = np.where(Map == 0)[0]
        #actionを空白からランダムで選択するようにコードを書いてください
        action = np.random.choice(empty_Map_list, 1)
        
        return action
```
#### 2.1.2 入力に用いるstateを準備
- ニューラルネットワークの入力値には、盤(Map)をそのまま用いるのではなく、直近の過去2回分とまとめた3回分のMapをさらに変換したものを用いる
    - 直近3回分のMapを古い順に並べた27要素の配列をstate変数に格納
    - 1エピソード開始時には過去のMapが存在しないので、過去のMapとして空白の配列を代用
- 実装例
```
# 初期設定
np.random.seed(0)
episodes = 1

# インスタンス作成
env=Environment()
agent=Agent()

for num_episode in range(episodes):
    Map = env.start()
    print("",Map[0:3],"\n",Map[3:6],"\n",Map[6:9])
    # stateの作成を行ってください
    state = np.append(Map,Map,axis=0)
    state = np.append(state,Map,axis=0)
    print(state[0:9],state[9:18],state[18:27])
    done = False
    step = 1
    while not done:
        # actionをここで指定します
        action = agent.get_action(Map)
        new_Map,reward,done = env.step(action)
        # next_stateの更新を行ってください
        next_state = np.append(state[9:27],new_Map, axis=0)
        Map = np.copy(new_Map)
        print("→",Map[0:3],"\n ",Map[3:6],"\n ",Map[6:9], "報酬", reward,"終了", done)
        state = np.copy(next_state)
        print(state[0:9],state[9:18],state[18:27])
        step += 1
```
#### 2.1.3 run関数およびtransform_state関数、初期化関数（init）を作成
- 実装例
```
    def run(self, state, action, reward, next_state, done):
        self.state = self.transform_state(state)
        self.next_state = self.transform_state(next_state)
        empty_Map = np.where(Map == 0)[0]
        
        # Replay Memory用データの保存
        self.replay_memory.append((self.state, action, reward, self.next_state, done, empty_Map)) 
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            # .popleft : deque の左側から要素をひとつ削除し、その要素を返します
            self.replay_memory.popleft()
        print("Replay Memory :", self.replay_memory[0])
        
        
    def transform_state(self, state):
        temp = np.copy(state)
        ret = np.array([])
        # ここから return temp までのコードを書いてください
        for k in range(27):
            # 0は00に分解する
            if temp[k] == 0:
                ret = np.append(ret, [0,0], axis=0)
            # 1は01に分解する
            if temp[k] == 1:
                ret = np.append(ret, [0,1], axis=0)
            # 2は10に分解する
            if temp[k] == 2:
                ret = np.append(ret, [1,0], axis=0)
        return ret
```
### 2.2 ニューラルネットワークによる学習
#### 2.2.1 QネットワークとTarget Qネットワークを作成、初期化、学習する
- colab添付
#### 2.2.2 ε-greedy法による行動選択
- 最初にQ値の高い行動を見つけてしまうと、その行動のみを取り続けてしまう
- それを防ぐためにε-greedy方を実装
- 実装例はcolab添付