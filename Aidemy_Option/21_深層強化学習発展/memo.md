# メモ
## 1.強化学習概論
### 1.1 強化学習概論
#### 1.1.1 はじめに
- 強化学習: 対象の問題について不完全な知識しかなく、また、対象へのはたらきかけによってその結果が変化する場合に、最適なはたらきかけ方を見つける問題
#### 1.1.2 強化学習の実例
- AlphaGo Zero: 人間の棋譜データを用いた教師あり学習を全く使わずに、深層強化学習のみで囲碁の攻略法を学習することで、人間を超えた囲碁の人工知能を実現
    - 囲碁のような試合やゲームに対して学習させるのに、強化学習は相性が良い
- Preferred Networks社とトヨタが開発した深層強化学習による自動駐車の取り組み
    - 強化学習は機械の自動制御問題とも相性が良い
    - この例では車載カメラによる車の主観画像と車の現在のスピードとステアリングの3つを環境情報とし、いかに車が長く駐車スペースの内側にいるかを報酬として学習
#### 1.1.3 強化学習の構成要素
- エージェント: 行動する主体
- 環境: はたらきかける対象
- 行動: エージェントが環境に対して行うはたらきかけ
- 状態: 行動によって変化する環境の各要素
- 報酬: 環境に対して働きかけて、即時に得られる評価
- 収益: 最終的に合計でどれくらいの報酬を得られたのか
#### 1.1.4 知識利用と探索のジレンマ
- 強化学習における行動の選択
    - エージェントは、選択した行動の結果を観測して次の行動を決める
    - このとき、高い報酬を獲得するように行動を選択する
    - そこでエージェントの行動選択の方策を、現在の環境の状態を入力とし、行動を出力とする関数の形式で表現する
- 知識利用と探索のジレンマ
    - 探索：知識を集めること
    - 利用：持っている知識を利用すること
    - 最適な行動を取りたいが、そのためには最適ではない可能性のある行動を探索のために行う必要がある
    - 短時間で少ない報酬を獲得するか、長い時間をかけて高い報酬を獲得するかの二者択一
### 1.2 強化学習の手法
#### 1.2.1 ε-greedy方策
- greedy方策
    - 選択した行動による報酬の期待値が既知なのであれば、期待値が最大となる行動を選択し続けること
    - しかし、一般的に最初から得られる報酬の期待値が既知の場合はほとんどない
- ε-greedy方策
    - 探索のための行動を追加したもの
    - 小さな値εの確率で、全ての行動から同じ確率で選択をし（探索）、それ以外の(1-ε)の確率で現時点で最も報酬の期待値が高い行動を選択(利用)
    - 試行回数と比例させてεを少しずつ減らしていくことで、より効率的に探索する
#### 1.2.2 ボルツマン選択
- 行動の選択確率がボルツマン分布に従う
- ソフトマックス方策とも
#### 1.2.3 DQN
- Q学習のQ関数を深層学習で表した方法
- Q学習
    - Q関数と呼ばれる行動価値関数を推定する強化学習アルゴリズム
    - 行動価値関数を推定することができれば最適方策もわかる
        - 行動価値関数: 入力として状態sで行動aをとり、その後は最適方策に従った時に得られる報酬の期待値を計算する関数
            1. 試しにある行動を一回行う
            2. 実際に得られた行動価値と、ひとつ先の状態で可能な行動をとりあえず行い、それで得られる行動価値の合算を観測
            3. 今の行動価値との差で、少しだけ（学習率で調整して）更新してみる
        - Q関数は状態sと行動aの全ての組み合わせを表したテーブル関数で表現される
            - 問題設定によってはこのサイズが非常に膨大となり、(s,a)に対応する値を全て保存しておく必要があるため、メモリの容量が足りなくなってしまう問題がある
    - DQNではこのQ関数を多層のニューラルネットワークで関数近似することで、Q関数のサイズ爆発問題を解決
- DQNの特徴
    - Experience Replay: データの時系列をシャッフルすることで、時系列の相関に対処。
    - Target Network: 正解との誤差を計算し,モデルが正解に近くなるように調整。データからランダムにバッチを作成し、バッチ学習。
    - CNN(畳み込みニューラルネットワーク)の導入: 画像特徴量を採用し、異なるゲームにおいても同じフレームでの学習が可能。
    - 報酬のclipping: 報酬を負なら−1、正なら+1、なしは0で固定。
#### 1.2.4 履歴のメモリ
- エージェントが経験した入力(s, a, r, s’)を全て、もしくは有限個数記録しておき、記録したサンプルをランダムで呼び出して学習に利用する手法
- これによって、時系列の入力値における強い相関性を取り除き、学習の収束性をよくしている
## 2. 強化学習の実装
### 2.1 強化学習の実装
#### 2.1.1 環境の作成
- TF-Agents: 強化学習のライブラリ
- Open AI Gym（以降、Gym）: 強化学習用の様々な環境を用意しているライブラリ
    - 環境の作成: `suite_gym.load(環境種別)`
    - 行動数の取得: `env.action_space.n`
    - 他の引数やAPIなど
        - https://gym.openai.com/envs/#classic_control
        - https://github.com/openai/gym/wiki
#### 2.1.2 モデルの構築
- TF-Agentsによるモデルの作成: `q_network.QNetwork(input_tensor_spec, action_spec, fc_layer_params)`
    - input_tensor_spec: 入力形式
    - action_spec: 出力(行動)形式
    - fc_layer_params: 全結合層の各ユニット数を指定(タプルで指定)
#### 2.1.3 エージェントの設定
- TF-Agentsによるエージェントの作成: `tf_agent = dqn_agent.DqnAgent(time_step_spec, action_spec, q_network, optimizer)`
    - time_step_spec: 状態や報酬などの形式
    - action_spec: 行動の形式
    - q_network: 多層ニューラルネットワークのモデル
    - optimizer: 最適化アルゴリズム
#### 2.1.4 履歴の設定
- TF-Agentsによる履歴の作成: `replay_buffer, batch_size, max_length)`
    - data_spec: 記憶する経験の形式
    - batch_size: 学習時に一度に得る経験の数
    - max_length: 記憶しておく経験の容量
#### 2.1.5 テストの実施
- 学習の実施: `train_loss = tf_agent.train(experience)`
- 評価: `compute_avg_return(eval_env, tf_agent.policy, episode)`
    - eval_env: 評価用の環境
    - policy: エージェントの方策
    - episode: どのエピソードを実行して評価するか
- 実装例: 
```
import numpy as np
from tensorflow.compat.v1.train import AdamOptimizer
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# 訓練用の環境と評価用の環境の作成
ENV_NAME = 'CartPole-v0'
env = suite_gym.load(ENV_NAME)
train_env = tf_py_environment.TFPyEnvironment(env)
eval_py_env = suite_gym.load(ENV_NAME)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# 多層ニューラルネットワーク(Qネットワーク)の構築
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(16,16,16))


# DQNエージェントの作成
tf_agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=AdamOptimizer(learning_rate=1e-3),
    td_errors_loss_fn=common.element_wise_squared_loss)
tf_agent.initialize()

# 履歴の構築
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
   data_spec=tf_agent.collect_data_spec,
   batch_size=train_env.batch_size,
   max_length=100000)


# 評価値を計算する関数です
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

# 経験を集めてリプレイバッファに蓄積する関数です
def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

# 初めにランダムな方策を実行してリプレイバッファに蓄積します
initial_collect_steps=1000
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy)
dataset = replay_buffer.as_dataset(
   num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)
iterator = iter(dataset)

num_eval_episodes = 5  
eval_interval = 1000 
num_iterations = 20000
# 訓練前にモデルを評価します
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]
for step in range(1, num_iterations+1):
    # エージェントが環境(CartPole-v0)と相互作用して得た経験をリプレイバッファに追加します
    collect_step(train_env, tf_agent.collect_policy)
    # リプレイバッファから経験を取り出して学習を行います
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)
    # step数に応じてモデルの評価(テスト)を行います
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
```
## 3.卓球ゲームによる強化学習実践
### 3.1 卓球ゲームによる強化学習実践
#### 3.1.1 環境の作成
- 今回は atari の卓球をターゲットにする
- `env = suite_atari.load('PongNoFrameskip-v4', gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)`
    - 学習しやすくするために状態を210×160画素の1枚のカラー画像から84×84画素の連続する4枚のグレースケール画像に変更するオプション引数`gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING` を使用
#### 3.1.2 層の作成
#### 3.1.3 エージェントの設定
#### 3.1.4 履歴の設定
#### 3.1.5 学習の実施
#### 3.1.6 テストの実施
- ここまでの実装例
```
import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym, suite_atari
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step 

# 訓練用の環境と評価用の環境の作成
ENV_NAME = 'PongNoFrameskip-v4'
env = suite_atari.load(
    ENV_NAME,
    gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
train_py_env = suite_atari.load(
    ENV_NAME,
    gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
eval_py_env = suite_atari.load(
    ENV_NAME,
    gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

class Norm_pixel(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs/255

# 多層ニューラルネットワーク(Qネットワーク)の構築
fc_layer_params = (512,)
conv_layer_params = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
q_net = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            preprocessing_layers=Norm_pixel(),
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params)
optimizer = tf.compat.v1.train.RMSPropOptimizer(
    learning_rate=2.5e-4,
    decay=0.95,
    momentum=0.0,
    epsilon=1e-2)

time_step_spec = time_step.time_step_spec(train_env.observation_spec())
action_spec = tensor_spec.from_spec(train_env.action_spec())

# DQNエージェントの作成
tf_agent = dqn_agent.DqnAgent(
    time_step_spec,
    action_spec,
    q_network=q_net,
    optimizer=optimizer)
tf_agent.initialize()
# 履歴の構築
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
   data_spec=tf_agent.collect_data_spec,
   batch_size=train_env.batch_size,
   max_length=1_000) # 1_000_000


# 評価値を計算する関数です
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

# 経験を集めてリプレイバッファに蓄積する関数です
def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

# 初めにランダムな方策を実行してリプレイバッファに蓄積します
initial_collect_steps = 10 # 50_000
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy)
dataset = replay_buffer.as_dataset(
   num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)
iterator = iter(dataset)

num_eval_episodes = 1 # 5
log_interval = 10 # 100_000
eval_interval = 50 # 100_000
num_iterations = 50 # 5_000_000
tf_agent.train = common.function(tf_agent.train)
returns = []
for step in range(1, num_iterations+1):
    # エージェントが環境と相互作用して得た経験をリプレイバッファに追加します
    collect_step(train_env, tf_agent.collect_policy)
    # リプレイバッファから経験を取り出して学習を行います
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)
    # step数に応じてモデルのロスを出力します
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
    # step数に応じてモデルの評価(テスト)を行います。
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
```
#### 3.1.7 Dueling DQNとは
- DQNを少し発展させたもので、DQNのネットワーク層の最後を変更したもの
- DQNではconvolution層３つのあとに全結合層を経て出力のQ値へつながる
- Dueling DQNではconv３層のあと２つの全結合に分かれ、一方が状態価値Vの算出、もう一方が行動Aの算出となり、それらからQ値を求める