import gymnasium as gym
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict

# 환경 설정
env = gym.make("Blackjack-v1", natural=False)

# SARSA 파라미터 설정
alpha = 0.1  # 학습률
gamma = 0.99  # 할인율
epsilon = 0.1  # 탐험율
num_episodes = 500000  # 에피소드 수

# Q 테이블 초기화
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

# SARSA 알고리즘
for episode in tqdm(range(num_episodes)):
    state, info = env.reset()
    action = choose_action(state, epsilon)
    done = False

    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        next_action = choose_action(next_state, epsilon)
        done = terminated or truncated

        if done:
            q_table[state][action] += alpha * (reward - q_table[state][action])
        else:
            td_target = reward + gamma * q_table[next_state][next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
        
        state = next_state
        action = next_action

# Q 테이블 저장
with open('q_table_agent2_sarsa.pkl', 'wb') as f:
    pickle.dump(dict(q_table), f)

env.close()
print("Training finished and Q-table saved.")
