import numpy as np
import random
from agents.agent_base import Agent
from tqdm import tqdm
class QLearningAgent(Agent):
    def __init__(self, name, board_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(name)
        self.board_size = board_size
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(map(tuple, state))

    def select_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            empty_positions = np.argwhere(state == 0)
            action = tuple(empty_positions[np.random.choice(len(empty_positions))])
        else:
            q_values = self.q_table.get(state_key, {})
            if not q_values:
                action = self.select_random_action(state)
            else:
                # 무효한 행동 필터링
                valid_q_values = {k: v for k, v in q_values.items() if state[k[0], k[1]] == 0}
                if valid_q_values:
                    action = max(valid_q_values, key=valid_q_values.get)
                else:
                    action = self.select_random_action(state)
        return action

    def select_random_action(self, state):
        empty_positions = np.argwhere(state == 0)
        action = tuple(empty_positions[np.random.choice(len(empty_positions))])
        return action

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        old_value = self.q_table.get(state_key, {}).get(action, 0.0)
        future_rewards = max(self.q_table.get(next_state_key, {}).values(), default=0.0)
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * future_rewards - old_value)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action] = new_value

    def learn(self, episodes, env):
        for episode in tqdm(range(episodes)):
            state = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                if "invalid" in info and info["invalid"]:
                    reward = -1
                    done = True
                self.update_q_value(state, action, reward, next_state)
                state = next_state
