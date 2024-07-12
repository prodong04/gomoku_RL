import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from agents.agent_base import Agent

class DQNAgent(Agent):
    def __init__(self, name, board_size, learning_rate=0.001, discount_factor=0.99, epsilon=0.1, batch_size=64):
        super().__init__(name)
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def build_model(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.board_size * self.board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.board_size * self.board_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        empty_positions = np.argwhere(state == 0)
        if random.random() < self.epsilon:
            action = tuple(empty_positions[np.random.choice(len(empty_positions))])
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().detach().numpy().flatten()
            valid_q_values = q_values[np.ravel_multi_index(empty_positions.T, (self.board_size, self.board_size))]
            best_action_index = np.argmax(valid_q_values)
            action = tuple(empty_positions[best_action_index])
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        action_indices = np.array([action[0] * self.board_size + action[1] for action in actions])
        action_indices = torch.LongTensor(action_indices).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        q_value = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.discount_factor * next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, episodes, env, update_target_every=10):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                if "invalid" in info and info["invalid"]:
                    reward = -1
                    done = True
                self.store_transition(state, action, reward, next_state, done)
                self.replay()
                state = next_state
            if episode % update_target_every == 0:
                self.update_target_model()
