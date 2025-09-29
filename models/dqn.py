# models/dqn.py

import numpy as np
from collections import deque
import random
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


class DQNAgent:
    def __init__(self, state_size=5, action_size=3, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        if next_state is not None and state is not None:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=16):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if next_state is not None and not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                q_future = self.model.predict(next_state, verbose=0)[0]
                target = reward + self.gamma * np.amax(q_future)

            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=100):
        rewards = []
        for _ in range(episodes):
            state = np.random.rand(1, self.state_size)
            total_reward = 0
            for _ in range(10):  # 10 steps per episode
                action = self.act(state)
                next_state = np.random.rand(1, self.state_size)
                reward = np.random.randn()
                done = np.random.rand() < 0.1

                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            self.replay()
            rewards.append(total_reward)

        return rewards
