# models/sarsa.py

import numpy as np

class SARSALearner:
    def __init__(self, n_states=5, n_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        predict = self.q_table[state][action]
        target = reward + self.gamma * self.q_table[next_state][next_action]
        self.q_table[state][action] += self.alpha * (target - predict)

    def train(self, episodes=1000):
        history = []
        for _ in range(episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            episode_steps = []
            for _ in range(10):
                next_state = np.random.randint(self.n_states)
                reward = np.random.randn()
                next_action = self.choose_action(next_state)
                self.update(state, action, reward, next_state, next_action)
                episode_steps.append((state, action, reward))
                state, action = next_state, next_action
            history.append(episode_steps)
        return self.q_table, history
