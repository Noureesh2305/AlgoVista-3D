import streamlit as st
import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras import layers
import gym
import plotly.graph_objs as go

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_dqn_experiment(episodes=10):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    rewards = []

    for e in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        rewards.append(total_reward)

    env.close()
    return rewards

def render():
    st.title("🧠 DQN – Deep Q Network (Experiment Mode)")
    st.markdown("""
    **DQN** combines Q-learning with deep neural networks.
    
    - Learns from experience replay
    - Uses epsilon-greedy policy for exploration
    - Targets are updated based on Bellman equation
    """)

    episodes = st.slider("Select number of training episodes", 5, 50, 10)

    if st.button("🚀 Run DQN Experiment"):
        with st.spinner("Training DQN agent..."):
            scores = run_dqn_experiment(episodes=episodes)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=scores,
            mode='lines+markers',
            name='Episode Rewards',
            line=dict(color='limegreen')
        ))
        fig.update_layout(
            title="DQN Training Progress",
            xaxis_title="Episode",
            yaxis_title="Reward",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("Change the number of episodes to observe different behaviors of the DQN agent.")

