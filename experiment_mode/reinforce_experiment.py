# experiment_mode/reinforce_experiment.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

def train_reinforce(env_name='CartPole-v1', hidden_size=128, lr=0.01, gamma=0.99, episodes=300):
    env = gym.make(env_name)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = PolicyNetwork(obs_size, hidden_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    total_rewards = []

    for ep in range(episodes):
        state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        log_probs = []
        rewards = []
        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _, _ = env.step(action.item()) if len(env.step(action.item())) == 5 else (*env.step(action.item()), None)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            ep_reward += reward

        total_rewards.append(ep_reward)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Loss
        loss = []
        for log_prob, Gt in zip(log_probs, returns):
            loss.append(-log_prob * Gt)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env.close()
    return total_rewards

def render():
    st.subheader("🎯 REINFORCE – Experiment Mode")

    st.markdown("""
    **REINFORCE** is a policy gradient method in Reinforcement Learning. It updates the policy weights using rewards from complete episodes.
    
    Tune:
    - Hidden Layer Size
    - Learning Rate
    - Discount Factor
    - Number of Episodes
    """)

    # Sidebar controls
    hidden_size = st.slider("Hidden Layer Size", 16, 256, 128, step=16)
    learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.01, step=0.0005, format="%.4f")
    gamma = st.slider("Discount Factor (Gamma)", 0.80, 0.999, 0.99, step=0.01)
    episodes = st.slider("Training Episodes", 100, 1000, 300, step=50)

    if st.button("Train REINFORCE Agent"):
        with st.spinner("Training in progress..."):
            rewards = train_reinforce(hidden_size=hidden_size, lr=learning_rate, gamma=gamma, episodes=episodes)

            fig, ax = plt.subplots()
            ax.plot(rewards, label="Reward per Episode")
            ax.set_xlabel("Episodes")
            ax.set_ylabel("Total Reward")
            ax.set_title("REINFORCE Learning Curve")
            ax.legend()
            st.pyplot(fig)

        st.success("Training Complete ✅")
