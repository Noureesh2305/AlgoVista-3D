# experiment_mode/sarsa_experiment.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go

def render():
    st.subheader("🚶 SARSA – Experiment Mode")

    st.markdown("""
    **SARSA** (State-Action-Reward-State-Action) is an on-policy reinforcement learning algorithm.
    
    It updates Q-values based on the action taken by the current policy, making it more conservative than Q-Learning.

    Use the sliders below to experiment with:
    - Learning Rate (α)
    - Discount Factor (γ)
    - Exploration Rate (ε)
    - Number of Episodes
    """)

    # Hyperparameter sliders
    alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, step=0.01)
    gamma = st.slider("Discount Factor (γ)", 0.1, 1.0, 0.9, step=0.1)
    epsilon = st.slider("Exploration Rate (ε)", 0.0, 1.0, 0.2, step=0.05)
    episodes = st.slider("Episodes", 100, 2000, 500, step=100)

    # Simple 4x4 GridWorld environment
    grid_size = 4
    goal_state = (3, 3)
    actions = ['up', 'down', 'left', 'right']
    q_table = np.zeros((grid_size, grid_size, len(actions)))

    def get_next_state(state, action):
        i, j = state
        if action == 'up' and i > 0: i -= 1
        elif action == 'down' and i < grid_size - 1: i += 1
        elif action == 'left' and j > 0: j -= 1
        elif action == 'right' and j < grid_size - 1: j += 1
        return (i, j)

    def choose_action(state):
        if np.random.rand() < epsilon:
            return np.random.choice(len(actions))
        else:
            i, j = state
            return np.argmax(q_table[i, j])

    rewards = []

    for ep in range(episodes):
        state = (0, 0)
        action_idx = choose_action(state)
        total_reward = 0
        while state != goal_state:
            next_state = get_next_state(state, actions[action_idx])
            next_action_idx = choose_action(next_state)
            i, j = state
            ni, nj = next_state

            reward = 1 if next_state == goal_state else -0.01
            q_table[i, j, action_idx] += alpha * (reward + gamma * q_table[ni, nj, next_action_idx] - q_table[i, j, action_idx])
            state = next_state
            action_idx = next_action_idx
            total_reward += reward
        rewards.append(total_reward)

    # Plot rewards over episodes
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rewards, mode='lines', name='Reward per Episode'))
    fig.update_layout(title="SARSA Learning Curve", xaxis_title="Episode", yaxis_title="Total Reward")
    st.plotly_chart(fig)

    st.markdown("✅ *Try adjusting α, γ, ε, and episodes to observe how SARSA performance changes over time!*")
