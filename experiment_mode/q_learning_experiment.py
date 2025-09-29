# experiment_mode/q_learning_experiment.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go

def render():
    st.subheader("🧠 Q-Learning – Experiment Mode")

    st.markdown("""
    **Q-Learning** is a model-free reinforcement learning algorithm that learns the value of actions in states.
    
    You can experiment with:
    - Learning rate (α)
    - Discount factor (γ)
    - Exploration rate (ε)
    - Number of episodes

    The goal is to learn the optimal policy for reaching the goal in a simple grid.
    """)

    # Controls
    alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
    gamma = st.slider("Discount Factor (γ)", 0.1, 1.0, 0.9)
    epsilon = st.slider("Exploration Rate (ε)", 0.0, 1.0, 0.1)
    episodes = st.slider("Number of Episodes", 100, 2000, 500, step=100)

    # Grid environment (4x4)
    n_states = 16
    n_actions = 4
    q_table = np.zeros((n_states, n_actions))

    # Transition table: up, right, down, left
    def get_next_state(state, action):
        row, col = divmod(state, 4)
        if action == 0 and row > 0: row -= 1
        elif action == 1 and col < 3: col += 1
        elif action == 2 and row < 3: row += 1
        elif action == 3 and col > 0: col -= 1
        return row * 4 + col

    # Reward function
    goal_state = 15
    rewards = np.full(n_states, -1)
    rewards[goal_state] = 10

    # Q-Learning algorithm
    for _ in range(episodes):
        state = np.random.randint(0, n_states)
        while state != goal_state:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = np.argmax(q_table[state])
            next_state = get_next_state(state, action)
            reward = rewards[next_state]
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state

    # Show learned policy
    arrows = ['↑', '→', '↓', '←']
    policy_grid = []
    for s in range(n_states):
        if s == goal_state:
            policy_grid.append("🏁")
        else:
            best_action = np.argmax(q_table[s])
            policy_grid.append(arrows[best_action])

    grid_text = ""
    for i in range(4):
        grid_text += " | ".join(policy_grid[i*4:(i+1)*4]) + "\n" + "-"*15 + "\n"

    st.text("Learned Policy:\n\n" + grid_text)

    # 3D Visualization of Q-values
    fig = go.Figure()
    for state in range(n_states):
        for action in range(n_actions):
            fig.add_trace(go.Scatter3d(
                x=[state],
                y=[action],
                z=[q_table[state, action]],
                mode='markers',
                marker=dict(size=5, color=q_table[state, action], colorscale='Viridis'),
                name=f"S{state}-A{action}"
            ))

    fig.update_layout(
        title="Q-Table Visualization",
        scene=dict(
            xaxis_title='State',
            yaxis_title='Action',
            zaxis_title='Q-Value'
        ),
        height=600
    )
    st.plotly_chart(fig)
