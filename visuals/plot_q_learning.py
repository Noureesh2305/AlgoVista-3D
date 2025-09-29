import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.q_learning import train_q_learning
from tutor.voice_guide import speak_theory

def plot_agent_path(path, grid_size):
    fig = go.Figure()

    # Draw all grid points
    for i in range(grid_size):
        for j in range(grid_size):
            fig.add_trace(go.Scatter3d(
                x=[i], y=[j], z=[0],
                mode='markers',
                marker=dict(size=4, color='lightgrey'),
                showlegend=False
            ))

    # Agent Path
    x_vals, y_vals = zip(*path)
    fig.add_trace(go.Scatter3d(
        x=x_vals, y=y_vals, z=[0.1] * len(path),
        mode='lines+markers',
        marker=dict(size=5, color='blue'),
        line=dict(color='blue', width=4),
        name="Agent Path"
    ))

    # Goal
    fig.add_trace(go.Scatter3d(
        x=[grid_size - 1], y=[grid_size - 1], z=[0.2],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Goal'
    ))

    fig.update_layout(
        title="Q-Learning Agent Path in 3D Grid World",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title=''),
            zaxis_range=[0, 1],
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def render():
    st.markdown("## 🤖 Q-Learning (Reinforcement Learning)")
    st.markdown("""
Q-Learning is a **model-free** reinforcement learning algorithm used to find the **optimal action-selection policy** for any given environment.

- Learns from **rewards**
- Uses a **Q-table** to estimate the utility of actions
- Balances **exploration** (trying new actions) and **exploitation** (choosing best-known actions)

#### 🔢 Q-Learning Formula:
Q(s, a) ← Q(s, a) + α [r + γ * max(Q(s', a')) - Q(s, a)]
Where:
- `α` is learning rate
- `γ` is discount factor
- `r` is reward received
- `s` is current state, `s'` is next state
""")

    # Experiment Mode Sliders
    st.sidebar.header("🎛️ Experiment Mode")
    alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1, step=0.01)
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.9, step=0.01)
    episodes = st.sidebar.slider("Episodes", 10, 1000, 200, step=10)

    Q, grid_size, best_path = train_q_learning(alpha=alpha, gamma=gamma, episodes=episodes)

    st.markdown("### 📊 Q-Table:")
    st.dataframe(Q)

    st.markdown("### 🌐 Agent Path in Grid World")
    fig = plot_agent_path(best_path, grid_size)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🔊 Voiceover Explanation"):
        speak_theory("Q-Learning")
