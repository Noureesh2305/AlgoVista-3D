# visuals/plot_actor_critic.py

import streamlit as st
import plotly.graph_objects as go
from models.reinforcement.actor_critic import train_actor_critic

def render():
    st.header("🎭 Actor-Critic Method – Reinforcement Learning")

    # Optional: Trigger training if user wants
    if st.button("Train Actor-Critic Model (Demo: CartPole-v1)"):
        with st.spinner("Training..."):
            train_actor_critic(episodes=100)
        st.success("✅ Training Complete (simplified for demo)")

    st.markdown("### 🎯 Theory – How Actor-Critic Works")

    st.markdown("""
**Actor-Critic** is a hybrid reinforcement learning technique that combines:

- 🧠 **Actor**: Chooses actions using a policy network.
- 🧮 **Critic**: Evaluates how good the current state-action is using a value function.

Together, they improve training **efficiency** and **stability**.

---

### 💡 Key Concepts

- **Policy Gradient** methods can be unstable on their own.
- **Value-Based** methods (like Q-learning) are stable but less flexible.
- Actor-Critic combines both: fast learning (actor) and stable evaluation (critic).

---

### ⚙️ Core Equation

The policy is updated via:  
`Loss_actor = -log(π(a|s)) * Advantage`

The critic learns using:  
`TD Target = r + γ * V(s')`

---
""")

    st.markdown("### 🧠 Interactive Visualization – Actor & Critic Feedback Loop")

    fig = go.Figure()

    # Actor Node
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        text=["Actor"],
        textposition="top center",
        marker=dict(size=10, color='red'),
        name="Actor"
    ))

    # Critic Node
    fig.add_trace(go.Scatter3d(
        x=[1], y=[1], z=[1],
        mode='markers+text',
        text=["Critic"],
        textposition="top center",
        marker=dict(size=10, color='blue'),
        name="Critic"
    ))

    # Feedback Loop (line)
    fig.add_trace(go.Scatter3d(
        x=[0, 1], y=[0, 1], z=[0, 1],
        mode="lines",
        line=dict(width=4, color='gray'),
        name="Feedback Loop"
    ))

    fig.update_layout(
        title="Actor-Critic: Cooperative Feedback System",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
