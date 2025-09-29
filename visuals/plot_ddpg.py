# visuals/plot_ddpg.py

import streamlit as st
import plotly.graph_objects as go

def render():
    st.subheader("🎯 DDPG – Deep Deterministic Policy Gradient")

    st.markdown("""
    **DDPG** is an advanced Reinforcement Learning algorithm tailored for **continuous action spaces**.

    - 🔁 Combines **Actor-Critic** structure.
    - 🧠 Uses **experience replay** and **target networks**.
    - 🦾 Ideal for **robotics** and **real-time control systems**.
    - 🎯 Unlike DQN (discrete actions), DDPG handles **high-dimensional continuous actions**.

    The **Actor** chooses what to do, while the **Critic** tells how good the decision was.
    """)

    fig = go.Figure()

    # Actor & Critic
    fig.add_trace(go.Scatter3d(
        x=[0, 1],
        y=[0, 1],
        z=[0, 0],
        mode='markers+text',
        text=["Actor", "Critic"],
        textposition="top center",
        marker=dict(size=10, color=['orange', 'blue']),
        name="DDPG Components"
    ))

    # Connection Line
    fig.add_trace(go.Scatter3d(
        x=[0, 1],
        y=[0, 1],
        z=[0, 0],
        mode='lines',
        line=dict(color='black', width=4),
        name="Interaction"
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=500
    )

    st.plotly_chart(fig)
