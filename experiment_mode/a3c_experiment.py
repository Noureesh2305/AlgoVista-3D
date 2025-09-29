# experiment_mode/a3c_experiment.py

import streamlit as st
import plotly.graph_objects as go

def render():
    st.subheader("🧠 A3C – Asynchronous Advantage Actor-Critic (Experiment Mode)")

    st.markdown("""
    **A3C** is an advanced reinforcement learning algorithm that:
    
    - Uses **multiple agents** running in **parallel environments**.
    - Updates a shared **global network** asynchronously.
    - Uses **advantage estimation** to improve training stability and convergence.

    A3C is highly efficient and faster than traditional deep RL methods like DQN.
    """)

    agent_count = st.slider("Number of Parallel Agents", min_value=2, max_value=8, step=1, value=4)

    # Visualizing multiple agents and the global network
    fig = go.Figure()

    for i in range(agent_count):
        fig.add_trace(go.Scatter3d(
            x=[i], y=[i % 2], z=[i // 2],
            mode='markers+text',
            text=[f"Agent {i+1}"],
            textposition="top center",
            marker=dict(size=8, color='green'),
            name=f"Agent {i+1}"
        ))

    # Global Network Center Node
    fig.add_trace(go.Scatter3d(
        x=[agent_count / 2], y=[1], z=[1],
        mode='markers+text',
        text=["Global Net"],
        textposition="top center",
        marker=dict(size=12, color='red'),
        name="Global Network"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=500
    )

    st.plotly_chart(fig)

    st.markdown(f"✅ **{agent_count} parallel agents** connected to a shared global policy network.")
