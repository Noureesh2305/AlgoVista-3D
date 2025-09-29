# experiment_mode/ddpg_experiment.py

import streamlit as st
import plotly.graph_objects as go

def render():
    st.subheader("🚗 DDPG – Deep Deterministic Policy Gradient (Experiment Mode)")

    st.markdown("""
    **DDPG** is a powerful Actor-Critic algorithm designed for **continuous action spaces**.  
    It combines:
    
    - **Actor**: Learns the policy function (which action to take).
    - **Critic**: Learns the Q-value function (how good an action is).
    - **Target Networks** and **Experience Replay** for training stability.

    DDPG is ideal for robotic control and continuous decision-making problems.
    """)

    learning_rate_actor = st.slider("Learning Rate (Actor)", 0.0001, 0.01, 0.001, step=0.0005, format="%.4f")
    learning_rate_critic = st.slider("Learning Rate (Critic)", 0.0001, 0.01, 0.001, step=0.0005, format="%.4f")
    tau = st.slider("Soft Update Rate (τ)", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
    gamma = st.slider("Discount Factor (γ)", 0.80, 0.99, 0.95, step=0.01)

    st.markdown(f"""
    ✅ **Current Hyperparameters:**
    - Actor LR: `{learning_rate_actor}`
    - Critic LR: `{learning_rate_critic}`
    - Soft Update Rate (τ): `{tau}`
    - Discount Factor γ: `{gamma}`
    """)

    # Visualization
    fig = go.Figure()

    # Actor
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        text=["Actor"],
        textposition="top center",
        marker=dict(size=10, color='blue'),
        name="Actor"
    ))

    # Critic
    fig.add_trace(go.Scatter3d(
        x=[1], y=[0], z=[0],
        mode='markers+text',
        text=["Critic"],
        textposition="top center",
        marker=dict(size=10, color='red'),
        name="Critic"
    ))

    # Environment
    fig.add_trace(go.Scatter3d(
        x=[0.5], y=[1], z=[0.5],
        mode='markers+text',
        text=["Environment"],
        textposition="top center",
        marker=dict(size=10, color='green'),
        name="Environment"
    ))

    # Replay Buffer
    fig.add_trace(go.Scatter3d(
        x=[1], y=[1], z=[1],
        mode='markers+text',
        text=["Replay Buffer"],
        textposition="top center",
        marker=dict(size=10, color='purple'),
        name="Replay Buffer"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig)
    st.success("This 3D diagram illustrates the flow between Actor, Critic, Environment, and Replay Buffer in DDPG.")
