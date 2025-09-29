# experiment_mode/actor_critic_experiment.py

import streamlit as st
import plotly.graph_objects as go

def render():
    st.subheader("🎭 Actor-Critic (Experiment Mode)")

    st.markdown("""
    **Actor-Critic** combines the benefits of both:
    
    - The **Actor**, which selects actions based on a policy.
    - The **Critic**, which evaluates the value of actions using a value function.

    This dual structure helps stabilize learning and improves performance compared to value-only or policy-only methods.
    """)

    learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, step=0.0005, value=0.001, format="%.4f")
    gamma = st.slider("Discount Factor (γ)", min_value=0.80, max_value=0.99, step=0.01, value=0.95)

    st.markdown(f"""
    ✅ Current Configuration:
    - **Learning Rate:** {learning_rate}
    - **Discount Factor γ:** {gamma}
    """)

    fig = go.Figure()

    # Actor Node
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        text=["Actor"],
        textposition="top center",
        marker=dict(size=10, color='blue'),
        name="Actor"
    ))

    # Critic Node
    fig.add_trace(go.Scatter3d(
        x=[1], y=[1], z=[1],
        mode='markers+text',
        text=["Critic"],
        textposition="top center",
        marker=dict(size=10, color='orange'),
        name="Critic"
    ))

    # Environment
    fig.add_trace(go.Scatter3d(
        x=[0.5], y=[0.5], z=[1.5],
        mode='markers+text',
        text=["Env"],
        textposition="top center",
        marker=dict(size=10, color='green'),
        name="Environment"
    ))

    # Arrows (can be logical)
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig)

    st.success("This simple 3D structure helps understand the Actor-Critic relationship during RL training.")
