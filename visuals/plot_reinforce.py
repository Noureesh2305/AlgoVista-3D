# visuals/plot_reinforce.py

import streamlit as st
import plotly.graph_objs as go
from models.reinforcement.reinforce import train_reinforce

def render():
    st.subheader("🎯 REINFORCE – Monte Carlo Policy Gradient Method")
    
    st.markdown("""
    **REINFORCE** is a Policy Gradient method that:
    
    - Uses full episodes to compute gradients.
    - Learns a parameterized policy directly via a neural network.
    - Optimizes the probability of taking actions that yield higher cumulative rewards.
    
    Used in simple environments like **CartPole**, and forms the basis of many modern RL algorithms.
    """)

    episodes = st.slider("Select Training Episodes", 10, 200, 50)

    if st.button("🚀 Train REINFORCE Agent"):
        with st.spinner("Training in progress..."):
            scores = train_reinforce(episodes=episodes)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=scores,
            mode='lines+markers',
            name='Episode Reward',
            line=dict(color='purple')
        ))
        fig.update_layout(
            title="REINFORCE Training Rewards",
            xaxis_title="Episode",
            yaxis_title="Total Reward",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("Click the button above to simulate REINFORCE on CartPole-v1.")
