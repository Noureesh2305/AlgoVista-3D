# visuals/plot_dqn.py

import streamlit as st
import plotly.graph_objects as go
from models.dqn import DQNAgent

def render():
    st.subheader("🔴 DQN – Deep Q-Network")
    
    st.markdown("""
    **DQN** is a Deep Reinforcement Learning algorithm that uses a neural network to approximate Q-values.

    It combines Q-learning with deep learning to handle high-dimensional state spaces.

    The core idea is:  
    **Q(s, a) ≈ NeuralNetwork(s)[a]**
    """)

    agent = DQNAgent()
    rewards = agent.train(episodes=100)

    st.markdown("📈 Below is the reward trend over episodes:")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rewards, mode='lines+markers', name='Episode Reward'))
    fig.update_layout(title="DQN Reward Trend", xaxis_title="Episode", yaxis_title="Total Reward")
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ DQN model trained and reward trend visualized!")
