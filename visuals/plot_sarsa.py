# visuals/plot_sarsa.py

import streamlit as st
import plotly.graph_objects as go
from models.sarsa import SARSALearner

def render():
    st.subheader("🔴 SARSA – State-Action-Reward-State-Action")
    
    st.markdown("""
    **SARSA** is an on-policy reinforcement learning algorithm that updates its Q-values based on the action actually taken.  
    It learns a policy while exploring the environment using the formula:
    
    \n**Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]**
    
    """)
    
    st.markdown("Here's a simple visualization of Q-values for each state-action pair:")

    agent = SARSALearner()
    q_table, history = agent.train(episodes=500)

    fig = go.Figure(data=[
        go.Surface(z=q_table, colorscale='Viridis')
    ])
    
    fig.update_layout(
        title='SARSA Q-Table Visualization',
        scene=dict(
            xaxis_title='State',
            yaxis_title='Action',
            zaxis_title='Q-Value'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ SARSA Q-values trained and visualized!")
