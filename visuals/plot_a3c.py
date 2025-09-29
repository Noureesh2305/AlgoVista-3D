# visuals/plot_a3c.py

import streamlit as st
import plotly.graph_objects as go
from models.reinforcement.a3c import train_a3c

def render():
    st.header("🤖 A3C – Asynchronous Advantage Actor-Critic")

    st.markdown("### 🧪 Simulation Concept (3D View of Parallel Agents)")

    fig = go.Figure()

    # Simulate 4 parallel agents
    for i in range(4):
        fig.add_trace(go.Scatter3d(
            x=[i], y=[i % 2], z=[i // 2],
            mode='markers+text',
            text=[f"Agent {i+1}"],
            textposition="top center",
            marker=dict(size=8, color='green'),
            name=f"Agent {i+1}"
        ))

    # Global shared network
    fig.add_trace(go.Scatter3d(
        x=[1.5], y=[1], z=[1],
        mode='markers+text',
        text=["Global Network"],
        textposition="top center",
        marker=dict(size=14, color='red'),
        name="Global Network"
    ))

    fig.update_layout(
        title="A3C Concept – Multiple Agents & Shared Global Net",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # 📘 Theory Section
    with st.expander("📘 A3C – Theory & Key Concepts"):
        st.markdown("""
### 🎯 What is A3C?

A3C stands for **Asynchronous Advantage Actor-Critic**. It's an advanced **reinforcement learning** algorithm that improves on traditional Actor-Critic methods by:

- Using **multiple agents in parallel environments**
- Updating a **shared global policy and value network**
- Performing **asynchronous updates** (non-blocking)

---

### 🧠 Core Concepts

- **Actor**: Learns the policy (decides what action to take).
- **Critic**: Learns the value function (estimates how good the current state is).
- **Advantage Function**: Helps reduce the variance of updates by comparing action quality vs expected.

---

### ⚡ Why Asynchronous?

- Speeds up training by using **multiple threads**
- Reduces training variance
- Improves exploration and generalization

---

### ⚠️ Note:

Due to its parallel nature and complexity, A3C is typically implemented in **TensorFlow or PyTorch** with multiprocessing. It’s not suitable for simple Streamlit demos but is shown here for conceptual clarity.
""")
