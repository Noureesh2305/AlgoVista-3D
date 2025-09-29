# tutor/learning_intro.py

import streamlit as st
from tutor.voice_guide import speak_theory

def show_learning_type_selector():
    st.markdown("## 🔍 Choose the Type of Machine Learning")

    theory = """
    ### 🧠 Types of Machine Learning

    **1. Supervised Learning**  
    Trains on labeled data. The algorithm learns from input-output pairs.  
    📌 Examples: Linear Regression, Decision Trees, SVM, etc.

    **2. Unsupervised Learning**  
    Works on unlabeled data to find patterns or groupings.  
    📌 Examples: K-Means, PCA, DBSCAN, etc.

    **3. Reinforcement Learning**  
    The agent learns by interacting with the environment and receiving rewards.  
    📌 Examples: Q-Learning, DQN, SARSA, etc.
    """
    st.markdown(theory)

    if st.button("🔊 Explain Types via Voice"):
        speak_theory(theory)

    choice = st.selectbox(
        "👇 Select Learning Type",
        ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"]
    )
    return choice
