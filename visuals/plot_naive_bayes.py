# visuals/plot_naive_bayes.py

import streamlit as st
import plotly.express as px
import pandas as pd
from models.naive_bayes import train_naive_bayes_model

def render():
    st.subheader("📌 Naive Bayes – Visualization & Theory")

    X_train, X_test, y_train, y_pred, model = train_naive_bayes_model()

    df_train = pd.DataFrame(X_train, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_train['Label'] = y_train

    df_test = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_test['Predicted'] = y_pred

    st.markdown("### 🎯 Training Data")
    fig_train = px.scatter_3d(df_train, x='Feature 1', y='Feature 2', z='Feature 3',
                              color=df_train['Label'].astype(str),
                              title="Naive Bayes - Training Data")
    st.plotly_chart(fig_train, use_container_width=True)

    st.markdown("### 🤖 Predictions on Test Data")
    fig_test = px.scatter_3d(df_test, x='Feature 1', y='Feature 2', z='Feature 3',
                             color=df_test['Predicted'].astype(str),
                             title="Naive Bayes - Predictions")
    st.plotly_chart(fig_test, use_container_width=True)

    # 📘 Theory
    st.markdown("### 📘 Naive Bayes – Theory")
    st.markdown("""
**Naive Bayes** is a **supervised learning** algorithm based on **Bayes' Theorem** with a strong assumption that features are independent.

#### 📎 Key Concepts:
- Calculates **posterior probability** for each class given input features.
- Chooses class with highest probability.

#### 🧠 Bayes' Theorem:
`P(Class | Data) = (P(Data | Class) * P(Class)) / P(Data)`

- Assumes:
  - Features are **independent**
  - Each feature contributes **equally and independently** to the outcome

#### ✅ Advantages:
- Extremely fast
- Works well with **text data** and **high dimensions**
- Requires small training data

#### ⚠️ Limitations:
- Assumes feature independence (not realistic)
- Performs poorly when features are correlated
""")
