# visuals/plot_logistic_regression.py

import streamlit as st
import plotly.express as px
import pandas as pd
from models.logistic_regression import train_logistic_regression_model

def render():
    st.subheader("📌 Logistic Regression – Visualization & Theory")

    X_train, X_test, y_train, y_pred, model = train_logistic_regression_model()

    df_train = pd.DataFrame(X_train, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_train['Label'] = y_train

    df_test = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_test['Predicted'] = y_pred

    st.markdown("### 🎯 Training Data")
    fig_train = px.scatter_3d(df_train, x='Feature 1', y='Feature 2', z='Feature 3',
                              color=df_train['Label'].astype(str),
                              title="Logistic Regression - Training Data")
    st.plotly_chart(fig_train, use_container_width=True)

    st.markdown("### 🤖 Predictions on Test Data")
    fig_test = px.scatter_3d(df_test, x='Feature 1', y='Feature 2', z='Feature 3',
                             color=df_test['Predicted'].astype(str),
                             title="Logistic Regression - Predictions")
    st.plotly_chart(fig_test, use_container_width=True)

    # 📘 Theory Explanation
    st.markdown("### 📘 Logistic Regression – Theory")
    st.markdown("""
**Logistic Regression** is a **supervised classification** algorithm used for **binary and multi-class classification** problems.

#### 📎 Key Concepts:
- Models the **probability** that a given input belongs to a class.
- Uses the **sigmoid function** to map values between 0 and 1.
- Decision boundary based on probability threshold (usually 0.5).

#### 🧠 Equation:
- `P(y=1|x) = 1 / (1 + e^-(b0 + b1x1 + ... + bnxn))`

#### ✅ Advantages:
- Works well for linearly separable data
- Interpretable and simple
- Fast training

#### ⚠️ Limitations:
- Assumes linear decision boundary
- Can underperform with complex or non-linear data
- Sensitive to outliers and multicollinearity
""")
