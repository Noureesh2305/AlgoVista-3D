# visuals/plot_linear_regression.py

import streamlit as st
import plotly.express as px
import pandas as pd
from models.linear_regression import train_linear_regression_model

def render():
    st.subheader("📌 Linear Regression – Visualization & Theory")

    X_train, X_test, y_train, y_pred, model = train_linear_regression_model()

    # Train data plot
    df_train = pd.DataFrame(X_train, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_train['Target'] = y_train

    df_test = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    df_test['Prediction'] = y_pred

    # 3D plot of training data
    st.markdown("### 🎯 Training Data")
    fig_train = px.scatter_3d(df_train, x='Feature 1', y='Feature 2', z='Feature 3',
                              color='Target', title="Linear Regression - Training Data", opacity=0.7)
    st.plotly_chart(fig_train, use_container_width=True)

    # 3D plot of test predictions
    st.markdown("### 🤖 Predictions on Test Data")
    fig_test = px.scatter_3d(df_test, x='Feature 1', y='Feature 2', z='Feature 3',
                             color='Prediction', title="Linear Regression - Predictions", opacity=0.7)
    st.plotly_chart(fig_test, use_container_width=True)

    # 📘 Theory Explanation
    st.markdown("### 📘 Linear Regression – Theory")
    st.markdown("""
**Linear Regression** is a **supervised learning** algorithm used to predict a continuous target variable based on input features.

#### 📎 Key Points:
- Models the relationship as a **linear equation**:  
  `Y = b0 + b1*X1 + b2*X2 + ... + bn*Xn`
- Tries to **minimize the error** between predicted and actual values using **least squares**.
- Evaluated using metrics like **Mean Squared Error (MSE)** and **R² score**.

#### ✅ Advantages:
- Easy to implement
- Works well when the relationship is truly linear
- Fast and interpretable

#### ⚠️ Limitations:
- Assumes linearity
- Sensitive to outliers
- Poor performance with non-linear patterns
""")
