# experiment_mode/linear_regression_experiment.py

import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def render():
    st.subheader("📈 Linear Regression – Experiment Mode")

    st.markdown("""
    **Linear Regression** models the relationship between a dependent variable and one or more independent variables by fitting a linear equation.

    Use the sliders below to experiment with:
    - Number of features (1 or 2 for plotting)
    - Number of samples
    - Noise in data
    - Test/train split
    """)

    # User parameters
    n_samples = st.slider("Number of Samples", 50, 1000, 300, step=50)
    n_features = st.selectbox("Number of Features (1 or 2 for plotting)", [1, 2])
    noise = st.slider("Noise Level", 0.0, 50.0, 10.0, step=1.0)
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.3)
    random_state = st.slider("Random State", 0, 100, 42)

    # Generate synthetic regression data
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2_score = model.score(X_test, y_test)
    st.success(f"✅ R² Score on Test Set: {r2_score:.4f}")

    # Visualization
    st.markdown("### 📊 Prediction Visualization")

    if n_features == 1:
        # 2D Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode='markers', name='True Values'))
        fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred, mode='markers', name='Predicted Values'))
        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Target",
            height=500
        )
        st.plotly_chart(fig)

    elif n_features == 2:
        # 3D Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=X_test[:, 0], y=X_test[:, 1], z=y_test,
            mode='markers',
            marker=dict(size=4),
            name='True'
        ))
        fig.add_trace(go.Scatter3d(
            x=X_test[:, 0], y=X_test[:, 1], z=y_pred,
            mode='markers',
            marker=dict(size=4),
            name='Predicted'
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Target'
            ),
            height=600
        )
        st.plotly_chart(fig)
