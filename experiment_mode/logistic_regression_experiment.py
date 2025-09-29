# experiment_mode/logistic_regression_experiment.py

import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

def render():
    st.subheader("🔍 Logistic Regression – Experiment Mode")

    st.markdown("""
    **Logistic Regression** is used for binary or multiclass classification tasks. It models the probability that a given input belongs to a particular class.

    Customize parameters to observe model behavior:
    - Number of features
    - Class separation
    - Noise (flip labels)
    - Number of informative features
    """)

    # Parameter controls
    n_samples = st.slider("Number of Samples", 50, 1000, 300, step=50)
    n_features = st.selectbox("Number of Features (1 or 2 for plotting)", [1, 2])
    n_classes = st.selectbox("Number of Classes", [2, 3])
    class_sep = st.slider("Class Separation", 0.1, 2.0, 1.0, step=0.1)
    flip_y = st.slider("Noise (Label Flip Fraction)", 0.0, 0.5, 0.01)
    random_state = st.slider("Random State", 0, 100, 42)

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state
    )

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Accuracy on Test Set: {acc:.4f}")

    # Visualization
    st.markdown("### 📊 Classification Visualization")

    if n_features == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_test.flatten(), y=y_test,
            mode='markers', name='True Labels'
        ))
        fig.add_trace(go.Scatter(
            x=X_test.flatten(), y=y_pred,
            mode='markers', name='Predicted Labels'
        ))
        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Class",
            height=500
        )
        st.plotly_chart(fig)

    elif n_features == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=X_test[:, 0],
            y=X_test[:, 1],
            z=y_test,
            mode='markers',
            marker=dict(size=4),
            name='True'
        ))
        fig.add_trace(go.Scatter3d(
            x=X_test[:, 0],
            y=X_test[:, 1],
            z=y_pred,
            mode='markers',
            marker=dict(size=4),
            name='Predicted'
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Class'
            ),
            height=600
        )
        st.plotly_chart(fig)
