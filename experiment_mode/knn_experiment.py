# experiment_mode/knn_experiment.py

import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def render():
    st.subheader("📍 K-Nearest Neighbors (KNN) – Experiment Mode")

    st.markdown("""
    **KNN** is a simple, instance-based learning algorithm that:
    
    - Classifies a sample based on **majority vote** of its *K* nearest neighbors.
    - No training phase, predictions are made during inference.

    Explore how varying **K**, sample size, and feature dimensions affects the classifier.
    """)

    # Parameters
    n_samples = st.slider("Number of Samples", 100, 1000, 300, step=50)
    n_features = st.selectbox("Number of Features", [2, 3])
    n_classes = st.slider("Number of Classes", 2, 5, 3)
    k_neighbors = st.slider("K (Number of Neighbors)", 1, 15, 5)
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.3)
    random_state = st.slider("Random State", 0, 100, 42)

    # Generate data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train KNN
    model = KNeighborsClassifier(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    acc = model.score(X_test, y_test)
    st.success(f"✅ Accuracy on Test Set: {acc * 100:.2f}%")

    # Visualization
    st.markdown("### 🔍 3D/2D Visualization of Predictions")

    if n_features == 3:
        fig = go.Figure()

        for label in np.unique(y_pred):
            points = X_test[y_pred == label]
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                name=f'Class {label}',
                marker=dict(size=4)
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3'
            ),
            height=500
        )
        st.plotly_chart(fig)

    else:
        fig = go.Figure()

        for label in np.unique(y_pred):
            points = X_test[y_pred == label]
            fig.add_trace(go.Scatter(
                x=points[:, 0], y=points[:, 1],
                mode='markers',
                name=f'Class {label}',
                marker=dict(size=6)
            ))

        fig.update_layout(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            height=500
        )
        st.plotly_chart(fig)
