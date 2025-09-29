# experiment_mode/pca_experiment.py

import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

def render():
    st.subheader("📉 PCA – Experiment Mode")

    st.markdown("""
    **Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms data to a new coordinate system
    where the greatest variance lies on the first principal component, the second greatest on the second, and so on.

    You can interactively:
    - Choose the number of principal components
    - Visualize the transformation in 3D space
    """)

    # Load dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Controls
    n_components = st.slider("Number of Principal Components", 2, 64, 3)

    # PCA transformation
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    st.success(f"🔍 Explained Variance (first {n_components} components): {np.sum(explained_variance):.4f}")

    # Visualization (only if 2D or 3D)
    if n_components == 2:
        fig = go.Figure()
        for label in np.unique(y):
            idx = y == label
            fig.add_trace(go.Scatter(
                x=X_pca[idx, 0], y=X_pca[idx, 1],
                mode='markers',
                name=str(label),
                marker=dict(size=5),
            ))
        fig.update_layout(title="PCA 2D Projection", xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig)

    elif n_components >= 3:
        fig = go.Figure()
        for label in np.unique(y):
            idx = y == label
            fig.add_trace(go.Scatter3d(
                x=X_pca[idx, 0],
                y=X_pca[idx, 1],
                z=X_pca[idx, 2],
                mode='markers',
                name=str(label),
                marker=dict(size=3)
            ))
        fig.update_layout(scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ), height=600)
        st.plotly_chart(fig)
