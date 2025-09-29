 # experiment_mode/kmeans_experiment.py

import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

def render():
    st.subheader("📊 K-Means Clustering – Experiment Mode")

    st.markdown("""
    **K-Means** is a centroid-based clustering algorithm that:
    
    - Divides data into **K non-overlapping clusters**.
    - Minimizes the **intra-cluster variance**.
    - Iteratively updates centroids until convergence.

    You can explore how the number of clusters and dimensions affect the results.
    """)

    # Parameter inputs
    n_samples = st.slider("Number of Samples", 50, 500, 150, step=10)
    n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
    n_features = st.selectbox("Feature Dimensions", [2, 3])
    random_state = st.slider("Random State", 0, 100, 42)

    # Generate synthetic data
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=random_state)

    # KMeans clustering
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    # Parameters Summary
    st.markdown(f"""
    ✅ **Experiment Settings**
    - Samples: `{n_samples}`
    - Clusters (K): `{n_clusters}`
    - Dimensions: `{n_features}D`
    """)

    # Plot
    if n_features == 3:
        fig = go.Figure()
        for label in np.unique(labels):
            cluster = X[labels == label]
            fig.add_trace(go.Scatter3d(
                x=cluster[:, 0], y=cluster[:, 1], z=cluster[:, 2],
                mode='markers',
                marker=dict(size=4),
                name=f'Cluster {label + 1}'
            ))

        # Add centroids
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
            mode='markers+text',
            marker=dict(size=6, color='black'),
            text=[f'C{i}' for i in range(len(centroids))],
            name="Centroids"
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
        for label in np.unique(labels):
            cluster = X[labels == label]
            fig.add_trace(go.Scatter(
                x=cluster[:, 0], y=cluster[:, 1],
                mode='markers',
                name=f'Cluster {label + 1}'
            ))

        # Add centroids
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers+text',
            text=[f'C{i}' for i in range(len(centroids))],
            marker=dict(size=8, color='black'),
            name="Centroids"
        ))

        fig.update_layout(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            height=500
        )
        st.plotly_chart(fig)
