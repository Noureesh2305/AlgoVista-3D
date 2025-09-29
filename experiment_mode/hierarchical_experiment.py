# experiment_mode/hierarchical_experiment.py

import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def render():
    st.subheader("🌳 Hierarchical Clustering – Experiment Mode")

    st.markdown("""
    **Hierarchical Clustering** builds a tree of clusters using either:
    
    - **Agglomerative (Bottom-Up)** – Start with individual points and merge them.
    - **Divisive (Top-Down)** – Start with one cluster and split it.

    It does not require the number of clusters beforehand and produces a dendrogram.
    """)

    n_samples = st.slider("Number of Samples", 20, 300, 100, step=10)
    n_features = st.selectbox("Feature Dimensions", [2, 3])
    n_clusters = st.slider("Number of Final Clusters", 2, 6, 3)
    linkage_method = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single'])

    # Generate synthetic data
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

    # Apply clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)

    st.markdown(f"""
    ✅ **Parameters Used:**
    - Samples: `{n_samples}`
    - Dimensions: `{n_features}D`
    - Clusters: `{n_clusters}`
    - Linkage: `{linkage_method}`
    """)

    # Plotting
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
        fig.update_layout(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            height=500
        )
        st.plotly_chart(fig)
